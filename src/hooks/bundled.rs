//! Bundled hook implementations and declarative hook registration.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::hooks::{
    Hook, HookContext, HookError, HookEvent, HookFailureMode, HookOutcome, HookPoint, HookRegistry,
};

const DEFAULT_RULE_PRIORITY: u32 = 100;
const DEFAULT_WEBHOOK_PRIORITY: u32 = 300;
const DEFAULT_WEBHOOK_TIMEOUT_MS: u64 = 2000;
const MAX_HOOK_TIMEOUT_MS: u64 = 30_000;

const ALL_HOOK_POINTS: [HookPoint; 6] = [
    HookPoint::BeforeInbound,
    HookPoint::BeforeToolCall,
    HookPoint::BeforeOutbound,
    HookPoint::OnSessionStart,
    HookPoint::OnSessionEnd,
    HookPoint::TransformResponse,
];

/// Errors while parsing or compiling declarative hook bundles.
#[derive(Debug, thiserror::Error)]
pub enum HookBundleError {
    #[error("Invalid hook bundle format: {0}")]
    InvalidFormat(String),

    #[error("Hook '{hook}' must declare at least one hook point")]
    MissingHookPoints { hook: String },

    #[error("Hook '{hook}' has invalid regex '{pattern}': {reason}")]
    InvalidRegex {
        hook: String,
        pattern: String,
        reason: String,
    },

    #[error("Hook '{hook}' timeout must be between 1 and {max_ms} ms")]
    InvalidTimeout { hook: String, max_ms: u64 },

    #[error("Outbound webhook hook '{hook}' has invalid url: {url}")]
    InvalidWebhookUrl { hook: String, url: String },
}

/// A declarative hook bundle loaded from workspace files or extension capabilities.
///
/// Supports two bundled hook types:
/// - Rule hooks (`rules`) for reject/regex transform/prepend/append logic
/// - Outbound webhook hooks (`outbound_webhooks`) for fire-and-forget event delivery
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HookBundleConfig {
    /// Declarative content/tool/session rules.
    #[serde(default)]
    pub rules: Vec<HookRuleConfig>,
    /// Fire-and-forget webhook notifications on selected hook points.
    #[serde(default)]
    pub outbound_webhooks: Vec<OutboundWebhookConfig>,
}

impl HookBundleConfig {
    /// Parse a hook bundle from JSON value.
    ///
    /// Accepts either:
    /// - object form: `{ "rules": [...], "outbound_webhooks": [...] }`
    /// - array form:  `[ {rule}, {rule} ]` (shorthand for rules only)
    pub fn from_value(value: &serde_json::Value) -> Result<Self, HookBundleError> {
        if value.is_array() {
            let rules: Vec<HookRuleConfig> = serde_json::from_value(value.clone())
                .map_err(|e| HookBundleError::InvalidFormat(e.to_string()))?;
            return Ok(Self {
                rules,
                outbound_webhooks: Vec::new(),
            });
        }

        serde_json::from_value(value.clone())
            .map_err(|e| HookBundleError::InvalidFormat(e.to_string()))
    }
}

/// Summary of hook registrations performed from a bundle.
#[derive(Debug, Default, Clone, Copy)]
pub struct HookRegistrationSummary {
    /// Number of non-webhook hook registrations (audit/rule hooks).
    pub hooks: usize,
    /// Number of outbound webhook hook registrations.
    pub outbound_webhooks: usize,
    /// Number of invalid/failed registrations skipped.
    pub errors: usize,
}

impl HookRegistrationSummary {
    /// Total number of hooks successfully registered.
    pub fn total_registered(&self) -> usize {
        self.hooks + self.outbound_webhooks
    }

    pub fn merge(&mut self, other: HookRegistrationSummary) {
        self.hooks += other.hooks;
        self.outbound_webhooks += other.outbound_webhooks;
        self.errors += other.errors;
    }
}

/// Register bundled built-in hooks that ship with IronClaw.
pub async fn register_bundled_hooks(registry: &Arc<HookRegistry>) -> HookRegistrationSummary {
    registry
        .register_with_priority(Arc::new(AuditLogHook), 25)
        .await;

    HookRegistrationSummary {
        hooks: 1,
        outbound_webhooks: 0,
        errors: 0,
    }
}

/// Register all hooks from a declarative bundle.
pub async fn register_bundle(
    registry: &Arc<HookRegistry>,
    source: &str,
    bundle: HookBundleConfig,
) -> HookRegistrationSummary {
    let mut summary = HookRegistrationSummary::default();

    for rule in bundle.rules {
        match RuleHook::from_config(source, rule) {
            Ok((hook, priority)) => {
                registry
                    .register_with_priority(Arc::new(hook), priority)
                    .await;
                summary.hooks += 1;
            }
            Err(err) => {
                summary.errors += 1;
                tracing::warn!(source = source, error = %err, "Skipping invalid declarative hook rule");
            }
        }
    }

    for webhook in bundle.outbound_webhooks {
        match OutboundWebhookHook::from_config(source, webhook) {
            Ok((hook, priority)) => {
                registry
                    .register_with_priority(Arc::new(hook), priority)
                    .await;
                summary.outbound_webhooks += 1;
            }
            Err(err) => {
                summary.errors += 1;
                tracing::warn!(source = source, error = %err, "Skipping invalid outbound webhook hook");
            }
        }
    }

    summary
}

/// Declarative regex/string rule hook.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookRuleConfig {
    /// Stable hook name (scoped with source during registration).
    pub name: String,
    /// Lifecycle points where this rule applies.
    pub points: Vec<HookPoint>,
    /// Optional priority override (lower runs first).
    #[serde(default)]
    pub priority: Option<u32>,
    /// Failure handling mode (default fail_open).
    #[serde(default)]
    pub failure_mode: Option<HookFailureMode>,
    /// Optional timeout override for this hook in milliseconds.
    #[serde(default)]
    pub timeout_ms: Option<u64>,
    /// Optional regex guard. If provided and no match, rule is a no-op.
    #[serde(default)]
    pub when_regex: Option<String>,
    /// Optional immediate reject reason if guard matches.
    #[serde(default)]
    pub reject_reason: Option<String>,
    /// Regex replacements applied in order.
    #[serde(default)]
    pub replacements: Vec<RegexReplacementConfig>,
    /// Text prepended to the event's primary content.
    #[serde(default)]
    pub prepend: Option<String>,
    /// Text appended to the event's primary content.
    #[serde(default)]
    pub append: Option<String>,
}

/// A single regex replacement step in a rule hook.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegexReplacementConfig {
    pub pattern: String,
    pub replacement: String,
}

/// Declarative fire-and-forget outbound webhook hook.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutboundWebhookConfig {
    /// Stable webhook hook name (scoped with source during registration).
    pub name: String,
    /// Lifecycle points that trigger this webhook.
    pub points: Vec<HookPoint>,
    /// Target URL.
    pub url: String,
    /// Optional static headers.
    #[serde(default)]
    pub headers: HashMap<String, String>,
    /// Optional timeout override in milliseconds.
    #[serde(default)]
    pub timeout_ms: Option<u64>,
    /// Optional priority override (lower runs first).
    #[serde(default)]
    pub priority: Option<u32>,
}

/// Built-in audit trail hook that logs lifecycle events.
struct AuditLogHook;

#[async_trait]
impl Hook for AuditLogHook {
    fn name(&self) -> &str {
        "builtin.audit_log"
    }

    fn hook_points(&self) -> &[HookPoint] {
        &ALL_HOOK_POINTS
    }

    async fn execute(
        &self,
        event: &HookEvent,
        _ctx: &HookContext,
    ) -> Result<HookOutcome, HookError> {
        tracing::debug!(
            target: "hooks::audit",
            hook = self.name(),
            point = event.hook_point().as_str(),
            user_id = %event_user_id(event),
            "Lifecycle hook event"
        );

        Ok(HookOutcome::ok())
    }
}

#[derive(Debug, Clone)]
struct CompiledReplacement {
    regex: Regex,
    replacement: String,
}

/// Runtime hook compiled from [`HookRuleConfig`].
struct RuleHook {
    name: String,
    points: Vec<HookPoint>,
    failure_mode: HookFailureMode,
    timeout: Duration,
    when_regex: Option<Regex>,
    reject_reason: Option<String>,
    replacements: Vec<CompiledReplacement>,
    prepend: Option<String>,
    append: Option<String>,
}

impl RuleHook {
    fn from_config(source: &str, config: HookRuleConfig) -> Result<(Self, u32), HookBundleError> {
        let scoped_name = format!("{}::{}", source, config.name);

        if config.points.is_empty() {
            return Err(HookBundleError::MissingHookPoints { hook: scoped_name });
        }

        let timeout = timeout_from_ms(config.timeout_ms, &scoped_name)?;

        let when_regex = match config.when_regex {
            Some(pattern) => {
                Some(
                    Regex::new(&pattern).map_err(|e| HookBundleError::InvalidRegex {
                        hook: scoped_name.clone(),
                        pattern,
                        reason: e.to_string(),
                    })?,
                )
            }
            None => None,
        };

        let mut replacements = Vec::with_capacity(config.replacements.len());
        for replacement in config.replacements {
            let compiled =
                Regex::new(&replacement.pattern).map_err(|e| HookBundleError::InvalidRegex {
                    hook: scoped_name.clone(),
                    pattern: replacement.pattern.clone(),
                    reason: e.to_string(),
                })?;

            replacements.push(CompiledReplacement {
                regex: compiled,
                replacement: replacement.replacement,
            });
        }

        let hook = Self {
            name: scoped_name,
            points: config.points,
            failure_mode: config.failure_mode.unwrap_or(HookFailureMode::FailOpen),
            timeout,
            when_regex,
            reject_reason: config.reject_reason,
            replacements,
            prepend: config.prepend,
            append: config.append,
        };

        Ok((hook, config.priority.unwrap_or(DEFAULT_RULE_PRIORITY)))
    }
}

#[async_trait]
impl Hook for RuleHook {
    fn name(&self) -> &str {
        &self.name
    }

    fn hook_points(&self) -> &[HookPoint] {
        &self.points
    }

    fn failure_mode(&self) -> HookFailureMode {
        self.failure_mode
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }

    async fn execute(
        &self,
        event: &HookEvent,
        _ctx: &HookContext,
    ) -> Result<HookOutcome, HookError> {
        let content = extract_primary_content(event);

        if let Some(ref guard) = self.when_regex
            && !guard.is_match(&content)
        {
            return Ok(HookOutcome::ok());
        }

        if let Some(ref reason) = self.reject_reason {
            return Ok(HookOutcome::reject(reason.clone()));
        }

        let mut modified = content.clone();

        for replacement in &self.replacements {
            modified = replacement
                .regex
                .replace_all(&modified, replacement.replacement.as_str())
                .into_owned();
        }

        if let Some(ref prefix) = self.prepend {
            modified = format!("{}{}", prefix, modified);
        }

        if let Some(ref suffix) = self.append {
            modified.push_str(suffix);
        }

        if modified != content {
            Ok(HookOutcome::modify(modified))
        } else {
            Ok(HookOutcome::ok())
        }
    }
}

/// Runtime outbound webhook hook.
struct OutboundWebhookHook {
    name: String,
    points: Vec<HookPoint>,
    client: reqwest::Client,
    url: String,
    headers: HashMap<String, String>,
    timeout: Duration,
}

impl OutboundWebhookHook {
    fn from_config(
        source: &str,
        config: OutboundWebhookConfig,
    ) -> Result<(Self, u32), HookBundleError> {
        let scoped_name = format!("{}::{}", source, config.name);

        if config.points.is_empty() {
            return Err(HookBundleError::MissingHookPoints { hook: scoped_name });
        }

        if reqwest::Url::parse(&config.url).is_err() {
            return Err(HookBundleError::InvalidWebhookUrl {
                hook: scoped_name,
                url: config.url,
            });
        }

        let timeout = timeout_from_ms(
            config.timeout_ms.or(Some(DEFAULT_WEBHOOK_TIMEOUT_MS)),
            &scoped_name,
        )?;

        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| HookBundleError::InvalidFormat(e.to_string()))?;

        let hook = Self {
            name: format!("{}::{}", source, config.name),
            points: config.points,
            client,
            url: config.url,
            headers: config.headers,
            timeout,
        };

        Ok((hook, config.priority.unwrap_or(DEFAULT_WEBHOOK_PRIORITY)))
    }
}

#[derive(Debug, Serialize)]
struct OutboundWebhookPayload {
    hook: String,
    point: String,
    timestamp: String,
    event: serde_json::Value,
    metadata: serde_json::Value,
}

#[async_trait]
impl Hook for OutboundWebhookHook {
    fn name(&self) -> &str {
        &self.name
    }

    fn hook_points(&self) -> &[HookPoint] {
        &self.points
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }

    async fn execute(
        &self,
        event: &HookEvent,
        ctx: &HookContext,
    ) -> Result<HookOutcome, HookError> {
        let payload = OutboundWebhookPayload {
            hook: self.name.clone(),
            point: event.hook_point().as_str().to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            event: serde_json::to_value(event).unwrap_or(serde_json::Value::Null),
            metadata: ctx.metadata.clone(),
        };

        let client = self.client.clone();
        let url = self.url.clone();
        let headers = self.headers.clone();
        let hook_name = self.name.clone();

        tokio::spawn(async move {
            let mut request = client.post(url).json(&payload);

            for (name, value) in headers {
                request = request.header(name, value);
            }

            if let Err(err) = request.send().await {
                tracing::warn!(hook = %hook_name, error = %err, "Outbound webhook delivery failed");
            }
        });

        Ok(HookOutcome::ok())
    }
}

fn timeout_from_ms(timeout_ms: Option<u64>, hook_name: &str) -> Result<Duration, HookBundleError> {
    if let Some(ms) = timeout_ms {
        if ms == 0 || ms > MAX_HOOK_TIMEOUT_MS {
            return Err(HookBundleError::InvalidTimeout {
                hook: hook_name.to_string(),
                max_ms: MAX_HOOK_TIMEOUT_MS,
            });
        }
        Ok(Duration::from_millis(ms))
    } else {
        Ok(Duration::from_secs(5))
    }
}

fn event_user_id(event: &HookEvent) -> &str {
    match event {
        HookEvent::Inbound { user_id, .. }
        | HookEvent::ToolCall { user_id, .. }
        | HookEvent::Outbound { user_id, .. }
        | HookEvent::SessionStart { user_id, .. }
        | HookEvent::SessionEnd { user_id, .. }
        | HookEvent::ResponseTransform { user_id, .. } => user_id,
    }
}

fn extract_primary_content(event: &HookEvent) -> String {
    match event {
        HookEvent::Inbound { content, .. } | HookEvent::Outbound { content, .. } => content.clone(),
        HookEvent::ToolCall { parameters, .. } => {
            serde_json::to_string(parameters).unwrap_or_default()
        }
        HookEvent::SessionStart { session_id, .. } | HookEvent::SessionEnd { session_id, .. } => {
            session_id.clone()
        }
        HookEvent::ResponseTransform { response, .. } => response.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inbound_event(content: &str) -> HookEvent {
        HookEvent::Inbound {
            user_id: "user-1".to_string(),
            channel: "test".to_string(),
            content: content.to_string(),
            thread_id: None,
        }
    }

    #[test]
    fn test_parse_bundle_array_shorthand() {
        let value = serde_json::json!([
            {
                "name": "append-bang",
                "points": ["beforeInbound"],
                "append": "!"
            }
        ]);

        let parsed = HookBundleConfig::from_value(&value).unwrap();
        assert_eq!(parsed.rules.len(), 1);
        assert!(parsed.outbound_webhooks.is_empty());
    }

    #[tokio::test]
    async fn test_rule_hook_modifies_content() {
        let registry = Arc::new(HookRegistry::new());

        let bundle = HookBundleConfig {
            rules: vec![HookRuleConfig {
                name: "redact-secret".to_string(),
                points: vec![HookPoint::BeforeInbound],
                priority: None,
                failure_mode: None,
                timeout_ms: None,
                when_regex: None,
                reject_reason: None,
                replacements: vec![RegexReplacementConfig {
                    pattern: "secret".to_string(),
                    replacement: "[redacted]".to_string(),
                }],
                prepend: None,
                append: None,
            }],
            outbound_webhooks: vec![],
        };

        let summary = register_bundle(&registry, "workspace:hooks/hooks.json", bundle).await;
        assert_eq!(summary.hooks, 1);
        assert_eq!(summary.errors, 0);

        let result = registry
            .run(&inbound_event("contains secret here"))
            .await
            .unwrap();
        match result {
            HookOutcome::Continue {
                modified: Some(value),
            } => {
                assert_eq!(value, "contains [redacted] here");
            }
            other => panic!("expected modified output, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_rule_hook_rejects() {
        let registry = Arc::new(HookRegistry::new());

        let bundle = HookBundleConfig {
            rules: vec![HookRuleConfig {
                name: "block-forbidden".to_string(),
                points: vec![HookPoint::BeforeInbound],
                priority: None,
                failure_mode: None,
                timeout_ms: None,
                when_regex: Some("forbidden".to_string()),
                reject_reason: Some("forbidden content".to_string()),
                replacements: vec![],
                prepend: None,
                append: None,
            }],
            outbound_webhooks: vec![],
        };

        let summary = register_bundle(&registry, "plugin:tool:test", bundle).await;
        assert_eq!(summary.hooks, 1);

        let result = registry.run(&inbound_event("this is forbidden")).await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            HookError::Rejected { reason } if reason == "forbidden content"
        ));
    }

    #[tokio::test]
    async fn test_outbound_webhook_hook_registers() {
        let registry = Arc::new(HookRegistry::new());

        let bundle = HookBundleConfig {
            rules: vec![],
            outbound_webhooks: vec![OutboundWebhookConfig {
                name: "notify".to_string(),
                points: vec![HookPoint::BeforeInbound],
                url: "http://127.0.0.1:9/hook".to_string(),
                headers: HashMap::new(),
                timeout_ms: Some(1000),
                priority: None,
            }],
        };

        let summary = register_bundle(&registry, "workspace:hooks/webhook.hook.json", bundle).await;
        assert_eq!(summary.outbound_webhooks, 1);

        // Should return immediately regardless of webhook delivery result.
        let result = registry.run(&inbound_event("hello")).await;
        assert!(result.is_ok());
    }
}
