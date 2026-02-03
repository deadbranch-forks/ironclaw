//! Memory document types.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::WorkspaceError;

/// Document type in the workspace.
///
/// Each type represents a different kind of persistent memory:
/// - **Memory**: Long-term curated facts and decisions (MEMORY.md)
/// - **DailyLog**: Append-only daily notes (memory/YYYY-MM-DD.md)
/// - **Identity**: Agent name and personality
/// - **Soul**: Core values and behavior principles
/// - **Agents**: Behavior instructions
/// - **User**: User context (name, preferences)
/// - **Heartbeat**: Periodic checklist for proactive execution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DocType {
    /// Long-term curated memory (MEMORY.md equivalent).
    Memory,
    /// Daily append-only logs.
    DailyLog,
    /// Agent identity (name, nature, vibe).
    Identity,
    /// Core values and principles (SOUL.md).
    Soul,
    /// Behavior instructions (AGENTS.md).
    Agents,
    /// User context (USER.md).
    User,
    /// Periodic checklist (HEARTBEAT.md).
    Heartbeat,
}

impl DocType {
    /// Get the string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            DocType::Memory => "memory",
            DocType::DailyLog => "daily_log",
            DocType::Identity => "identity",
            DocType::Soul => "soul",
            DocType::Agents => "agents",
            DocType::User => "user",
            DocType::Heartbeat => "heartbeat",
        }
    }

    /// Check if this document type is a singleton (one per user/agent).
    pub fn is_singleton(&self) -> bool {
        match self {
            DocType::Memory
            | DocType::Identity
            | DocType::Soul
            | DocType::Agents
            | DocType::User
            | DocType::Heartbeat => true,
            DocType::DailyLog => false,
        }
    }

    /// Check if this document should be included in the system prompt.
    pub fn is_identity_document(&self) -> bool {
        matches!(
            self,
            DocType::Identity | DocType::Soul | DocType::Agents | DocType::User
        )
    }
}

impl TryFrom<&str> for DocType {
    type Error = WorkspaceError;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        match s {
            "memory" => Ok(DocType::Memory),
            "daily_log" => Ok(DocType::DailyLog),
            "identity" => Ok(DocType::Identity),
            "soul" => Ok(DocType::Soul),
            "agents" => Ok(DocType::Agents),
            "user" => Ok(DocType::User),
            "heartbeat" => Ok(DocType::Heartbeat),
            _ => Err(WorkspaceError::InvalidDocType {
                doc_type: s.to_string(),
            }),
        }
    }
}

impl std::fmt::Display for DocType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// A memory document stored in the database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryDocument {
    /// Unique document ID.
    pub id: Uuid,
    /// User identifier.
    pub user_id: String,
    /// Optional agent ID for multi-agent isolation.
    pub agent_id: Option<Uuid>,
    /// Document type.
    pub doc_type: DocType,
    /// Optional title (e.g., date for daily logs).
    pub title: Option<String>,
    /// Full document content.
    pub content: String,
    /// Creation timestamp.
    pub created_at: DateTime<Utc>,
    /// Last update timestamp.
    pub updated_at: DateTime<Utc>,
    /// Flexible metadata.
    pub metadata: serde_json::Value,
}

impl MemoryDocument {
    /// Create a new document (not persisted yet).
    pub fn new(
        user_id: impl Into<String>,
        agent_id: Option<Uuid>,
        doc_type: DocType,
        title: Option<String>,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            user_id: user_id.into(),
            agent_id,
            doc_type,
            title,
            content: String::new(),
            created_at: now,
            updated_at: now,
            metadata: serde_json::Value::Object(serde_json::Map::new()),
        }
    }

    /// Check if the document is empty.
    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }

    /// Get word count.
    pub fn word_count(&self) -> usize {
        self.content.split_whitespace().count()
    }
}

/// A chunk of a memory document for search indexing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryChunk {
    /// Unique chunk ID.
    pub id: Uuid,
    /// Parent document ID.
    pub document_id: Uuid,
    /// Position in the document (0-based).
    pub chunk_index: i32,
    /// Chunk text content.
    pub content: String,
    /// Embedding vector (if generated).
    pub embedding: Option<Vec<f32>>,
    /// Creation timestamp.
    pub created_at: DateTime<Utc>,
}

impl MemoryChunk {
    /// Create a new chunk (not persisted yet).
    pub fn new(document_id: Uuid, chunk_index: i32, content: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            document_id,
            chunk_index,
            content: content.into(),
            embedding: None,
            created_at: Utc::now(),
        }
    }

    /// Set the embedding.
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_doc_type_roundtrip() {
        for doc_type in [
            DocType::Memory,
            DocType::DailyLog,
            DocType::Identity,
            DocType::Soul,
            DocType::Agents,
            DocType::User,
            DocType::Heartbeat,
        ] {
            let s = doc_type.as_str();
            let parsed = DocType::try_from(s).unwrap();
            assert_eq!(parsed, doc_type);
        }
    }

    #[test]
    fn test_singleton_types() {
        assert!(DocType::Memory.is_singleton());
        assert!(DocType::Heartbeat.is_singleton());
        assert!(!DocType::DailyLog.is_singleton());
    }

    #[test]
    fn test_identity_documents() {
        assert!(DocType::Soul.is_identity_document());
        assert!(DocType::Agents.is_identity_document());
        assert!(!DocType::Memory.is_identity_document());
        assert!(!DocType::DailyLog.is_identity_document());
    }

    #[test]
    fn test_memory_document_word_count() {
        let mut doc = MemoryDocument::new("user1", None, DocType::Memory, None);
        assert_eq!(doc.word_count(), 0);

        doc.content = "Hello world, this is a test.".to_string();
        assert_eq!(doc.word_count(), 6);
    }
}
