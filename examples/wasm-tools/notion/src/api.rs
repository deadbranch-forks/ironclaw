//! Notion API implementation.
//!
//! All API calls go through the host's HTTP capability, which handles
//! credential injection and rate limiting. The WASM tool never sees
//! the actual API token.

use crate::near::agent::host;
use crate::types::*;

const NOTION_API_BASE: &str = "https://api.notion.com/v1";
const NOTION_VERSION: &str = "2022-06-28";

/// Make a Notion API GET request.
fn notion_get(path: &str) -> Result<serde_json::Value, String> {
    let url = format!("{}{}", NOTION_API_BASE, path);

    let headers = serde_json::json!({
        "Notion-Version": NOTION_VERSION
    });
    let headers_str = serde_json::to_string(&headers).map_err(|e| e.to_string())?;

    host::log(host::LogLevel::Debug, &format!("Notion GET: {}", path));

    let response = host::http_request("GET", &url, &headers_str, None)?;

    if response.status < 200 || response.status >= 300 {
        let body = String::from_utf8_lossy(&response.body);
        return Err(format!(
            "Notion API returned status {}: {}",
            response.status, body
        ));
    }

    let body = String::from_utf8(response.body)
        .map_err(|e| format!("Invalid UTF-8 in response: {}", e))?;
    serde_json::from_str(&body).map_err(|e| format!("Invalid JSON in response: {}", e))
}

/// Make a Notion API POST request.
fn notion_post(path: &str, body: &serde_json::Value) -> Result<serde_json::Value, String> {
    let url = format!("{}{}", NOTION_API_BASE, path);

    let headers = serde_json::json!({
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json"
    });
    let headers_str = serde_json::to_string(&headers).map_err(|e| e.to_string())?;
    let body_str = serde_json::to_string(body).map_err(|e| e.to_string())?;

    host::log(host::LogLevel::Debug, &format!("Notion POST: {}", path));

    let response = host::http_request("POST", &url, &headers_str, Some(body_str.as_bytes()))?;

    if response.status < 200 || response.status >= 300 {
        let body = String::from_utf8_lossy(&response.body);
        return Err(format!(
            "Notion API returned status {}: {}",
            response.status, body
        ));
    }

    let body = String::from_utf8(response.body)
        .map_err(|e| format!("Invalid UTF-8 in response: {}", e))?;
    serde_json::from_str(&body).map_err(|e| format!("Invalid JSON in response: {}", e))
}

/// Make a Notion API PATCH request.
fn notion_patch(path: &str, body: &serde_json::Value) -> Result<serde_json::Value, String> {
    let url = format!("{}{}", NOTION_API_BASE, path);

    let headers = serde_json::json!({
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json"
    });
    let headers_str = serde_json::to_string(&headers).map_err(|e| e.to_string())?;
    let body_str = serde_json::to_string(body).map_err(|e| e.to_string())?;

    host::log(host::LogLevel::Debug, &format!("Notion PATCH: {}", path));

    let response = host::http_request("PATCH", &url, &headers_str, Some(body_str.as_bytes()))?;

    if response.status < 200 || response.status >= 300 {
        let body = String::from_utf8_lossy(&response.body);
        return Err(format!(
            "Notion API returned status {}: {}",
            response.status, body
        ));
    }

    let body = String::from_utf8(response.body)
        .map_err(|e| format!("Invalid UTF-8 in response: {}", e))?;
    serde_json::from_str(&body).map_err(|e| format!("Invalid JSON in response: {}", e))
}

/// Make a Notion API DELETE request.
fn notion_delete(path: &str) -> Result<serde_json::Value, String> {
    let url = format!("{}{}", NOTION_API_BASE, path);

    let headers = serde_json::json!({
        "Notion-Version": NOTION_VERSION
    });
    let headers_str = serde_json::to_string(&headers).map_err(|e| e.to_string())?;

    host::log(host::LogLevel::Debug, &format!("Notion DELETE: {}", path));

    let response = host::http_request("DELETE", &url, &headers_str, None)?;

    if response.status < 200 || response.status >= 300 {
        let body = String::from_utf8_lossy(&response.body);
        return Err(format!(
            "Notion API returned status {}: {}",
            response.status, body
        ));
    }

    let body = String::from_utf8(response.body)
        .map_err(|e| format!("Invalid UTF-8 in response: {}", e))?;
    serde_json::from_str(&body).map_err(|e| format!("Invalid JSON in response: {}", e))
}

// ==================== Search ====================

pub fn search(
    query: &str,
    filter: Option<&SearchFilter>,
    page_size: u32,
    start_cursor: Option<&str>,
) -> Result<serde_json::Value, String> {
    let mut body = serde_json::json!({
        "page_size": page_size.min(100)
    });

    if !query.is_empty() {
        body["query"] = serde_json::Value::String(query.to_string());
    }

    if let Some(f) = filter {
        body["filter"] = serde_json::json!({
            "property": f.property,
            "value": f.value
        });
    }

    if let Some(cursor) = start_cursor {
        body["start_cursor"] = serde_json::Value::String(cursor.to_string());
    }

    notion_post("/search", &body)
}

// ==================== Pages ====================

pub fn get_page(page_id: &str) -> Result<serde_json::Value, String> {
    let page_id = normalize_uuid(page_id);
    notion_get(&format!("/pages/{}", page_id))
}

pub fn create_page(
    parent: &serde_json::Value,
    properties: &serde_json::Value,
    children: Option<&Vec<serde_json::Value>>,
    icon: Option<&serde_json::Value>,
    cover: Option<&serde_json::Value>,
) -> Result<serde_json::Value, String> {
    let mut body = serde_json::json!({
        "parent": parent,
        "properties": properties
    });

    if let Some(c) = children {
        if !c.is_empty() {
            body["children"] = serde_json::Value::Array(c.clone());
        }
    }

    if let Some(i) = icon {
        body["icon"] = i.clone();
    }

    if let Some(c) = cover {
        body["cover"] = c.clone();
    }

    notion_post("/pages", &body)
}

pub fn update_page(
    page_id: &str,
    properties: &serde_json::Value,
    icon: Option<&serde_json::Value>,
    cover: Option<&serde_json::Value>,
) -> Result<serde_json::Value, String> {
    let page_id = normalize_uuid(page_id);

    let mut body = serde_json::json!({
        "properties": properties
    });

    if let Some(i) = icon {
        body["icon"] = i.clone();
    }

    if let Some(c) = cover {
        body["cover"] = c.clone();
    }

    notion_patch(&format!("/pages/{}", page_id), &body)
}

pub fn archive_page(page_id: &str) -> Result<serde_json::Value, String> {
    let page_id = normalize_uuid(page_id);
    notion_patch(
        &format!("/pages/{}", page_id),
        &serde_json::json!({ "archived": true }),
    )
}

pub fn restore_page(page_id: &str) -> Result<serde_json::Value, String> {
    let page_id = normalize_uuid(page_id);
    notion_patch(
        &format!("/pages/{}", page_id),
        &serde_json::json!({ "archived": false }),
    )
}

// ==================== Blocks ====================

pub fn get_blocks(
    block_id: &str,
    page_size: u32,
    start_cursor: Option<&str>,
) -> Result<serde_json::Value, String> {
    let block_id = normalize_uuid(block_id);
    let mut path = format!(
        "/blocks/{}/children?page_size={}",
        block_id,
        page_size.min(100)
    );

    if let Some(cursor) = start_cursor {
        path.push_str(&format!("&start_cursor={}", cursor));
    }

    notion_get(&path)
}

pub fn append_blocks(
    block_id: &str,
    children: &[serde_json::Value],
    after: Option<&str>,
) -> Result<serde_json::Value, String> {
    let block_id = normalize_uuid(block_id);

    let mut body = serde_json::json!({
        "children": children
    });

    if let Some(after_id) = after {
        body["after"] = serde_json::Value::String(normalize_uuid(after_id));
    }

    notion_patch(&format!("/blocks/{}/children", block_id), &body)
}

pub fn get_block(block_id: &str) -> Result<serde_json::Value, String> {
    let block_id = normalize_uuid(block_id);
    notion_get(&format!("/blocks/{}", block_id))
}

pub fn update_block(
    block_id: &str,
    content: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let block_id = normalize_uuid(block_id);
    notion_patch(&format!("/blocks/{}", block_id), content)
}

pub fn delete_block(block_id: &str) -> Result<serde_json::Value, String> {
    let block_id = normalize_uuid(block_id);
    notion_delete(&format!("/blocks/{}", block_id))
}

// ==================== Databases ====================

pub fn get_database(database_id: &str) -> Result<serde_json::Value, String> {
    let database_id = normalize_uuid(database_id);
    notion_get(&format!("/databases/{}", database_id))
}

pub fn query_database(
    database_id: &str,
    filter: Option<&serde_json::Value>,
    sorts: Option<&Vec<serde_json::Value>>,
    page_size: u32,
    start_cursor: Option<&str>,
) -> Result<serde_json::Value, String> {
    let database_id = normalize_uuid(database_id);

    let mut body = serde_json::json!({
        "page_size": page_size.min(100)
    });

    if let Some(f) = filter {
        body["filter"] = f.clone();
    }

    if let Some(s) = sorts {
        if !s.is_empty() {
            body["sorts"] = serde_json::Value::Array(s.clone());
        }
    }

    if let Some(cursor) = start_cursor {
        body["start_cursor"] = serde_json::Value::String(cursor.to_string());
    }

    notion_post(&format!("/databases/{}/query", database_id), &body)
}

pub fn create_database(
    parent: &serde_json::Value,
    title: &[serde_json::Value],
    properties: &serde_json::Value,
    icon: Option<&serde_json::Value>,
    cover: Option<&serde_json::Value>,
    is_inline: bool,
) -> Result<serde_json::Value, String> {
    let mut body = serde_json::json!({
        "parent": parent,
        "title": title,
        "properties": properties,
        "is_inline": is_inline
    });

    if let Some(i) = icon {
        body["icon"] = i.clone();
    }

    if let Some(c) = cover {
        body["cover"] = c.clone();
    }

    notion_post("/databases", &body)
}

pub fn update_database(
    database_id: &str,
    title: Option<&Vec<serde_json::Value>>,
    properties: Option<&serde_json::Value>,
) -> Result<serde_json::Value, String> {
    let database_id = normalize_uuid(database_id);

    let mut body = serde_json::json!({});

    if let Some(t) = title {
        body["title"] = serde_json::Value::Array(t.clone());
    }

    if let Some(p) = properties {
        body["properties"] = p.clone();
    }

    notion_patch(&format!("/databases/{}", database_id), &body)
}

// ==================== Comments ====================

pub fn get_comments(
    block_id: &str,
    page_size: u32,
    start_cursor: Option<&str>,
) -> Result<serde_json::Value, String> {
    let block_id = normalize_uuid(block_id);
    let mut path = format!(
        "/comments?block_id={}&page_size={}",
        block_id,
        page_size.min(100)
    );

    if let Some(cursor) = start_cursor {
        path.push_str(&format!("&start_cursor={}", cursor));
    }

    notion_get(&path)
}

pub fn add_comment(
    parent: &serde_json::Value,
    rich_text: &[serde_json::Value],
) -> Result<serde_json::Value, String> {
    let body = serde_json::json!({
        "parent": parent,
        "rich_text": rich_text
    });

    notion_post("/comments", &body)
}

// ==================== Users ====================

pub fn list_users(page_size: u32, start_cursor: Option<&str>) -> Result<serde_json::Value, String> {
    let mut path = format!("/users?page_size={}", page_size.min(100));

    if let Some(cursor) = start_cursor {
        path.push_str(&format!("&start_cursor={}", cursor));
    }

    notion_get(&path)
}

pub fn get_user(user_id: &str) -> Result<serde_json::Value, String> {
    let user_id = normalize_uuid(user_id);
    notion_get(&format!("/users/{}", user_id))
}

pub fn get_me() -> Result<serde_json::Value, String> {
    notion_get("/users/me")
}

// ==================== Helpers ====================

/// Normalize a UUID by removing dashes if needed.
/// Notion accepts both formats, but we normalize for consistency.
fn normalize_uuid(id: &str) -> String {
    // If it already has dashes in the right places, return as-is
    if id.len() == 36 && id.chars().filter(|c| *c == '-').count() == 4 {
        return id.to_string();
    }

    // If it's 32 characters without dashes, add them
    let clean: String = id.chars().filter(|c| c.is_ascii_hexdigit()).collect();
    if clean.len() == 32 {
        return format!(
            "{}-{}-{}-{}-{}",
            &clean[0..8],
            &clean[8..12],
            &clean[12..16],
            &clean[16..20],
            &clean[20..32]
        );
    }

    // Otherwise return as-is (let the API handle validation)
    id.to_string()
}
