//! Types for Notion API requests and responses.

use serde::{Deserialize, Serialize};

/// Input parameters for the Notion tool.
#[derive(Debug, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum NotionAction {
    // ==================== Search ====================
    /// Search across all pages and databases.
    Search {
        /// Text query to search for.
        #[serde(default)]
        query: String,
        /// Filter by object type: "page" or "database".
        #[serde(default)]
        filter: Option<SearchFilter>,
        /// Max results (default: 10, max: 100).
        #[serde(default = "default_page_size")]
        page_size: u32,
        /// Pagination cursor from previous response.
        #[serde(default)]
        start_cursor: Option<String>,
    },

    // ==================== Pages ====================
    /// Retrieve a page by ID.
    GetPage {
        /// Page ID (UUID format, with or without dashes).
        page_id: String,
    },

    /// Create a new page.
    CreatePage {
        /// Parent reference: { "database_id": "..." } or { "page_id": "..." }.
        parent: serde_json::Value,
        /// Page properties matching the parent database schema.
        properties: serde_json::Value,
        /// Optional page content as an array of block objects.
        #[serde(default)]
        children: Option<Vec<serde_json::Value>>,
        /// Optional icon: { "emoji": "..." } or { "external": { "url": "..." } }.
        #[serde(default)]
        icon: Option<serde_json::Value>,
        /// Optional cover: { "external": { "url": "..." } }.
        #[serde(default)]
        cover: Option<serde_json::Value>,
    },

    /// Update page properties.
    UpdatePage {
        /// Page ID.
        page_id: String,
        /// Properties to update.
        properties: serde_json::Value,
        /// Optional new icon.
        #[serde(default)]
        icon: Option<serde_json::Value>,
        /// Optional new cover.
        #[serde(default)]
        cover: Option<serde_json::Value>,
    },

    /// Archive (soft-delete) a page.
    ArchivePage {
        /// Page ID.
        page_id: String,
    },

    /// Restore an archived page.
    RestorePage {
        /// Page ID.
        page_id: String,
    },

    // ==================== Blocks ====================
    /// Get child blocks of a block or page.
    GetBlocks {
        /// Block or page ID.
        block_id: String,
        /// Max results (default: 50, max: 100).
        #[serde(default = "default_block_page_size")]
        page_size: u32,
        /// Pagination cursor.
        #[serde(default)]
        start_cursor: Option<String>,
    },

    /// Append blocks to a page or block.
    AppendBlocks {
        /// Block or page ID to append to.
        block_id: String,
        /// Array of block objects to append.
        children: Vec<serde_json::Value>,
        /// Append after this block ID (for ordering).
        #[serde(default)]
        after: Option<String>,
    },

    /// Retrieve a single block.
    GetBlock {
        /// Block ID.
        block_id: String,
    },

    /// Update a block's content.
    UpdateBlock {
        /// Block ID.
        block_id: String,
        /// Block content (varies by type, e.g., { "paragraph": { "rich_text": [...] } }).
        content: serde_json::Value,
    },

    /// Delete a block.
    DeleteBlock {
        /// Block ID.
        block_id: String,
    },

    // ==================== Databases ====================
    /// Retrieve a database schema.
    GetDatabase {
        /// Database ID.
        database_id: String,
    },

    /// Query a database with filters and sorts.
    QueryDatabase {
        /// Database ID.
        database_id: String,
        /// Filter object (Notion filter format).
        #[serde(default)]
        filter: Option<serde_json::Value>,
        /// Sort configuration array.
        #[serde(default)]
        sorts: Option<Vec<serde_json::Value>>,
        /// Max results (default: 10, max: 100).
        #[serde(default = "default_page_size")]
        page_size: u32,
        /// Pagination cursor.
        #[serde(default)]
        start_cursor: Option<String>,
    },

    /// Create a new database.
    CreateDatabase {
        /// Parent reference: { "page_id": "..." } or { "type": "workspace", "workspace": true }.
        parent: serde_json::Value,
        /// Database title as rich text array.
        title: Vec<serde_json::Value>,
        /// Property schema: { "Name": { "title": {} }, "Status": { "select": { "options": [...] } } }.
        properties: serde_json::Value,
        /// Optional icon.
        #[serde(default)]
        icon: Option<serde_json::Value>,
        /// Optional cover.
        #[serde(default)]
        cover: Option<serde_json::Value>,
        /// Make database inline (default: false).
        #[serde(default)]
        is_inline: bool,
    },

    /// Update database title or properties schema.
    UpdateDatabase {
        /// Database ID.
        database_id: String,
        /// New title (optional).
        #[serde(default)]
        title: Option<Vec<serde_json::Value>>,
        /// Properties to add or update (optional).
        #[serde(default)]
        properties: Option<serde_json::Value>,
    },

    // ==================== Comments ====================
    /// Get comments on a block or page.
    GetComments {
        /// Block or page ID.
        block_id: String,
        /// Max results (default: 50, max: 100).
        #[serde(default = "default_block_page_size")]
        page_size: u32,
        /// Pagination cursor.
        #[serde(default)]
        start_cursor: Option<String>,
    },

    /// Add a comment to a page or discussion thread.
    AddComment {
        /// Parent reference: { "page_id": "..." } or { "discussion_id": "..." }.
        parent: serde_json::Value,
        /// Comment text as rich text array.
        rich_text: Vec<serde_json::Value>,
    },

    // ==================== Users ====================
    /// List all users in the workspace.
    ListUsers {
        /// Max results (default: 50, max: 100).
        #[serde(default = "default_block_page_size")]
        page_size: u32,
        /// Pagination cursor.
        #[serde(default)]
        start_cursor: Option<String>,
    },

    /// Get a specific user.
    GetUser {
        /// User ID.
        user_id: String,
    },

    /// Get the bot user (the integration itself).
    GetMe,
}

/// Search filter for limiting results to pages or databases.
#[derive(Debug, Deserialize, Serialize)]
pub struct SearchFilter {
    /// "object" field.
    pub property: String,
    /// "page" or "database".
    pub value: String,
}

fn default_page_size() -> u32 {
    10
}

fn default_block_page_size() -> u32 {
    50
}

// ==================== Response Types ====================

/// Generic Notion API response with pagination.
#[derive(Debug, Serialize)]
pub struct PaginatedResponse {
    pub object: String,
    pub results: Vec<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<String>,
    pub has_more: bool,
}

/// Single object response.
#[derive(Debug, Serialize)]
pub struct ObjectResponse {
    pub object: String,
    #[serde(flatten)]
    pub data: serde_json::Value,
}

/// Success response for mutations.
#[derive(Debug, Serialize)]
pub struct MutationResponse {
    pub success: bool,
    #[serde(flatten)]
    pub data: serde_json::Value,
}

/// Minimal success response.
#[derive(Debug, Serialize)]
pub struct SuccessResponse {
    pub success: bool,
}
