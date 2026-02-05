//! Notion WASM Tool for NEAR Agent.
//!
//! This is a standalone WASM component that provides Notion integration.
//! It follows the MCP server pattern with domain-grouped operations.
//!
//! # Capabilities Required
//!
//! - HTTP: `api.notion.com/v1/*` (GET, POST, PATCH, DELETE)
//! - Secrets: `notion_api_token` (injected automatically as Bearer token)
//!
//! # Supported Actions
//!
//! ## Search
//! - `search`: Search across pages and databases
//!
//! ## Pages
//! - `get_page`: Retrieve a page by ID
//! - `create_page`: Create a new page in a database or as child of another page
//! - `update_page`: Update page properties
//! - `archive_page`: Archive (soft-delete) a page
//! - `restore_page`: Restore an archived page
//!
//! ## Blocks
//! - `get_blocks`: Get child blocks of a page/block
//! - `append_blocks`: Append content blocks to a page/block
//! - `get_block`: Get a single block
//! - `update_block`: Update a block's content
//! - `delete_block`: Delete a block
//!
//! ## Databases
//! - `get_database`: Get database schema
//! - `query_database`: Query with filters and sorts
//! - `create_database`: Create a new database
//! - `update_database`: Update database title/properties
//!
//! ## Comments
//! - `get_comments`: Get comments on a page/block
//! - `add_comment`: Add a comment
//!
//! ## Users
//! - `list_users`: List workspace users
//! - `get_user`: Get a specific user
//! - `get_me`: Get the bot user
//!
//! # Example Usage
//!
//! ```json
//! {"action": "search", "query": "meeting notes", "page_size": 5}
//! ```
//!
//! ```json
//! {
//!   "action": "query_database",
//!   "database_id": "abc123...",
//!   "filter": {"property": "Status", "select": {"equals": "Done"}},
//!   "sorts": [{"property": "Created", "direction": "descending"}]
//! }
//! ```

mod api;
mod types;

use types::NotionAction;

// Generate bindings from the WIT interface.
wit_bindgen::generate!({
    world: "sandboxed-tool",
    path: "../../../wit/tool.wit",
});

/// Implementation of the tool interface.
struct NotionTool;

impl exports::near::agent::tool::Guest for NotionTool {
    fn execute(req: exports::near::agent::tool::Request) -> exports::near::agent::tool::Response {
        match execute_inner(&req.params) {
            Ok(result) => exports::near::agent::tool::Response {
                output: Some(result),
                error: None,
            },
            Err(e) => exports::near::agent::tool::Response {
                output: None,
                error: Some(e),
            },
        }
    }

    fn schema() -> String {
        TOOL_SCHEMA.to_string()
    }

    fn description() -> String {
        "Notion workspace integration for searching, managing pages/databases/blocks, \
         comments, and users. Requires a Notion integration token with appropriate \
         capabilities (read content, update content, insert content)."
            .to_string()
    }
}

/// Inner execution logic with proper error handling.
fn execute_inner(params: &str) -> Result<String, String> {
    // Check if the Notion token is configured
    if !crate::near::agent::host::secret_exists("notion_api_token") {
        return Err(
            "Notion API token not configured. Please add the 'notion_api_token' secret."
                .to_string(),
        );
    }

    // Parse the action from JSON
    let action: NotionAction =
        serde_json::from_str(params).map_err(|e| format!("Invalid parameters: {}", e))?;

    crate::near::agent::host::log(
        crate::near::agent::host::LogLevel::Info,
        &format!("Executing Notion action: {:?}", action),
    );

    // Dispatch to the appropriate handler
    let result = match action {
        // Search
        NotionAction::Search {
            query,
            filter,
            page_size,
            start_cursor,
        } => api::search(&query, filter.as_ref(), page_size, start_cursor.as_deref())?,

        // Pages
        NotionAction::GetPage { page_id } => api::get_page(&page_id)?,

        NotionAction::CreatePage {
            parent,
            properties,
            children,
            icon,
            cover,
        } => api::create_page(
            &parent,
            &properties,
            children.as_ref(),
            icon.as_ref(),
            cover.as_ref(),
        )?,

        NotionAction::UpdatePage {
            page_id,
            properties,
            icon,
            cover,
        } => api::update_page(&page_id, &properties, icon.as_ref(), cover.as_ref())?,

        NotionAction::ArchivePage { page_id } => api::archive_page(&page_id)?,

        NotionAction::RestorePage { page_id } => api::restore_page(&page_id)?,

        // Blocks
        NotionAction::GetBlocks {
            block_id,
            page_size,
            start_cursor,
        } => api::get_blocks(&block_id, page_size, start_cursor.as_deref())?,

        NotionAction::AppendBlocks {
            block_id,
            children,
            after,
        } => api::append_blocks(&block_id, &children, after.as_deref())?,

        NotionAction::GetBlock { block_id } => api::get_block(&block_id)?,

        NotionAction::UpdateBlock { block_id, content } => api::update_block(&block_id, &content)?,

        NotionAction::DeleteBlock { block_id } => api::delete_block(&block_id)?,

        // Databases
        NotionAction::GetDatabase { database_id } => api::get_database(&database_id)?,

        NotionAction::QueryDatabase {
            database_id,
            filter,
            sorts,
            page_size,
            start_cursor,
        } => api::query_database(
            &database_id,
            filter.as_ref(),
            sorts.as_ref(),
            page_size,
            start_cursor.as_deref(),
        )?,

        NotionAction::CreateDatabase {
            parent,
            title,
            properties,
            icon,
            cover,
            is_inline,
        } => api::create_database(
            &parent,
            &title,
            &properties,
            icon.as_ref(),
            cover.as_ref(),
            is_inline,
        )?,

        NotionAction::UpdateDatabase {
            database_id,
            title,
            properties,
        } => api::update_database(&database_id, title.as_ref(), properties.as_ref())?,

        // Comments
        NotionAction::GetComments {
            block_id,
            page_size,
            start_cursor,
        } => api::get_comments(&block_id, page_size, start_cursor.as_deref())?,

        NotionAction::AddComment { parent, rich_text } => api::add_comment(&parent, &rich_text)?,

        // Users
        NotionAction::ListUsers {
            page_size,
            start_cursor,
        } => api::list_users(page_size, start_cursor.as_deref())?,

        NotionAction::GetUser { user_id } => api::get_user(&user_id)?,

        NotionAction::GetMe => api::get_me()?,
    };

    serde_json::to_string(&result).map_err(|e| format!("Failed to serialize response: {}", e))
}

// Export the tool implementation.
export!(NotionTool);

/// JSON Schema for the tool's parameters.
///
/// This schema uses `oneOf` to describe each action's specific parameters.
const TOOL_SCHEMA: &str = r#"{
  "type": "object",
  "required": ["action"],
  "oneOf": [
    {
      "properties": {
        "action": { "const": "search" },
        "query": { "type": "string", "description": "Text to search for" },
        "filter": {
          "type": "object",
          "properties": {
            "property": { "type": "string", "const": "object" },
            "value": { "type": "string", "enum": ["page", "database"] }
          },
          "description": "Filter by object type"
        },
        "page_size": { "type": "integer", "minimum": 1, "maximum": 100, "default": 10 },
        "start_cursor": { "type": "string", "description": "Pagination cursor" }
      },
      "required": ["action"]
    },
    {
      "properties": {
        "action": { "const": "get_page" },
        "page_id": { "type": "string", "description": "Page ID (UUID)" }
      },
      "required": ["action", "page_id"]
    },
    {
      "properties": {
        "action": { "const": "create_page" },
        "parent": {
          "type": "object",
          "description": "Parent reference: {\"database_id\": \"...\"} or {\"page_id\": \"...\"}"
        },
        "properties": { "type": "object", "description": "Page properties" },
        "children": {
          "type": "array",
          "items": { "type": "object" },
          "description": "Block children for page content"
        },
        "icon": { "type": "object", "description": "Page icon" },
        "cover": { "type": "object", "description": "Page cover image" }
      },
      "required": ["action", "parent", "properties"]
    },
    {
      "properties": {
        "action": { "const": "update_page" },
        "page_id": { "type": "string" },
        "properties": { "type": "object", "description": "Properties to update" },
        "icon": { "type": "object" },
        "cover": { "type": "object" }
      },
      "required": ["action", "page_id", "properties"]
    },
    {
      "properties": {
        "action": { "const": "archive_page" },
        "page_id": { "type": "string" }
      },
      "required": ["action", "page_id"]
    },
    {
      "properties": {
        "action": { "const": "restore_page" },
        "page_id": { "type": "string" }
      },
      "required": ["action", "page_id"]
    },
    {
      "properties": {
        "action": { "const": "get_blocks" },
        "block_id": { "type": "string", "description": "Block or page ID" },
        "page_size": { "type": "integer", "minimum": 1, "maximum": 100, "default": 50 },
        "start_cursor": { "type": "string" }
      },
      "required": ["action", "block_id"]
    },
    {
      "properties": {
        "action": { "const": "append_blocks" },
        "block_id": { "type": "string", "description": "Block or page ID to append to" },
        "children": {
          "type": "array",
          "items": { "type": "object" },
          "description": "Block objects to append"
        },
        "after": { "type": "string", "description": "Insert after this block ID" }
      },
      "required": ["action", "block_id", "children"]
    },
    {
      "properties": {
        "action": { "const": "get_block" },
        "block_id": { "type": "string" }
      },
      "required": ["action", "block_id"]
    },
    {
      "properties": {
        "action": { "const": "update_block" },
        "block_id": { "type": "string" },
        "content": { "type": "object", "description": "Block content by type" }
      },
      "required": ["action", "block_id", "content"]
    },
    {
      "properties": {
        "action": { "const": "delete_block" },
        "block_id": { "type": "string" }
      },
      "required": ["action", "block_id"]
    },
    {
      "properties": {
        "action": { "const": "get_database" },
        "database_id": { "type": "string" }
      },
      "required": ["action", "database_id"]
    },
    {
      "properties": {
        "action": { "const": "query_database" },
        "database_id": { "type": "string" },
        "filter": { "type": "object", "description": "Notion filter object" },
        "sorts": {
          "type": "array",
          "items": { "type": "object" },
          "description": "Sort configuration"
        },
        "page_size": { "type": "integer", "minimum": 1, "maximum": 100, "default": 10 },
        "start_cursor": { "type": "string" }
      },
      "required": ["action", "database_id"]
    },
    {
      "properties": {
        "action": { "const": "create_database" },
        "parent": { "type": "object", "description": "Parent page or workspace" },
        "title": {
          "type": "array",
          "items": { "type": "object" },
          "description": "Database title as rich text"
        },
        "properties": { "type": "object", "description": "Property schema" },
        "icon": { "type": "object" },
        "cover": { "type": "object" },
        "is_inline": { "type": "boolean", "default": false }
      },
      "required": ["action", "parent", "title", "properties"]
    },
    {
      "properties": {
        "action": { "const": "update_database" },
        "database_id": { "type": "string" },
        "title": { "type": "array", "items": { "type": "object" } },
        "properties": { "type": "object" }
      },
      "required": ["action", "database_id"]
    },
    {
      "properties": {
        "action": { "const": "get_comments" },
        "block_id": { "type": "string", "description": "Block or page ID" },
        "page_size": { "type": "integer", "minimum": 1, "maximum": 100, "default": 50 },
        "start_cursor": { "type": "string" }
      },
      "required": ["action", "block_id"]
    },
    {
      "properties": {
        "action": { "const": "add_comment" },
        "parent": {
          "type": "object",
          "description": "{\"page_id\": \"...\"} or {\"discussion_id\": \"...\"}"
        },
        "rich_text": {
          "type": "array",
          "items": { "type": "object" },
          "description": "Comment content as rich text"
        }
      },
      "required": ["action", "parent", "rich_text"]
    },
    {
      "properties": {
        "action": { "const": "list_users" },
        "page_size": { "type": "integer", "minimum": 1, "maximum": 100, "default": 50 },
        "start_cursor": { "type": "string" }
      },
      "required": ["action"]
    },
    {
      "properties": {
        "action": { "const": "get_user" },
        "user_id": { "type": "string" }
      },
      "required": ["action", "user_id"]
    },
    {
      "properties": {
        "action": { "const": "get_me" }
      },
      "required": ["action"]
    }
  ]
}"#;
