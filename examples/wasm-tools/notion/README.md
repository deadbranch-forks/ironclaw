# Notion WASM Tool

A WASM-sandboxed Notion integration for NEAR Agent. Follows the MCP server pattern with domain-grouped operations.

## Building

```bash
cargo build --release --target wasm32-wasip1
```

The compiled module will be at `target/wasm32-wasip1/release/notion_tool.wasm`.

## Capabilities

This tool requires:

- **HTTP**: Access to `api.notion.com/v1/*` (GET, POST, PATCH, DELETE)
- **Secrets**: `notion_api_token` (injected as Bearer authorization)

See `notion_tool.capabilities.json` for the full capability configuration.

## Setup

1. Create a Notion integration at https://www.notion.so/my-integrations
2. Copy the "Internal Integration Secret"
3. Add it to the agent's secrets store as `notion_api_token`
4. Share your Notion pages/databases with the integration

## Supported Actions

### Search

| Action | Required Params | Optional Params |
|--------|-----------------|-----------------|
| `search` | - | `query`, `filter`, `page_size`, `start_cursor` |

### Pages

| Action | Required Params | Optional Params |
|--------|-----------------|-----------------|
| `get_page` | `page_id` | - |
| `create_page` | `parent`, `properties` | `children`, `icon`, `cover` |
| `update_page` | `page_id`, `properties` | `icon`, `cover` |
| `archive_page` | `page_id` | - |
| `restore_page` | `page_id` | - |

### Blocks

| Action | Required Params | Optional Params |
|--------|-----------------|-----------------|
| `get_blocks` | `block_id` | `page_size`, `start_cursor` |
| `append_blocks` | `block_id`, `children` | `after` |
| `get_block` | `block_id` | - |
| `update_block` | `block_id`, `content` | - |
| `delete_block` | `block_id` | - |

### Databases

| Action | Required Params | Optional Params |
|--------|-----------------|-----------------|
| `get_database` | `database_id` | - |
| `query_database` | `database_id` | `filter`, `sorts`, `page_size`, `start_cursor` |
| `create_database` | `parent`, `title`, `properties` | `icon`, `cover`, `is_inline` |
| `update_database` | `database_id` | `title`, `properties` |

### Comments

| Action | Required Params | Optional Params |
|--------|-----------------|-----------------|
| `get_comments` | `block_id` | `page_size`, `start_cursor` |
| `add_comment` | `parent`, `rich_text` | - |

### Users

| Action | Required Params | Optional Params |
|--------|-----------------|-----------------|
| `list_users` | - | `page_size`, `start_cursor` |
| `get_user` | `user_id` | - |
| `get_me` | - | - |

## Examples

### Search for pages

```json
{
  "action": "search",
  "query": "meeting notes",
  "filter": { "property": "object", "value": "page" },
  "page_size": 10
}
```

### Query a database with filters

```json
{
  "action": "query_database",
  "database_id": "abc123-def456-...",
  "filter": {
    "property": "Status",
    "select": { "equals": "Done" }
  },
  "sorts": [
    { "property": "Created", "direction": "descending" }
  ],
  "page_size": 20
}
```

### Create a page in a database

```json
{
  "action": "create_page",
  "parent": { "database_id": "abc123-def456-..." },
  "properties": {
    "Name": {
      "title": [{ "text": { "content": "New Task" } }]
    },
    "Status": {
      "select": { "name": "To Do" }
    }
  },
  "children": [
    {
      "object": "block",
      "type": "paragraph",
      "paragraph": {
        "rich_text": [{ "text": { "content": "Task description here." } }]
      }
    }
  ]
}
```

### Append content to a page

```json
{
  "action": "append_blocks",
  "block_id": "page-id-here",
  "children": [
    {
      "object": "block",
      "type": "heading_2",
      "heading_2": {
        "rich_text": [{ "text": { "content": "New Section" } }]
      }
    },
    {
      "object": "block",
      "type": "paragraph",
      "paragraph": {
        "rich_text": [{ "text": { "content": "Some content..." } }]
      }
    }
  ]
}
```

### Add a comment

```json
{
  "action": "add_comment",
  "parent": { "page_id": "page-id-here" },
  "rich_text": [
    { "text": { "content": "This is a comment from the agent." } }
  ]
}
```

## Security

- Runs in WASM sandbox with fuel metering and memory limits
- API token injected at host boundary (never visible to WASM)
- Only `api.notion.com/v1/*` endpoints allowed
- Responses scanned for secret leakage
- Rate limited: 50 req/min, 1000 req/hour

## References

- [Notion API Documentation](https://developers.notion.com/reference/intro)
- [Notion MCP Server](https://github.com/makenotion/notion-mcp-server)
- [NEAR Agent WASM Tool System](../../../src/tools/wasm/)
