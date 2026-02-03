//! Workspace and memory system (OpenClaw-inspired).
//!
//! The workspace provides persistent memory for agents:
//! - **MEMORY.md**: Long-term curated memory (facts, decisions, preferences)
//! - **Daily logs**: Append-only daily notes (raw context)
//! - **Identity files**: Agent personality and user context
//! - **HEARTBEAT.md**: Periodic checklist for proactive execution
//!
//! Memory is searchable via hybrid search (FTS + semantic embeddings).
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                        Workspace                             │
//! │  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐  │
//! │  │ MemoryDocument │  │  MemoryChunk   │  │   Search     │  │
//! │  │ (full docs)    │──│ (chunked)      │──│ (FTS+vector) │  │
//! │  └────────────────┘  └────────────────┘  └──────────────┘  │
//! │           │                   │                  │          │
//! │           └───────────────────┴──────────────────┘          │
//! │                           │                                  │
//! │                    ┌──────┴──────┐                          │
//! │                    │  Repository │                          │
//! │                    │ (PostgreSQL)│                          │
//! │                    └─────────────┘                          │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Key Patterns
//!
//! 1. **Memory is persistence**: If you want to remember something, write it
//! 2. **Two-tier memory**: Daily logs (raw) + MEMORY.md (curated)
//! 3. **Hybrid search**: Vector similarity + BM25 full-text via RRF

mod chunker;
mod document;
mod embeddings;
mod repository;
mod search;

pub use chunker::{ChunkConfig, chunk_document};
pub use document::{DocType, MemoryChunk, MemoryDocument};
pub use embeddings::{EmbeddingProvider, OpenAiEmbeddings};
pub use repository::Repository;
pub use search::{SearchConfig, SearchResult};

use std::sync::Arc;

use chrono::{NaiveDate, Utc};
use deadpool_postgres::Pool;
use uuid::Uuid;

use crate::error::WorkspaceError;

/// Workspace provides database-backed memory storage for an agent.
///
/// Each workspace is scoped to a user (and optionally an agent).
/// Documents are persisted to PostgreSQL and indexed for search.
pub struct Workspace {
    /// User identifier (from channel).
    user_id: String,
    /// Optional agent ID for multi-agent isolation.
    agent_id: Option<Uuid>,
    /// Database repository.
    repo: Repository,
    /// Embedding provider for semantic search.
    embeddings: Option<Arc<dyn EmbeddingProvider>>,
}

impl Workspace {
    /// Create a new workspace for a user.
    pub fn new(user_id: impl Into<String>, pool: Pool) -> Self {
        Self {
            user_id: user_id.into(),
            agent_id: None,
            repo: Repository::new(pool),
            embeddings: None,
        }
    }

    /// Create a workspace with a specific agent ID.
    pub fn with_agent(mut self, agent_id: Uuid) -> Self {
        self.agent_id = Some(agent_id);
        self
    }

    /// Set the embedding provider for semantic search.
    pub fn with_embeddings(mut self, provider: Arc<dyn EmbeddingProvider>) -> Self {
        self.embeddings = Some(provider);
        self
    }

    /// Get the user ID.
    pub fn user_id(&self) -> &str {
        &self.user_id
    }

    /// Get the agent ID.
    pub fn agent_id(&self) -> Option<Uuid> {
        self.agent_id
    }

    // ==================== Document Access ====================

    /// Get the main MEMORY.md document (long-term curated memory).
    ///
    /// Creates it if it doesn't exist.
    pub async fn memory(&self) -> Result<MemoryDocument, WorkspaceError> {
        self.repo
            .get_or_create_document(&self.user_id, self.agent_id, DocType::Memory, None)
            .await
    }

    /// Get today's daily log.
    ///
    /// Daily logs are append-only and keyed by date.
    pub async fn today_log(&self) -> Result<MemoryDocument, WorkspaceError> {
        let today = Utc::now().date_naive();
        self.daily_log(today).await
    }

    /// Get a daily log for a specific date.
    pub async fn daily_log(&self, date: NaiveDate) -> Result<MemoryDocument, WorkspaceError> {
        let title = date.format("%Y-%m-%d").to_string();
        self.repo
            .get_or_create_document(
                &self.user_id,
                self.agent_id,
                DocType::DailyLog,
                Some(&title),
            )
            .await
    }

    /// Get the heartbeat checklist (HEARTBEAT.md).
    pub async fn heartbeat_checklist(&self) -> Result<Option<String>, WorkspaceError> {
        match self
            .repo
            .get_document(&self.user_id, self.agent_id, DocType::Heartbeat, None)
            .await
        {
            Ok(doc) => Ok(Some(doc.content)),
            Err(WorkspaceError::DocumentNotFound { .. }) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Get a document by type.
    pub async fn get_document(
        &self,
        doc_type: DocType,
        title: Option<&str>,
    ) -> Result<MemoryDocument, WorkspaceError> {
        self.repo
            .get_document(&self.user_id, self.agent_id, doc_type, title)
            .await
    }

    // ==================== Memory Operations ====================

    /// Append an entry to the main MEMORY.md document.
    ///
    /// This is for important facts, decisions, and preferences worth
    /// remembering long-term.
    pub async fn append_memory(&self, entry: &str) -> Result<(), WorkspaceError> {
        let doc = self.memory().await?;
        let new_content = if doc.content.is_empty() {
            entry.to_string()
        } else {
            format!("{}\n\n{}", doc.content, entry)
        };
        self.repo.update_document(doc.id, &new_content).await?;
        self.reindex_document(doc.id).await?;
        Ok(())
    }

    /// Append an entry to today's daily log.
    ///
    /// Daily logs are raw, append-only notes for the current day.
    pub async fn append_daily_log(&self, entry: &str) -> Result<(), WorkspaceError> {
        let doc = self.today_log().await?;
        let timestamp = Utc::now().format("%H:%M:%S");
        let timestamped_entry = format!("[{}] {}", timestamp, entry);

        let new_content = if doc.content.is_empty() {
            timestamped_entry
        } else {
            format!("{}\n{}", doc.content, timestamped_entry)
        };
        self.repo.update_document(doc.id, &new_content).await?;
        self.reindex_document(doc.id).await?;
        Ok(())
    }

    /// Update a document's content entirely.
    pub async fn update_document(
        &self,
        doc_type: DocType,
        title: Option<&str>,
        content: &str,
    ) -> Result<(), WorkspaceError> {
        let doc = self
            .repo
            .get_or_create_document(&self.user_id, self.agent_id, doc_type, title)
            .await?;
        self.repo.update_document(doc.id, content).await?;
        self.reindex_document(doc.id).await?;
        Ok(())
    }

    // ==================== System Prompt ====================

    /// Build the system prompt from identity files.
    ///
    /// Loads AGENTS.md, SOUL.md, USER.md, and IDENTITY.md to compose
    /// the agent's system prompt.
    pub async fn system_prompt(&self) -> Result<String, WorkspaceError> {
        let mut parts = Vec::new();

        // Load identity files in order of importance
        let identity_types = [
            (DocType::Agents, "## Agent Instructions"),
            (DocType::Soul, "## Core Values"),
            (DocType::User, "## User Context"),
            (DocType::Identity, "## Identity"),
        ];

        for (doc_type, header) in identity_types {
            if let Ok(doc) = self
                .repo
                .get_document(&self.user_id, self.agent_id, doc_type, None)
                .await
            {
                if !doc.content.is_empty() {
                    parts.push(format!("{}\n\n{}", header, doc.content));
                }
            }
        }

        // Add today's memory context (last 2 days of daily logs)
        let today = Utc::now().date_naive();
        let yesterday = today.pred_opt().unwrap_or(today);

        for date in [today, yesterday] {
            if let Ok(doc) = self.daily_log(date).await {
                if !doc.content.is_empty() {
                    let header = if date == today {
                        "## Today's Notes"
                    } else {
                        "## Yesterday's Notes"
                    };
                    parts.push(format!("{}\n\n{}", header, doc.content));
                }
            }
        }

        Ok(parts.join("\n\n---\n\n"))
    }

    // ==================== Search ====================

    /// Hybrid search across all memory documents.
    ///
    /// Combines full-text search (BM25) with semantic search (vector similarity)
    /// using Reciprocal Rank Fusion (RRF).
    pub async fn search(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<SearchResult>, WorkspaceError> {
        self.search_with_config(query, SearchConfig::default().with_limit(limit))
            .await
    }

    /// Search with custom configuration.
    pub async fn search_with_config(
        &self,
        query: &str,
        config: SearchConfig,
    ) -> Result<Vec<SearchResult>, WorkspaceError> {
        // Generate embedding for semantic search if provider available
        let embedding = if let Some(ref provider) = self.embeddings {
            Some(
                provider
                    .embed(query)
                    .await
                    .map_err(|e| WorkspaceError::EmbeddingFailed {
                        reason: e.to_string(),
                    })?,
            )
        } else {
            None
        };

        self.repo
            .hybrid_search(
                &self.user_id,
                self.agent_id,
                query,
                embedding.as_deref(),
                &config,
            )
            .await
    }

    // ==================== Indexing ====================

    /// Re-index a document (chunk and generate embeddings).
    async fn reindex_document(&self, document_id: Uuid) -> Result<(), WorkspaceError> {
        // Get the document
        let doc = self.repo.get_document_by_id(document_id).await?;

        // Chunk the content
        let chunks = chunk_document(&doc.content, ChunkConfig::default());

        // Delete old chunks
        self.repo.delete_chunks(document_id).await?;

        // Insert new chunks
        for (index, content) in chunks.into_iter().enumerate() {
            // Generate embedding if provider available
            let embedding = if let Some(ref provider) = self.embeddings {
                match provider.embed(&content).await {
                    Ok(emb) => Some(emb),
                    Err(e) => {
                        tracing::warn!("Failed to generate embedding: {}", e);
                        None
                    }
                }
            } else {
                None
            };

            self.repo
                .insert_chunk(document_id, index as i32, &content, embedding.as_deref())
                .await?;
        }

        Ok(())
    }

    /// Generate embeddings for chunks that don't have them yet.
    ///
    /// This is useful for backfilling embeddings after enabling the provider.
    pub async fn backfill_embeddings(&self) -> Result<usize, WorkspaceError> {
        let Some(ref provider) = self.embeddings else {
            return Ok(0);
        };

        let chunks = self
            .repo
            .get_chunks_without_embeddings(&self.user_id, self.agent_id, 100)
            .await?;

        let mut count = 0;
        for chunk in chunks {
            match provider.embed(&chunk.content).await {
                Ok(embedding) => {
                    self.repo
                        .update_chunk_embedding(chunk.id, &embedding)
                        .await?;
                    count += 1;
                }
                Err(e) => {
                    tracing::warn!("Failed to embed chunk {}: {}", chunk.id, e);
                }
            }
        }

        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_doc_type_display() {
        assert_eq!(DocType::Memory.as_str(), "memory");
        assert_eq!(DocType::DailyLog.as_str(), "daily_log");
        assert_eq!(DocType::Heartbeat.as_str(), "heartbeat");
    }

    #[test]
    fn test_doc_type_parse() {
        assert_eq!(DocType::try_from("memory").unwrap(), DocType::Memory);
        assert_eq!(DocType::try_from("daily_log").unwrap(), DocType::DailyLog);
        assert!(DocType::try_from("invalid").is_err());
    }
}
