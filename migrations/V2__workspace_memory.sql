-- NEAR Agent Database Schema
-- V2: Workspace and memory system (OpenClaw-inspired)
--
-- This migration adds:
-- 1. Persistent memory documents (MEMORY.md, daily logs, identity files)
-- 2. Chunked content for hybrid search (FTS + vector)
-- 3. Heartbeat state for proactive execution

-- Enable pgvector extension for semantic search
-- NOTE: This requires pgvector to be installed on the PostgreSQL server
-- Install via: CREATE EXTENSION vector; (requires superuser or rds_superuser)
CREATE EXTENSION IF NOT EXISTS vector;

-- ==================== Memory Documents ====================
-- Stores full documents like MEMORY.md, daily logs, identity files
-- Think of this as the filesystem equivalent, but in PostgreSQL

CREATE TABLE memory_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- Ownership: who this document belongs to
    user_id TEXT NOT NULL,              -- User identifier (from channel)
    agent_id UUID,                      -- NULL = shared across all agents for this user

    -- Document type and content
    doc_type TEXT NOT NULL,             -- 'memory', 'daily_log', 'identity', 'soul', 'agents', 'user', 'heartbeat'
    title TEXT,                         -- Optional title (e.g., date for daily logs)
    content TEXT NOT NULL,              -- Full document content

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Flexible metadata (tags, source, etc.)
    metadata JSONB NOT NULL DEFAULT '{}',

    -- Ensure one document per type per user (for singleton docs like MEMORY.md)
    -- Daily logs use title as the date discriminator
    CONSTRAINT unique_doc_per_user_type UNIQUE (user_id, agent_id, doc_type, title)
);

-- Indexes for common queries
CREATE INDEX idx_memory_documents_user ON memory_documents(user_id);
CREATE INDEX idx_memory_documents_user_type ON memory_documents(user_id, doc_type);
CREATE INDEX idx_memory_documents_updated ON memory_documents(updated_at DESC);

-- ==================== Memory Chunks ====================
-- Documents are chunked for search. Each chunk has:
-- 1. Full-text search vector (tsvector) for keyword matching
-- 2. Embedding vector for semantic similarity
--
-- Hybrid search combines both using Reciprocal Rank Fusion (RRF)

CREATE TABLE memory_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES memory_documents(id) ON DELETE CASCADE,

    -- Chunk position and content
    chunk_index INT NOT NULL,           -- Position in document (0-based)
    content TEXT NOT NULL,              -- Chunk text (~800 tokens with 15% overlap)

    -- Full-text search: auto-generated tsvector
    content_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,

    -- Semantic search: embedding vector (OpenAI text-embedding-ada-002 = 1536 dims)
    -- NULL until embeddings are generated
    embedding VECTOR(1536),

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Each chunk index unique per document
    CONSTRAINT unique_chunk_per_doc UNIQUE (document_id, chunk_index)
);

-- GIN index for full-text search
CREATE INDEX idx_memory_chunks_tsv ON memory_chunks USING GIN(content_tsv);

-- HNSW index for vector similarity (cosine distance)
-- HNSW is faster than IVFFlat for reads, slightly slower for writes
CREATE INDEX idx_memory_chunks_embedding ON memory_chunks
    USING hnsw(embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Index for document lookups
CREATE INDEX idx_memory_chunks_document ON memory_chunks(document_id);

-- ==================== Heartbeat State ====================
-- Tracks periodic heartbeat execution per user/agent

CREATE TABLE heartbeat_state (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    agent_id UUID,                      -- NULL = default agent for user

    -- Timing
    last_run TIMESTAMPTZ,               -- When heartbeat last executed
    next_run TIMESTAMPTZ,               -- Scheduled next execution
    interval_seconds INT NOT NULL DEFAULT 1800,  -- 30 minutes default

    -- State
    enabled BOOLEAN NOT NULL DEFAULT true,
    consecutive_failures INT NOT NULL DEFAULT 0,

    -- Last check timestamps (for batched monitoring)
    -- e.g., {"email": "2024-01-15T10:00:00Z", "calendar": "2024-01-15T10:00:00Z"}
    last_checks JSONB NOT NULL DEFAULT '{}',

    -- Ensure one heartbeat config per user/agent
    CONSTRAINT unique_heartbeat_per_user UNIQUE (user_id, agent_id)
);

CREATE INDEX idx_heartbeat_user ON heartbeat_state(user_id);
CREATE INDEX idx_heartbeat_next_run ON heartbeat_state(next_run) WHERE enabled = true;

-- ==================== Helper Functions ====================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger to auto-update updated_at on memory_documents
CREATE TRIGGER update_memory_documents_updated_at
    BEFORE UPDATE ON memory_documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ==================== Views ====================

-- View for documents with chunk counts (useful for debugging)
CREATE VIEW memory_documents_summary AS
SELECT
    d.id,
    d.user_id,
    d.doc_type,
    d.title,
    d.created_at,
    d.updated_at,
    COUNT(c.id) as chunk_count,
    COUNT(c.embedding) as embedded_chunk_count
FROM memory_documents d
LEFT JOIN memory_chunks c ON c.document_id = d.id
GROUP BY d.id;

-- View for pending embedding work
CREATE VIEW chunks_pending_embedding AS
SELECT
    c.id as chunk_id,
    c.document_id,
    d.user_id,
    d.doc_type,
    LENGTH(c.content) as content_length
FROM memory_chunks c
JOIN memory_documents d ON d.id = c.document_id
WHERE c.embedding IS NULL;
