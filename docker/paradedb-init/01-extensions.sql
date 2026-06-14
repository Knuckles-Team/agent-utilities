-- ParadeDB (pgvector + pg_search) initialization for agent-utilities KG
-- This runs on first container startup via docker-entrypoint-initdb.d

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ParadeDB extensions (included in paradedb/paradedb image)
-- pg_search provides BM25 scoring
CREATE EXTENSION IF NOT EXISTS pg_search;

-- Postgres extension (must be installed separately if not in image)
