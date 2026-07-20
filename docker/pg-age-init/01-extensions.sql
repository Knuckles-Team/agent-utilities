-- Bootstrap the Apache AGE graph + pgvector (+ ParadeDB pg_search where present)
-- for the agent-utilities pg-age tier. Runs once on first container init
-- (docker-entrypoint-initdb.d). Shared by docker/pg-age and docker/pg-age-full —
-- pg_search only resolves on the -full image (needs shared_preload_libraries), so
-- it is guarded and skipped (with a notice) on the AGE+pgvector-only image.
CREATE EXTENSION IF NOT EXISTS age CASCADE;
CREATE EXTENSION IF NOT EXISTS vector;
LOAD 'age';
SET search_path = ag_catalog, "$user", public;

-- ParadeDB BM25 full-text — present only on the combined pg-age-full image.
DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS pg_search;
EXCEPTION
    WHEN others THEN
        RAISE NOTICE 'pg_search not installed (AGE+pgvector-only image): %', SQLERRM;
END
$$;

-- Create the AGE graph (idempotent: ignore "already exists" on re-init).
DO $$
BEGIN
    PERFORM ag_catalog.create_graph('agent_graph');
EXCEPTION
    WHEN others THEN
        RAISE NOTICE 'create_graph(agent_graph) skipped: %', SQLERRM;
END
$$;

-- Embeddings live in a side table keyed by node id (pgvector), separate from the
-- AGE property graph. The backend creates/uses this; declared here so a fresh DB
-- has it available immediately.
CREATE TABLE IF NOT EXISTS kg_embeddings (
    node_id TEXT PRIMARY KEY,
    embedding vector(768)
);
