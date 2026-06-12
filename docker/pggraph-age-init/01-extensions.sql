-- Bootstrap the Apache AGE graph + pgvector for the agent-utilities pggraph tier.
-- Runs once on first container init (docker-entrypoint-initdb.d).
CREATE EXTENSION IF NOT EXISTS age CASCADE;
CREATE EXTENSION IF NOT EXISTS vector;
LOAD 'age';
SET search_path = ag_catalog, "$user", public;

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
