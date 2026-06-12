"""DDL for the usage analytics store (CONCEPT:ECO-4.39).

SQLite (default) uses an FTS5 virtual table for message search; Postgres uses a
``tsvector`` GIN column. The base tables are identical so analytics/aggregation
SQL is shared and query-shape parity holds (agentsview backend-parity rule).
"""

from __future__ import annotations

# Shared base tables (valid on SQLite + Postgres; INTEGER/TEXT/REAL are accepted
# by Postgres, BOOLEAN stored as INTEGER 0/1 for parity).
_BASE_TABLES = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    project TEXT NOT NULL DEFAULT '',
    machine TEXT NOT NULL DEFAULT 'local',
    agent TEXT NOT NULL DEFAULT 'claude',
    first_message TEXT NOT NULL DEFAULT '',
    started_at TEXT,
    ended_at TEXT,
    message_count INTEGER NOT NULL DEFAULT 0,
    user_message_count INTEGER NOT NULL DEFAULT 0,
    total_output_tokens INTEGER NOT NULL DEFAULT 0,
    peak_context_tokens INTEGER NOT NULL DEFAULT 0,
    is_automated INTEGER NOT NULL DEFAULT 0,
    outcome TEXT NOT NULL DEFAULT 'unknown',
    health_grade TEXT,
    termination_status TEXT,
    parent_session_id TEXT,
    relationship_type TEXT NOT NULL DEFAULT '',
    origin TEXT NOT NULL DEFAULT 'ingested',
    tenant_id TEXT NOT NULL DEFAULT '',
    correlation_id TEXT NOT NULL DEFAULT '',
    file_path TEXT,
    file_hash TEXT,
    file_mtime INTEGER,
    file_inode INTEGER,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_usage_sessions_started ON sessions(started_at);
CREATE INDEX IF NOT EXISTS idx_usage_sessions_project ON sessions(project);
CREATE INDEX IF NOT EXISTS idx_usage_sessions_agent ON sessions(agent);
CREATE INDEX IF NOT EXISTS idx_usage_sessions_tenant ON sessions(tenant_id);
CREATE INDEX IF NOT EXISTS idx_usage_sessions_origin ON sessions(origin);

CREATE TABLE IF NOT EXISTS messages (
    session_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    thinking_text TEXT NOT NULL DEFAULT '',
    timestamp TEXT,
    model TEXT NOT NULL DEFAULT '',
    context_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    has_tool_use INTEGER NOT NULL DEFAULT 0,
    content_length INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (session_id, ordinal)
);
CREATE INDEX IF NOT EXISTS idx_usage_messages_session ON messages(session_id);

CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    message_ordinal INTEGER,
    tool_name TEXT NOT NULL DEFAULT '',
    category TEXT NOT NULL DEFAULT 'other',
    tool_use_id TEXT,
    input_json TEXT,
    skill_name TEXT,
    result_content_length INTEGER,
    subagent_session_id TEXT,
    status TEXT NOT NULL DEFAULT '',
    occurred_at TEXT,
    origin TEXT NOT NULL DEFAULT 'ingested',
    tenant_id TEXT NOT NULL DEFAULT '',
    correlation_id TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_usage_tool_calls_session ON tool_calls(session_id);
CREATE INDEX IF NOT EXISTS idx_usage_tool_calls_category ON tool_calls(category);
CREATE INDEX IF NOT EXISTS idx_usage_tool_calls_skill ON tool_calls(skill_name);

CREATE TABLE IF NOT EXISTS usage_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    message_ordinal INTEGER,
    source TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cache_creation_input_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_input_tokens INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL,
    cost_status TEXT NOT NULL DEFAULT '',
    cost_source TEXT NOT NULL DEFAULT '',
    occurred_at TEXT,
    dedup_key TEXT NOT NULL DEFAULT '',
    origin TEXT NOT NULL DEFAULT 'ingested',
    tenant_id TEXT NOT NULL DEFAULT '',
    correlation_id TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_usage_events_session ON usage_events(session_id);
CREATE INDEX IF NOT EXISTS idx_usage_events_model ON usage_events(model);
CREATE INDEX IF NOT EXISTS idx_usage_events_occurred ON usage_events(occurred_at);

CREATE TABLE IF NOT EXISTS model_pricing (
    model_pattern TEXT PRIMARY KEY,
    input_per_mtok REAL NOT NULL DEFAULT 0,
    output_per_mtok REAL NOT NULL DEFAULT 0,
    cache_creation_per_mtok REAL NOT NULL DEFAULT 0,
    cache_read_per_mtok REAL NOT NULL DEFAULT 0,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS skipped_files (
    path TEXT PRIMARY KEY,
    mtime INTEGER NOT NULL DEFAULT 0,
    size INTEGER NOT NULL DEFAULT 0
);
"""

# SQLite-only: a dedup unique index (partial) + FTS5 virtual table + triggers.
_SQLITE_EXTRA = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_usage_events_dedup
    ON usage_events(session_id, source, dedup_key) WHERE dedup_key != '';

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    session_id UNINDEXED,
    ordinal UNINDEXED,
    role UNINDEXED,
    tokenize = 'porter unicode61'
);
"""

# Postgres-only: dedup unique index + tsvector column + GIN index.
_POSTGRES_EXTRA = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_usage_events_dedup
    ON usage_events(session_id, source, dedup_key) WHERE dedup_key != '';
ALTER TABLE messages ADD COLUMN IF NOT EXISTS content_tsv tsvector;
CREATE INDEX IF NOT EXISTS idx_messages_tsv ON messages USING GIN(content_tsv);
"""


def sqlite_ddl() -> str:
    # SQLite uses AUTOINCREMENT as written; valid as-is.
    return _BASE_TABLES + _SQLITE_EXTRA


def postgres_ddl() -> str:
    # Postgres: SERIAL-style autoincrement. Rewrite the SQLite token.
    base = _BASE_TABLES.replace(
        "INTEGER PRIMARY KEY AUTOINCREMENT", "BIGSERIAL PRIMARY KEY"
    )
    return base + _POSTGRES_EXTRA
