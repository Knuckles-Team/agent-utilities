# Native Database Traversal (CONCEPT:ECO-4.33)

## Overview
Agent tools that let an agent (including RLM-driven recursive agents, ORCH-1.1)
**natively traverse a database** — list tables, inspect schema, and run live
read queries — over the single `UniversalConnector` abstraction (KG-2.9). One set
of tools works **universally** across **PostgreSQL, MySQL/MariaDB, MS SQL Server,
Oracle, SQLite, and MongoDB**; the DSN scheme selects the backend
(`postgresql://`, `mariadb://`, `mssql://`, `oracle://`, `sqlite:///`,
`mongodb://`). This is a capability Onyx lacks entirely — Onyx ships **zero**
database connectors. Here a database is both an *ingestion* source (the
`database` document-source connector, ECO-4.25) and an *interactive* agent tool.

Safety: read-only by default (a deny-list blocks DDL/DML); writes require
`DB_TOOLS_ALLOW_WRITE=1` and are routed through a committing path. Connection
strings resolve from `{ALIAS}_DSN` env vars so secrets never enter agent text.

## Implementation Details
- **Source Code**: `agent_utilities/tools/db_tools.py` (`db_tables`, `db_schema`,
  `db_query`), gated by `DB_TOOLS` in `agent_utilities/tools/tool_registry.py`
- **Backend abstraction**: `agent_utilities/protocols/universal_connector.py` (KG-2.9)
- **Pillar**: ECO
