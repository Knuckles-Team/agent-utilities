---
name: kg-sessions
description: >-
  Durable session management plus usage/cost/observability analytics and agent session-log
  ingestion. Use to inspect or steer runs and telemetry — "list sessions", "reply to a
  session", "token/cost usage by model", "ingest agent chat logs".
license: MIT
tags: [graph-os, sessions, observability]
tier: core
wraps: [graph_sessions, ingest_sessions, usage_query]
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-sessions

This skill fronts three verbs:
- **`graph_sessions`** — manage durable sessions: `list`, `get`, `delete`, `reply` (`user_reply` to a `session_id`), `cancel`.
- **`usage_query`** — usage/cost analytics (CONCEPT:ECO-4.41): `summary`, `by_model`/`by_project`/`by_agent`, `tools`, `activity`, `sessions`/`session_detail`/`top_sessions`, `search`, `traces`, `series` (filter by date/project/agent/model).
- **`ingest_sessions`** — ingest agent chat/session history (CONCEPT:ECO-4.42): `collect` (auto-detect local agents), `upload` (pre-parsed `bundles_json` to a remote engine), `paths` (explicit files).

## Invoke
- **MCP:** `load_tools(tools=["graph_sessions"])` (or `usage_query` / `ingest_sessions`), then call it.
- **REST twin:** `POST /graph/sessions` · `POST /usage/query` · `POST /usage/ingest-sessions`.

## Example
```
usage_query(action="by_model", from_date="2026-07-01")
graph_sessions(action="reply", session_id="s1", user_reply="approve")
```
