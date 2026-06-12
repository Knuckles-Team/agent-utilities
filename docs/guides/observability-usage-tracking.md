# Usage, Cost & Observability

Track token usage, cost, model/tool/skill/db-call metrics, and Langfuse traces
across **every AI coding agent you run** — and across our own agent runtime —
through one gateway API and native views in all three frontends (agent-webui,
agent-terminal-ui, geniusbot).

This assimilates the capabilities of [agentsview](https://github.com/) natively
into agent-utilities: a 36-agent session parser, a LiteLLM pricing catalog, a
backend-abstracted analytics store, and a REST + MCP surface.

> **TL;DR — it's zero-config.** Start the gateway. Installed agents are
> auto-detected, their logs are parsed and priced on a schedule, your own runs
> are recorded automatically, and the UIs light up. Nothing to configure for the
> local single-host case.

---

## Concepts at a glance

| Piece | What it does | Concept |
| --- | --- | --- |
| Pricing catalog | LiteLLM rates + offline fallback, model-name resolution | ECO-4.40 |
| UsageStore | Backend-abstracted fact store (SQLite/Postgres/DuckDB) | ECO-4.39 |
| Agent-source registry | Auto-detects + parses 36 agents' session logs | ECO-4.38 |
| Runtime instrumentation | Records our own graph runs + tool/skill/db calls | OS-5.31 |
| Gateway API + MCP tools | `/api/observability/*` + `usage_query`/`ingest_sessions` | ECO-4.41 |
| Remote ingest transport | Client-parses, server-sinks (no server FS access) | ECO-4.42 |

**Two data planes, one store.** Plane A = ingested *external* agent logs
(historical). Plane B = our *own* runtime telemetry (live). Both land in the same
store keyed by an `origin` column, so one API and one set of views serve both.

---

## Quick start (local, zero-config)

1. Run the gateway (agent-webui backend or `graph-os-daemon`). On startup the
   consolidated daemon registers two jobs automatically:
   - `usage_log_sync` (every 15 min) — auto-detects installed agents and syncs
     their logs into the store.
   - `usage_pricing_refresh` (daily) — refreshes the LiteLLM pricing catalog.
2. Open any frontend's **Usage & Cost** view. Done.

To force an immediate sync instead of waiting for the tick:

```bash
curl -X POST http://localhost:9000/api/observability/sync
```

or from an agent via MCP: `ingest_sessions(action="collect")`.

Check what was auto-detected:

```python
from agent_utilities.ingestion.agent_sources import ensure_parsers_loaded, detect_installed
ensure_parsers_loaded()
print([s.agent_type for s in detect_installed()])
# e.g. ['claude', 'gemini', 'opencode', 'antigravity']
```

### What "auto-config" means here

You do **not** list which agents you use, where their logs live, which models
cost what, or where to store data. The system:

- **Auto-detects agents** by probing each source's default dirs (`~/.claude/projects`,
  `~/.codex/sessions`, …). Only agents whose logs actually exist are synced.
- **Auto-prices** every model from the bundled offline table (no network, no keys)
  and refreshes from LiteLLM when online.
- **Auto-selects storage**: per-host SQLite+FTS5 by default (no external deps).
- **Auto-records our own runs**: every graph run is recorded beside the existing
  Langfuse export — no instrumentation calls in your code.
- **Auto-registers** the sync + pricing daemon jobs.

---

## The API

Mounted at `/api/observability` (auth + metrics + rate-limit inherited from the
gateway). All endpoints accept `from`, `to`, `project`, `agent`, `model`,
`origin` (`ingested|runtime`), `tenant_id` filters.

| Endpoint | Returns |
| --- | --- |
| `GET /summary` | tokens, cost, cache-hit, session count |
| `GET /comparison` | period-over-period cost delta |
| `GET /by-model`, `/by-project`, `/by-agent` | cost+tokens breakdown |
| `GET /analytics/tools` | tool/skill/db call freq + success rate |
| `GET /analytics/activity` (alias `/heatmap`) | day×hour heatmap |
| `GET /analytics/session-shape` | quick/standard/deep/marathon archetypes |
| `GET /top-sessions`, `/sessions`, `/sessions/{id}` | session browser + detail |
| `GET /search?q=` | full-text search over messages |
| `GET /traces` | Langfuse trace links (gated on credentials) |
| `POST /sessions/upload` | ingest pre-parsed bundles (remote transport) |
| `POST /sync` | trigger an immediate local sync |

### MCP tools (for agents)

- `usage_query(action=…)` — `summary | by_model | by_project | by_agent | tools |
  activity | sessions | session_detail | top_sessions | search | traces`.
- `ingest_sessions(action=…)` — `collect` (auto-detect + parse local logs),
  `upload` (push pre-parsed bundles JSON), `paths` (explicit files/dirs).

---

## Remote / central knowledge graph

When the engine/knowledge-graph is hosted on a **different** machine than where
your agent logs live, the logs are not on the server. agent-utilities closes this
gap with a **client-parses, server-sinks** model — the parser runs where the
files are, and only normalized rows travel to the server.

The collector auto-detects this: if `KG_DAEMON_ROLE=client` or a remote
`GRAPH_ENGINE_ENDPOINT` is set, `collect_local_sessions()` **pushes** instead of
writing locally. Point it at the central gateway:

```bash
export USAGE_GATEWAY_URL=https://graph-os.arpa     # central engine
export USAGE_TENANT_ID=my-team                       # optional tenant scope
```

Then either let the daemon tick handle it, run `POST /api/observability/sync`, or
from an agent call `ingest_sessions(action="collect")`. Under the hood each batch
is POSTed to `/api/observability/sessions/upload` (or sent via the
`ingest_sessions(action="upload")` MCP tool) — the server never reads the
client's filesystem.

**Team mirror (à la `agentsview pg push`).** Because the store is
backend-abstracted, a host can keep a local SQLite store and replicate to a
central Postgres backend by pointing `USAGE_DB_BACKEND=postgres` +
`STATE_DB_URI=postgresql://…` at the shared instance — same interface, different
target.

---

## Storage backends

`USAGE_DB_BACKEND` selects the store (all share one query-shape):

| Value | When | Notes |
| --- | --- | --- |
| `sqlite` (default) | single host, zero deps | SQLite + FTS5, per-host XDG file |
| `postgres` | enterprise / multi-host shared | `tsvector` search; via `STATE_DB_URI` |
| `duckdb` | heavy columnar analytics mirror | `pip install duckdb`; substring search |

---

## Configuration reference (all optional)

| Env / flag | Default | Purpose |
| --- | --- | --- |
| `USAGE_TRACKING_ENABLED` | `true` | Master switch for runtime recording (plane B) |
| `USAGE_DB_BACKEND` | `sqlite` | `sqlite` \| `postgres` \| `duckdb` |
| `USAGE_DB_URI` | — | Explicit store path/URI (else derived) |
| `USAGE_DB_PATH` | `~/.local/share/agent-utilities/usage.db` | SQLite file |
| `PRICING_LITELLM_URL` | BerriAI JSON | Pricing source (offline fallback if unreachable) |
| `USAGE_SYNC_INTERVAL` | `900` | Local-log sync cadence (s) |
| `USAGE_PRICING_REFRESH_INTERVAL` | `86400` | Pricing refresh cadence (s) |
| `USAGE_GATEWAY_URL` | — | Central gateway for remote push |
| `USAGE_TENANT_ID` | — | Tenant scope for pushed/recorded rows |
| `<AGENT>_DIR` / `CLAUDE_PROJECTS_DIR` etc. | per-agent defaults | Override a source's log dir |

Every flag has a sensible default — none are required for the local case.

---

## Frontends

All three consume the same `/api/observability/*` surface and present the same
feature set (no divergence): usage/cost summary, cost by model/project/agent,
token counts, tool/skill/db-call metrics, activity heatmap, session browser,
session detail/timeline, top sessions, session-shape, FTS search, and Langfuse
traces (when enabled).

- **agent-webui** — `Usage` view (`src/components/views/UsageView.tsx`).
- **agent-terminal-ui** — `UsageScreen` (`Alt+U` or `/usage`); reconciles the
  live local `cost_tracker` against gateway-historical.
- **geniusbot** — `Usage & Cost` cockpit panel.
