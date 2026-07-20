# Usage, Cost & Observability

Track token usage, cost, model/tool/skill/db-call metrics, and Langfuse traces
across **every AI coding agent you run** — and across our own agent runtime —
through one gateway API and native views in all three frontends (agent-webui,
agent-terminal-ui, geniusbot).

This assimilates the capabilities of [agentsview](https://github.com/) natively
into agent-utilities: a 36-agent session parser, a LiteLLM pricing catalog, a
backend-abstracted analytics store, and a REST + MCP surface.

> **TL;DR — usage collection is configure-by-default.** After the platform identity
> and secret references are configured, start the REST gateway. Installed agents are
> auto-detected, their logs are parsed and priced on a schedule, your own runs
> are recorded automatically, and the UIs light up. No additional source inventory
> is required for the local single-host case.

---

## Concepts at a glance

| Piece | What it does | Concept |
| --- | --- | --- |
| Pricing catalog | LiteLLM rates + offline fallback, model-name resolution | ECO-4.40 |
| UsageStore | Backend-abstracted fact store (SQLite/Postgres/DuckDB) | AU-OS.observability.usage-analytics-store |
| Agent-source registry | Auto-detects + parses 36 agents' session logs | AU-ECO.connector.agent-source-ingestion |
| Runtime instrumentation | Records our own graph runs + tool/skill/db calls | AU-OS.observability.persist-this-graph-run |
| Gateway API + MCP tools | `/api/observability/*` + `usage_query`/`ingest_sessions` | AU-ECO.mcp.usage-cost-observability-surface |
| Remote ingest transport | Client-parses, server-sinks (no server FS access) | AU-ECO.mcp.client-side-chat-session |

**Two data planes, one privacy boundary.** Plane A = ingested *external* agent
facts (historical). Plane B = our *own* runtime telemetry (live). Both pass
through the same persistence boundary and land in the same store keyed by an
`origin` column. Tenant, run, correlation, parent, tool-use, file, and dedup
identities are stored as tenant-qualified opaque references. Host names and
filesystem locations are never retained.

The default `USAGE_CONTENT_RETENTION=metadata` stores counts, timestamps,
models, outcomes, and costs—not prompts, message/thinking text, or tool inputs.
`sanitized` retention is an explicit local-development opt-in for a separately
governed store; the production profile rejects it.

On first startup with this privacy schema, rows written by older versions are
purged because their raw identities/content cannot be proven policy-safe.
Pricing configuration is retained; subsequent facts use the governed boundary.

---

## Quick start (local)

1. Run the REST gateway with `python -m agent_utilities`. On startup its
   consolidated daemon registers two jobs automatically:
   - `usage_log_sync` (every 15 min) — auto-detects installed agents and syncs
     their logs into the store.
   - `usage_pricing_refresh` (daily) — refreshes the LiteLLM pricing catalog.
2. Open any frontend's **Usage & Cost** view.

`graph-os-daemon` can host the queue and background jobs when the REST process is
not the host, but it serves no HTTP API or frontend. `graph-os` is the MCP server.

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

- **Auto-detects agents** through each registered source adapter's platform-aware
  discovery contract. Only sources with readable records are synchronized. Source
  locations are runtime state and never enter durable facts, traces, or reports.
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
`origin` (`ingested|runtime`), `tenant_id` filters. In the served profile the
verified JWT/GraphSession tenant is authoritative: a caller filter cannot widen
scope, and a usage administrator must name a different tenant explicitly.

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
| `GET /traces` | opaque Langfuse trace references/counts (no host or raw link) |
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
files are, and only metadata-retained, privacy-normalized rows travel to the
server. Source paths, host names, raw identities, and transcript/tool content do
not cross the transport under the default policy.

The collector auto-detects this: if `KG_DAEMON_ROLE=client` or
`GRAPH_SERVICE_ENDPOINTS` is configured, `collect_local_sessions()` **pushes** instead of
writing locally. Point it at the central gateway through AgentConfig or a runtime
deployment override:

```json
{
  "USAGE_GATEWAY_URL": "https://gateway.example.invalid",
  "PERSISTENCE_IDENTITY_HMAC_KEY_REF": "secret://observability/identity-hmac-key"
}
```

Tenant scope comes from the verified request identity. Do not persist a personal or
environment-specific tenant label in source control.

Then either let the daemon tick handle it, run `POST /api/observability/sync`, or
from an agent call `ingest_sessions(action="collect")`. Under the hood each batch
is POSTed to `/api/observability/sessions/upload` (or sent via the
`ingest_sessions(action="upload")` MCP tool) — the server never reads the
client's filesystem.

**Client-side trigger for a remote engine (no gateway URL needed).** When the
engine runs on another host, `collect` runs *engine-side* and can't see this
client's logs (it fails with `"no gateway url"` if `USAGE_GATEWAY_URL` is unset).
The direct fix is the client-side upload command — it parses THIS host's logs
(Claude + Antigravity + every other detected agent) and pushes the bundles to the
remote graph-os over MCP, reusing the fleet client (server resolved from
`mcp_config.json`), so no gateway URL or bespoke HTTP client is required:

```bash
# parse local claude/antigravity/... logs → push to the remote engine via MCP
agent-utilities ingest-sessions --upload --server graph-os
agent-utilities ingest-sessions --upload --url "$SESSION_INGEST_URL" --all
```

It calls `upload_local_sessions()` (`agent_utilities/ingestion/collector.py`),
which drives the remote `ingest_sessions(action="upload", bundles_json=…)` tool in
batches. `--all` re-parses every file; the default syncs only changed files.

**Shared production store.** Because the store is backend-abstracted, a
deployment selects a central Postgres backend with `USAGE_DB_BACKEND=postgres`
and a runtime-injected `STATE_DB_URI`. This switches the authoritative target;
it is not an implicit SQLite replication feature. The production profile guard
rejects per-process SQLite/DuckDB authorities and disabled usage tracking.
It also requires metadata-only usage/Langfuse capture and a secret-backed
`PERSISTENCE_IDENTITY_HMAC_KEY_REF`.

---

## Storage backends

`USAGE_DB_BACKEND` selects the store (all share one query-shape):

| Value | When | Notes |
| --- | --- | --- |
| `sqlite` (development default) | single host, zero deps | SQLite + FTS5, per-host XDG file |
| `postgres` | production / multi-host shared | `tsvector` search; runtime-injected `STATE_DB_URI` |
| `duckdb` | heavy columnar analytics mirror | `pip install duckdb`; substring search |

---

## Langfuse, TLS, and failure evolution

Self-hosted and hosted Langfuse use the same AgentConfig contract:

```json
{
  "LANGFUSE_HOST": "https://observability.example.invalid",
  "LANGFUSE_PUBLIC_KEY_REF": "secret://observability/langfuse-public-key",
  "LANGFUSE_SECRET_KEY_REF": "secret://observability/langfuse-secret-key",
  "LANGFUSE_TLS_PROFILE_REF": "secret://tls/langfuse-profile",
  "LANGFUSE_CAPTURE_CONTENT": false,
  "LANGFUSE_KG_AUTO_INGEST": false
}
```

The TLS profile is resolved at runtime and can contain a complete private CA chain,
mTLS identity, and proxy policy. Verification is mandatory. Do not commit a
certificate path or add a host-specific verification bypass. GraphOS projects the
same resolved trust into the SDK, Requests, SSL, OTLP exporter, and native Langfuse
MCP child.

The MCP child runs from the same installed `agent-utilities[serving]` artifact and
interpreter as GraphOS; it does not invoke a package-on-demand bootstrap. When both
Langfuse credential references resolve:

- `LANGFUSE_MCP_ENABLED` enables automatically unless explicitly set to `false`.
- `KG_FAILURE_EVOLUTION` enables automatically unless explicitly set to `false`;
  it remains propose-only.
- `TRACE_EXPORT_ENABLED` remains an explicit authorization gate.
- `LANGFUSE_CAPTURE_CONTENT` remains `false`; production stays metadata-only.
- `LANGFUSE_KG_AUTO_INGEST` remains `false`. Enabling it also requires a separate
  `LANGFUSE_PERSISTENCE_HMAC_KEY_REF` so project credentials are never reused as
  durable identity material.

When GraphOS mounts the Langfuse child, graph persistence is parent-mediated.
The child receives `LANGFUSE_KG_AUTO_INGEST=false` and remains a read-only API
adapter. After a successful supported read, GraphOS maps and writes the bounded
result under the caller's already-verified `GraphSession`, requiring `kg:write`.
No graph token, identity claim, engine credential, or authority reference is
delegated to the child; missing or read-only authority fails the forwarded call.

Validate reference resolution, trust, the current MCP provider contract, the mounted
child's metadata-only posture and bounded API read, and a metadata-only trace round
trip without displaying resolved material:

```bash
agent-utilities-doctor --only langfuse
agent-utilities-doctor --live
```

See [Failure-Driven Evolution](../architecture/failure_driven_evolution.md) for the
read/write integration and proposal workflow.

---

## Configuration reference

| AgentConfig setting | Default | Purpose |
| --- | --- | --- |
| `USAGE_TRACKING_ENABLED` | `true` | Master switch for runtime recording (plane B) |
| `USAGE_DB_BACKEND` | `sqlite` | `sqlite` \| `postgres` \| `duckdb` |
| `USAGE_CONTENT_RETENTION` | `metadata` | `metadata` (default/production) \| `sanitized` (governed local opt-in) |
| `USAGE_DB_URI` | — | Runtime-injected shared-store connection; otherwise XDG discovery selects the local store |
| `USAGE_DB_PATH` | XDG data location | Optional runtime override; do not commit a machine path |
| `PRICING_LITELLM_URL` | BerriAI JSON | Pricing source (offline fallback if unreachable) |
| `USAGE_SYNC_INTERVAL` | `900` | Local-log sync cadence (s) |
| `USAGE_PRICING_REFRESH_INTERVAL` | `86400` | Pricing refresh cadence (s) |
| `USAGE_GATEWAY_URL` | — | Central gateway for remote push |
| `USAGE_TENANT_ID` | — | Runtime-only collector scope; served requests derive tenant scope from verified identity |
| `PERSISTENCE_IDENTITY_HMAC_KEY_REF` | — | Secret reference for stable opaque durable identities; required in production |
| `ENABLE_OTEL` | `false` | Enable metadata-only OpenTelemetry; GraphOS activates it at startup, and production supplies an OTLP endpoint or a canonical Langfuse credential-reference pair from which one is derived |
| `TRACE_EXPORT_ENABLED` | `false` | Explicitly authorize trace export; credentials alone do not enable emission |
| `LANGFUSE_MCP_ENABLED` | `auto` | Lazy Langfuse MCP child auto-enables when both credentials are ready; explicit `false` opts out |
| `LANGFUSE_CAPTURE_CONTENT` | `false` | Opt in to sanitized trace content across sinks; production remains metadata-only |
| `LANGFUSE_HOST` | hosted service | Langfuse base URL; private trust is selected separately |
| `LANGFUSE_PUBLIC_KEY_REF` / `LANGFUSE_SECRET_KEY_REF` | — | Runtime project credential references; configure as a pair |
| `LANGFUSE_TLS_PROFILE_REF` | — | Runtime verified trust-profile reference |
| `KG_FAILURE_EVOLUTION` | `auto` | Propose-only failure evolution auto-enables with the credential pair; explicit `false` opts out |
| `LANGFUSE_KG_AUTO_INGEST` | `false` | Explicit graph-persistence opt-in; requires `LANGFUSE_PERSISTENCE_HMAC_KEY_REF` |
| `LANGFUSE_PERSISTENCE_HMAC_KEY_REF` | — | Independent HMAC-key reference for opaque persisted trace identities |

Every flag has a sensible local-development default. Production requires the
shared Postgres backend, usage tracking, and an enabled OTLP exporter with its
endpoint injected at runtime. It additionally requires metadata-only content
retention and a secret-backed identity HMAC key reference.

---

## Frontends

All three consume the same `/api/observability/*` surface and present the same
feature set (no divergence): usage/cost summary, cost by model/project/agent,
token counts, tool/skill/db-call metrics, activity heatmap, session browser,
metadata-only session detail/timeline, top sessions, session-shape, governed
search (when sanitized content retention is explicitly enabled), and opaque
Langfuse trace references (when enabled).

- **agent-webui** — `Usage` view (`src/components/views/UsageView.tsx`).
- **agent-terminal-ui** — `UsageScreen` (`Alt+U` or `/usage`); reconciles the
  live local `cost_tracker` against gateway-historical.
- **geniusbot** — `Usage & Cost` cockpit panel.
