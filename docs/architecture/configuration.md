# Configuration Reference & Flag Audit

This is the single, authoritative inventory of every environment variable
`agent-utilities` reads, with a **verdict** for each: is the flag actually needed,
or should the system detect and self-configure instead?

It exists because the codebase had grown to **~96 distinct `KG_*` / `GRAPH_*` /
`EPISTEMIC_*` / `AGENT_UTILITIES_*` flags** — over-configuration that is overwhelming
to operate and a frequent source of footguns. The rule for adding new flags is in
`AGENTS.md` → *Configuration discipline*. The CI gate `scripts/check_no_env_sprawl.py`
enforces that flags are declared on `AgentConfig` (`core/config.py`), not read with bare
`os.environ.get()` scattered across modules.

## How configuration is read (two centralized paths)

`core/config.py` (and `core/paths.py`) are the **only** files that touch `os.environ`.
Every other module reads through one of two paths, both **driven by `config.json`** —
the XDG loader injects `~/.config/agent-utilities/config.json` (or
`AGENT_UTILITIES_CONFIG_DIR`) into `os.environ` before anything reads it, so a JSON key
`graph_db_uri` becomes `GRAPH_DB_URI`:

1. **Typed `AgentConfig` field** (`Field(alias="MY_VAR")`, read `config.my_var`) — for
   static settings parsed once at import.
2. **`config.setting("MY_VAR", default, cast=…)`** — the sanctioned accessor for reads
   that must stay **live** (daemon cadences, test-varied flags, runtime toggles). It
   reads `os.environ` at call time with a declared default and type coercion (inferred
   from the default's type, or pass `cast`).

**Decision:** field for static, `setting()` for dynamic — **never** a bare
`os.environ.get` / `os.getenv` / `os.environ[...]` read in a module.

**The fold is complete: ZERO bare env reads remain anywhere in `agent_utilities/`**
(every prefix — `KG_*`, `GRAPH_*`, `AGENT_*`, `VAULT_*`, `OTEL_*`, connector creds, …).
Deployment-varying/behavioral flags → `setting()` (or a typed field); pure load/cadence
tunables → auto-sized via `compute_ingest_worker_count()` or named module constants.
`scripts/check_no_env_sprawl.py` covers **every prefix** and its baseline
(`scripts/env_flag_baseline.txt`) is **empty** — any new bare read fails CI.
`setting()` itself lives in the dependency-free `core/_env.py` (re-exported by
`config`) so it stays importable while `config` is still initializing.

**Verdict legend**
- **KEEP** — legitimate deployment config (path / DSN / secret / port / socket). Must be
  read via the central `config` object, not bare `os.environ`.
- **AUTO** — should be auto-detected/auto-sized from the runtime; the flag should be
  removed (or kept only as an override with an auto default).
- **REMOVE** — always-on behavior or stale experiment; delete the flag.

## A. Deployment config — KEEP (read via `AgentConfig`)

| Flag | Default | What it sets |
|---|---|---|
| `GRAPH_DB_URI` / `PGGRAPH_DSN` | none | Postgres/pg-age **mirror** DSN |
| `GRAPH_BACKEND` | `epistemic_graph` | Engine mode: `epistemic_graph` (engine authority alone) or `fanout` (engine + mirrors). The `tiered`/`GRAPH_BACKEND_L1` scheme is removed |
| `GRAPH_AUTHORITY` / `GRAPH_MIRROR_TARGETS` | `epistemic_graph` / unset | Read source-of-truth + the fan-out mirror set (mirror names from `KG_CONNECTIONS`); `GRAPH_MIRROR_TARGETS` supersedes the removed `GRAPH_BACKEND_L2` |
| `EPISTEMIC_GRAPH_SOCKET` | `/tmp/epistemic-graph.sock` | Rust engine UDS |
| `GRAPH_PERSISTENCE_PATH`, `GRAPH_SERVICE_PERSIST_DIR` | data dir | Engine durable snapshot dir |
| `GRAPH_DB_HOST/PORT/NAME/USER/PASSWORD/PATH` | — | DSN parts (legacy; prefer `GRAPH_DB_URI`) |
| `GRAPH_FUSEKI_URL/USER/PASSWORD/DATASET` | — | SPARQL endpoint (optional backend) |
| `GITLAB_INSTANCES` | none | JSON list of GitLab instances to index/query — the multi-tenant source of truth shared by the KG GitLab indexer and the `gitlab-api` connector registry. Each entry `{"name":<str>,"url":<str>,"token":<str>,"verify_ssl":<bool>}`. Unset → single-host `GITLAB_URL`/`GITLAB_TOKEN` (CONCEPT:KG-2.9g) |
| `GRAPH_PGGRAPH_SCHEMA` | `public` | Postgres schema |
| `AGENT_UTILITIES_{CONFIG,DATA,CACHE,LOG,MEMORY,RUNTIME}_DIR` | XDG | Path overrides (resolved in `core/paths.py`) |
| `AGENT_UTILITIES_TOKEN_SECRET` | — | Run-scoped tool-token secret |
| `KG_DAEMON_ROLE` | `auto` | host/client/auto election (topology) |
| `STATE_DB_URI` | none | Externalize ALL durable state (durable-exec checkpoints, sessions/turns/goals, KG task queue) to a shared Postgres; unset keeps the zero-infra per-host SQLite files (CONCEPT:OS-5.16) |
| `STATE_DB_POOL_SIZE` | `8` | Max connections in the ONE shared state-store psycopg pool (CONCEPT:OS-5.16) |
| `TASK_QUEUE_BACKEND` | none (auto) | Ingest task queue: `sqlite`\|`postgres`\|`kafka`. Unset = auto (postgres when `STATE_DB_URI` set, else sqlite). Explicit kafka/postgres is FAIL-LOUD at startup (CONCEPT:KG-2.55) |
| `KG_TASKS_PARTITIONS` | `6` | Partitions ensured on the `kg_tasks` topic at startup (grow-only, never shrinks); bounds kg-ingest consumer-group parallelism (CONCEPT:KG-2.56) |
| `AGENT_DISPATCH_BACKEND` | `inline` | How agent turns (goal runs / orchestrator jobs) dispatch: `inline` keeps the in-process execution; `queue` publishes a session-keyed envelope onto the `agent_turns` queue (transport follows `TASK_QUEUE_BACKEND`) and returns a job handle for the `agent-dispatch-worker` fleet (CONCEPT:ORCH-1.45) |
| `AGENT_TURNS_PARTITIONS` | `6` | Partitions ensured on the `agent_turns` topic when Kafka carries dispatched agent turns (grow-only); bounds fleet-wide concurrent-session parallelism (CONCEPT:ORCH-1.45) |
| `EPISTEMIC_GRAPH_AUTOSTART` | — | Auto-spawn the engine (local `unix://` endpoint only; never remote shards) |
| `GRAPH_SERVICE_ENDPOINTS` | unset | Engine shard endpoints (comma/JSON list). 2+ entries = tenant-partitioned sharding via HRW over graph names; unset/1 = single-engine zero-infra default (CONCEPT:KG-2.58) |
| `KG_DEFAULT_GRAPH` | `__bus__` | Default named graph; in sharded mode the ambient ActorContext tenant maps it to `tenant__<t>__<base>` before HRW (CONCEPT:KG-2.58) |
| `KG_WATCH_DIRS` | unset | Operator document directories the file-watcher auto-ingests **recursively**, unified with the built-in ScholarX/research download dirs. New files are ingested and modified files re-ingested on the 5s watch tick; unchanged files delta-skip by content hash (CONCEPT:KG-2.8). Value is a JSON array or an `os.pathsep`/comma-separated list of paths (`~` expanded), e.g. `~/Documents`. config.json key: `kg_watch_dirs`. Resolved by `sdd/watcher.py:get_watched_directories()` |
| `GRAPH_SERVICE_AUTH_SECRET` | auto-generated | Engine HMAC secret; unset → per-install secret persisted at `data_dir()/engine_secret` (0600) (CONCEPT:OS-5.14) |
| `KG_ENGINE_INSECURE` | `false` | Dev opt-out of engine HMAC auth; sets `EPISTEMIC_GRAPH_ALLOW_INSECURE=1` on spawned engines (CONCEPT:OS-5.14) |
| `KG_AUTH_REQUIRED` | `false` | Require server-validated JWT identity for KG access — 401 without it; caller `_actor`/`_roles`/`_tenant` kwargs ignored (CONCEPT:OS-5.14) |
| `KG_AUTH_TOKEN` | — | JWT minting the stdio MCP process identity (validated against `AUTH_JWT_JWKS_URI`) (CONCEPT:OS-5.14) |
| `KG_ACL_DEFAULT_ALLOW` | `false` | With `KG_BRAIN_ENFORCE` on: allow nodes WITHOUT an ACL (escape hatch from the fail-closed default-deny) (CONCEPT:OS-5.14) |
| `KG_SERVED_PROFILE` | `true` | Fail-closed served-security profile for network MCP transports: refuse to start without `AUTH_JWT_JWKS_URI`, auto-enable auth + enforcement. `0` = serve open (local dev only) (CONCEPT:OS-5.14) |
| `KG_ENGINE_POOL_SIZE` | `0` | Max warm per-tenant engine clients per process (elastic LRU pool over KG-2.58 routing); `0` = per-use construction (CONCEPT:KG-2.62) |
| `KG_ENGINE_POOL_DROP_ON_EVICT` | `false` | On pool eviction also unload the tenant's named graph from the engine to reclaim memory — **only safe when a durable mirror holds the data** (CONCEPT:KG-2.62) |
| `GATEWAY_METRICS` | `true` | Python-tier Prometheus middleware + `GET /metrics` on the gateway (CONCEPT:OS-5.23) |
| `GATEWAY_RATE_LIMIT` | `0` (off) | Per-tenant token-bucket rate limit, sustained req/s; buckets are per-process (CONCEPT:OS-5.23) |
| `GATEWAY_RATE_BURST` | `0` (→ 2× rate) | Token-bucket burst capacity (CONCEPT:OS-5.23) |
| `GATEWAY_WORKERS` | `1` | Pre-forked gateway worker processes on one shared listen socket; the flock host-lock elects ONE KG host among them (CONCEPT:OS-5.23) |
| `ENGINE_BREAKER_THRESHOLD` | `5` | Consecutive engine connect/timeout failures before the client circuit opens (0 = off) (CONCEPT:OS-5.23) |
| `ENGINE_BREAKER_COOLDOWN` | `15` | Seconds an open engine circuit waits before the half-open probe (CONCEPT:OS-5.23) |
| `MCP_CHILD_MAX_CONCURRENCY` | `8` | Max in-flight tool calls per multiplexer child (0 = unlimited); per-server `max_concurrency` override in `mcp_config.json` (CONCEPT:ECO-4.34) |
| `MCP_CHILD_QUEUE_TIMEOUT` | `30` | Seconds an excess call queues for a child slot before the typed `MCPChildBusyError`; per-server `queue_timeout` override (CONCEPT:ECO-4.34) |
| `MCP_CHILD_POOL_SIZE` | `1` | Session-pool size for remote (streamable-http/SSE) children — N round-robin connections for parallel calls; stdio stays single-pipe; per-server `pool_size` override (CONCEPT:ECO-4.34) |
| `MCP_CHILD_MAX_RESTARTS` | `5` | Auto-restarts a crashed child may consume inside the window before being parked `failed` (0 = no auto-restart); per-server `max_restarts` override (CONCEPT:ECO-4.34) |
| `MCP_CHILD_RESTART_WINDOW` | `300` | Sliding window (s) for the restart budget; older restarts are forgiven; per-server `restart_window` override (CONCEPT:ECO-4.34) |
| `MCP_CHILD_BREAKER_THRESHOLD` | `5` | Consecutive transport failures/timeouts before a child's circuit opens (typed `MCPChildCircuitOpenError`, 0 = off); per-server `breaker_threshold` override (CONCEPT:ECO-4.34) |
| `MCP_CHILD_BREAKER_COOLDOWN` | `15` | Seconds an open child circuit waits before the half-open probe; per-server `breaker_cooldown` override (CONCEPT:ECO-4.34) |
| `MCP_MULTIPLEXER_MODE` | `eager` | Tool-exposure strategy: `eager` spawns every child and exposes all tools at boot (historical); `dynamic` exposes only the `find_tools`/`load_tools`/`unload_tools`/`multiplexer_status` meta-tools + always-on children, mounting other tools on demand with a `tools/list_changed` notification (CONCEPT:ECO-4.36) |
| `MCP_DYNAMIC_ALWAYS_ON` | `["graph-os"]` | Child servers mounted at boot in `dynamic` mode (in addition to meta-tools); defaults to the KG server so `find_tools` can rank semantically (CONCEPT:ECO-4.36) |
| `MCP_DYNAMIC_TOP_K` | `8` | Default number of ranked candidates `find_tools` returns when `top_k` is unspecified (CONCEPT:ECO-4.36) |
| `MCP_TOOL_MODE` | `condensed` | Per-agent tool surface: `condensed` (action-routed tools, default), `verbose` (one 1:1 tool per API method), or `both`. Read fleet-wide via `tool_mode()`; verbose tools are tagged `verbose` for `--tools`/tag filtering. See [MCP Tool Modes](../guides/mcp-tool-modes.md) (CONCEPT:ECO-4.82) |
| `ACTION_POLICY_PATH` | shipped default | Operational ActionPolicy YAML; empty → conservative `deploy/action-policy.default.yml` (everything mutating = approval_required). KG `governance_rule` overrides win (CONCEPT:OS-5.24) |
| `FLEET_RECONCILER` | `false` | Opt-in leader-only desired-state fleet reconciler tick — diff registry vs observed, converge through the ActionPolicy gate + actuator seam (CONCEPT:OS-5.25) |
| `FLEET_RECONCILER_INTERVAL` | `120` | Seconds between fleet-reconciler ticks (CONCEPT:OS-5.25) |
| `FLEET_RECONCILER_MAX_ACTIONS` | `5` | Storm guard: max convergence actions per tick, rest deferred (CONCEPT:OS-5.25) |
| `FLEET_REGISTRY_PATH` | shipped registry | Fleet service registry YAML; empty → `deploy/mcp-fleet.registry.yml` (CONCEPT:OS-5.25) |
| `FLEET_DESIRED_STATE_PATH` | unset | Optional desired-state override YAML (per-service `replicas`/`desired`/`version`) layered on the registry (CONCEPT:OS-5.25) |
| `FLEET_ACTUATOR` | `dryrun` | Actuator selection: `dryrun` (records intent, mutates nothing) or `docker` (reference CLI actuator). Portainer/Swarm actuators are deployment-wired via `set_fleet_actuator()` (CONCEPT:OS-5.25) |
| `DEPLOY_WATCH_WINDOW` | `300` | Health-watch window (s) after a deploy/restart; failure inside the window triggers the policy-gated rollback (CONCEPT:OS-5.27) |
| `DEPLOY_WATCH_POLL` | `15` | Seconds between health probes inside a deploy watch (CONCEPT:OS-5.27) |
| `FLEET_AUTOSCALER` | `false` | Opt-in leader-only reactive replica autoscaler tick — load signal → registry-declared min/max bounds → policy-gated `scale_service` + deploy watch (CONCEPT:OS-5.29) |
| `FLEET_AUTOSCALER_INTERVAL` | `60` | Seconds between autoscaler ticks (CONCEPT:OS-5.29) |
| `SCALING_PROMETHEUS_URL` | unset | Prometheus base URL for autoscaling signals (instant `/api/v1/query` GETs); unset → zero-infra in-process gauges; injected provider via `set_scaling_signal_provider()` wins (CONCEPT:OS-5.29) |

These genuinely vary per host and aren't derivable. **Action:** ensure each is a typed
`AgentConfig` field; remove duplicate bare reads (`GRAPH_DB_URI` is read in 4 places,
`AGENT_UTILITIES_CONFIG_DIR` in 5).

## B. Daemon on/off toggles, all default ON — REMOVED (Phase 3) ✓

**Done.** The six always-on toggles below were deleted and collapsed behind a single
`KG_DEV_MODE` switch on `AgentConfig`, read through one `engine_tasks._kg_dev_mode()` helper
that gates the maintenance scheduler + embedding-backfill startup. Production keeps every daemon
on; `KG_DEV_MODE=1` silences the lot. Removed: `KG_EMBED_BACKFILL`, `KG_ENRICH_DAEMON`,
`KG_FILE_WATCH`, `KG_HYGIENE_DAEMON`, `KG_TASK_REAPER_DAEMON`, `KG_RECONCILE_DURABLE`. Also fixed
the `KG_EMBED_BACKFILL_BATCH` dual-default bug → two named constants
(`_EMBED_BACKFILL_BUDGET=256`, `_EMBED_BACKFILL_FETCH=512`). Sprawl baseline 95 → 88. (The
`KG_CONCEPT_CODE_LINK` toggle lives in another module and is still pending.)

Original inventory (for reference):

| Flag | Default | Gated thread/job (`engine_tasks.py`) |
|---|---|---|
| `KG_EMBED_BACKFILL` | `1` | vector-embedding backfill drain (L402) |
| `KG_ENRICH_DAEMON` | `1` | semantic enrichment tick (L618) |
| `KG_FILE_WATCH` | `1` | SDD/skills/config file-watch (L635) |
| `KG_HYGIENE_DAEMON` | `1` | memory decay/dedup (L645) |
| `KG_TASK_REAPER_DAEMON` | `1` | zombie-task recovery (L657) |
| `KG_RECONCILE_DURABLE` | `1` | engine→mirror autoheal (L609) |
| `KG_CONCEPT_CODE_LINK` | `1` | concept↔code bridge |
| `GRAPH_DIRECT_DISPATCH` | `true` | sync dispatch |
| `KG_RETRIEVAL_QUALITY_GATE` | `true` | relevance filter |

Nobody runs these off in production. **Action:** delete the env gates; if a dev escape
hatch is wanted, a single `KG_DEV_MODE=1` disables *all* background daemons.

### B.1 Safety overrides — KEEP (typed on `AgentConfig`)

| Flag | Default | What it gates |
|---|---|---|
| `KG_ALLOW_FULL_SCAN` | `false` | Permit an unscoped Cypher query to enumerate the whole graph (ORCH-1.40). Off by default so a buggy unscoped query can never silently full-scan; deliberate opt-in only. Typed `config.kg_allow_full_scan`, read in `backends/epistemic_graph_backend.py`. |

## C. Ingest-throughput knobs — REMOVED ✓

**Done.** All three deleted:
- `KG_INGEST_FEATURES` / `KG_INGEST_PROFILE` → per-repo call-graph community detection is now
  **always on**. The hang risk that motivated the opt-out is fixed at the source: the engine's
  `community_detection` is deterministically bounded (15s wall-clock + iteration cap,
  epistemic-graph `algorithms.rs`), and `make_community_fn` loads its scratch tenant in **one
  `batch_update` round-trip** instead of per-element RPCs.
  *Correction (verified against code):* one `KG_INGEST_PROFILE` read survives — pipeline
  phase selection in `knowledge_graph/pipeline/__init__.py` (`select_phases`; values
  `structural` | `full`, unset = full). It no longer gates community detection. It is
  tracked in the bare-read baseline (see section H).
- `KG_BULK_INGEST` → the maintenance scheduler **auto-detects** a bulk ingest from the durable
  submission-queue depth (`_submission_queue.get_queue_size() > _BULK_QUEUE_THRESHOLD`) and defers
  its whole-graph passes per-tick, instead of a manual startup flag.

## D. Performance tunables — keep as constants / deployment config (auto-sizing deferred)

The one real defect here — the `KG_EMBED_BACKFILL_BATCH` dual-default — is **fixed** (Phase 3:
two named constants). The rest are left as-is by design: the worker pool **already auto-sizes**
(`engine_tasks.py` CPU 36% + mem cap), and the remaining batch sizes / intervals have correct
universal defaults. Per the *Configuration discipline* rule, a tunable with a good default should
be a constant, not a knob — but mechanically converting every one to a CPU-derived auto-sizer is
speculative churn with behaviour-change risk and little payoff, so it is **not** pursued. `GRAPH_TIMEOUT`'s
20-minute default is noted (it made the old community-detection hang look infinite); now moot since
the engine bounds the call itself.

| Flag | Default | Notes |
|---|---|---|
| `KG_LLM_CONCURRENCY` | 4 | typed on `AgentConfig` (`kg_llm_concurrency`); the **total** parallel capacity of the local inference endpoint — the one knob for local-model parallelism. ORCH-1.59 always reserves 1 slot for the interactive path (messaging responder + graph-os-spawned agents, which share the default model); background KG work (enrichment/analysis/embeddings) is bounded to `background_llm_concurrency()` = capacity − 1. Subsumes the former `KG_ENRICH_CONCURRENCY`. |
| `KG_PARSE_BATCH` | 128 | constant |
| `KG_ENRICH_BATCH` / `KG_ENRICH_MAX_BATCHES` | 16 / 8 | constants |
| `KG_EMBED_BACKFILL_BATCH` | **256 *and* 512 (BUG)** | read twice with different defaults (L1040, L1155) — unify |
| `KG_BACKGROUND_MAX_CONCURRENT` | 2 | auto |
| `GRAPH_POOL_MIN/MAX` | 2 / 10 | auto from cpu |
| `KG_CHAT_CONCURRENCY` | 8 | auto |
| `KG_*_INTERVAL` (enrich/file_watch/embed/evolution/golden) | 20–3600 | constants unless deployment-varying |
| `GRAPH_TIMEOUT` | 1200000 ms | 20-min RPC timeout — far too long; the root of "hangs look infinite" |

## E. Experiment / feature gates

**`KG_GOLDEN_*` (10 flags) — collapsed onto `AgentConfig` ✓.** Every `KG_GOLDEN_*` /
`KG_BREADTH_*` read was moved off bare `os.environ` onto typed `AgentConfig` fields —
opt-in, all off by default, single typed source of truth.

| Flag | Default | What it gates |
|---|---|---|
| `KG_LOOP` | `False` | enable the autonomous golden-loop daemon cycle (intake→acquire→resolve→distil→synthesize) |
| `KG_LOOP_DISTILL` | `False` | distillation step of the golden loop |
| `KG_LOOP_BREADTH` | `False` | breadth ingest step (auto-ingest of the roots below) |
| `KG_LOOP_STANDARDIZE` | `False` | standardization pass of the golden loop |
| `KG_GOLDEN_AUTO_MERGE` | `False` | governed auto-merge of promoted proposals |
| `KG_GOLDEN_MERGE_THRESHOLD` | `None` | score threshold gating auto-merge |
| `KG_LOOP_INTERVAL` | `3600` | daemon tick interval (s) |
| `KG_LOOP_TOPICS` | `5` | hot topics processed per cycle |
| `KG_BREADTH_LIBRARY_ROOTS` | `""` | comma-separated OSS library roots auto-ingested by the breadth step (deployment-specific; empty = no-op) |
| `KG_BREADTH_REPO_ROOTS` | `""` | comma-separated code-repo roots auto-ingested by the breadth step |
| `AGENT_THINKING_EFFORT` | `""` | pydantic-ai v2 native extended thinking on every built agent: `low`/`medium`/`high` (empty = off). Opt-in because reasoning is expensive (more tokens/latency); runs natively where the provider supports it and no-ops elsewhere. Read via `setting()` in `agent/factory.py`; per-agent override = the `create_agent(thinking_effort=...)` arg. |

| Family | Count | Notes |
|---|---|---|
| `KG_EA_WRITEBACK`, `KG_ENABLE_HARD_NEGATIVE_MINING`, `KG_BRAIN_ENFORCE`, `KG_RESEARCH_EXTERNAL`, `KG_PROCESS_WRITEBACK` | 5 | remaining experiment gates — graduate (always-on) or delete |

**`KG_PROCESS_WRITEBACK` — outbound process-intelligence writeback (`CONCEPT:KG-2.8`,
default off).** Opt-in because it performs *outbound* mutating calls into external
systems (Camunda process-instance variables, ARIS model attributes) — exactly the
"expensive / side-effecting on every run" case the opt-in tier is reserved for. When
set, `resolve_process_writeback` pushes the KG's per-process intelligence (capability/
code lineage, OWL inferences, operational signals, glossary/data lineage) back onto the
live process world via the `graph_analyze action=process_writeback` surface. The
connector connection/credentials themselves are **not** agent-utilities config — they
live in the `camunda-mcp` / `aris-mcp` packages' own environment
(`CAMUNDA7_*`/`CAMUNDA8_*`, `ARIS_*`), read by each connector's `auth.get_client()`.

**`EVOLUTION_WORKTREE_ROOT` — evolution→branch bridge (`CONCEPT:AHE-3.21`), typed on
`AgentConfig` (`evolution_worktree_root`).** Root directory the `LocalBranchPublisher`
creates fresh git worktrees under when publishing a promoted proposal as a reviewable
local branch. Empty (default) resolves to `data_dir()/evolution_worktrees` — publication
never writes into a canonical checkout's working tree.

**`KG_FAILURE_*` — Failure-Driven Evolution (`CONCEPT:AHE-3.18`), typed on `AgentConfig`,
opt-in, all off by default.** The boolean gates are parsed via `to_boolean`
(`"True"`/`"False"`, consistent with the fleet's other toggles). See
[`failure_driven_evolution.md`](./failure_driven_evolution.md).

| Flag | Default | Notes |
|---|---|---|
| `KG_FAILURE_EVOLUTION` | `False` | enable the daemon `failure_ingest` tick (pull Langfuse failures → remediation) |
| `KG_FAILURE_EVOLUTION_INTERVAL` | `3600` | daemon tick interval (s) |
| `KG_FAILURE_EVOLUTION_WINDOW` | `86400` | telemetry look-back window (s) |
| `KG_FAILURE_REGRESSION_DATASET` | `False` | enable the dataset-based regression path |
| `KG_DSPY_OPTIMIZATION` | `False` | enable the daemon `dspy_optimization` tick — propose-only DSPy optimization sweep over the self-supervised targets (CONCEPT:AHE-3.46) |
| `KG_DSPY_OPTIMIZATION_INTERVAL` | `3600` | DSPy optimization sweep interval (s) |

**`KG_FUSEKI_*` — Ontology distribution to Apache Jena Fuseki (`CONCEPT:KG-2.52`), typed on
`AgentConfig`, opt-in.** The `fuseki_publish` maintenance tick pushes the bundled ontology
modules (the authoritative TBox) to an optional enterprise Fuseki triplestore for SPARQL
federation. Off by default — Fuseki is optional infrastructure.

| Flag | Default | Notes |
|---|---|---|
| `KG_FUSEKI_PUBLISH` | `False` | enable the daemon `fuseki_publish` tick |
| `KG_FUSEKI_ENDPOINT` | `None` | Fuseki URL; `None` defers to the publisher (`FUSEKI_ENDPOINT`, then localhost) |
| `KG_FUSEKI_PUBLISH_INTERVAL` | `3600` | daemon tick interval (s) |

**`KG_WORKFLOW_SHAPE_GATE` — execution-time workflow ontology gate (`CONCEPT:ORCH-1.42`),
typed on `AgentConfig`, default ON.** `execute_workflow` AND its background twin
`dispatch_workflow` (REST twin `/api/graph/orchestrate/dispatch-workflow`) SHACL-validate
the stored `WorkflowDefinition` (+ steps) against the governance shapes before dispatch and
refuse malformed definitions with a structured violation report; cheap and LLM-free. The
companion permission gate (ontology permissioning ACL on the workflow node) is governed by
the existing `KG_BRAIN_ENFORCE` flag (OS-5.14 fail-closed semantics), not a new one.

| Flag | Default | Notes |
|---|---|---|
| `KG_WORKFLOW_SHAPE_GATE` | `True` | SHACL-validate stored workflows before execution |

**Langfuse (`CONCEPT:AHE-3.18` / `AHE-3.0`) — official SDK variable names only.** The host
variable is **`LANGFUSE_HOST`** (the non-standard `LANGFUSE_BASE_URL` fallback was removed —
greenfield). Resolved through `AgentConfig.langfuse_host` / `langfuse_public_key` /
`langfuse_secret_key`.

| Flag | Default | Notes |
|---|---|---|
| `LANGFUSE_HOST` | `https://cloud.langfuse.com` | Langfuse base URL (read + OTEL write paths) |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | `None` | project API keypair |

## F. Testing — KEEP

| Flag | Default | Notes |
|---|---|---|
| `AGENT_UTILITIES_TESTING` | `False` | test-mode guard (read in ~8 places — centralize) |
| `AGENT_UTILITIES_GWT_STRICT` | `""` | global-workspace strict test mode |

## G. Complete `AgentConfig` inventory — platform fields beyond the KG/graph flag audit

Sections A–F are the original `KG_*`/`GRAPH_*` sprawl audit. `AgentConfig`
(`core/config.py`, pydantic-settings) additionally carries the platform's general
configuration surface. **Totals, extracted programmatically from
`AgentConfig.model_fields`: 244 fields, 242 distinct environment variables**
(`SECRETS_VAULT_URL` and `SECRETS_VAULT_MOUNT` each bind two fields — `vault_url`/
`secrets_vault_url` and `vault_mount`/`secrets_vault_mount` — a legacy duplication
kept for compatibility).

Environment-name resolution: every field declares an explicit `alias` which IS its
environment variable name (no `env_prefix`; matching is case-insensitive). Sources in
precedence order: constructor args → environment → `.env` → nested secrets file
(`AGENT_SECRETS_FILE`) → Docker secrets (`/run/secrets`). All fields below are
therefore environment-settable; none are internal-only.

### G.1 Model providers & routing

| Flag | Default | What it sets |
|---|---|---|
| `CHAT_MODELS` | `[]` | JSON list of chat-model configs (id/provider/base_url/api_key/intelligence_level); drives `default_chat_model` / `lite_chat_model` / `super_chat_model` |
| `EMBEDDING_MODELS` | `[]` | JSON list of embedding-model configs (first entry = default) |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` | `None` | OpenAI fallback credentials for ad-hoc model creation |
| `ANTHROPIC_API_KEY` | `None` | Anthropic fallback API key |
| `GEMINI_API_KEY` | `None` | Google Gemini fallback API key |
| `GROQ_API_KEY` | `None` | Groq fallback API key |
| `MISTRAL_API_KEY` | `None` | Mistral fallback API key |
| `HUGGING_FACE_API_KEY` | `None` | Hugging Face fallback API key |
| `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` | `None` | DeepSeek fallback credentials |
| `VLLM_BASE_URL` | `None` | Dedicated vLLM inference server base URL |
| `MODEL_REGISTRY_PATH` | `None` | YAML/JSON model-registry file |
| `MODEL_ROLE_ROUTING` | `{}` | Role→{tier,tags} overrides for planner/generator/learner/judge selection (CONCEPT:ORCH-1.27); empty roles use `models/model_registry.py` defaults |
| `ROUTING_STRATEGY` | `hybrid` | Graph routing strategy |
| `ROUTING_PERCENTILE` | `50.0` | Routing percentile tunable |

### G.2 LLM request defaults

| Flag | Default | What it sets |
|---|---|---|
| `MAX_TOKENS` | `16384` | Default completion token cap |
| `TEMPERATURE` | `0.7` | Sampling temperature |
| `TOP_P` | `1.0` | Nucleus sampling |
| `TIMEOUT` | `32400` | LLM request timeout (s) |
| `TOOL_TIMEOUT` | `32400` | Tool-call timeout (s) |
| `PARALLEL_TOOL_CALLS` | `true` | Allow parallel tool calls |
| `SEED` | `None` | Deterministic sampling seed |
| `PRESENCE_PENALTY` / `FREQUENCY_PENALTY` | `0.0` | Repetition penalties |
| `LOGIT_BIAS` | `None` | Token logit-bias map (JSON) |
| `STOP_SEQUENCES` | `None` | Stop sequences (JSON list) |
| `EXTRA_HEADERS` / `EXTRA_BODY` | `None` | Extra provider request headers/body (JSON) |

### G.3 Agent identity & HTTP server

| Flag | Default | What it sets |
|---|---|---|
| `DEFAULT_AGENT_NAME` | package name | Agent display name |
| `AGENT_DESCRIPTION` | package description | Agent description |
| `AGENT_SYSTEM_PROMPT` | `None` | System prompt override |
| `WORKSPACE_PATH` | `None` | Workspace root override |
| `HOST` | `0.0.0.0` | Gateway bind address |
| `PORT` | `9000` | Gateway port |
| `DEBUG` | `false` | Debug mode |
| `ENABLE_WEB_UI` | `false` | Serve the web UI |
| `ENABLE_TERMINAL_UI` | `false` | Terminal UI mode (disables `GATEWAY_WORKERS>1`) |
| `ENABLE_WEB_LOGS` | `true` | Web log streaming |
| `ENABLE_ACP` | `false` | Agent Client Protocol adapter |
| `ACP_PORT` | `8001` | ACP port |
| `ACP_SESSION_ROOT` | `.acp-sessions` | ACP session storage dir |
| `DEFAULT_TERMINAL_AGENT` | `agent-terminal-ui` | Terminal agent binary |
| `MCP_URL` | `None` | Remote MCP server URL the agent attaches to |
| `MCP_CONFIG` | `None` | Path to `mcp_config.json` |
| `AGENT_API_KEY` | `None` | Static API key for gateway auth |
| `ENABLE_API_AUTH` | `false` | Require the API key |
| `MAX_UPLOAD_SIZE` | `10485760` | Upload cap (bytes) |
| `ALLOWED_ORIGINS` | `None` (= `*`) | CORS origins, comma-separated |
| `ALLOWED_HOSTS` | `None` | TrustedHostMiddleware hosts, comma-separated |

### G.4 Identity, JWT & delegation

`AUTH_JWT_JWKS_URI` is in section A's orbit (OS-5.14); its companions:

| Flag | Default | What it sets |
|---|---|---|
| `AUTH_JWT_ISSUER` | `None` | Expected JWT issuer claim |
| `AUTH_JWT_AUDIENCE` | `None` | Expected JWT audience claim |
| `OIDC_CONFIG_URL` | `None` | OIDC discovery URL (any compliant IdP) |
| `OIDC_CLIENT_ID` / `OIDC_CLIENT_SECRET` | `None` | OAuth 2.0 client credentials |
| `ENABLE_DELEGATION` | `false` | RFC 8693 token exchange for downstream APIs (CONCEPT:ECO-4.0) |
| `AUDIENCE` | `None` | Target audience for delegated tokens |
| `DELEGATED_SCOPES` | `api` | Space-separated delegation scopes |
| `FLEET_EVENTS_TOKEN` | `None` | Shared secret for `POST /api/fleet/events` webhook ingress (`X-Fleet-Events-Token` header); unset = no token required (CONCEPT:OS-5.15) |

### G.5 Secrets backends

| Flag | Default | What it sets |
|---|---|---|
| `SECRETS_BACKEND` | `inmemory` | `inmemory` \| `sqlite` \| `vault` |
| `SECRETS_SQLITE_PATH` | `None` | SQLite secrets DB path |
| `SECRETS_VAULT_URL` | `None` | HashiCorp Vault / OpenBao URL (binds both `vault_url` and `secrets_vault_url`) |
| `SECRETS_VAULT_MOUNT` | `secret` | KV v2 mount (binds both `vault_mount` and `secrets_vault_mount`) |
| `VAULT_AUTH_METHOD` | `auto` | `oidc` \| `approle` \| `token` \| `kubernetes` \| `auto` |
| `VAULT_AUTH_MOUNT` | `jwt` | Auth-method mount path |
| `VAULT_ROLE` | `None` | Role for OIDC/JWT or Kubernetes login |
| `VAULT_PATH_PREFIX` | `None` | KV path prefix (e.g. `agents/mcp/`) |

### G.6 Graph service & KG runtime (fields not already in A–F)

| Flag | Default | What it sets |
|---|---|---|
| `GRAPH_PERSISTENCE_TYPE` | `file` | Engine durable-persistence mode |
| `GRAPH_BACKEND_L2` | — | **Removed.** Replaced by `GRAPH_MIRROR_TARGETS` (the fan-out mirror set; LadybugDB or PostgreSQL named in `KG_CONNECTIONS`) |
| `GRAPH_COMPUTE_BACKEND` | `rust` | Compute tier selection |
| `GRAPH_SERVICE_SOCKET` | `None` (XDG runtime dir) | Engine UDS path; default `$XDG_RUNTIME_DIR/epistemic-graph.sock` |
| `GRAPH_SERVICE_TCP_ADDR` | `None` | Engine TCP address (e.g. `0.0.0.0:9100`); `GRAPH_SERVICE_ENDPOINTS` overrides both |
| `GRAPH_SERVICE_CHECKPOINT_SECS` | `300` | Engine auto-checkpoint interval (0 = off) |
| `GRAPH_SERVICE_PERSIST_ON_SHUTDOWN` | `true` | Serialize all graphs on engine shutdown |
| `GRAPH_DIRECT_EXECUTION` | `true` | AG-UI/ACP adapters bypass the LLM tool-call hop and invoke graph execution directly |
| `GRAPH_ROUTER_TIMEOUT` / `GRAPH_VERIFIER_TIMEOUT` | `300` | Router/verifier timeouts (s) |
| `ENABLE_LLM_VALIDATION` | `false` | LLM validation pass |
| `ENABLE_KG_EMBEDDINGS` | `true` | KG embedding generation |
| `KG_EMBEDDING_DIM` | `768` | Must match the embedding model's output dimension; the schema vector column size derives from it |
| `KG_BACKUPS` | `3` | KG backup retention count |
| `KG_INGESTION_WORKERS` | `None` (auto) | Ingestion worker count override; unset auto-sizes |
| `KG_ADAPTIVE_CONCURRENCY` | `true` | Auto-scale per-model LLM/embedding fan-out to real vLLM serving capacity (CONCEPT:KG-2.145). Off → static `model_capacity` only. See `architecture/adaptive_model_concurrency.md` |
| `MODEL_MAX_CONCURRENCY` | `512` | Ceiling the adaptive per-model concurrency target can ramp to (no hardcoded small cap; static `model_capacity` is the floor) |
| `GPU_CONCURRENCY_BUDGETS` | `{}` (unset → no cap) | JSON `group -> int`: total concurrent in-flight calls across all models sharing one physical GPU (CONCEPT:KG-2.146). Caps the SUM of member adaptive targets so they can't jointly oversubscribe the device, reserving latency-sensitive roles' floors first. Unset → per-model behaviour (no regression). See `architecture/distributed_gpu_concurrency.md` |
| `GPU_RESERVED_ROLES` | `chat,generator,default,lite,super` | Roles whose static floor is reserved off the top of a GPU budget (CONCEPT:KG-2.146); best-effort roles (embedding/batch) get the remainder and yield to these under contention |
| `KG_ANALYSIS_MAX_DEPTH` | `2` | Max recursion depth for background research daemons |
| `MAX_RECURSION_DEPTH` | `2` | Graph recursion depth tunable |
| `KNOWLEDGE_GRAPH_SYNC_BACKGROUND` | `true` | Background task workers for the KG pipeline |
| `ENABLE_SDD_WATCHER` | `true` | Plan/task watcher thread in the KG MCP server |
| `KG_ANOMALY_CONSUMER` | `true` | Drain unconsumed PerformanceAnomaly nodes into failure_gap topics; LLM-free, bounded, propose-only (CONCEPT:AHE-3.19) |
| `SPARQL_ENDPOINTS` | `["https://query.wikidata.org/sparql"]` | External SPARQL endpoints to federate (CONCEPT:KG-2.7) |
| `JENA_FUSEKI_URL` | `None` | Local Jena Fuseki URL (distinct from the `KG_FUSEKI_*` publish tick in section E) |
| `KAFKA_BOOTSTRAP_SERVERS` | `None` | Kafka brokers (task-queue/event transport; one of the three scale knobs in `docs/scaling/capacity_model.md`) |
| `KAFKA_TOPIC` | `None` | Default Kafka topic for messaging/event ingestion |
| `NATS_URL` | `None` | NATS broker URL |

### G.7 Observability exporters

The gateway Prometheus flags (`GATEWAY_METRICS` etc.) are in section A; the metric
series themselves are catalogued in [`../reference/metrics.md`](../reference/metrics.md).

| Flag | Default | What it sets |
|---|---|---|
| `ENABLE_OTEL` | `false` | OpenTelemetry tracing |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `None` | OTLP collector endpoint |
| `OTEL_EXPORTER_OTLP_HEADERS` | `None` | OTLP headers |
| `OTEL_EXPORTER_OTLP_PUBLIC_KEY` / `OTEL_EXPORTER_OTLP_SECRET_KEY` | `None` | OTLP keypair (Langfuse-style basic auth) |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `http/protobuf` | OTLP protocol |
| `LANGFUSE_DATASET_CAPTURE_THRESHOLD` | `0.0` | Score threshold for dataset capture (AHE-3.18 regression datasets) |
| `LANGFUSE_LATENCY_BASELINE_SECONDS` | `60` | Latency baseline for anomaly scoring |
| `LANGFUSE_TOKEN_BASELINE` | `20000` | Token-usage baseline for anomaly scoring |
| `LANGFUSE_VERIFIER_FALLBACK_LIMIT` | `1` | Verifier fallback attempts |

### G.8 A2A (agent-to-agent)

| Flag | Default | What it sets |
|---|---|---|
| `A2A_BROKER` | `in-memory` | A2A broker backend |
| `A2A_BROKER_URL` | `None` | Broker URL when not in-memory |
| `A2A_STORAGE` | `in-memory` | A2A storage backend |
| `A2A_STORAGE_URL` | `None` | Storage URL when not in-memory |
| `A2A_CONFIG` | `None` | `a2a_config.json` path for external agent discovery (CONCEPT:ECO-4.0) |
| `A2A_REFRESH_INTERVAL` | `300` | Agent-card re-fetch interval (s) |

### G.9 Orchestration, scheduler & guardrails

| Flag | Default | What it sets |
|---|---|---|
| `MIN_CONFIDENCE` | `0.4` | Minimum confidence gate |
| `VALIDATION_MODE` | `false` | Validation-only execution mode |
| `APPROVAL_TIMEOUT` | `0` | Approval wait timeout (s; 0 = no wait) |
| `COGNITIVE_SCHEDULER_ENABLED` | `true` | Priority-aware agent scheduler (CONCEPT:OS-5.2) |
| `MAX_CONCURRENT_AGENTS` | `5` | Concurrent specialist agents (CONCEPT:OS-5.2) |
| `AGENT_TOKEN_QUOTA` | `100000` | Per-agent token budget before preemption (CONCEPT:OS-5.2) |
| `PREEMPTION_THRESHOLD_PCT` | `0.85` | Quota usage triggering preemption warning |
| `AGENT_POLICIES_PATH` | `None` | `agent_policies.json` for identity-based governance |
| `PERMISSIONS_SIGNING_KEY` | `None` (auto) | HMAC key for agent identity tokens; auto-generated if unset |
| `SPECIALIST_REGISTRY_PATH` | `None` | Local specialist registry dir |
| `MAX_PARALLEL_AGENTS` | `60` | Global engine-wide execution semaphore (CONCEPT:ORCH-1.8) |
| `WORKER_POOL_SIZE` | `8` | Workers per node for agent turns / graph mutations; active-concurrency scale knob (CONCEPT:ORCH-1.8) |
| `PARALLEL_BATCH_SIZE` | `25` | Agents per execution wave |
| `SYNTHESIS_STRATEGY` | `auto` | `auto` \| `flat` \| `hierarchical` \| `progressive` \| `rlm` (CONCEPT:ORCH-1.26) |
| `SYNTHESIS_RATIO` | `10` | Outputs per hierarchical synthesis sub-node |
| `AGENT_EXECUTION_TIMEOUT` | `120` | Per-agent timeout (s) |
| `CIRCUIT_BREAKER_THRESHOLD` | `3` | Consecutive failures disabling an agent type |
| `ENABLE_PROGRESSIVE_SYNTHESIS` | `true` | Streaming synthesis as agents complete |
| `HOMEOSTATIC_DOWNGRADE_ENABLED` | `true` | Auto model-tier downgrade under budget pressure (CONCEPT:OS-5.2) |
| `ADVERSARIAL_VERIFICATION` | `false` | Adversarial verification pass — opt-in, doubles verification cost (CONCEPT:AHE-3.1) |
| `MAINTENANCE_TOKEN_BUDGET` | `0` (unlimited) | Token budget for the autonomous maintenance cron |
| `MAINTENANCE_PRIORITY` | `LOW` | Maintenance task priority (LOW/MEDIUM/HIGH) |
| `WATCHDOG_PATTERNS` | `pyproject.toml, mcp_config.json, requirements*.txt` | File patterns for the file-watcher trigger (CONCEPT:OS-5.0) |
| `TOOL_GUARD_MODE` | `strict` | Sensitive-tool guard mode |
| `SENSITIVE_TOOL_PATTERNS` | 67 regexes | Tool-name patterns treated as mutating/sensitive (delete/exec/deploy/...); override only to extend |

### G.10 Skills

| Flag | Default | What it sets |
|---|---|---|
| `CUSTOM_SKILLS_DIRECTORY` | `None` | Extra skills directory |
| `SKILL_TYPES` | `None` | Skill-type filter (JSON list) |

### G.11 Native messaging backends (CONCEPT:ECO-4.0)

| Flag | Default | What it sets |
|---|---|---|
| `MESSAGING_ENABLED_BACKENDS` | `[]` | Backend IDs to auto-connect (e.g. `["discord","slack"]`) |
| `MESSAGING_KG_INGEST` | `true` | Auto-ingest all inbound/outbound messages into the KG |
| `MESSAGING_KG_MEMORY_TYPE` | `episodic` | KG memory tier for inbound messages (`episodic`/`semantic`/`procedural`) |
| `MESSAGING_ROUTE_TO_PLANNER` | `true` | Route inbound events to the Planner Graph Agent |
| `MESSAGING_DISCORD_TOKEN` | `None` | Discord bot token (also reads `DISCORD_BOT_TOKEN`) |
| `MESSAGING_SLACK_TOKEN` | `None` | Slack bot token `xoxb-...` (also reads `SLACK_BOT_TOKEN`) |
| `MESSAGING_SLACK_APP_TOKEN` | `None` | Slack app-level token `xapp-...` (Socket Mode) |
| `MESSAGING_TELEGRAM_TOKEN` | `None` | Telegram bot token (also reads `TELEGRAM_BOT_TOKEN`) |
| `MESSAGING_WHATSAPP_TOKEN` | `None` | WhatsApp API token (also reads `WHATSAPP_TOKEN`) |
| `MESSAGING_WHATSAPP_PHONE_NUMBER_ID` | `None` | WhatsApp Business phone number ID |
| `MESSAGING_WHATSAPP_USE_BUSINESS_API` | `false` | Official Business API vs neonize bridge |
| `MESSAGING_TEAMS_APP_ID` / `MESSAGING_TEAMS_APP_SECRET` | `None` | Microsoft Teams Bot Framework credentials |
| `MESSAGING_GOOGLECHAT_TOKEN` | `None` | Google Chat service-account JSON path |
| `MESSAGING_GOOGLEMEET_TOKEN` | `None` | Google Meet service-account JSON path |
| `MESSAGING_MATTERMOST_TOKEN` / `MESSAGING_MATTERMOST_URL` | `None` | Mattermost token + server URL |
| `MESSAGING_MATRIX_TOKEN` / `MESSAGING_MATRIX_HOMESERVER` / `MESSAGING_MATRIX_USER_ID` | `None` | Matrix credentials |
| `MESSAGING_IRC_SERVER` | `None` | IRC server |
| `MESSAGING_IRC_PORT` | `6667` | IRC port |
| `MESSAGING_IRC_NICKNAME` | `agent_bot` | IRC nickname |
| `MESSAGING_IRC_CHANNELS` | `[]` | IRC channels to auto-join |
| `MESSAGING_SIGNAL_TOKEN` | `None` | Signal phone number (semaphore-bot) |
| `MESSAGING_LINE_TOKEN` | `None` | LINE channel access token |
| `MESSAGING_TWITCH_TOKEN` | `None` | Twitch OAuth token |
| `MESSAGING_TWITCH_CHANNELS` | `[]` | Twitch channels to join |
| `MESSAGING_SYNOLOGY_WEBHOOK_URL` | `None` | Synology Chat webhook URL |
| `MESSAGING_VOICECALL_APP_ID` / `MESSAGING_VOICECALL_TOKEN` / `MESSAGING_VOICECALL_FROM_NUMBER` | `None` | Twilio voice/SMS credentials (account SID / auth token / from number) |
| `MESSAGING_NEXTCLOUD_URL` / `MESSAGING_NEXTCLOUD_TOKEN` / `MESSAGING_NEXTCLOUD_APP_ID` | `None` | Nextcloud Talk credentials (URL / app token / username) |

## H. Former bare `KG_*`/`GRAPH_*`/`EPISTEMIC_*` reads (now folded)

Every governed (`KG_*`/`GRAPH_*`/`EPISTEMIC_*`) bare `os.environ` read has been
**folded off `os.environ`** onto a centralized path, per *Configuration discipline*:

- **Deployment-varying / behavioral / test-varied** → `config.setting("VAR", default)`
  (live, config.json-driven). Still fully settable — set `var` in `config.json` or
  `VAR` in the environment.
- **Pure load tunables** (concurrency/batch) → **auto-sized** via
  `compute_ingest_worker_count()`.
- **Single-value cadences/limits/timeouts** → **named module constants** (no knob).

`scripts/check_no_env_sprawl.py` now ratchets **all-prefix** bare reads against
`scripts/env_flag_baseline.txt` (the governed reads are gone from it; what remains is
the non-KG burn-down — `AGENT_*`, `VAULT_*`, `OTEL_*`, connector creds, …). The table
below is the reference for these settings and how each is now resolved (defaults
unchanged); for the `setting()` rows, the value is config.json-/env-overridable:

| Flag | Default | Read in | What it sets |
|---|---|---|---|
| `KG_SERVER_HOST` / `KG_SERVER_PORT` | `127.0.0.1` / `8100` | `agent/factory.py`, `mcp/kg_coordinator.py`, `backends/contrib/ladybug_backend.py`, `core/config.py` | KG coordinator server address |
| `KG_DAEMON_LOG_LEVEL` | `INFO` | `gateway/daemon.py` | Daemon log level |
| `GRAPH_ROUTING_STRATEGY` | `hybrid` | `knowledge_graph/core/engine.py` | Engine-side routing strategy (overlaps `ROUTING_STRATEGY` on `AgentConfig`) |
| `KG_CARD_MODEL` | `lite` | `core/engine_tasks.py` | `lite` or `heavy` model for enrichment cards |
| `KG_LLM_TIMEOUT` / `KG_LLM_MAX_RETRIES` | `30` / `1` | `enrichment/cards.py` | Enrichment LLM call timeout (s) / retries |
| `KG_EMBED_BACKFILL_INTERVAL` / `KG_EMBED_BACKFILL_BUSY_SLEEP` | `30` / `1` | `core/engine_tasks.py` | Embedding-backfill idle/busy sleep (s) |
| `KG_RECONCILE_INTERVAL` | `900` | `core/engine_tasks.py` | engine→mirror reconcile tick (s) |
| `KG_HYGIENE_INTERVAL` | `86400` | `core/engine_tasks.py` | Memory decay/dedup tick (s) |
| `KG_TASK_REAPER_INTERVAL` | `120` | `core/engine_tasks.py` | Zombie-task reaper tick (s) |
| `KG_TASK_ORPHAN_GRACE_SEC` | `90` | `core/engine_tasks.py` | Grace before an orphaned task is reclaimed |
| `KG_TASK_MAX_RUNTIME_SEC` | `7200` | `core/engine_tasks.py` | Max task runtime before requeue |
| `KG_TASK_MAX_REQUEUE` | `3` | `core/engine_tasks.py` | Max requeues before a task is failed |
| `GRAPH_SERVICE_CHECKPOINT_INTERVAL` | `60` | `core/graph_compute.py` | Spawned-engine checkpoint interval (distinct from `GRAPH_SERVICE_CHECKPOINT_SECS`) |
| `KG_GRAPH_NAME` | `__bus__` | `distillation/skill_graph_distiller.py` | Target graph for skill-graph distillation |
| `KG_INGEST_INFLIGHT` | `40` | `ingestion/batch_orchestrator.py` | Max in-flight ingest submissions |
| `KG_INGEST_PROFILE` | unset (= `full`) | `pipeline/__init__.py` | Pipeline phase profile (`structural` \| `full`) — residual read, see the section C correction |
| `KG_EVAL_CAPTURE` | off | `memory/optimization_engine.py` | Capture retrieval evals |
| `KG_MIN_RELEVANCE_THRESHOLD` | unset (arg/schema-pack) | `retrieval/retrieval_quality.py` | Relevance-gate threshold override |
| `KG_TRUST_HIERARCHY` | built-in defaults | `core/company_brain_runtime.py` | JSON trust-hierarchy entries (with `KG_BRAIN_ENFORCE`) |
| `GRAPH_SCHEMA_PACK` | unset | `models/schema_pack_loader.py` | Schema-pack selection override |
| `GRAPH_SCHEMA_AUDIT_DIR` / `GRAPH_SCHEMA_AUDIT_VERBOSE` | unset / off | `models/schema_pack_audit.py` | Schema-audit output dir / verbosity |
| `KG_PROVIDER_ADAPTER_BACKEND` | `static` | `prompting/provider_adapter.py` | Prompting provider-adapter backend |

Disposition: the address / model-choice / behavioral / profile / schema rows
(`KG_SERVER_*`, `GRAPH_ROUTING_STRATEGY`, `KG_CARD_MODEL`, `KG_GRAPH_NAME`,
`KG_INGEST_*`, `KG_EVAL_CAPTURE`, `KG_MIN_RELEVANCE_THRESHOLD`, `KG_TRUST_HIERARCHY`,
`GRAPH_SCHEMA_*`, `KG_PROVIDER_ADAPTER_BACKEND`, `KG_DAEMON_LOG_LEVEL`) are now
`config.setting(...)` reads (config.json-/env-overridable). The cadence/limit/timeout
rows (`KG_*_INTERVAL`, `KG_TASK_*`, `KG_LLM_TIMEOUT`/`_MAX_RETRIES`,
`KG_EMBED_BACKFILL_*`, `GRAPH_SERVICE_CHECKPOINT_INTERVAL`) are now named module
constants. `MCP_CHILD_*` flags were already fully typed on `AgentConfig` with no bare
reads (`mcp/child_resilience.py` consumes the config object).

The agent toolset gates in `tools/tool_registry.py` are also bare reads (not
KG-prefixed, so outside the ratchet). The optional-infra toolsets all default
OFF and are opt-in:

| Flag | Default | What it gates |
|---|---|---|
| `X_TOOLS` | `False` | X/Grok social search + post browsing via xAI (needs `XAI_API_KEY`); production X/Grok deployments must set `X_TOOLS=1` explicitly |
| `MEDIA_TOOLS` | `False` | Media generation / transcription services (ECO-4.30/4.31) |
| `DB_TOOLS` | `False` | Native database traversal tools (ECO-4.33) |

(The always-available local toolsets — `WORKSPACE_TOOLS`, `GIT_TOOLS`,
`A2A_TOOLS`, `SCHEDULER_TOOLS`, `BROWSER_TOOLS`, `DEVELOPER_TOOLS` — default
ON in the same registry.)

## Coverage statement

Verified against `agent_utilities/core/config.py` on this branch by extracting
`AgentConfig.model_fields` programmatically: **244 fields / 242 distinct env
variables, every one documented above** — sections A–F cover the KG/graph audit
surface, section G the remaining platform fields (no field was deemed
internal-only: every `AgentConfig` field declares an env alias and is settable
from the environment). Section H additionally documents the user-facing flags
that exist only as baseline-frozen bare `os.environ` reads. Drift fixed in this
pass: `KG_LLM_CONCURRENCY` default is `4` (doc previously said 6), and
`KG_INGEST_PROFILE` retains one phase-selection read despite the section C
removal note.

## Known bugs surfaced by this audit

1. **`KG_EMBED_BACKFILL_BATCH` dual default** — 256 (`engine_tasks.py:1040`) vs 512
   (`:1155`). Same flag, two meanings (per-tick budget vs DB fetch batch). Split into two
   named constants or one config field.
2. **Scattered duplicate reads** — `GRAPH_DB_URI` (4×), `AGENT_UTILITIES_CONFIG_DIR` (5×),
   `KG_DAEMON_ROLE`/`KG_INGEST_PROFILE` (2× each). No single source of truth.
3. **`GRAPH_TIMEOUT=1200000` (20 min)** makes a non-converging engine call look like an
   infinite hang for 20 minutes before erroring.

## Target end-state

~96 flags → roughly the **~27 KEEP** deployment items (all typed on `AgentConfig`),
behavior otherwise auto-detected, and `scripts/check_no_env_sprawl.py` blocking
regressions. A fresh `graph-os-daemon` with **zero `KG_*` env vars set** should ingest a
full corpus correctly.
