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

**Verdict legend**
- **KEEP** — legitimate deployment config (path / DSN / secret / port / socket). Must be
  read via the central `config` object, not bare `os.environ`.
- **AUTO** — should be auto-detected/auto-sized from the runtime; the flag should be
  removed (or kept only as an override with an auto default).
- **REMOVE** — always-on behavior or stale experiment; delete the flag.

## A. Deployment config — KEEP (read via `AgentConfig`)

| Flag | Default | What it sets |
|---|---|---|
| `GRAPH_DB_URI` / `PGGRAPH_DSN` | none | Durable L3 Postgres/pggraph DSN |
| `GRAPH_BACKEND` / `GRAPH_BACKEND_L1` | `tiered` / `epistemic_graph` | Backend selection |
| `EPISTEMIC_GRAPH_SOCKET` | `/tmp/epistemic-graph.sock` | Rust engine UDS |
| `GRAPH_PERSISTENCE_PATH`, `GRAPH_SERVICE_PERSIST_DIR` | data dir | L1 snapshot dir |
| `GRAPH_DB_HOST/PORT/NAME/USER/PASSWORD/PATH` | — | DSN parts (legacy; prefer `GRAPH_DB_URI`) |
| `GRAPH_FUSEKI_URL/USER/PASSWORD/DATASET` | — | SPARQL endpoint (optional backend) |
| `GRAPH_PGGRAPH_SCHEMA` | `public` | Postgres schema |
| `AGENT_UTILITIES_{CONFIG,DATA,CACHE,LOG,MEMORY,RUNTIME}_DIR` | XDG | Path overrides (resolved in `core/paths.py`) |
| `AGENT_UTILITIES_TOKEN_SECRET` | — | Run-scoped tool-token secret |
| `KG_DAEMON_ROLE` | `auto` | host/client/auto election (topology) |
| `EPISTEMIC_GRAPH_AUTOSTART` | — | Auto-spawn the engine |
| `GRAPH_SERVICE_AUTH_SECRET` | auto-generated | Engine HMAC secret; unset → per-install secret persisted at `data_dir()/engine_secret` (0600) (CONCEPT:OS-5.14) |
| `KG_ENGINE_INSECURE` | `false` | Dev opt-out of engine HMAC auth; sets `EPISTEMIC_GRAPH_ALLOW_INSECURE=1` on spawned engines (CONCEPT:OS-5.14) |
| `KG_AUTH_REQUIRED` | `false` | Require server-validated JWT identity for KG access — 401 without it; caller `_actor`/`_roles`/`_tenant` kwargs ignored (CONCEPT:OS-5.14) |
| `KG_AUTH_TOKEN` | — | JWT minting the stdio MCP process identity (validated against `AUTH_JWT_JWKS_URI`) (CONCEPT:OS-5.14) |
| `KG_ACL_DEFAULT_ALLOW` | `false` | With `KG_BRAIN_ENFORCE` on: allow nodes WITHOUT an ACL (escape hatch from the fail-closed default-deny) (CONCEPT:OS-5.14) |

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
| `KG_RECONCILE_DURABLE` | `1` | L1→L2/L3 autoheal (L609) |
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
| `KG_LLM_CONCURRENCY` | 6 | auto from cpu |
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
`KG_BREADTH_*` read was moved off bare `os.environ` onto typed `AgentConfig` fields
(`kg_golden_loop`, `kg_golden_distill`, `kg_golden_breadth`, `kg_golden_standardize`,
`kg_golden_auto_merge`, `kg_golden_merge_threshold`, `kg_golden_loop_interval`,
`kg_golden_loop_topics`, `kg_breadth_library_roots`, `kg_breadth_repo_roots`) — opt-in, all off
by default, single typed source of truth.

| Family | Count | Notes |
|---|---|---|
| `KG_EA_WRITEBACK`, `KG_ENABLE_HARD_NEGATIVE_MINING`, `KG_BRAIN_ENFORCE`, `KG_RESEARCH_EXTERNAL` | 4 | remaining experiment gates — graduate (always-on) or delete |

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
typed on `AgentConfig`, default ON.** `execute_workflow` SHACL-validates the stored
`WorkflowDefinition` (+ steps) against the governance shapes before dispatch and refuses
malformed definitions with a structured violation report; cheap and LLM-free. The companion
permission gate (ontology permissioning ACL on the workflow node) is governed by the existing
`KG_BRAIN_ENFORCE` flag (OS-5.14 fail-closed semantics), not a new one.

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
