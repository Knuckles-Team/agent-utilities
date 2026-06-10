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

## C. Ingest-throughput knobs — AUTO-DETECT (Phase 2)

| Flag | Default | Purpose | Plan |
|---|---|---|---|
| `KG_INGEST_FEATURES` | `1` | per-repo call-graph community detection | auto |
| `KG_INGEST_PROFILE` | `""` | `structural` defers features | auto |
| `KG_BULK_INGEST` | off | skip maintenance scheduler during bulk | auto-detect via queue depth |

The system already knows when it's bulk-loading (durable queue depth via
`self._submission_queue.get_queue_size()`, `engine_tasks.py:188`) and already has a
drain-completion hook (`_maybe_build_vector_indexes`, `engine_tasks.py:2642`). **Caveat
found during audit:** the per-repo community detection (`enrichment/features.py:90` →
Rust label-propagation) can *hang* (non-convergence), and `Code` nodes do **not** persist
`calls` (`pipeline.py:_write_code`), so a naive "defer + reconstruct on drain" would drop
features. The correct fix is either a convergence bound in the Rust engine's label
propagation or a `CALLS`-edge-based full-graph pass — tracked as a focused follow-up, not
a one-line flag swap.

## D. Performance tunables — AUTO-SIZE from CPU/mem/load

Reuse the existing sizer (`engine_tasks.py:1683-1709`: CPU 36% + 3 GB/worker mem cap).

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

## E. Experiment / feature gates — GRADUATE or DELETE

| Family | Count | Notes |
|---|---|---|
| `KG_GOLDEN_*` (`LOOP`, `DISTILL`, `BREADTH`, `STANDARDIZE`, `AUTO_MERGE`, `MERGE_THRESHOLD`, `LOOP_INTERVAL`, `LOOP_TOPICS`, `BREADTH_LIBRARY_ROOTS`, `BREADTH_REPO_ROOTS`) | 10 | collapse into one nested `GoldenLoopConfig` |
| `KG_EA_WRITEBACK`, `KG_ENABLE_HARD_NEGATIVE_MINING`, `KG_BRAIN_ENFORCE`, `KG_RESEARCH_EXTERNAL` | 4 | graduate (always-on) or delete |

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
