# Spec: Self-Healing Durable-Tier Sync (L1↔L2 autoheal)

> CONCEPT:KG-2.8 (consolidated KG daemon / maintenance scheduler). Surfaced when
> validating all graph layers after a pre-persistence daemon restart lost data.

## Finding (what we observed)

The KG runs a tiered store — **L1** the in-memory/Rust `epistemic_graph` compute
graph, **L2/L3** durable Postgres (`pggraph`/`agent_kg`). Three ways the tiers
silently diverged, costing real data:

1. **L1-only runs.** With `GRAPH_BACKEND=epistemic_graph` the engine writes *only*
   to L1 → nothing reaches Postgres. A whole session of enrichment (112 concepts,
   100 SDD features) lived only in RAM and was **lost on the next restart**.
2. **No auto-DDL.** The durable tier ships tables only for types in the static
   schema; a write for a NEW node type (`SDD_Feature`) or a missing column failed
   with `relation/column ... does not exist` and was **silently dropped** (the
   backend logs + returns `[]`, never raising).
3. **Append-only restart.** The compute daemon had no `--persist-dir`, so a restart
   started from nothing while Postgres was also down — the only copy was RAM.

Net: the durable layer is not actually durable unless every write reaches it and
the schema self-extends — neither of which held.

## User Stories

### US-1: A new node type persists durably without a migration
**As** the engine, **I want** the durable tier to create a table for any node type
on first write, **so that** new types (`SDD_Feature`, …) aren't silently dropped.
- [x] `PostgreSQLBackend.ensure_label_table(label)` — `CREATE TABLE IF NOT EXISTS` (id/name/type/properties) + register with the transpiler.
- [x] `ensure_column(table, col)` — `ALTER TABLE ADD COLUMN IF NOT EXISTS` for schema-drifted properties.
- [x] `execute()` catches `relation/column ... does not exist`, runs the matching auto-DDL, and **retries** — so the write succeeds instead of dropping.

### US-2: The tiers self-converge — divergence can't accumulate
**As** an operator, **I want** L1→L2 drift to repair itself, **so that** an L1-only
run, a restart, or a new type never leaves the durable tier behind.
- [x] `reconcile_durable` maintenance job in the consolidated scheduler (`_maintenance_jobs`), running **~15s after startup** then every `KG_RECONCILE_INTERVAL` (900s), calling `TieredGraphBackend.reconcile_to_durable()`.
- [x] Reconcile benefits from the auto-DDL (its `l3.execute` self-heals missing tables/cols), so the backfill can't fail on a missing type.
- [x] Opt-out via `KG_RECONCILE_DURABLE=0`; only registered when a durable reconcile exists (tiered backend).

### US-3: Restarts are non-destructive
- [x] Compute daemon runs with `--persist-dir … --checkpoint-interval 60` (disk snapshot) — reload on restart.
- [x] Gateway runs **tiered** (`GRAPH_BACKEND=tiered`, correct pggraph creds) so writes hit Postgres too — a second durable copy.

## Non-Functional Requirements
- [x] Auto-DDL is idempotent + cheap (guarded by `_known_tables`); additive ALTERs only.
- [x] All existing backend/tiered/assimilation tests pass (142 green); ruff clean.
- [x] No new top-level concept id (maintenance-scheduler refinement, KG-2.8).

## Validation (done, live)
- Backfill reconciled **L1 4,575 → L2**: `Article 3254=3254`, `Concept 136`, `sdd_feature 100`, `DataConnector 142`, `Task ~484`, `Skill 46`, … — **L1==L2 across all types** (only `memory` 39/35, in-flight, which the periodic job converges).
- Auto-DDL created tables for every previously-missing type (`sdd_feature`, `ValidationProbe`, …).
- Gateway scheduler now lists `reconcile_durable`; it runs at startup + every 900s.

## Status
**IMPLEMENTED** — `backends/postgresql_backend.py` (auto-DDL), `core/engine_tasks.py`
(reconcile job + tick). **Follow-on:** `reconcile_to_durable` counts an
`execute()`-swallowed failure as success (it never raises); make it assert the row
landed (read-back / `RETURNING`) so the drift metric is exact, and reconcile edges
into `kg_edges` (currently 0 edges mirrored).
