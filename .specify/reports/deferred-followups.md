# Deferred Follow-Ups — Memory Synergy Work (agent-oss + memory-os)

> Captured after implementing ORCH-1.27, KG-2.11–2.15/2.17–2.19, AHE-3.12 and the docs sweep.
> Everything below is **known-incomplete and intentionally deferred** to a later plan. The shipped
> features are coded, wired (≤3 hops, verified), and unit-tested; the items here are the gaps.

## 1. Empirical / live-LLM validation (highest priority)
The LLM-dependent runtime paths are unit-tested via deterministic fakes/fallbacks, **not** against a
live model pool + real data:
- HyDE planner (`hybrid_retriever._generate_hyde_plan`), learner extraction (`extract_edits`),
  benchmark `generator`/`judge` roles, the Ground-Truth preamble in a real agent loop, and the real
  wiki-ingest path (`curate_wiki` → `IngestionEngine.ingest`).
- **Action:** run each end-to-end against a configured provider pool; confirm role routing
  (ORCH-1.27) resolves real models and the outputs parse.

## 2. LongMemEval-S benchmark run (AHE-3.12)
The harness is built + unit-tested but the actual 500-question eval was never executed.
- **Action:** start the FastAPI server, point Quarq's `quarqlabs/benchmarks` runner at `/benchmark/*`,
  capture a baseline score, then re-run after the memory features to defend the "meets/beats 98.2%"
  claim. Wire `scripts/check_longmemeval.py` into CI on a frozen subset.

## 3. Partially-implemented feature internals
- ✅ **KG-2.17 semantic merge — DONE (was detect-only).** `MemoryHygiene.run` now applies the merge:
  `merge_plan` computes survivor union(tags)+max(importance); the survivor is updated and each duplicate
  is soft-retired (`status=MERGED` + `valid_to` + `MERGED_INTO` edge, never deleted). Tests in
  `test_kg_2_17_memory_hygiene.py`.
- **AU-KG.memory.ground-truth-preamble-declaring cross-turn injection dedup.** The authority hierarchy + preamble shipped; the
  *per-session, cross-turn* dedup of injected context (memory-os `multi-source-surgical-injection`)
  was not added because `StartupContextBuilder` is a one-shot startup payload, not a per-turn hook.
  **Action:** add a per-turn injection hook with a session-scoped seen-set if/when a live pre-call
  injection path exists.

## 4. SDD scaffold stubs are skeletons
The 11 `.specify/design|specs/<id>/` dirs scaffolded from the memory-os ledger
(`ground-truth-hierarchy`, `four-level-fallback-cascade`, `semantic-dedup-merge`, etc.) are template
stubs with TODOs. The **per-concept docs** under `docs/pillars/**` are complete, but these SDD
artifacts are not fully authored.
- ✅ **DONE.** `kg-2-14-ground-truth-hierarchy` SDD authored (design/spec/tasks, 0 TODOs); the 19
  redundant ledger-id stubs were pruned (superseded by the per-concept docs in `docs/pillars/**`). No
  TODO-laden stubs remain. The shipped KG-2.15/2.17/2.18/2.19 + ORCH-1.28–1.31 concepts are recorded via
  their per-concept docs rather than separate `.specify/design` dirs.

## 5. Pre-existing repo-wide static-analysis debt (NOT introduced by this work)
Full-repo runs this round (net-new findings from my code = **0** after fixes):
- **mypy:** `723 errors in 53 files (checked 673 source files)` — all pre-existing. The ones in files
  I edited are in code I did not touch: `core/model_factory.py:31-81` (pydantic-ai version-compat
  import shim), `mcp/kg_server.py:77` (method reassignment), `server/app.py:397` (FastAPI
  `description` typing). My added code is mypy-clean.
- **vulture:** 0 findings repo-wide (min-confidence 95).
- **bandit:** 0 findings after fixing the 2 Low (B112) in my code.
- **RESOLVED → tracked plan:** characterized and triaged in
  [`mypy-remediation-plan.md`](mypy-remediation-plan.md). Finding: **92% (665/723) live in 5
  graph/checkpoint files; 596 are `attr-defined` false positives** from pydantic-graph generic
  state/deps access (mypy can't resolve `ctx.state.query`, `node.plan`, `NodeSnapshot.run_id`
  through `BaseNode[StateT, …]`). **Phase 0 LANDED:** a *scoped* `[[tool.mypy.overrides]]` in
  `pyproject.toml` disables **only `attr-defined`** in **only those 5 files** → **723 → 170
  errors (−76%)**, masking nothing real (attr-defined stays active everywhere else; new code fully
  checked). Phases 1–3 (drain ≤2-error long tail → typed State/Deps protocols that remove the
  override → targeted real-bug fixes) are sequenced in the plan, intentionally **not bundled** into
  the pre-live-testing checkpoint since they touch others' framework code.

## 6. Build / process gaps
- **mkdocs site not rebuilt** — `mkdocs` isn't installed in this environment. Run `mkdocs build`
  (or `mkdocs gh-deploy`) in the docs CI env to refresh `site/`.
- **Full `pre-commit run --all-files` not executed end-to-end** — ran ruff (touched files), mypy
  (scoped), bandit (full), vulture (full), and the guardrail gates
  (`check_no_stub`/`check_coupling`/`check_concepts`) individually. The codespell / nbQA / other
  hooks were not run.
- **Not committed** — all changes are in the working tree (plus 2 staged `git mv` doc renames).

## 7. Wiring opportunities noted but not taken
- **KG-2.18 trust feedback is built but not auto-invoked on the retrieval hot path.** `UsageTelemetry`
  + `build_lineage` exist and are tested, but nothing in `search_hybrid`/`plan_and_retrieve`
  currently records recall/usage automatically. **Action:** wire `record_recall` at retrieval and
  `record_usage`/`build_lineage` at generation, then `flush_to_engine` periodically (could ride the
  KG-2.17 hygiene daemon tick).

---

## Overall wired-in audit (all sessions, mine + others)

Ran `check_wiring.py` over all 50 changed/new production modules + per-feature live-path invocation
spot-checks (reachable != invoked).

**My modules — all wired AND invoked on a live path** after fixing two gaps found during this audit:
- **KG-2.18 (evidence-weighted memory): FIXED.** `UsageTelemetry`/`build_lineage` were API-only;
  now `HybridRetriever.plan_and_retrieve` calls `usage_telemetry.record_recall(...)` on every
  retrieval, and `record_answer_usage()` closes the loop (usage → trust persist → lineage). Live-path
  test: `tests/unit/knowledge_graph/test_kg_2_18_live_path.py`.
- (RLM C/A/D/E were similarly fixed earlier this session.)

**Others' modules (company-brain / enrichment / routing — NOT mine):**
- ✅ `workflow_context.py` (registered `WorkflowContextRouter`), `camunda.py`/`erpnext.py` extractors
  (self-register via `register_source`, auto-imported by `discover_extractors()` @
  `engine_federation.py:239`), `realizes.py` (used by `pipeline.py`/`distill.py`).
- ✅ **`knowledge_graph/enrichment/capability_writeback.py` — FIXED (was orphaned).** Added
  `resolve_writeback_fn(backend)` (gated by `KG_EA_WRITEBACK`, default off → no-op) and wired it into
  `EnrichmentPipeline(writeback_fn=...)` at both `ingestion/engine.py:642` (graph_ingest codebase path)
  and `enrichment/__main__.py` (CLI). The pipeline already called `writeback_fn(minted)` — nothing
  built the callable; now it does. Tests: `tests/unit/knowledge_graph/test_capability_writeback_wiring.py`.

**Tooling finding:** `check_wiring.py` (import-graph) has a **blind spot for plugin/decorator-based
dynamic registration** (`register_source` + `pkgutil` discovery) — it false-flagged camunda/erpnext as
unreachable when they wire via discovery. The audit must therefore ALSO grep for registration/discovery
mechanisms, not rely on import reachability alone. ✅ RESOLVED — `check_wiring.py` now detects plugin/
decorator self-registration (`registered_via_plugin`) and no longer false-flags self-registering modules.

**Update:** the deferred `run_full_rlm` RunTrace integration (item under ORCH-1.29) is now DONE — the
live REPL loop populates a structured `RunTrace` (per-iteration step + `FailureClass`) exposed as
`env.last_run_trace`. Live-path test: `tests/unit/rlm/test_orch_1_29_runtrace_live_path.py`.

**Fan-out per-target timeout (CONCEPT:AU-KG.backend.multi-connection-registry).** A `graph_query/graph_search/graph_write target='all'`
fan-out iterated connections *sequentially with no timeout*, so one slow/unreachable backend (e.g. a
mirror under heavy drain) stalled the whole call (~300s). ✅ RESOLVED — `kg_server.fanout_execute()` runs
every target concurrently under a shared per-target wall-clock budget (`GRAPH_FANOUT_TIMEOUT`, default
30s, live-tunable via `graph_configure set_config`); a slow/raising target lands in `errors` while the
rest still return (partial-success contract). Tests: `tests/unit/mcp/test_fanout_timeout.py`.
