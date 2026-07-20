# Spec: LongMemEval-S Validation Harness (AHE-3.12)

> References design: `.specify/design/ahe-3.12-longmemeval-harness/design.md`

## Pre-Flight Checklist
- [x] Design exists; KG-nearest-concepts table completed.
- [x] Extension target identified (AHE-3.2/3.4, similarity ≥ 0.70); reuses EvaluationCorpus, KG-2.12, ORCH-1.27, router pattern.
- [x] New CONCEPT:AU-AHE.evaluation.longmemeval-validation-harness justified as the standalone reusable harness (user decision).
- [x] Wire-First confirmed: `POST /benchmark/query` → router → `search_hybrid(mode="hyde")` → `plan_and_retrieve` ≤ 3 hops.

## User Stories

### US-1 — Quarq-runner-compatible HTTP surface
**As** the external Quarq benchmark runner, **I want** `/benchmark/session`, `/benchmark/query`,
`/benchmark/report/{run_id}` over HTTP, **so that** it drives agent-utilities unmodified.
- **AC1**: Router mounts under `/benchmark` and `GET /benchmark/health` returns `{status:"ok", concept:"AHE-3.12"}`.
- **AC2**: `POST /benchmark/session` ingests haystack messages as **episodic** memory and creates + **freezes** an `EvaluationCorpus`, returning `{session_id, corpus_id, ingested}`.
- **AC3**: `POST /benchmark/query` runs the KG-2.12 pipeline (`mode="hyde"`, `self_correct`, corpus-scoped), judges the answer, and accumulates the result under `run_id`.

### US-2 — Reproducible, gated scoring
**As** CI, **I want** deterministic pure scoring + a floor gate, **so that** regressions fail fast.
- **AC4**: `normalize_answer`, `judge_binary`, `aggregate_report` are pure and LLM-free; `aggregate_report` yields accuracy + per-category breakdown.
- **AC5**: `scripts/check_longmemeval.py` imports the *same* scoring helpers, exits non-zero below `--floor` (default 0.95), and ships a `--self-test`.

### US-3 — Role integration
- **AC6**: answer synthesis uses the ORCH-1.27 `generator` role; judging uses the `judge` role, each with a deterministic fallback so the harness never hard-fails offline.

## Non-Functional Requirements
- `tests/integration/server/test_benchmark_router.py` (`@pytest.mark.concept(id="AHE-3.12")`), ≤60s, no live engine/LLM (TestClient + pure helpers).
- `pre-commit` green; concepts.yaml regenerated; 7-artifact mandate satisfied.
- Full 500-question LongMemEval-S run is nightly/on-demand; CI gates a frozen subset.
