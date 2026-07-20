# Tasks: LongMemEval-S Validation Harness (AHE-3.12)

## T1 — Benchmark router (US-1,3)  [code]
- [x] `server/routers/benchmark.py`: `/benchmark/{health,session,query,report}` + generator/judge role glue.

## T2 — Pure scoring (US-2)  [code]
- [x] `normalize_answer`, `judge_binary`, `aggregate_report` in the router module.

## T3 — Registration (US-1)  [code]
- [x] `include_router(benchmark.router)` in `server/app.py`; export in `server/routers/__init__.py`.

## T4 — CI gate (US-2)  [code]
- [x] `scripts/check_longmemeval.py` (shared scoring import, `--floor`, `--self-test`).

## T5 — Tests (NFR)  [test]
- [x] `tests/integration/server/test_benchmark_router.py` — AC1–AC5 (5 tests).

## T6 — Artifacts (NFR)  [docs]
- [x] concepts.yaml regen (AHE-3.12); CHANGELOG; README AHE-3 count; pillar doc note.
