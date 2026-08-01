# Design Document: SWE-bench instances live in the general EvalCorpus, explicitly NOT the finance-shaped BacktestHarness schema

CONCEPT:AU-AHE.harness.swebench-instance-corpus

> `agent_utilities/harness/swebench_corpus.py`.

## Decision — a thin, framework-agnostic instance loader persisted through the general `EvalCorpus`, not a domain-specific schema reused by analogy

The module docstring (`swebench_corpus.py:1-7`) states the decision and its
rejected alternative in the same breath: `SweBenchInstance` (repo, base
commit, problem statement, gold `test_patch`, FAIL_TO_PASS/PASS_TO_PASS
selectors) persists through the general `EvalCorpus` (graph-first, memory
fallback) "so instances live in the same place as every other eval case —
we do NOT reuse the finance-shaped `BacktestHarness` schema, which would be
a forced fit."

**The rejected alternative is named explicitly: reusing `BacktestHarness`**
— an existing, working evaluation-run schema that superficially looks
reusable (it already models runs, metrics, comparisons). The decision
rejects that reuse specifically because `BacktestHarness` is shaped around
finance backtesting semantics (trades, positions, benchmark comparisons)
that don't map cleanly onto "did this code change make FAIL_TO_PASS tests
pass without breaking PASS_TO_PASS tests" — forcing the fit would mean
either contorting SWE-bench semantics into finance-shaped fields or leaving
several `BacktestHarness` fields meaningless for every SWE-bench instance.
Instead, `SweBenchInstance` is its own minimal dataclass, following the
upstream SWE-bench dataset's own field-naming convention, and rides the
domain-agnostic `EvalCorpus` alongside every other kind of eval case.

## Risk Assessment

- **Blast Radius**: `agent_utilities/harness/swebench_corpus.py`,
  `agent_utilities/harness/eval_corpus.py`,
  `agent_utilities/harness/swebench_harness.py` (consumes `SweBenchInstance`).
- **Backward Compatible**: Yes — an additive, self-contained loader.
- **Known weak point**: field names deliberately follow the SWE-bench dataset
  convention rather than this codebase's own naming style, so
  `SweBenchInstance` reads inconsistently next to other `EvalCorpus`-adjacent
  types — a documented, accepted tradeoff for staying framework-agnostic
  with the upstream dataset shape, not an oversight.
