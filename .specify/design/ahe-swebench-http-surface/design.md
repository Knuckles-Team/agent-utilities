# Design Document: The SWE-bench harness reuses the LongMemEval benchmark-router shape, not a bespoke API

CONCEPT:AU-AHE.harness.swebench-http-surface

> `agent_utilities/harness/swebench_harness.py`,
> `agent_utilities/server/routers/swebench.py`.

## Decision — one injectable, LLM-free-scoring harness behind an HTTP surface that deliberately mirrors an existing benchmark router's shape

Both files carry the same concept id because they are one decision viewed
from two layers. `swebench_harness.py`'s docstring states the per-instance
flow: provision a developer workspace, clone the repo at `base_commit`,
optionally KG-ingest it for grounding, run the agent, apply the gold
`test_patch`, run the FAIL_TO_PASS/PASS_TO_PASS selectors, score "resolved."
Orchestration is injectable — `solve` defaults to the KG-grounded SWE agent
but a test can pass a scripted solver — and the scoring helpers
(`is_resolved`, `aggregate_report`) are pure and LLM-free, so scoring
correctness doesn't depend on a model call. `routers/swebench.py`'s
docstring states the HTTP contract explicitly: "Mirrors the LongMemEval
benchmark router shape: POST a set of instances to run, GET the aggregate
report."

**The rejected alternative is designing a one-off HTTP shape specific to
SWE-bench's own domain concepts (instances, patches, FAIL_TO_PASS
selectors).** Instead, reusing the already-established LongMemEval router
pattern (`server/routers/benchmark.py`) means an operator who already knows
how to drive one benchmark harness over HTTP already knows how to drive
this one — same POST-to-run / GET-to-aggregate shape, different domain
payload. The router optionally files failure-gap Concepts for unresolved
instances (AHE-3.23), directly wiring this HTTP surface into
`swe-failure-remediation`'s golden-loop feedback rather than treating the
HTTP layer as a dead-end reporting endpoint.

## Risk Assessment

- **Blast Radius**: `agent_utilities/harness/swebench_harness.py`,
  `agent_utilities/server/routers/swebench.py`,
  `agent_utilities/server/routers/benchmark.py` (the shape this mirrors),
  `agent_utilities/harness/swebench_corpus.py`.
- **Backward Compatible**: Yes — an additive router; scoring helpers are
  pure functions independently testable.
- **Known weak point**: `_RUNS: dict[str, dict[str, Any]]` in the router is
  an in-process dict — run state does not survive a process restart and
  isn't shared across multiple server instances, so a POST-then-GET
  workflow only works reliably against the same long-lived process.
