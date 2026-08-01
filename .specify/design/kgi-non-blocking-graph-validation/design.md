# Design Document: Post-ingestion validation auto-fixes and logs, but never blocks the pipeline on anything short of a truly unusable graph

> `agent_utilities/knowledge_graph/pipeline/phases/validate.py`.

CONCEPT:AU-KG.ingest.non-blocking-graph-validation

## Decision — validation runs as a non-blocking post-ingestion phase, tiered by severity

`validate.py:1-30`.

**The rejected alternative**: a blocking pre-commit (or blocking
post-commit) validation gate that fails the whole ingestion pipeline run on
any detected integrity/quality issue. That is the conventional "validate
before you trust it" design, and it is explicitly NOT what this phase does.

**The design chosen**: `execute_validate` runs `GraphValidator` as a
non-blocking step AFTER ingestion completes, with issues split into tiers:
**Tier 1** issues are auto-fixed inline; **Tier 2/3/4** issues are logged
(for later inspection/trend analysis via `EvaluationCapture`) but do not
themselves stop the pipeline. Completion is blocked ONLY when Tier 4 fatals
are detected **AND** the graph is truly unusable (zero nodes) — the
conjunction of both conditions, not either alone, is what triggers a hard
stop.

**Why this tradeoff was made**: ingestion at scale will always produce SOME
imperfect material (a malformed source record, a partially-resolved
reference). A blocking gate on every integrity issue would make ingestion
runs brittle against noise that is real but not fatal — trading pipeline
availability for a validation-purity guarantee the ingestion volume can't
actually sustain. The chosen design keeps the pipeline available and
observable (validation metrics persist to `eval_capture` for trend analysis)
while still auto-healing what can be auto-healed and refusing to silently
accept a genuinely empty/broken graph.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/pipeline/phases/validate.py`,
  `agent_utilities/knowledge_graph/security/graph_validator.py`,
  `agent_utilities/knowledge_graph/memory/EvaluationCapture`.
- **Backward Compatible**: Yes — validation is additive to the pipeline; a
  pipeline that never had this phase simply gains logged metrics.
- **Breaking Changes**: None, except the explicit zero-nodes+Tier-4 hard
  stop, which is a deliberate new failure mode.
- **Known weak point**: Tier 2/3 issues are logged but never surfaced as an
  actionable alert by this phase itself — a slow accumulation of non-fatal
  integrity issues across many ingestion runs relies on someone actively
  reviewing the trend data in `eval_capture`, rather than the pipeline
  raising a threshold-based alarm on its own.
