# Design Document: The planner's shape is authoritative over the verifier's own local heuristic when it opts a job out of verification

CONCEPT:AU-ORCH.execution.dynamic-shaper-authority

> `agent_utilities/graph/verification.py:236-255` (the verifier graph node).

## Decision — when `shape.run_verifier is False` and a result already exists, the verifier skips its quality gate on the shape's say-so, not by re-deriving the decision locally

The verifier node already had a LOCAL proportional-verification heuristic
(tagged `CONCEPT:AU-ORCH.execution.orchestration-flow-mermaid (perf)` in the
same block): a trivial, single-step or plan-less direct-dispatch read whose
result is non-empty skips the full LLM quality-gate + re-plan machinery,
because that machinery was previously scoring genuinely correct answers
`0.00` (for not volunteering fields the query never asked about) and then
looping until context overflow trying to "fix" a non-existent problem. That
local heuristic is a real, independent decision — but it can only see what
the verifier node itself observes (is the result non-empty, was there a
plan) at the moment it runs.

`dynamic-shaper-authority` is the newer, distinct decision layered on top:
the per-job `ExecutionProfile` the escalating planner already built
(`CONCEPT:AU-ORCH.execution.dynamic-execution-profile`) carries a
`run_verifier` flag decided from BROADER per-job context the verifier node
never sees directly (recipe-cache history, the learned shape policy, the
lexical/structural cascade). When that shape explicitly opted a job OUT of
verification (`run_verifier=False`) and a result exists, the verifier
DEFERS to that upstream decision and routes straight to synthesis — the
local proportional heuristic is explicitly kept only "as a floor for
shape-less callers" (a caller that never went through the planner at all).

**The rejected alternative is letting the verifier keep making this call
unilaterally from its own local signals, ignoring the shape entirely.** That
would mean the verifier could disagree with a shape the planner already spent
real cost computing — re-deriving from a narrower view of the job a decision
the planner made with strictly more context, and potentially running the
expensive quality-gate machinery on a job the planner had already determined
didn't need it (a `direct_complete`/lean job, for instance). Giving the
shape authority means the two layers agree by construction: whichever
component decided the job's altitude gets the final say, rather than two
independent heuristics both trying to answer "does this need verification?"
from different information.

## Risk Assessment

- **Blast Radius**: `agent_utilities/graph/verification.py` (the verifier
  node only — synthesis and every other node are unaffected).
- **Backward Compatible**: Yes — a caller with no `execution_shape` on
  `ctx.deps` (the pre-planner code path) falls through to the unchanged local
  proportional heuristic.
- **Known weak point**: the authority handoff is one-directional and
  silent — if the shape's `run_verifier=False` decision was wrong for a
  specific job (e.g. the planner's cascade mis-classified a job as lean when
  it genuinely needed verification), the verifier has no mechanism to
  override or flag disagreement; it trusts the shape unconditionally
  whenever a non-empty result exists.
