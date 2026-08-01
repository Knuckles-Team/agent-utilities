# Design Document: Unattended overnight runs stay propose-only; the loop never auto-approves what it cannot ask about

CONCEPT:AU-AHE.harness.overnight-loop-driver

> `agent_utilities/claude_harness/overnight_runner.py`.

## Decision — no human to answer an `ask` mid-run means the loop never auto-approves anything; it halts to the permission fence instead

The module docstring (`overnight_runner.py:4-19`) states the constraint and
the resolution directly. This core drives the existing `LoopController` once
per iteration, commits after each productive cycle, stops when the loop
converges (no new progress) or a cap is hit, and writes a morning summary
into `MEMORY.md` so the existing memory bridge surfaces it on the next
session. But "there is no human to answer an `ask` mid-run," so **the core
never auto-approves anything**: the permission fence halts `ask`/`deny` tool
calls, and the loop only ever advances the propose-only Loop cycle — it
writes proposals, never executes high-stakes actions.

**The rejected alternative is auto-approving prompts during unattended runs
to get more done overnight** — the obvious way to make an unattended loop
maximally productive is to remove the human-approval bottleneck entirely
while no human is present. That's explicitly rejected: instead of widening
what the loop can do unattended, the design narrows what an unattended loop
is ALLOWED to do to propose-only work, so a run that would normally need an
`ask` simply halts rather than silently getting itself approved. The safety
budget is spent on convergence detection and morning-summary write-back
instead — the loop stops itself when nothing new is happening rather than
running indefinitely, and the human sees a written record on the next
session regardless of how the run ended.

## Risk Assessment

- **Blast Radius**: `agent_utilities/claude_harness/overnight_runner.py`,
  `agent_utilities/claude_harness/__init__.py`,
  `agent_utilities/cli/__init__.py`,
  `agent_utilities/knowledge_graph/research/loop_controller.py`.
- **Backward Compatible**: Yes — this is an additive CLI-invoked mode, not a
  change to the underlying `LoopController`'s default behavior.
- **Known weak point**: "propose-only" is enforced by the loop only ever
  advancing the Loop cycle's propose path — if a future code change to
  `LoopController` adds a new action type that isn't propose-only by
  default, this driver has no independent check that would catch it; the
  containment relies on the Loop cycle's own contract staying propose-only.
