# Design Document: The eight agent-loop exits are enforced by the harness, never merely requested of the model

CONCEPT:AU-AHE.harness.loop-exit-conditions

> `agent_utilities/orchestration/loop_guards.py` (primary — the shared
> primitives), `agent_utilities/orchestration/agent_runner.py:176-225`
> (exit 7 wired into the execution path).

## Decision — loop-exit conditions are small, reusable, side-effect-free primitives enforced in code, shared by both loop drivers

The module docstring (`loop_guards.py:4-10`) states the decision directly:
the eight agent-loop exit conditions are *enforced* by the harness, not
merely requested of the model in a prompt. The primitives are deliberately
generic so both `LoopController.run_loop` (the KG-driven loop) and
`orchestration.agent_runner`'s execution path share them rather than each
reimplementing its own version of "when do we stop":

- `ConsecutiveFailureGuard` lifts the transport layer's
  `engine_breaker.CircuitBreaker` threshold+reset semantics to *loop
  iterations* — the same counter/reset idiom the fan-out backend already
  uses, generalized rather than reinvented for loops.
- `GoalEvaluation`/`build_goal_evaluator` gives a measured pass/fail (0..1
  score) so a loop stops on a *real, verified* success — deterministic
  validation for `develop` loops, an LLM-judge rubric for `research`/`skill`
  loops.
- `progress_signature` hashes `(status, output, checkpoint)` to detect a
  stalled loop.
- `resolve_deadline` turns a relative or absolute duration into one
  monotonic wall-clock deadline.

**The rejected alternative, named directly by the concept's framing, is
trusting the model's own self-declaration of "done."** A loop that stops
because the callee said it finished has no independent verification — a
model can declare success on a task it didn't actually complete. Wiring exit
7 (error threshold) into `agent_runner.py:176-225` shows the same discipline
applied to delegation outcomes: a process-wide per-agent
`ConsecutiveFailureGuard` tracks degraded/no-data runs across repeated
`run_agent` calls, but deliberately does NOT abort a single `run_agent` call
itself (that would change its one-shot contract) — it only tracks the signal
so a *driving loop* can choose to halt, keeping the guard's scope
intentionally narrow.

## Risk Assessment

- **Blast Radius**: `agent_utilities/orchestration/loop_guards.py`,
  `agent_utilities/orchestration/agent_runner.py`,
  `agent_utilities/knowledge_graph/research/loop_controller.py`,
  `agent_utilities/graph/state.py`,
  `agent_utilities/capabilities/content_guardrails.py`.
- **Backward Compatible**: Yes — the primitives are additive; a loop that
  doesn't consult them behaves as before.
- **Known weak point**: `GoalEvaluation`'s deterministic-vs-rubric split
  degrades to "trusting the callee" when no model endpoint is reachable for
  `research`/`skill` loops — the exact self-declaration failure mode the
  concept exists to avoid, reintroduced as a fallback under a specific
  infrastructure-unavailability condition.
