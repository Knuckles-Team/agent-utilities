# Design Document: A plausible-but-empty answer is a DEGRADED outcome, not a success

CONCEPT:AU-ORCH.execution.degraded-no-data-outcome ·
CONCEPT:AU-ORCH.execution.all-tool-calls-errored

> `agent_utilities/graph/verification.py` (`no_data`/`execution_success` flag,
> the synthesizer step) and `agent_utilities/orchestration/agent_runner.py`
> (`_delegation_degraded`, `_result_looks_like_error`). Introduced by commit
> `f41c59c0` ("fail-loud + self-healing feedback on degraded delegations").

## Decision — a run that gathered nothing still returns `status="completed"` at the transport level, but is tagged `degraded=True` so nothing downstream mistakes it for a real answer

`CONCEPT:AU-ORCH.execution.degraded-no-data-outcome`

`verification.py:645-649` names the problem directly: a fall-through to the
generic "…unable to find specific data…" sentinel was previously recorded as
a plain success. That poisoned every learning surface that reads outcome as a
reward signal — the reward flywheel, the Self-Model, TeamConfig — because a
non-answer scored `1.0`, and it hid the failure from the `:RunTrace` entirely
(`verification.py:744-751`, `execution_success` explicitly excludes `no_data`
and `structured_failure` from counting as success for the post-execution
feedback loop, `AU-KG.maintenance.post-execution-feedback`).

The fix does not change the user-facing transport status (a "no data found"
answer is still a completed turn from the caller's point of view — it isn't
an exception). It adds a **structured `degraded` flag** read out of
`GraphResponse.metadata`, set by the graph synthesizer itself rather than
inferred after the fact from string-matching the output text. `agent_runner.
_delegation_degraded()` (`agent_runner.py:3793-3810`) reads that flag first;
only if it's absent does it fall back to an output-text sentinel/empty-output
check, so single-server and focused-tools paths (which never touch the
synthesizer) are still covered.

**The rejected alternative was output-text sentinel matching as the ONLY
signal** — brittle (a rephrased "I couldn't find that" sentence silently stops
being detected) and late (it only catches degradation that surfaces as
specific known phrases, `_DELEGATION_DEGRADED_SENTINELS =
("unable to find specific data", "could not be generated")`). The structured
flag from the producer (the synthesizer, which actually knows whether it
gathered zero results) is authoritative; the sentinel match is kept only as a
fallback for paths that never run the synthesizer at all.

### Pointer — `CONCEPT:AU-ORCH.execution.all-tool-calls-errored`

`agent_runner.py:3728-3752` and `3793-3810`. The second, independent half of
the same detector: a run that DID call tools, but **every** call errored
(the example in the docstring: 13 Kubernetes calls that all raised `"has no
attribute"`), produced no tool-grounded result either — and the graph
synthesizer's `degraded` flag doesn't see this case, because from the
synthesizer's point of view the run "completed" with tool output, it just
never checked whether that output was actually error text. `_TOOL_ERROR_MARKERS`
(`"error executing"`, `"traceback (most recent"`, `"has no attribute"`,
`"exception:"`, `"failed:"`, `"is not defined"`) and `_result_looks_like_error()`
scan each tool call's result string; `_delegation_degraded` treats "every
call in the list looks like an error" as degraded even when
`metadata.degraded` was never set. Same trust category as Decision 1 — a
tool-call-shaped failure that would otherwise read as "ran fine, used tools,
returned an answer" — but a structurally distinct detection path (result-text
pattern match on `:ToolCall` results, not a synthesizer-set flag), which is
why it earns its own concept rather than folding silently into the flag
check.

## Risk Assessment

- **Blast Radius**: `agent_utilities/graph/verification.py`,
  `agent_utilities/orchestration/agent_runner.py`,
  `tests/unit/test_delegation_degraded_outcome.py`.
- **Backward Compatible**: Yes — transport-level status is unchanged; only the
  internal degraded/success accounting used for learning and `:RunTrace`
  changed.
- **Known weak point**: `_TOOL_ERROR_MARKERS` is a fixed English-language
  substring list. A tool that returns an error in a shape/language none of
  the six markers match (e.g. a JSON error envelope with no matching prose)
  is invisible to the all-tool-calls-errored detector and relies on the
  synthesizer's own `degraded` flag or the sentinel fallback to catch it
  instead.
