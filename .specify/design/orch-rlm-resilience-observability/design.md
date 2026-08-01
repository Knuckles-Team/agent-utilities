# Design Document: RLM failures are classified, not just caught — and the same recoverable/fatal discipline is what lets AHE point RLM at its own failure corpus

CONCEPT:AU-ORCH.execution.rlm-resilience-telemetry · CONCEPT:AU-ORCH.execution.rlm-experience-observability

> `agent_utilities/rlm/telemetry.py` (the taxonomy + structured trace) and
> `agent_utilities/harness/continuous_evaluation_engine.py` (AHE's own failure-clustering
> pipeline applying the same RLM-with-fallback posture at a different layer).

## Decision — `AU-ORCH.execution.rlm-resilience-telemetry`: a typed failure taxonomy replaces free-text reflection, and recoverable errors are structurally distinct from fatal ones

`agent_utilities/rlm/telemetry.py:1-14` states this was "assimilated from predict-rlm
(`Trampoline-AI/predict-rlm@edaddfe`, `src/predict_rlm/trace.py`, `telemetry.py`,
`interpreter.py`)" to solve two concerns at once: a structured `RunTrace` (per-iteration code,
output, reasoning, finish reason, token usage) "replacing free-text reflections" so the GEPA
proposer reflects on classified data instead of prose; and a failure taxonomy that separates
*recoverable* from *fatal* errors.

The taxonomy is a closed `FailureClass` literal (`telemetry.py:35-43`:
`model_generated_bad_code`, `host_tool_timeout`, `sandbox_exec_timeout`, `sandbox_fatal`,
`sandbox_escalated`, `evaluator_reject`, `unknown`) with an explicit precedence order
(`telemetry.py:49-57`) so that when several failure signals co-occur in one run, the dominant
one is deterministic rather than "whichever exception was caught last." `classify_failure()`
(`telemetry.py:60-89`) pattern-matches exception type and text into that taxonomy, and
`SandboxFatalError` (`telemetry.py:27-32`) is deliberately a plain `RuntimeError` rather than
a recoverable-tool-error subtype, "so the RLM loop does not catch it and keep iterating on a
dead sandbox (which would silently burn the iteration budget)" — the docstring names the
failure mode being prevented directly. `with_tool_timeout()` (`telemetry.py:140-157`) is the
concrete recoverable case: a per-tool wall-clock budget (`TOOL_CALL_TIMEOUT_SEC = 180.0`)
returns `(False, "<timeout msg>")` on timeout so "the caller surfaces the message to the model
and keeps the sandbox alive, rather than killing it" — while any `SandboxFatalError` raised
inside the awaited coroutine is explicitly re-raised, never swallowed.

**The rejected alternative** is the two failure modes this replaces, both named in the module
docstring and the code comments: (1) **free-text reflection** — the GEPA proposer working
from unstructured prose about what went wrong, which is exactly the "parent re-reads and
re-classifies dozens of unstructured blurbs" problem this codebase repeatedly rejects
elsewhere (see the sibling `predict-rlm-runtime` decision); and (2) **uniform error handling**
— treating every exception as equally recoverable (which would keep iterating on a dead
sandbox and silently burn the iteration budget) or equally fatal (which would kill the run on
a transient per-tool timeout that a retry would have survived). The taxonomy's whole value is
in refusing to collapse those two very different failure shapes into one catch-all.

### Pointer — `CONCEPT:AU-ORCH.execution.rlm-experience-observability`

`agent_utilities/harness/continuous_evaluation_engine.py:281-338` is this same resilience
posture applied one layer up, inside AHE's own trace distillation pipeline. `_cluster_failures`
(`continuous_evaluation_engine.py:281-316`) has two implementations of "group failures by root
cause": `_cluster_failures_keyword` (always available) and `_cluster_failures_rlm`
(`continuous_evaluation_engine.py:318-338`+, marked "RLM for AHE Experience Observability" at
line 325) which spawns an `RLMEnvironment` to do deep semantic clustering over the KG when the
failure count is large. The routing is threshold-gated through the *same*
`RLMConfig.should_trigger(trace_count=...)` hierarchy from the paradigm decision
(`continuous_evaluation_engine.py:300-309`, `rlm_config.should_trigger(trace_count=len(failures))`),
not a bespoke size check — and on any exception from the RLM path, the code falls straight
back to keyword clustering with a logged warning (`continuous_evaluation_engine.py:308-316`:
`except Exception as e: logger.warning(...); return self._cluster_failures_keyword(failures)`).
That is structurally identical to `rlm-resilience-telemetry`'s core bet: use the expensive,
semantically-richer path when it is justified and available, but never let its failure take
the whole pipeline down — degrade to the cheaper deterministic path instead. It groups here,
not with the eval-set-compounding decision, because nothing about this code concerns the
evaluation *corpus*; it is entirely about how AHE decides whether and how safely to spend RLM
on its own failure-trace analysis.

## Risk Assessment

- **Blast Radius**: `agent_utilities/rlm/telemetry.py`,
  `agent_utilities/harness/continuous_evaluation_engine.py`, and any code importing
  `classify_failure`/`RunTrace`/`SandboxFatalError` from `rlm.telemetry` (notably
  `agent_utilities/rlm/repl.py` and `agent_utilities/rlm/runner.py`, both of which import this
  module under the `typed-failure-classification` marker).
- **Backward Compatible**: Yes — additive typed telemetry over what was previously ad hoc
  exception handling.
- **Known weak point**: `classify_failure()`'s text-matching branches (`"timeout" in text`,
  `"reject" in text`) are string-heuristic, not structural, for every case except
  `SandboxFatalError`/`asyncio.TimeoutError`/`SandboxRejected` — a differently-worded exception
  message from a future sandbox backend silently falls into `"unknown"` rather than erroring
  loudly.
