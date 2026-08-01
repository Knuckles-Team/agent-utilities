# Design Document: The RLM REPL loop classifies every failure into a ranked taxonomy instead of reporting free text, and fatal sandbox death always fast-fails

CONCEPT:AU-ORCH.execution.typed-failure-classification

> `agent_utilities/rlm/repl.py` (live loop wiring, primary) and
> `agent_utilities/rlm/runner.py` (entry-surface wiring), both consuming
> `agent_utilities/rlm/telemetry.py::classify_failure` / `SandboxFatalError` /
> `RunTrace` (that module's own concept id is
> `AU-ORCH.execution.rlm-resilience-telemetry`, a distinct id not in this
> batch — cited here only as supporting evidence for what the `repl.py`/
> `runner.py` marker sites actually wire up).

## The real decision

`repl.py` and `runner.py` both wire the RLM's iteration loop and top-level
entry point to a **fixed, ranked failure taxonomy** rather than free-text
error reflection. `telemetry.py`'s own docstring states what this replaces
directly: *"Structured RunTrace — per-iteration steps (code, output, reasoning,
finish reason) + token usage, **replacing free-text reflections**"*
(`telemetry.py:6-7`). The taxonomy itself is a closed `Literal`:

```
telemetry.py:35-43
FailureClass = Literal[
    "model_generated_bad_code", "host_tool_timeout", "sandbox_exec_timeout",
    "sandbox_fatal", "sandbox_escalated", "evaluator_reject", "unknown",
]
```

with an explicit precedence order (`_PRECEDENCE`, `telemetry.py:49-56`) so
that when multiple failure signals co-occur in one run, `dominant_failure`
returns the single highest-precedence one — `sandbox_fatal` dominates
everything, `sandbox_escalated` (a benign router tier-escalation event, not a
real failure) sits near the bottom.

**Every classification point in the live loop records the class into the
`RunTrace` rather than just logging or re-raising bare.**
`repl.py:686-689` populates a `RunTrace()` at the start of each run
(*"populate a structured RunTrace as the live loop runs for canonical outcome
analysis"*); `repl.py:738-748` wraps each code-execution iteration so that
**both** a fatal sandbox death and a generic exception get
`run_trace.add_step(code=..., failure_class=classify_failure(e))` before the
fatal case re-raises and the generic case sets `final_status = "failure"`;
`repl.py:832` stamps `final_status = "success"` on the one clean-exit path.

**The entry surface (`runner.py:88-90`) inverts the swallow/raise decision by
failure class, not uniformly.** `run_rlm`'s docstring-adjacent comment states
the rule directly:

```
runner.py:88-90
    except SandboxFatalError:
        raise  # fatal sandbox death must fast-fail, never be swallowed.
    except Exception as e:  # noqa: BLE001 - entry surface must not raise
        failure = classify_failure(e)
        return {"ok": False, "error": str(e), "failure_class": failure, "task": task}
```

Recoverable failures are caught and returned as a typed `{"ok": False,
"failure_class": ...}` result so the caller/optimizer gets a structured
signal instead of a raised exception (*"entry surface must not raise"*) — but
`SandboxFatalError` is the one class explicitly re-raised, never swallowed
into that typed result.

## The rejected alternative

Two rejected alternatives, one per surface:

**In the live loop:** free-text error reflection, where the GEPA proposer
(the RLM's prompt-optimization consumer) would have to re-derive a failure
category from prose logs. The taxonomy exists specifically so *"the GEPA
proposer reflects on this classified trace"* (`telemetry.py:7`) instead of
unstructured text — a fixed vocabulary is what makes the trace usable as a
training/reflection signal rather than just a log.

**At the entry surface:** the uniform alternative in either direction —
either let every exception propagate (bad DX for the optimizer/caller, which
now has to handle raw exceptions of unknown shape from every call site) or
swallow every exception uniformly into the typed result (which would hide an
irreversible sandbox death behind the same `{"ok": False}` shape as an
ordinary recoverable failure, letting the optimizer keep iterating on what it
believes is a live sandbox that is actually dead — silently burning the
iteration budget on a run that cannot recover). The chosen design is
class-dependent: swallow-and-classify for everything recoverable, re-raise
unconditionally for the one class that is not.

## Risk Assessment

- **Blast Radius**: `agent_utilities/rlm/repl.py`,
  `agent_utilities/rlm/runner.py`, `agent_utilities/rlm/telemetry.py`.
- **Backward Compatible**: Yes — `classify_failure` falls back to `"unknown"`
  for anything it cannot pattern-match, so an unrecognized exception still
  produces a valid (if uninformative) classification rather than crashing the
  classifier itself.
- **Known weak point**: `classify_failure` (`telemetry.py:60-89`) matches
  largely on `str(exc).lower()` substring checks (`"timeout"`, `"reject"`,
  `"mount"`, …) rather than exception type for most classes — a legitimate
  error message that happens to contain one of these substrings (e.g. a tool
  error whose message mentions "timeout" for an unrelated reason) is
  misclassified, and the precedence order then determines which
  misclassification wins if more than one substring matches.
