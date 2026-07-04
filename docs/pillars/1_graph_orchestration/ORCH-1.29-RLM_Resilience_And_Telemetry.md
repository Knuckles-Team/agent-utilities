# RLM Resilience + Structured Telemetry (CONCEPT:AU-ORCH.execution.typed-failure-classification)

## Overview

Makes the RLM robust and gives GEPA a high-signal feedback channel: a **structured RunTrace** (per
iteration + token usage) replaces free-text reflections, a **failure taxonomy** classifies errors,
and **recoverable-vs-fatal** error handling stops the RLM burning iterations on a dead sandbox.
Assimilated from predict-rlm (`trace.py`, `telemetry.py`, `interpreter.py`). Extends **ORCH-1.12**.

## How it works

- **`RunTrace` / `IterationStep` / `LMUsage`** (`rlm/telemetry.py`) — structured per-iteration capture
  (code, output, reasoning, finish reason) + main/sub-LM token usage; `failure_summary()` returns the
  dominant failure class for the proposer.
- **`FailureClass`** taxonomy (`model_generated_bad_code`, `host_tool_timeout`, `sandbox_exec_timeout`,
  `sandbox_fatal`, `sandbox_escalated`, `evaluator_reject`, `unknown`) with precedence ordering;
  `classify_failure` maps an exception/text to a class (`sandbox_escalated` = a benign ORCH-1.38
  router escalation, low precedence).
- **Recoverable vs fatal** — `with_tool_timeout` gives each host tool a wall-clock budget and returns a
  *recoverable* error on timeout (sandbox survives), while `SandboxFatalError` (raised from `repl.py`'s
  container path on irreversible sandbox death) **fast-fails** the run.

## Key files / API

| Piece | Location |
|---|---|
| Telemetry + resilience | `rlm/telemetry.py` (`RunTrace`, `FailureClass`, `classify_failure`, `dominant_failure`, `with_tool_timeout`, `SandboxFatalError`) |
| Fatal wiring | `rlm/sandboxes/docker_backend.py` (`DockerSandbox` raises `SandboxFatalError` on dead container/timeout); the router propagates it without escalating (ORCH-1.38) |

## Wiring (≤3 hops)
`graph_orchestrate(rlm_run)` → `runner` → `repl`/`telemetry` (≤3 hops).

## Research provenance
predict-rlm `src/predict_rlm/trace.py`, `telemetry.py`, `interpreter.py` (`SandboxFatalError`, `TOOL_CALL_TIMEOUT_SEC`) — verified.
