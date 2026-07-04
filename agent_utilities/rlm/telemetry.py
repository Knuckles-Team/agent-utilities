"""CONCEPT:AU-ORCH.execution.rlm-resilience-telemetry — RLM Resilience + Structured Telemetry.

Assimilated from predict-rlm (`Trampoline-AI/predict-rlm@edaddfe`, `src/predict_rlm/trace.py`,
`telemetry.py`, `interpreter.py`). Two concerns that make the RLM robust and give GEPA a
high-signal feedback channel:

1. **Structured RunTrace** — per-iteration steps (code, output, reasoning, finish reason) + token
   usage, replacing free-text reflections. The GEPA proposer reflects on this classified trace.
2. **Failure taxonomy + recoverable-vs-fatal errors** — a per-tool wall-clock timeout returns a
   *recoverable* error (the sandbox survives, the model can retry), while irreversible sandbox death
   raises :class:`SandboxFatalError` to **fast-fail** the run instead of iterating on a dead sandbox.

Pure models + helpers; fully unit-testable.
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from pydantic import BaseModel, Field

# Per-host-tool wall-clock budget. A tool exceeding it is RECOVERABLE (sandbox stays alive).
TOOL_CALL_TIMEOUT_SEC: float = 180.0


class SandboxFatalError(RuntimeError):
    """An irreversible sandbox failure (subprocess killed, mount lost) — fast-fail the run.

    Deliberately a ``RuntimeError`` and NOT a recoverable tool error, so the RLM loop does not catch
    it and keep iterating on a dead sandbox (which would silently burn the iteration budget).
    """


FailureClass = Literal[
    "model_generated_bad_code",
    "host_tool_timeout",
    "sandbox_exec_timeout",
    "sandbox_fatal",
    "sandbox_escalated",
    "evaluator_reject",
    "unknown",
]

# Precedence: a fatal sandbox failure dominates a tool timeout dominates a code error, etc.
# ``sandbox_escalated`` (a backend rejected the snippet and the router moved to the next tier,
# CONCEPT:AU-ORCH.sandbox.tiered-rlm-sandbox) sits near the bottom — it is a benign routing event, not a run failure,
# and is dominated by any real failure that co-occurred.
_PRECEDENCE: tuple[FailureClass, ...] = (
    "sandbox_fatal",
    "sandbox_exec_timeout",
    "host_tool_timeout",
    "model_generated_bad_code",
    "evaluator_reject",
    "sandbox_escalated",
    "unknown",
)


def classify_failure(exc: BaseException | str) -> FailureClass:
    """Classify a failure into the taxonomy the GEPA proposer keys off (CONCEPT:AU-ORCH.execution.rlm-resilience-telemetry)."""
    if isinstance(exc, SandboxFatalError):
        return "sandbox_fatal"
    if isinstance(exc, asyncio.TimeoutError):
        return "host_tool_timeout"
    # ORCH-1.38: a router escalation (SandboxRejected). Matched by class name to avoid a
    # circular import (this module is imported by the sandboxes layer). Must precede the
    # generic "reject" text match below so it is not mis-bucketed as ``evaluator_reject``.
    if type(exc).__name__ == "SandboxRejected" and not isinstance(exc, str):
        return "sandbox_escalated"
    text = str(exc).lower()
    if (
        "sandboxfatal" in text
        or "subprocess has been killed" in text
        or "mount" in text
    ):
        return "sandbox_fatal"
    if "timeout" in text and "sandbox" in text:
        return "sandbox_exec_timeout"
    if "timeout" in text:
        return "host_tool_timeout"
    if any(
        k in text
        for k in ("syntaxerror", "nameerror", "indentationerror", "unterminated")
    ):
        return "model_generated_bad_code"
    if "reject" in text or "invalid output" in text:
        return "evaluator_reject"
    return "unknown"


def dominant_failure(classes: list[FailureClass]) -> FailureClass:
    """Return the highest-precedence failure class in a list (or 'unknown')."""
    for c in _PRECEDENCE:
        if c in classes:
            return c
    return "unknown"


class LMUsage(BaseModel):
    """Token usage for a trace (main LM + sub-LM)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    sub_lm_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens + self.sub_lm_tokens


class IterationStep(BaseModel):
    """One RLM REPL iteration."""

    index: int
    code: str = ""
    output: str = ""
    reasoning: str = ""
    finish_reason: str = ""
    failure_class: FailureClass | None = None


class RunTrace(BaseModel):
    """Structured trace of a full RLM run — the GEPA proposer's high-signal feedback (ORCH-1.29)."""

    steps: list[IterationStep] = Field(default_factory=list)
    usage: LMUsage = Field(default_factory=LMUsage)
    final_status: Literal["success", "failure", "partial"] = "partial"

    def add_step(self, **kwargs: Any) -> IterationStep:
        step = IterationStep(index=len(self.steps), **kwargs)
        self.steps.append(step)
        return step

    def failure_summary(self) -> FailureClass | None:
        classes = [s.failure_class for s in self.steps if s.failure_class]
        return dominant_failure(classes) if classes else None


async def with_tool_timeout(
    coro: Any, *, seconds: float = TOOL_CALL_TIMEOUT_SEC
) -> tuple[bool, Any]:
    """Await a host tool with a wall-clock budget; a timeout is RECOVERABLE (CONCEPT:AU-ORCH.execution.rlm-resilience-telemetry).

    Returns ``(ok, value_or_error)``: ``(True, result)`` on success, ``(False, "<timeout msg>")`` on
    timeout — the caller surfaces the message to the model and keeps the sandbox alive, rather than
    killing it. ``SandboxFatalError`` from ``coro`` is re-raised (never swallowed).
    """
    try:
        return True, await asyncio.wait_for(coro, timeout=seconds)
    except TimeoutError:
        return (
            False,
            f"Tool call exceeded {seconds:.0f}s budget (recoverable — retry or adjust).",
        )
    except SandboxFatalError:
        raise  # fatal: must propagate to fast-fail the run
