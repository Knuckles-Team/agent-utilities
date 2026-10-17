"""Failure classes of a recursive run, so the caller gets a typed signal."""

from __future__ import annotations

from typing import Literal

FailureClass = Literal[
    "model_generated_bad_code",
    "host_tool_timeout",
    "sandbox_exec_timeout",
    "sandbox_fatal",
    "sandbox_escalated",
    "evaluator_reject",
    "unknown",
]

#: Highest precedence first: a fatal failure dominates a timeout, and so on.
PRECEDENCE: tuple[FailureClass, ...] = (
    "sandbox_fatal",
    "sandbox_exec_timeout",
    "host_tool_timeout",
    "model_generated_bad_code",
    "evaluator_reject",
    "sandbox_escalated",
    "unknown",
)


class SandboxFatalError(RuntimeError):
    """An irreversible execution failure; the run fails fast and never retries."""


def classify_failure(exc: BaseException | str) -> FailureClass:
    """Map an exception (or its message) to one failure class."""
    if isinstance(exc, SandboxFatalError):
        return "sandbox_fatal"
    if isinstance(exc, TimeoutError):
        return "host_tool_timeout"
    if not isinstance(exc, str) and type(exc).__name__ == "SandboxRejected":
        return "sandbox_escalated"
    text = str(exc).lower()
    if "sandboxfatal" in text or "subprocess has been killed" in text:
        return "sandbox_fatal"
    if "timeout" in text:
        return "sandbox_exec_timeout" if "sandbox" in text else "host_tool_timeout"
    if any(k in text for k in ("syntaxerror", "nameerror", "indentationerror")):
        return "model_generated_bad_code"
    if "reject" in text or "invalid output" in text:
        return "evaluator_reject"
    return "unknown"


def dominant_failure(classes: list[FailureClass]) -> FailureClass:
    """The highest-precedence class in ``classes`` (``"unknown"`` when empty)."""
    return next((c for c in PRECEDENCE if c in classes), "unknown")


__all__ = [
    "PRECEDENCE",
    "FailureClass",
    "SandboxFatalError",
    "classify_failure",
    "dominant_failure",
]
