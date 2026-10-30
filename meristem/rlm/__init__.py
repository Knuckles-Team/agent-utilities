"""Recursive run loop: one root agent that reads a long input through tools
and delegates bounded sub-queries to child runs, up to a depth it is given."""

from .runner import RLMConfig, run_rlm
from .telemetry import FailureClass, SandboxFatalError, classify_failure

__all__ = [
    "FailureClass",
    "RLMConfig",
    "SandboxFatalError",
    "classify_failure",
    "run_rlm",
]
