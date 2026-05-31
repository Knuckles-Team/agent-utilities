"""Orchestration Module.

CONCEPT:ORCH-2.0 — Orchestration Engine
"""

from typing import Any

from .engine import AgentOrchestrationEngine

__all__ = [
    "AgentOrchestrationEngine",
    "ParallelEngine",
    "WorkflowRunner",
]


def __getattr__(name: str) -> Any:
    if name == "ParallelEngine":
        from agent_utilities.graph.parallel_engine import ParallelEngine

        return ParallelEngine
    if name == "WorkflowRunner":
        from agent_utilities.workflows.runner import WorkflowRunner

        return WorkflowRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
