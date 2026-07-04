"""Orchestration Module.

CONCEPT:AU-ORCH.execution.orchestration — Orchestration Engine
"""

from typing import Any

__all__ = [
    "AgentOrchestrationEngine",
    "ParallelEngine",
    "WorkflowRunner",
]


def __getattr__(name: str) -> Any:
    if name == "AgentOrchestrationEngine":
        # ``engine`` pulls the pydantic-ai/pydantic-graph agent runtime (via
        # ``graph.mermaid``/``graph.state`` and the model factory). Import it
        # lazily — like the two exports below — so importing a *lean* sibling
        # (e.g. ``orchestration.action_policy``) does not drag the ``[agent]``-
        # extra deps onto the import path. Keeps the package importable in the
        # lean serving/CI install (Dependency discipline).
        from .engine import AgentOrchestrationEngine

        return AgentOrchestrationEngine
    if name == "ParallelEngine":
        from agent_utilities.graph.parallel_engine import ParallelEngine

        return ParallelEngine
    if name == "WorkflowRunner":
        from agent_utilities.workflows.runner import WorkflowRunner

        return WorkflowRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
