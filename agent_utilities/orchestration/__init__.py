"""Orchestration Module.

CONCEPT:ORCH-2.0 — Unified Orchestration Engine
"""

from .engine import AgentOrchestrationEngine

__all__ = [
    "AgentOrchestrationEngine",
]

# Backward compatibility aliases that raise DeprecationWarnings if possible,
# or just alias directly for now to not break tests.
AgentOrchestrationEngine = AgentOrchestrationEngine
KGDrivenExecutionEngine = AgentOrchestrationEngine
ParallelEngine = AgentOrchestrationEngine
WorkflowRunner = AgentOrchestrationEngine
