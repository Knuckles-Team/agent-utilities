"""Unified Orchestration Protocol — CONCEPT:AU-ORCH.execution.unified-orchestration-protocol.

Defines the structural typing contract that all task-dispatching
orchestrators must satisfy.  Uses ``typing.Protocol`` (PEP 544) so
existing classes gain conformance without inheritance changes — they
just need to implement ``dispatch()`` and ``get_status()``.

Domain-specific facades (SDDOrchestrator, EngineeringPatternOrchestrator,
DynamicToolOrchestrator) are *consumers* of orchestration, not task
dispatchers, and therefore do NOT need to implement this protocol.

Conforming classes:
    - ``Orchestrator`` (orchestration/manager.py)
    - ``AgentOrchestrationEngine`` (graph/graph_orchestrator.py)
    - ``KGDrivenExecutionEngine`` (graph/dynamic_graph_orchestrator.py)
    - ``ParallelEngine`` (graph/parallel_engine.py)
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class OrchestratorProtocol(Protocol):
    """Interface contract that all task-dispatching orchestrators must satisfy.

    CONCEPT:AU-ORCH.execution.unified-orchestration-protocol — Unified Orchestration Protocol

    Any class that implements ``dispatch()`` and ``get_status()`` with
    compatible signatures automatically satisfies this protocol via
    structural subtyping — no explicit inheritance required.

    Example::

        def run_with_any_orchestrator(orch: OrchestratorProtocol):
            result = await orch.dispatch("analyze codebase", context=ctx)
            status = orch.get_status(result["job_id"])
    """

    async def dispatch(self, task: str, **kwargs: Any) -> dict[str, Any]:
        """Dispatch a task for execution.

        Args:
            task: Natural language or structured task description.
            **kwargs: Orchestrator-specific options (e.g., dependencies,
                context, timeout, model overrides).

        Returns:
            Dict containing at minimum ``job_id`` and ``status`` keys.
        """
        ...

    def get_status(self, job_id: str) -> dict[str, Any]:
        """Query the current status of a dispatched task.

        Args:
            job_id: The identifier returned by ``dispatch()``.

        Returns:
            Dict with ``job_id``, ``status``, and optional ``output``.
        """
        ...
