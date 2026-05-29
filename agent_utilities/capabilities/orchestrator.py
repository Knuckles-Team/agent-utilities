"""Capability Orchestrator.

Satisfies ``OrchestratorProtocol`` via structural typing.
"""

from typing import Any

from ..protocols.capability import CapabilityContext, CapabilityHandlerProtocol


class CapabilityOrchestrator:
    """Dispatches tasks to registered capability handlers.

    Satisfies ``OrchestratorProtocol`` via structural typing.
    """

    def __init__(self):
        self.capabilities: list[CapabilityHandlerProtocol] = []

    def register(self, cap: CapabilityHandlerProtocol):
        self.capabilities.append(cap)

    async def dispatch_capabilities(self, context: CapabilityContext) -> dict[str, Any]:
        """Execute all matching capability handlers for a context."""
        results = {}
        for cap in self.capabilities:
            if cap.can_handle(context):
                results[cap.capability_name] = await cap.execute(context)
        return results

    # ── OrchestratorProtocol conformance ──────────────────────────

    async def dispatch(self, task: str, **kwargs: Any) -> dict[str, Any]:
        """Unified dispatch entry point (OrchestratorProtocol).

        Routes through capability handlers matching the task context.

        Args:
            task: Task description.
            **kwargs: Additional context data.

        Returns:
            Dict of capability_name → result mappings.
        """
        context = CapabilityContext(
            trigger_data={"task": task, **kwargs},
            state={},
            metadata={},
        )
        results = await self.dispatch_capabilities(context)
        return {
            "job_id": "capability_dispatch",
            "status": "complete",
            "results": results,
        }

    def get_status(self, job_id: str) -> dict[str, Any]:
        """Capability dispatch is stateless (OrchestratorProtocol)."""
        return {"job_id": job_id, "status": "stateless"}
