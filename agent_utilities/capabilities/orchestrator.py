"""Capability Orchestrator."""
from typing import Any
from ..protocols.capability import CapabilityContext, CapabilityHandlerProtocol

class CapabilityOrchestrator:
    def __init__(self):
        self.capabilities: list[CapabilityHandlerProtocol] = []

    def register(self, cap: CapabilityHandlerProtocol):
        self.capabilities.append(cap)

    async def dispatch(self, context: CapabilityContext) -> dict[str, Any]:
        results = {}
        for cap in self.capabilities:
            if cap.can_handle(context):
                results[cap.capability_name] = await cap.execute(context)
        return results
