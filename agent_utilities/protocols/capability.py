"""Unified Capability Protocol."""

from dataclasses import dataclass
from typing import Any


@dataclass
class CapabilityContext:
    trigger_data: dict[str, Any]
    state: dict[str, Any]
    metadata: dict[str, Any]


class CapabilityHandlerProtocol:
    @property
    def capability_name(self) -> str:
        # Default name derived from class name
        return self.__class__.__name__

    def can_handle(self, context: CapabilityContext) -> bool:
        # Default implementation check based on presence of trigger data
        return bool(context.trigger_data)

    async def execute(self, context: CapabilityContext) -> dict[str, Any]:
        # Default execution returns the input state as a success status
        return {"status": "success", "state": context.state}
