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
        raise NotImplementedError

    def can_handle(self, context: CapabilityContext) -> bool:
        raise NotImplementedError

    async def execute(self, context: CapabilityContext) -> dict[str, Any]:
        raise NotImplementedError
