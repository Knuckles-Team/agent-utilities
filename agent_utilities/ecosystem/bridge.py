"""Ecosystem Bridge (ECO-4.0 / ECO-4.2).

Provides the primary routing layer between the internal agent-utilities
graph orchestrator and the external multi-agent ecosystem (e.g. systems-manager,
tunnel-manager, genius-agent).
"""

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class EcosystemBridge:
    """Routes payloads between local agents and external ecosystem components."""

    def __init__(self):
        self._handlers: dict[str, list[Callable]] = {}
        self._agents: dict[str, Any] = {}

    def register_agent(self, agent_id: str, agent_instance: Any) -> None:
        """Register an agent with the ecosystem bridge."""
        self._agents[agent_id] = agent_instance
        logger.info(f"Registered agent {agent_id} with EcosystemBridge.")

    def register_handler(self, event_type: str, handler: Callable) -> None:
        """Register a callback for specific event types."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def dispatch(self, event_type: str, payload: dict[str, Any]) -> None:
        """Dispatch an event payload to external systems or listening handlers."""
        logger.debug(f"EcosystemBridge dispatching {event_type} with payload {payload}")
        if event_type in self._handlers:
            for handler in self._handlers[event_type]:
                try:
                    handler(payload)
                except Exception as e:
                    logger.error(f"Error handling event {event_type}: {e}")

    def send_a2a_message(
        self, target_agent: str, message: str, context: dict | None = None
    ) -> dict:
        """Send a message across the A2A network using JSON-RPC semantics."""
        logger.info(f"Sending A2A message to {target_agent}: {message}")
        if target_agent in self._agents:
            # Simulate local routing if agent is resident in the same process
            agent = self._agents[target_agent]
            if hasattr(agent, "receive_message"):
                return agent.receive_message(message, context or {})

        # In a real distributed system, this would drop down to ZeroMQ or HTTP
        # based on the routing table.
        return {"status": "dispatched", "target": target_agent, "network": "external"}
