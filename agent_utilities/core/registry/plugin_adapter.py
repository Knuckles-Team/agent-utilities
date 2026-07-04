#!/usr/bin/python
from __future__ import annotations

"""Plugin Registry (CONCEPT:AU-ECO.messaging.native-backend-abstraction).

Provides dynamic tool hydration at runtime. Agents can request capabilities
and the plugin registry dynamically resolves and attaches them to the Pydantic AI agent,
removing the need for hardcoded tools.
Assimilates the "hot-swappable capability extension" innovation from Pydantic AI.
"""

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class PluginRegistry:
    """Dynamic tool hydration registry."""

    _instance: PluginRegistry | None = None

    def __init__(self):
        self._plugins: dict[str, dict[str, Callable[..., Any]]] = {}

    @classmethod
    def instance(cls) -> PluginRegistry:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register_plugin(self, domain: str, name: str, func: Callable[..., Any]) -> None:
        """Register a new tool dynamically."""
        if domain not in self._plugins:
            self._plugins[domain] = {}
        self._plugins[domain][name] = func
        logger.debug(f"Registered plugin {name} for domain {domain}")

    def hydrate_agent(self, agent: Any, domains: list[str]) -> None:
        """Hydrate a Pydantic AI agent with tools from specified domains."""
        count = 0
        for domain in domains:
            if domain in self._plugins:
                for name, func in self._plugins[domain].items():
                    # Check if agent has the tool decorator method from pydantic_ai
                    if hasattr(agent, "tool"):
                        agent.tool(func)
                        count += 1
        logger.info(
            f"Hydrated agent '{getattr(agent, 'name', 'unknown')}' with {count} dynamic tools."
        )

    def get_available_domains(self) -> list[str]:
        return list(self._plugins.keys())
