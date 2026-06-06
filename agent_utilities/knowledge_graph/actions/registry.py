#!/usr/bin/python
from __future__ import annotations

"""Ontology Action System — registry (CONCEPT:KG-2.25).

The :class:`ActionRegistry` is the discovery surface for governed verbs: it
binds an :class:`~agent_utilities.knowledge_graph.actions.models.OntologyAction`
definition to a concrete handler callable, rejects duplicate names, and supports
lookup by name or by the ontology object type an action ``acts_on``.
"""

import logging
from collections.abc import Callable
from typing import Any

from .models import OntologyAction

logger = logging.getLogger(__name__)

# A handler takes the validated params dict and returns an arbitrary result that
# the executor summarizes into the ActionInvocation record.
ActionHandler = Callable[[dict[str, Any]], Any]


class ActionRegistry:
    """Registry of governed ontology actions and their handlers. CONCEPT:KG-2.25."""

    def __init__(self) -> None:
        self._actions: dict[str, OntologyAction] = {}
        self._handlers: dict[str, ActionHandler] = {}

    def register(self, action: OntologyAction, handler: ActionHandler) -> None:
        """Register an action definition with its handler.

        Raises:
            ValueError: If an action with the same name is already registered.
        """
        if action.name in self._actions:
            raise ValueError(f"Action already registered: {action.name!r}")
        if not callable(handler):
            raise TypeError(f"handler for {action.name!r} must be callable")
        self._actions[action.name] = action
        self._handlers[action.name] = handler
        logger.debug(
            "Registered ontology action: %s (verb=%s, cap=%s, effect=%s)",
            action.name,
            action.verb,
            action.required_capability,
            action.produces_effect,
        )

    def get(self, name: str) -> OntologyAction | None:
        """Return the action definition for ``name``, or ``None``."""
        return self._actions.get(name)

    def get_handler(self, name: str) -> ActionHandler | None:
        """Return the handler callable for ``name``, or ``None``."""
        return self._handlers.get(name)

    def list_actions(self) -> list[OntologyAction]:
        """Return all registered action definitions."""
        return list(self._actions.values())

    def actions_for_type(self, object_type: str) -> list[OntologyAction]:
        """Return all actions whose ``acts_on`` includes ``object_type``.

        Case-insensitive on the ontology object type.
        """
        ot = object_type.lower()
        return [
            a for a in self._actions.values() if any(t.lower() == ot for t in a.acts_on)
        ]

    def __contains__(self, name: object) -> bool:
        return name in self._actions

    def __len__(self) -> int:
        return len(self._actions)
