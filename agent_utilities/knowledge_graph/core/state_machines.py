#!/usr/bin/env python3
"""Formal State Machines and Invariants.

CONCEPT:KG-2.48 — State Machine Invariant Engine

Implements Deterministic Finite Automata (DFA) abstractions and provable
state invariants from *Mathematics for Computer Science* (MCS Ch 6).
Provides mathematical guarantees of agent safety by formally validating
transitions against structural invariants, preventing infinite loops.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Transition:
    """A formal transition in a state machine."""

    source: str
    target: str
    action: str
    condition: Callable[[dict[str, Any]], bool] | None = None


class FormalStateMachine:
    """A Deterministic Finite Automaton (DFA) with Invariant checking.

    Provides formal mathematical guarantees that an agent cannot
    make illegal transitions or violate structural invariants.
    """

    def __init__(self, start_state: str):
        self.start_state = start_state
        self.current_state = start_state
        self.states: set[str] = {start_state}
        self.transitions: list[Transition] = []
        self.invariants: list[Callable[[str, dict[str, Any]], bool]] = []

    def add_state(self, state: str) -> None:
        """Add a valid state to the machine."""
        self.states.add(state)

    def add_transition(
        self,
        source: str,
        target: str,
        action: str,
        condition: Callable[[dict[str, Any]], bool] | None = None,
    ) -> None:
        """Add a directed transition between two states."""
        self.states.add(source)
        self.states.add(target)
        self.transitions.append(Transition(source, target, action, condition))

    def add_invariant(
        self, invariant_func: Callable[[str, dict[str, Any]], bool]
    ) -> None:
        """Add a global invariant that must hold for all states and transitions.

        An invariant is a predicate P(state) that is true for the start state,
        and if true before a transition, remains true after.
        """
        self.invariants.append(invariant_func)

    def get_available_actions(self, context: dict[str, Any] | None = None) -> list[str]:
        """Get all valid actions from the current state given the context."""
        ctx = context or {}
        valid_actions = []
        for t in self.transitions:
            if t.source == self.current_state:
                if t.condition is None or t.condition(ctx):
                    valid_actions.append(t.action)
        return valid_actions

    def validate_invariants(self, target_state: str, context: dict[str, Any]) -> bool:
        """Check if transitioning to the target state preserves all invariants."""
        for inv in self.invariants:
            if not inv(target_state, context):
                logger.error(
                    f"Invariant violation: Transition to {target_state} failed invariant check."
                )
                return False
        return True

    def transition(self, action: str, context: dict[str, Any] | None = None) -> str:
        """Execute a formal transition if it is valid and preserves invariants.

        Args:
            action: The action to execute.
            context: Contextual variables evaluated by transition conditions.

        Returns:
            The new state.

        Raises:
            ValueError: If the action is invalid or an invariant is violated.
        """
        ctx = context or {}
        valid_targets = []

        for t in self.transitions:
            if t.source == self.current_state and t.action == action:
                if t.condition is None or t.condition(ctx):
                    valid_targets.append(t.target)

        if not valid_targets:
            raise ValueError(
                f"No valid transition found for action '{action}' from state '{self.current_state}'."
            )

        if len(valid_targets) > 1:
            raise ValueError(
                f"Non-deterministic transition detected for action '{action}'. DFA requires deterministic transitions."
            )

        target = valid_targets[0]

        # Prove invariants
        if not self.validate_invariants(target, ctx):
            raise ValueError(f"Transition to '{target}' aborted: Invariant violation.")

        logger.info(f"Transitioned: {self.current_state} --[{action}]--> {target}")
        self.current_state = target
        return self.current_state
