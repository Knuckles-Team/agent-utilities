"""Hierarchical Constraint Engine.

CONCEPT:AU-012 — Agentic Harness Engineering (Constraint Hierarchy)

Implements the AHE constraint hierarchy:
    tool_implementation → middleware → tool_description → prompt

If a constraint is violated at the prompt level (soft), it escalates
to enforcement at the middleware/runtime level (hard). If the agent
"forgets" a constraint, the system blocks the action automatically.

The constraint hierarchy works with the existing guardrails.py
PolicyEngine and tool_guard.py — it adds the concept of
**automatic escalation** based on observed failures.

Levels (from softest to hardest):
    1. PROMPT: Constraint is included in the system prompt (advisory)
    2. TOOL_DESCRIPTION: Constraint is in tool metadata (descriptive)
    3. MIDDLEWARE: Constraint is enforced via pre/post hooks (blocking)
    4. TOOL_IMPLEMENTATION: Constraint is in the code itself (hardcoded)
"""

from __future__ import annotations

import logging
import time
from enum import IntEnum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ConstraintLevel(IntEnum):
    """Enforcement levels from softest (1) to hardest (4).

    Higher levels provide stronger guarantees but less flexibility.
    Constraints escalate upward when violations are detected at
    lower levels.
    """

    PROMPT = 1  # Soft: included in system prompt
    TOOL_DESCRIPTION = 2  # Medium: in tool metadata
    MIDDLEWARE = 3  # Hard: pre/post execution hooks
    TOOL_IMPLEMENTATION = 4  # Hardest: code-level enforcement


class HierarchicalConstraint(BaseModel):
    """A constraint that escalates through enforcement levels.

    When a constraint is violated at its current level, it can be
    escalated to a higher (harder) enforcement level. The escalation
    history is tracked for the Evolve Agent to learn from.

    Attributes:
        id: Unique constraint identifier.
        description: Human-readable description of the constraint.
        current_level: The current enforcement level.
        max_level: The maximum level this constraint can escalate to.
        violation_count: Number of observed violations.
        escalation_history: Log of when/why the constraint was escalated.
        applies_to: List of tool names or patterns this constraint covers.
        condition: Optional condition expression for evaluation.
        action: What to do when violated ("block", "warn", "log").
    """

    id: str
    description: str
    current_level: ConstraintLevel = ConstraintLevel.PROMPT
    max_level: ConstraintLevel = ConstraintLevel.TOOL_IMPLEMENTATION
    violation_count: int = 0
    escalation_threshold: int = 3  # Violations before auto-escalation
    escalation_history: list[dict[str, Any]] = Field(default_factory=list)
    applies_to: list[str] = Field(default_factory=list)
    condition: str | None = None
    action: str = "block"  # block, warn, log
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConstraintViolation(BaseModel):
    """Record of a constraint violation event.

    Attributes:
        constraint_id: The violated constraint.
        tool_name: The tool call that triggered the violation.
        violation_context: What specifically was violated.
        timestamp: When the violation occurred.
        auto_blocked: Whether the system automatically blocked the action.
    """

    constraint_id: str
    tool_name: str
    violation_context: str
    timestamp: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    auto_blocked: bool = False


class ConstraintEngine:
    """Manages the constraint hierarchy with automatic escalation.

    Integrates with guardrails.py PolicyEngine and tool_guard.py
    to provide AHE-style runtime enforcement.

    The engine:
        1. Evaluates tool calls against active constraints
        2. Records violations
        3. Auto-escalates constraints after repeated violations
        4. Reports constraint state to the Evolve Agent

    Args:
        constraints: Initial set of constraints to enforce.
        knowledge_engine: Optional KG for persisting constraint state.
    """

    def __init__(
        self,
        constraints: list[HierarchicalConstraint] | None = None,
        knowledge_engine: Any = None,
    ) -> None:
        self._constraints: dict[str, HierarchicalConstraint] = {}
        self._violations: list[ConstraintViolation] = []
        self.knowledge_engine = knowledge_engine

        if constraints:
            for c in constraints:
                self._constraints[c.id] = c

    def add_constraint(self, constraint: HierarchicalConstraint) -> None:
        """Register a new constraint."""
        self._constraints[constraint.id] = constraint
        logger.info(
            f"ConstraintEngine: Added constraint '{constraint.id}' "
            f"at level {constraint.current_level.name}"
        )

    def get_constraint(self, constraint_id: str) -> HierarchicalConstraint | None:
        """Retrieve a constraint by ID."""
        return self._constraints.get(constraint_id)

    def get_all_constraints(self) -> list[HierarchicalConstraint]:
        """Return all registered constraints."""
        return list(self._constraints.values())

    def get_constraints_at_level(
        self, level: ConstraintLevel
    ) -> list[HierarchicalConstraint]:
        """Get all constraints at a specific enforcement level."""
        return [c for c in self._constraints.values() if c.current_level == level]

    def check_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
    ) -> tuple[bool, list[ConstraintViolation]]:
        """Check if a tool call violates any active constraints.

        Only checks constraints at MIDDLEWARE level or higher — lower
        levels are advisory and don't block.

        Args:
            tool_name: The name of the tool being called.
            args: Optional tool call arguments.

        Returns:
            Tuple of (allowed, violations). If allowed is False,
            the tool call should be blocked.
        """
        violations: list[ConstraintViolation] = []
        blocked = False

        for constraint in self._constraints.values():
            if not self._applies_to_tool(constraint, tool_name):
                continue

            # Only enforce at MIDDLEWARE level or higher
            if constraint.current_level >= ConstraintLevel.MIDDLEWARE:
                if self._evaluate_constraint(constraint, tool_name, args):
                    violation = ConstraintViolation(
                        constraint_id=constraint.id,
                        tool_name=tool_name,
                        violation_context=(
                            f"Tool '{tool_name}' violates constraint "
                            f"'{constraint.description}' at level "
                            f"{constraint.current_level.name}"
                        ),
                        auto_blocked=constraint.action == "block",
                    )
                    violations.append(violation)
                    self._violations.append(violation)

                    # Record violation
                    constraint.violation_count += 1
                    logger.warning(
                        f"ConstraintEngine: Violation of '{constraint.id}' "
                        f"by tool '{tool_name}' "
                        f"(count: {constraint.violation_count})"
                    )

                    if constraint.action == "block":
                        blocked = True

        return (not blocked, violations)

    def escalate_constraint(
        self, constraint_id: str, reason: str = ""
    ) -> ConstraintLevel | None:
        """Escalate a constraint to the next enforcement level.

        Args:
            constraint_id: The constraint to escalate.
            reason: Why the escalation is happening.

        Returns:
            The new enforcement level, or None if already at max.
        """
        constraint = self._constraints.get(constraint_id)
        if not constraint:
            return None

        if constraint.current_level >= constraint.max_level:
            logger.info(
                f"ConstraintEngine: Constraint '{constraint_id}' already "
                f"at max level {constraint.max_level.name}"
            )
            return constraint.current_level

        old_level = constraint.current_level
        new_level = ConstraintLevel(constraint.current_level + 1)
        constraint.current_level = new_level

        constraint.escalation_history.append(
            {
                "from_level": old_level.name,
                "to_level": new_level.name,
                "reason": reason
                or f"Auto-escalated after {constraint.violation_count} violations",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )

        logger.warning(
            f"ConstraintEngine: Escalated '{constraint_id}' from "
            f"{old_level.name} to {new_level.name}. Reason: {reason}"
        )

        return new_level

    def auto_escalate_all(self) -> list[str]:
        """Check all constraints and auto-escalate those exceeding threshold.

        Returns:
            List of constraint IDs that were escalated.
        """
        escalated: list[str] = []
        for cid, constraint in self._constraints.items():
            if (
                constraint.violation_count >= constraint.escalation_threshold
                and constraint.current_level < constraint.max_level
            ):
                self.escalate_constraint(
                    cid,
                    reason=(
                        f"Auto-escalation: {constraint.violation_count} violations "
                        f"exceeded threshold of {constraint.escalation_threshold}"
                    ),
                )
                escalated.append(cid)
                # Reset violation count after escalation
                constraint.violation_count = 0

        if escalated:
            logger.info(
                f"ConstraintEngine: Auto-escalated {len(escalated)} constraints: "
                f"{escalated}"
            )
        return escalated

    def get_prompt_constraints(self) -> str:
        """Generate constraint text for inclusion in system prompts.

        Returns constraints at PROMPT level as a formatted string
        suitable for injection into agent system prompts.
        """
        prompt_constraints = self.get_constraints_at_level(ConstraintLevel.PROMPT)
        if not prompt_constraints:
            return ""

        lines = ["### ACTIVE CONSTRAINTS (AHE)", ""]
        for c in prompt_constraints:
            lines.append(f"- **{c.id}**: {c.description}")
            if c.applies_to:
                lines.append(f"  - Applies to: {', '.join(c.applies_to)}")

        return "\n".join(lines)

    def get_violations_report(self) -> str:
        """Generate a report of recent violations for the Evolve Agent."""
        if not self._violations:
            return "No constraint violations recorded."

        lines = ["# Constraint Violations Report", ""]
        for v in self._violations[-20:]:  # Last 20 violations
            lines.append(
                f"- [{v.timestamp}] {v.constraint_id}: {v.violation_context} "
                f"(blocked: {v.auto_blocked})"
            )
        return "\n".join(lines)

    def _applies_to_tool(
        self, constraint: HierarchicalConstraint, tool_name: str
    ) -> bool:
        """Check if a constraint applies to a specific tool."""
        if not constraint.applies_to:
            return True  # Global constraint
        return any(
            pattern in tool_name or tool_name in pattern
            for pattern in constraint.applies_to
        )

    def _evaluate_constraint(
        self,
        constraint: HierarchicalConstraint,
        tool_name: str,
        args: dict[str, Any] | None,
    ) -> bool:
        """Evaluate whether a constraint is violated.

        Returns True if the constraint IS violated.
        """
        # For now, constraints with conditions are evaluated as simple
        # keyword matches. Full expression evaluation comes later.
        if constraint.condition:
            # Simple keyword-based evaluation
            condition_lower = constraint.condition.lower()
            tool_name.lower()
            args_str = str(args).lower() if args else ""

            if "not_allowed" in condition_lower:
                return True  # Always violated
            if "requires_approval" in condition_lower:
                return True  # Flag for approval

            # Check if the condition keyword appears in the args
            keywords = condition_lower.split()
            return any(kw in args_str for kw in keywords)

        return False
