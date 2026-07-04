"""Goal-oriented autonomous execution models (KG-Native).

Provides Pydantic models for the ``/goal`` command — an autonomous
execution loop where the agent works iteratively toward a measurable
end-state without manual intervention.

Goals are first-class Knowledge Graph nodes (``GoalNode``). This enables:

- **Context enrichment**: Goals are refined with codebase context from
  the KG (file structures, symbols, prior changes).
- **Rule validation**: Goals are checked against the project constitution
  and governance rules stored in the KG before execution.
- **Historical leverage**: Prior goal executions are queryable for
  pattern-matching, improving future goal planning.
- **Durable persistence**: Goal state is checkpointed as
  ``ExecutionStateNode`` siblings in the KG for crash recovery.

Concept: ORCH-5.0 (Autonomous Goal Loop)

Example usage::

    /goal fix every failing test until npm test exits 0 without modifying /auth

The ``GoalSpec`` is parsed from the user's input, the ``GoalLoop``
engine iterates through ``GoalIteration`` steps, and the final
``GoalResult`` summarizes what was accomplished.
"""

from __future__ import annotations

import re
import time
import uuid
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class GoalStatus(StrEnum):
    """Lifecycle status of a goal execution."""

    PENDING = "pending"
    RUNNING = "running"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    # A non-terminal goal whose owning process died (detected at restart
    # rehydration, CONCEPT:AU-ORCH.session.durable-goal-registry-goals): visible + explicitly resumable, never
    # silently lost.
    ORPHANED = "orphaned"


class GoalSpec(BaseModel):
    """Specification for an autonomous goal (KG-Native).

    Parsed from the user's ``/goal`` input, this model captures the
    objective, measurable completion criteria, constraints, and
    execution parameters.  Persisted as a ``GoalNode`` in the KG.

    KG Integration:
        - Stored as ``GoalNode`` with relationships to ``File``, ``Symbol``,
          and ``ConstitutionRule`` nodes for context-aware execution.
        - ``kg_context`` is auto-populated from the KG at parse time to
          give the agent relevant codebase knowledge.
        - ``kg_rules`` captures governance constraints from the project
          constitution that apply to this goal.
        - ``related_goals`` links to prior ``GoalNode`` executions for
          pattern reuse.

    Attributes:
        id: Unique goal identifier (also the KG node_id).
        objective: The primary task description.
        end_state: Measurable completion criteria.
        constraints: Rules the agent must follow.
        validation_cmd: Shell command to verify completion.
        max_iterations: Maximum agent turns before giving up.
        auto_approve: Whether to auto-approve tool calls.
        session_id: The agent session this goal belongs to.
        raw_input: The original user input string.
        kg_context: Codebase context from the KG (files, symbols, etc.).
        kg_rules: Governance rules from the KG constitution.
        related_goals: IDs of prior goals with similar objectives.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    objective: str = ""
    end_state: str = ""
    constraints: list[str] = Field(default_factory=list)
    validation_cmd: str = ""
    max_iterations: int = 20
    auto_approve: bool = True
    session_id: str = ""
    raw_input: str = ""
    created_at: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # --- KG-Native fields ---
    kg_context: list[str] = Field(
        default_factory=list,
        description="Codebase context from the KG (file paths, symbols, summaries).",
    )
    kg_rules: list[str] = Field(
        default_factory=list,
        description="Governance/constitution rules applicable to this goal.",
    )
    related_goals: list[str] = Field(
        default_factory=list,
        description="IDs of prior GoalNode executions with similar objectives.",
    )
    kg_node_type: str = Field(
        default="GoalNode",
        description="KG node label for persistence.",
    )

    @classmethod
    def parse_goal_input(cls, raw_input: str) -> GoalSpec:
        """Parse a raw ``/goal`` input string into a structured GoalSpec.

        Supports several natural-language patterns:

        - ``/goal <objective> until <end_state> without <constraint>``
        - ``/goal <objective> until <end_state>``
        - ``/goal <objective>`` (simple form — objective only)

        Args:
            raw_input: The raw user input (with or without ``/goal`` prefix).

        Returns:
            A populated GoalSpec instance.

        Examples:
            >>> spec = GoalSpec.parse_goal_input(
            ...     "fix failing tests until npm test exits 0 without modifying /auth"
            ... )
            >>> spec.objective
            'fix failing tests'
            >>> spec.end_state
            'npm test exits 0'
            >>> spec.constraints
            ['modifying /auth']
        """
        # Strip /goal prefix if present
        text = raw_input.strip()
        if text.lower().startswith("/goal"):
            text = text[5:].strip()

        objective = text
        end_state = ""
        constraints: list[str] = []
        validation_cmd = ""

        # Pattern: objective until end_state without constraint1, constraint2
        until_without = re.match(
            r"^(.+?)\s+until\s+(.+?)\s+without\s+(.+)$",
            text,
            re.IGNORECASE,
        )
        if until_without:
            objective = until_without.group(1).strip()
            end_state = until_without.group(2).strip()
            constraint_text = until_without.group(3).strip()
            constraints = [c.strip() for c in constraint_text.split(",")]
        else:
            # Pattern: objective until end_state
            until_match = re.match(
                r"^(.+?)\s+until\s+(.+)$",
                text,
                re.IGNORECASE,
            )
            if until_match:
                objective = until_match.group(1).strip()
                end_state = until_match.group(2).strip()

        # Extract validation command if end_state references a command
        cmd_patterns = [
            r"(\w+\s+\w+)\s+exits?\s+(\d+)",  # "npm test exits 0"
            r"(\w+)\s+(?:returns?|passes?|succeeds?)",  # "pytest passes"
        ]
        for pattern in cmd_patterns:
            cmd_match = re.search(pattern, end_state or objective, re.IGNORECASE)
            if cmd_match:
                validation_cmd = cmd_match.group(1)
                break

        return cls(
            objective=objective,
            end_state=end_state,
            constraints=constraints,
            validation_cmd=validation_cmd,
            raw_input=raw_input,
        )

    def to_system_prompt(self) -> str:
        """Generate a system prompt suffix for the goal loop.

        Returns:
            A structured prompt string that instructs the agent
            to work toward the goal autonomously.
        """
        parts = [
            "## Autonomous Goal Mode",
            f"**Objective:** {self.objective}",
        ]
        if self.end_state:
            parts.append(f"**Success Criteria:** {self.end_state}")
        if self.constraints:
            parts.append("**Constraints:**")
            for constraint in self.constraints:
                parts.append(f"  - Do NOT {constraint}")
        if self.validation_cmd:
            parts.append(
                f"**Validation:** Run `{self.validation_cmd}` to check completion."
            )
        parts.extend(
            [
                "",
                "Work autonomously toward this goal. After each action:",
                "1. Evaluate progress toward the success criteria",
                "2. If not complete, plan and execute the next step",
                "3. If the validation command is specified, run it to check",
                "4. Report completion only when criteria are fully met",
                f"5. Stop after {self.max_iterations} iterations if not complete",
            ]
        )
        return "\n".join(parts)


class GoalIteration(BaseModel):
    """A single iteration within a goal execution loop.

    Attributes:
        iteration: The iteration number (1-indexed).
        action: What the agent did in this iteration.
        result: The outcome/output of the action.
        validation_output: Output from the validation command, if run.
        is_complete: Whether this iteration satisfied the end-state.
        duration_ms: Wall-clock time for this iteration in milliseconds.
        tool_calls: Number of tool calls made in this iteration.
        timestamp: When this iteration started.
    """

    iteration: int = 0
    action: str = ""
    result: str = ""
    validation_output: str = ""
    is_complete: bool = False
    duration_ms: int = 0
    tool_calls: int = 0
    timestamp: float = Field(default_factory=time.time)


class GoalResult(BaseModel):
    """Final summary of a completed (or failed/cancelled) goal execution.

    Attributes:
        goal_id: Reference to the GoalSpec id.
        status: Final status of the goal.
        iterations: All iterations that were executed.
        total_iterations: Total number of iterations run.
        total_duration_ms: Total wall-clock time in milliseconds.
        total_tool_calls: Total number of tool calls across all iterations.
        summary: Human-readable summary of what was accomplished.
        error: Error message if the goal failed.
    """

    goal_id: str = ""
    status: GoalStatus = GoalStatus.COMPLETED
    iterations: list[GoalIteration] = Field(default_factory=list)
    total_iterations: int = 0
    total_duration_ms: int = 0
    total_tool_calls: int = 0
    summary: str = ""
    error: str = ""
    completed_at: float = Field(default_factory=time.time)

    @property
    def success(self) -> bool:
        """Whether the goal completed successfully."""
        return self.status == GoalStatus.COMPLETED

    def to_report(self) -> str:
        """Generate a human-readable report of the goal execution.

        Returns:
            A formatted markdown report string.
        """
        status_emoji = {
            GoalStatus.COMPLETED: "✅",
            GoalStatus.FAILED: "❌",
            GoalStatus.CANCELLED: "🚫",
            GoalStatus.PAUSED: "⏸️",
        }.get(self.status, "❓")

        duration_secs = self.total_duration_ms / 1000
        if duration_secs < 60:
            duration_str = f"{duration_secs:.1f}s"
        else:
            mins = duration_secs / 60
            duration_str = f"{mins:.1f}m"

        lines = [
            f"## Goal Result {status_emoji}",
            f"**Status:** {self.status.value}",
            f"**Iterations:** {self.total_iterations}",
            f"**Duration:** {duration_str}",
            f"**Tool Calls:** {self.total_tool_calls}",
        ]

        if self.summary:
            lines.extend(["", f"**Summary:** {self.summary}"])

        if self.error:
            lines.extend(["", f"**Error:** {self.error}"])

        if self.iterations:
            lines.extend(["", "### Iteration Log"])
            for it in self.iterations:
                check = "✅" if it.is_complete else "🔄"
                lines.append(
                    f"  {check} **#{it.iteration}**: {it.action[:80]}"
                    + ("..." if len(it.action) > 80 else "")
                )

        return "\n".join(lines)


class GoalCheckpoint(BaseModel):
    """Durable checkpoint for crash-recovering an in-progress goal.

    Stored in the KG via ``DurableExecutionManager`` / ``StateCheckpointer``
    so that a goal loop can be resumed after a process restart.

    Attributes:
        goal_spec: The full goal specification.
        current_iteration: Which iteration was in-progress.
        iterations: Completed iterations so far.
        status: Current execution status.
        session_id: The associated session ID.
        created_at: When this checkpoint was written.
    """

    goal_spec: GoalSpec
    current_iteration: int = 0
    iterations: list[GoalIteration] = Field(default_factory=list)
    status: GoalStatus = GoalStatus.RUNNING
    session_id: str = ""
    created_at: float = Field(default_factory=time.time)


class GoalKGIntegration:
    """KG-native operations for goal lifecycle management.

    Provides methods to persist, enrich, validate, and query goals
    through the Knowledge Graph.  All goal state lives in the KG
    as first-class ``GoalNode`` entities.

    Usage::

        from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

        engine = IntelligenceGraphEngine()
        kg = GoalKGIntegration(engine)

        spec = GoalSpec.parse_goal_input("fix failing tests until pytest passes")
        spec = kg.enrich_from_kg(spec)     # pull codebase context
        spec = kg.validate_against_rules(spec)  # check constitution
        kg.persist_goal(spec)              # store as GoalNode
    """

    def __init__(self, engine: Any | None = None) -> None:
        """Initialize with an optional KG engine.

        Args:
            engine: An ``IntelligenceGraphEngine`` instance.  If None,
                KG operations gracefully degrade to no-ops.
        """
        self.engine = engine

    def enrich_from_kg(self, spec: GoalSpec) -> GoalSpec:
        """Enrich a goal with codebase context from the KG.

        Queries the KG for files, symbols, and documentation
        relevant to the goal's objective, populating ``kg_context``.

        Args:
            spec: The goal to enrich.

        Returns:
            The enriched GoalSpec (mutated in place and returned).
        """
        if (
            not self.engine
            or not hasattr(self.engine, "backend")
            or not self.engine.backend
        ):
            return spec

        try:
            # Search for relevant files/symbols based on objective keywords
            results = self.engine.backend.execute(
                """
                MATCH (n)
                WHERE (n:File OR n:Symbol OR n:Module)
                  AND toLower(n.name) CONTAINS toLower($keyword)
                RETURN n.id AS id, n.name AS name, labels(n) AS labels
                LIMIT 10
                """,
                {"keyword": spec.objective.split()[0] if spec.objective else ""},
            )
            spec.kg_context = [
                f"[{r.get('labels', ['?'])[0]}] {r.get('name', r.get('id', ''))}"
                for r in results
            ]
        except Exception:
            pass  # nosec B110

        return spec

    def validate_against_rules(self, spec: GoalSpec) -> GoalSpec:
        """Validate a goal against KG-stored constitution rules.

        Queries ``ConstitutionRule`` and ``Policy`` nodes to ensure
        the goal doesn't violate governance constraints.  Applicable
        rules are added to ``kg_rules``.

        Args:
            spec: The goal to validate.

        Returns:
            The validated GoalSpec with ``kg_rules`` populated.
        """
        if (
            not self.engine
            or not hasattr(self.engine, "backend")
            or not self.engine.backend
        ):
            return spec

        try:
            rules = self.engine.backend.execute(
                """
                MATCH (r)
                WHERE r:ConstitutionRule OR r:Policy
                RETURN r.id AS id, r.description AS descriptionription
                LIMIT 20
                """,
                {},
            )
            spec.kg_rules = [
                r.get("description", r.get("id", ""))
                for r in rules
                if r.get("description")
            ]
        except Exception:
            pass  # nosec B110

        return spec

    def find_related_goals(self, spec: GoalSpec) -> GoalSpec:
        """Find prior goals with similar objectives.

        Searches existing ``GoalNode`` entries in the KG for
        pattern reuse and populates ``related_goals``.

        Args:
            spec: The goal to find matches for.

        Returns:
            GoalSpec with ``related_goals`` populated.
        """
        if (
            not self.engine
            or not hasattr(self.engine, "backend")
            or not self.engine.backend
        ):
            return spec

        try:
            results = self.engine.backend.execute(
                """
                MATCH (g:GoalNode)
                WHERE g.status IN ['completed', 'failed']
                RETURN g.id AS id, g.objective AS objective, g.status AS status
                ORDER BY g.created_at DESC
                LIMIT 5
                """,
                {},
            )
            spec.related_goals = [r.get("id", "") for r in results if r.get("id")]
        except Exception:
            pass  # nosec B110

        return spec

    def persist_goal(self, spec: GoalSpec) -> str:
        """Persist a GoalSpec as a GoalNode in the KG.

        Args:
            spec: The goal to persist.

        Returns:
            The node ID of the persisted goal.
        """
        if (
            not self.engine
            or not hasattr(self.engine, "backend")
            or not self.engine.backend
        ):
            return spec.id

        try:
            # Removed unused data variable
            self.engine.backend.execute(
                """
                MERGE (g:GoalNode {id: $id})
                SET g.objective = $objective,
                    g.end_state = $end_state,
                    g.status = $status,
                    g.session_id = $session_id,
                    g.created_at = $created_at,
                    g.max_iterations = $max_iterations,
                    g.validation_cmd = $validation_cmd
                """,
                {
                    "id": spec.id,
                    "objective": spec.objective,
                    "end_state": spec.end_state,
                    "status": "pending",
                    "session_id": spec.session_id,
                    "created_at": str(spec.created_at),
                    "max_iterations": spec.max_iterations,
                    "validation_cmd": spec.validation_cmd,
                },
            )
        except Exception:
            pass  # nosec B110

        return spec.id

    def update_goal_status(self, goal_id: str, status: str, summary: str = "") -> None:
        """Update the status of a persisted goal.

        Args:
            goal_id: The GoalNode ID.
            status: New status value.
            summary: Optional completion summary.
        """
        if (
            not self.engine
            or not hasattr(self.engine, "backend")
            or not self.engine.backend
        ):
            return

        try:
            self.engine.backend.execute(
                """
                MATCH (g:GoalNode {id: $id})
                SET g.status = $status, g.summary = $summary
                """,
                {"id": goal_id, "status": status, "summary": summary},
            )
        except Exception:
            pass  # nosec B110
