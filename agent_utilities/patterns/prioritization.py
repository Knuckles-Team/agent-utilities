#!/usr/bin/python
from __future__ import annotations

"""Task Prioritization Engine — CONCEPT:ORCH-1.1

Multi-factor task prioritization with dynamic re-scoring, priority
inheritance from blocking tasks, and capability-based specialist assignment.

Design-pattern source: Chapter 20 — Prioritization.

OWL: :PrioritizedTask rdfs:subClassOf :Action
     :blocks owl:TransitiveProperty
See docs/pillars/architecture_c4.md §CONCEPT:ORCH-1.1
"""


import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Global default weights for composite priority scoring
DEFAULT_URGENCY_WEIGHT = 0.35
DEFAULT_IMPACT_WEIGHT = 0.30
DEFAULT_EFFORT_WEIGHT = 0.20  # Inverse — lower effort = higher priority
DEFAULT_RISK_WEIGHT = 0.15


class PriorityScore(BaseModel):
    """Multi-factor priority score for a task."""

    urgency: float = Field(default=0.5, ge=0.0, le=1.0)
    impact: float = Field(default=0.5, ge=0.0, le=1.0)
    effort: float = Field(default=0.5, ge=0.0, le=1.0)
    risk: float = Field(default=0.5, ge=0.0, le=1.0)
    composite: float = Field(default=0.0, ge=0.0, le=1.0)

    def compute_composite(
        self,
        w_urgency: float = DEFAULT_URGENCY_WEIGHT,
        w_impact: float = DEFAULT_IMPACT_WEIGHT,
        w_effort: float = DEFAULT_EFFORT_WEIGHT,
        w_risk: float = DEFAULT_RISK_WEIGHT,
    ) -> float:
        """Compute weighted composite priority score.

        Effort is inverted (lower effort = higher priority).
        """
        effort_inv = 1.0 - self.effort
        self.composite = (
            w_urgency * self.urgency
            + w_impact * self.impact
            + w_effort * effort_inv
            + w_risk * self.risk
        )
        return self.composite


class PrioritizedTask(BaseModel):
    """A task with multi-factor priority and dependency tracking."""

    id: str
    description: str
    priority: PriorityScore = Field(default_factory=PriorityScore)
    status: str = "pending"  # pending, in_progress, completed, blocked
    assigned_specialist: str | None = None
    blocking_ids: list[str] = Field(default_factory=list)
    blocked_by_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class PrioritizationEngine:
    """Dynamic task prioritization with KG-backed history.

    Parameters
    ----------
    kg_engine : optional
        If provided, tasks are persisted to the KG.
    """

    def __init__(self, kg_engine: Any = None) -> None:
        self._engine = kg_engine
        self._tasks: dict[str, PrioritizedTask] = {}

    def add_task(self, task: PrioritizedTask) -> PrioritizedTask:
        """Add a task and compute its composite priority."""
        task.priority.compute_composite()
        self._tasks[task.id] = task
        return task

    def score_task(
        self,
        task_id: str,
        urgency: float | None = None,
        impact: float | None = None,
        effort: float | None = None,
        risk: float | None = None,
    ) -> PriorityScore:
        """Update and recompute a task's priority score."""
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"Task '{task_id}' not found")
        if urgency is not None:
            task.priority.urgency = urgency
        if impact is not None:
            task.priority.impact = impact
        if effort is not None:
            task.priority.effort = effort
        if risk is not None:
            task.priority.risk = risk
        task.priority.compute_composite()
        return task.priority

    def reprioritize(self, event: str = "context_change") -> list[PrioritizedTask]:
        """Re-score and re-sort all tasks based on current state.

        Parameters
        ----------
        event : str
            Description of the triggering event for logging.

        Returns
        -------
        list[PrioritizedTask]
            Tasks sorted by composite priority (highest first).
        """
        for task in self._tasks.values():
            task.priority.compute_composite()
            # Apply priority inheritance from blocked tasks
            self._inherit_priority(task)
        sorted_tasks = sorted(
            self._tasks.values(),
            key=lambda t: t.priority.composite,
            reverse=True,
        )
        logger.debug("Reprioritized %d tasks (event=%s)", len(sorted_tasks), event)
        return sorted_tasks

    def _inherit_priority(self, task: PrioritizedTask) -> None:
        """Inherit urgency from tasks this one blocks (priority inheritance)."""
        for blocked_id in task.blocking_ids:
            blocked = self._tasks.get(blocked_id)
            if blocked and blocked.priority.urgency > task.priority.urgency:
                task.priority.urgency = min(1.0, task.priority.urgency + 0.1)
                task.priority.compute_composite()

    def get_ready_tasks(self) -> list[PrioritizedTask]:
        """Get tasks that are ready to execute (not blocked)."""
        ready = []
        for task in self._tasks.values():
            if task.status == "pending":
                # Check if all blockers are completed
                all_done = all(
                    self._tasks.get(bid, PrioritizedTask(id="", description="")).status
                    == "completed"
                    for bid in task.blocked_by_ids
                )
                if all_done:
                    ready.append(task)
        return sorted(ready, key=lambda t: t.priority.composite, reverse=True)

    def assign_specialist(
        self,
        task_id: str,
        specialist_id: str,
    ) -> PrioritizedTask:
        """Assign a task to a specialist."""
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"Task '{task_id}' not found")
        task.assigned_specialist = specialist_id
        task.status = "in_progress"
        return task

    def complete_task(self, task_id: str) -> PrioritizedTask:
        """Mark a task as completed."""
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"Task '{task_id}' not found")
        task.status = "completed"
        return task

    def get_execution_order(self) -> list[PrioritizedTask]:
        """Get optimal execution order respecting dependencies and priority."""
        completed = {t.id for t in self._tasks.values() if t.status == "completed"}
        remaining = [t for t in self._tasks.values() if t.status != "completed"]
        ordered: list[PrioritizedTask] = []
        visited: set[str] = set()

        def _can_execute(t: PrioritizedTask) -> bool:
            return all(bid in completed or bid in visited for bid in t.blocked_by_ids)

        while remaining:
            ready = [t for t in remaining if _can_execute(t)]
            if not ready:
                # Deadlock — add remaining in priority order
                ordered.extend(
                    sorted(remaining, key=lambda t: t.priority.composite, reverse=True)
                )
                break
            ready.sort(key=lambda t: t.priority.composite, reverse=True)
            for t in ready:
                ordered.append(t)
                visited.add(t.id)
                remaining.remove(t)

        return ordered
