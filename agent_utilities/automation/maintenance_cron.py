#!/usr/bin/python
from __future__ import annotations

"""CONCEPT:OS-5.2 — Autonomous Maintenance Cron.

Provides scheduled autonomous maintenance capabilities that leverage
the existing graph infrastructure for codebase health monitoring.

Architecture:
    - **MaintenanceCron**: Defines scheduled tasks with configurable
      frequency, token budget (0 = unlimited), and priority.
    - **MaintenanceTask**: Individual task definitions with queries
      for graph execution.
    - Integrates with the ``CognitiveScheduler`` (CONCEPT:OS-5.2) for priority
      management and the ``ResourceOptimizer`` (CONCEPT:OS-5.2) for budget caps.

Task Categories:
    - **Nightly**: Pre-commit analysis, stale import detection,
      security advisories, documentation drift
    - **Hourly**: MCP server health checks via circuit breaker stats
    - **On-demand**: Documentation sync, dependency audit

Integrates with:
    - CONCEPT:OS-5.2 (CognitiveScheduler): Priority-aware scheduling
    - CONCEPT:OS-5.2 (ResourceOptimizer): Token budget enforcement
    - CONCEPT:OS-5.2 (AgentRegistry): Package health monitoring
    - CONCEPT:OS-5.0 (FileWatcher): Can be triggered by file changes

See docs/maintenance-cron.md §CONCEPT:OS-5.2.
"""


import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Budget: 0 = unlimited (per user request)
DEFAULT_TOKEN_BUDGET = int(os.getenv("MAINTENANCE_TOKEN_BUDGET", "0"))
DEFAULT_PRIORITY = os.getenv("MAINTENANCE_PRIORITY", "LOW")


class MaintenanceFrequency(str, Enum):  # noqa: UP042
    """How often a maintenance task should run."""

    HOURLY = "hourly"
    SIX_HOURLY = "six_hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    ON_DEMAND = "on_demand"


class MaintenanceTask(BaseModel):
    """Definition of an autonomous maintenance task.

    Attributes:
        id: Unique task identifier.
        name: Human-readable task name.
        query: The graph query to execute.
        frequency: How often to run.
        priority: Execution priority (LOW/MEDIUM/HIGH).
        token_budget: Max tokens for this task (0 = unlimited).
        enabled: Whether this task is currently active.
        last_run: Epoch timestamp of last execution.
        last_status: Result of last execution.
        tags: Searchable tags.
    """

    id: str
    name: str
    query: str
    frequency: MaintenanceFrequency = MaintenanceFrequency.DAILY
    priority: str = DEFAULT_PRIORITY
    token_budget: int = 0
    enabled: bool = True
    last_run: float = 0.0
    last_status: str = "never_run"
    tags: list[str] = Field(default_factory=list)


# Default maintenance tasks
DEFAULT_TASKS: list[MaintenanceTask] = [
    MaintenanceTask(
        id="precommit_analysis",
        name="Pre-commit Analysis",
        query=(
            "Run pre-commit hooks on the entire codebase and report "
            "any new lint violations, security findings, or formatting issues. "
            "Focus on actionable findings only."
        ),
        frequency=MaintenanceFrequency.DAILY,
        priority="LOW",
        tags=["lint", "quality"],
    ),
    MaintenanceTask(
        id="dependency_audit",
        name="Dependency Audit",
        query=(
            "Audit all project dependencies for: "
            "1) Known CVEs using pip-audit or equivalent, "
            "2) Outdated packages with available updates, "
            "3) Unused dependencies that can be removed. "
            "Produce a summary table with severity ratings."
        ),
        frequency=MaintenanceFrequency.DAILY,
        priority="LOW",
        tags=["security", "dependencies"],
    ),
    MaintenanceTask(
        id="mcp_health_check",
        name="MCP Server Health Check",
        query=(
            "Check the health of all registered MCP servers. "
            "Report any servers that are unreachable, have high "
            "error rates, or show circuit breaker activity."
        ),
        frequency=MaintenanceFrequency.HOURLY,
        priority="MEDIUM",
        tags=["mcp", "health"],
    ),
    MaintenanceTask(
        id="stale_import_detection",
        name="Stale Import Detection",
        query=(
            "Scan the codebase for unused imports, deprecated function "
            "calls, and stale references. Report findings grouped by "
            "severity (error vs. warning)."
        ),
        frequency=MaintenanceFrequency.WEEKLY,
        priority="LOW",
        tags=["quality", "cleanup"],
    ),
    MaintenanceTask(
        id="documentation_drift",
        name="Documentation Drift Check",
        query=(
            "Compare the current codebase structure against the docs/ "
            "directory. Identify: 1) New modules without documentation, "
            "2) Documentation referencing deleted code, "
            "3) README sections that are outdated."
        ),
        frequency=MaintenanceFrequency.WEEKLY,
        priority="LOW",
        tags=["docs", "quality"],
    ),
    MaintenanceTask(
        id="scholarx_paper_discovery",
        name="ScholarX Paper Discovery",
        query=(
            "Use scholarx tools to check for new papers published in the "
            "last 6 hours across cs.AI, cs.MA, cs.SE, cs.LG, cs.CL, "
            "q-bio, and related categories. Check the Knowledge Graph for "
            "any custom research watchlists. For each new paper: "
            "1) Summarize the key contribution, "
            "2) Assess relevance to existing KG concepts, "
            "3) If relevance_score > 0.6, ingest into the Knowledge Graph, "
            "4) Produce a digest of actionable findings."
        ),
        frequency=MaintenanceFrequency.SIX_HOURLY,
        priority="MEDIUM",
        tags=["research", "scholarx", "innovation"],
    ),
    MaintenanceTask(
        id="topological_community_partitioning",
        name="Topological Community Partitioning",
        query=(
            "Execute detect_communities to find emergent topological clusters "
            "in the Knowledge Graph and persist stable ones as COMMUNITY nodes."
        ),
        frequency=MaintenanceFrequency.DAILY,
        priority="LOW",
        tags=["knowledge_graph", "topology"],
    ),
]


@dataclass
class MaintenanceCron:
    """CONCEPT:OS-5.2 — Autonomous maintenance scheduler.

    Manages a registry of maintenance tasks and determines which
    ones are due for execution based on frequency and last-run time.

    Args:
        tasks: List of ``MaintenanceTask`` definitions.
        token_budget: Global token budget for all maintenance (0 = unlimited).
        priority: Default priority for maintenance tasks.
        tokens_used: Running total of tokens consumed.
    """

    tasks: list[MaintenanceTask] = field(
        default_factory=lambda: [t.model_copy() for t in DEFAULT_TASKS]
    )
    token_budget: int = DEFAULT_TOKEN_BUDGET
    priority: str = DEFAULT_PRIORITY
    tokens_used: int = 0

    # Frequency → minimum interval in seconds
    _INTERVALS: dict[str, int] = field(
        default_factory=lambda: {
            "hourly": 3600,
            "six_hourly": 21600,
            "daily": 86400,
            "weekly": 604800,
            "on_demand": 0,
        },
        repr=False,
    )

    def get_due_tasks(self) -> list[MaintenanceTask]:
        """Return all tasks that are due for execution.

        A task is due when:
        1. It is enabled
        2. Its frequency is not ``on_demand``
        3. Sufficient time has elapsed since last_run

        Returns:
            List of ``MaintenanceTask`` instances that should be executed.
        """
        now = time.time()
        due: list[MaintenanceTask] = []

        for task in self.tasks:
            if not task.enabled:
                continue
            if task.frequency == MaintenanceFrequency.ON_DEMAND:
                continue

            interval = self._INTERVALS.get(task.frequency.value, 86400)
            if now - task.last_run >= interval:
                due.append(task)

        # Sort by priority (HIGH first)
        priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        due.sort(key=lambda t: priority_order.get(t.priority, 2))

        return due

    def is_budget_available(self) -> bool:
        """Check if the maintenance token budget has remaining capacity.

        Returns:
            True if budget is unlimited (0) or tokens remain.
        """
        if self.token_budget == 0:
            return True  # Unlimited
        return self.tokens_used < self.token_budget

    def record_execution(
        self,
        task_id: str,
        status: str = "success",
        tokens_used: int = 0,
    ) -> None:
        """Record the execution of a maintenance task.

        Args:
            task_id: ID of the task that was executed.
            status: Result status (``success``, ``failure``, ``skipped``).
            tokens_used: Tokens consumed during execution.
        """
        for task in self.tasks:
            if task.id == task_id:
                task.last_run = time.time()
                task.last_status = status
                break

        self.tokens_used += tokens_used
        logger.info(
            "[CONCEPT:OS-5.2] Maintenance task '%s' completed: %s (tokens: %d, total: %d/%s)",
            task_id,
            status,
            tokens_used,
            self.tokens_used,
            self.token_budget if self.token_budget > 0 else "unlimited",
        )

    def add_task(self, task: MaintenanceTask) -> None:
        """Add a custom maintenance task.

        Args:
            task: The ``MaintenanceTask`` to add.
        """
        # Prevent duplicates
        existing_ids = {t.id for t in self.tasks}
        if task.id in existing_ids:
            logger.warning(
                f"[CONCEPT:OS-5.2] Task '{task.id}' already exists — skipping"
            )
            return
        self.tasks.append(task)

    def remove_task(self, task_id: str) -> bool:
        """Remove a maintenance task by ID.

        Args:
            task_id: ID of the task to remove.

        Returns:
            True if the task was found and removed.
        """
        before = len(self.tasks)
        self.tasks = [t for t in self.tasks if t.id != task_id]
        return len(self.tasks) < before

    def summary(self) -> dict[str, Any]:
        """Return a summary of the maintenance cron state.

        Returns:
            Dict with task counts, budget usage, and due tasks.
        """
        due = self.get_due_tasks()
        return {
            "total_tasks": len(self.tasks),
            "enabled_tasks": sum(1 for t in self.tasks if t.enabled),
            "due_tasks": len(due),
            "due_task_ids": [t.id for t in due],
            "tokens_used": self.tokens_used,
            "token_budget": self.token_budget if self.token_budget > 0 else "unlimited",
            "budget_available": self.is_budget_available(),
        }
