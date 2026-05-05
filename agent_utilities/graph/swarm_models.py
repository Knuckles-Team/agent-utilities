#!/usr/bin/python
"""CONCEPT:ORCH-1.0 — Swarm Orchestration Models.

Pydantic models for task decomposition, swarm hierarchy, and
swarm execution results used by :class:`SwarmOrchestrator`.

See docs/emergent-architecture.md §AU-014.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TaskTree(BaseModel):
    """Recursive task decomposition tree.

    Each node represents a subtask that may be further decomposed
    or executed directly by a specialist agent.
    """

    task: str = Field(description="The subtask description")
    subtasks: list[TaskTree] = Field(
        default_factory=list,
        description="Child subtasks (empty = leaf node, directly executable)",
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="Task descriptions this subtask depends on (must complete first)",
    )
    parallelizable: bool = Field(
        default=True,
        description="Whether this subtask can run in parallel with siblings",
    )
    assigned_agent: str | None = Field(
        default=None,
        description="ID of the specialist agent assigned to this subtask",
    )


class SwarmHierarchy(BaseModel):
    """The agent hierarchy formed by a swarm for a given task.

    Captures coordinator/specialist/aggregator roles and their
    relationships for observability and KG persistence.
    """

    coordinator: str = Field(description="Agent ID of the top-level coordinator")
    specialists: list[str] = Field(
        default_factory=list,
        description="Agent IDs of leaf-node specialists",
    )
    aggregators: list[str] = Field(
        default_factory=list,
        description="Agent IDs of intermediate aggregation agents",
    )
    edges: list[tuple[str, str, str]] = Field(
        default_factory=list,
        description="(from_id, to_id, relation) tuples",
    )


class SwarmResult(BaseModel):
    """Result of a swarm execution.

    Captures the full lifecycle of a dynamically spawned swarm:
    how many agents participated, the depth of recursion achieved,
    and the ratio of parallel vs sequential execution.
    """

    swarm_id: str = Field(description="Unique swarm coalition ID")
    agents_spawned: int = Field(default=0, description="Total agents created")
    depth_reached: int = Field(default=0, description="Max recursion depth hit")
    results: dict[str, Any] = Field(
        default_factory=dict,
        description="Task → result mapping from all subtasks",
    )
    parallelism_achieved: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Ratio of parallel vs sequential execution",
    )
    hierarchy: SwarmHierarchy | None = Field(
        default=None,
        description="The formed agent hierarchy (if available)",
    )
    error: str | None = None
