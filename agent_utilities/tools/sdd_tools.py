#!/usr/bin/python
# coding: utf-8
"""SDD (Spec-Driven Development) Toolset.

This module provides Pydantic AI tool wrappers for the SDDManager, enabling agents
to natively interact with structured SDD artifacts (Constitutions, Specs, Tasks).
"""

from typing import List, Optional
from pydantic_ai import RunContext

from ..models import (
    AgentDeps,
    ProjectConstitution,
    FeatureSpec,
    TaskList,
)
from ..sdd import SDDManager


async def save_constitution(
    ctx: RunContext[AgentDeps], constitution: ProjectConstitution
) -> str:
    """Save the project constitution to the workspace agent_data.

    Use this once at the start of a project to establish governance rules and tech stack.
    """
    manager = SDDManager(ctx.deps.workspace_path)
    path = manager.save(constitution)
    return f"Constitution saved to {path}"


async def load_constitution(ctx: RunContext[AgentDeps]) -> ProjectConstitution:
    """Load the project constitution from the workspace."""
    manager = SDDManager(ctx.deps.workspace_path)
    constitution = manager.load(ProjectConstitution)
    if not constitution:
        return ProjectConstitution()
    return constitution


async def save_feature_spec(
    ctx: RunContext[AgentDeps], spec: FeatureSpec, feature_id: str
) -> str:
    """Save a feature specification.

    Decomposes user intent into formal requirements, user stories, and success criteria.
    """
    manager = SDDManager(ctx.deps.workspace_path)
    path = manager.save(spec, feature_id=feature_id)
    return f"Feature spec '{feature_id}' saved to {path}"


async def load_feature_spec(
    ctx: RunContext[AgentDeps], feature_id: str
) -> Optional[FeatureSpec]:
    """Load a feature specification by ID."""
    manager = SDDManager(ctx.deps.workspace_path)
    return manager.load(FeatureSpec, feature_id=feature_id)


async def save_task_list(
    ctx: RunContext[AgentDeps], task_list: TaskList, feature_id: str
) -> str:
    """Save the executable task list for a feature.

    This includes phases, tasks, dependencies, and file-path affinity for parallel execution.
    """
    manager = SDDManager(ctx.deps.workspace_path)
    path = manager.save(task_list, feature_id=feature_id)
    return f"Task list for '{feature_id}' saved to {path}"


async def load_task_list(
    ctx: RunContext[AgentDeps], feature_id: str
) -> Optional[TaskList]:
    """Load the task list for a feature."""
    manager = SDDManager(ctx.deps.workspace_path)
    return manager.load(TaskList, feature_id=feature_id)


async def get_sdd_parallel_batches(
    ctx: RunContext[AgentDeps], feature_id: str
) -> List[List[str]]:
    """Analyze the current task list and return batches of tasks that can run in parallel.

    Uses dependency analysis and file collision detection.
    """
    manager = SDDManager(ctx.deps.workspace_path)
    task_list = manager.load(TaskList, feature_id=feature_id)
    if not task_list:
        return []
    return manager.get_parallel_opportunities(task_list)


sdd_tools = [
    save_constitution,
    load_constitution,
    save_feature_spec,
    load_feature_spec,
    save_task_list,
    load_task_list,
    get_sdd_parallel_batches,
]
