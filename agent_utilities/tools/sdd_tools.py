#!/usr/bin/python
"""SDD (Spec-Driven Development) Toolset.

This module provides Pydantic AI tool wrappers for the SDDManager, enabling agents
to natively interact with structured SDD artifacts (Constitutions, Specs, Tasks).
"""

from typing import Any

from pydantic_ai import RunContext

from ..models import (
    AgentDeps,
    ImplementationPlan,
    ProjectConstitution,
    Spec,
    Tasks,
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


async def save_spec(ctx: RunContext[AgentDeps], spec: Spec, feature_id: str) -> str:
    """Save a feature specification.

    Decomposes user intent into formal requirements, user stories, and success criteria.
    """
    manager = SDDManager(ctx.deps.workspace_path)
    path = manager.save(spec, feature_id=feature_id)
    return f"Spec '{feature_id}' saved to {path}"


async def load_spec(ctx: RunContext[AgentDeps], feature_id: str) -> Spec | None:
    """Load a feature specification by ID."""
    manager = SDDManager(ctx.deps.workspace_path)
    return manager.load(Spec, feature_id=feature_id)


async def save_tasks(ctx: RunContext[AgentDeps], tasks: Tasks, feature_id: str) -> str:
    """Save the executable tasks for a feature.

    This includes tasks, dependencies, and file-path affinity for parallel execution.
    """
    manager = SDDManager(ctx.deps.workspace_path)
    path = manager.save(tasks, feature_id=feature_id)
    return f"Tasks for '{feature_id}' saved to {path}"


async def load_tasks(ctx: RunContext[AgentDeps], feature_id: str) -> Tasks | None:
    """Load the tasks for a feature."""
    manager = SDDManager(ctx.deps.workspace_path)
    return manager.load(Tasks, feature_id=feature_id)


async def save_implementation_plan(
    ctx: RunContext[AgentDeps], plan: ImplementationPlan, feature_id: str
) -> str:
    """Save a technical implementation plan.

    Describes the architectural approach, trade-offs, and risks for a feature.
    """
    manager = SDDManager(ctx.deps.workspace_path)
    path = manager.save(plan, feature_id=feature_id)
    return f"Implementation plan '{feature_id}' saved to {path}"


async def load_implementation_plan(
    ctx: RunContext[AgentDeps], feature_id: str
) -> ImplementationPlan | None:
    """Load an implementation plan by ID."""
    manager = SDDManager(ctx.deps.workspace_path)
    return manager.load(ImplementationPlan, feature_id=feature_id)


async def get_sdd_parallel_batches(
    ctx: RunContext[AgentDeps], feature_id: str
) -> list[list[str]]:
    """Analyze the current task list and return batches of tasks that can run in parallel.

    Uses dependency analysis and file collision detection.
    """
    manager = SDDManager(ctx.deps.workspace_path)
    task_list = manager.load(Tasks, feature_id=feature_id)
    if not task_list:
        return []
    return manager.get_parallel_opportunities(task_list)


async def setup_sdd(ctx: RunContext[AgentDeps]) -> str:
    """Initialize the SDD (Spec-Driven Development) structure in the workspace.

    Creates the .specify/ directory and subdirectories for specs, plans, and tasks.
    Provides full spec-kit parity for project initialization.
    """
    manager = SDDManager(ctx.deps.workspace_path)
    # SDDManager.save handles directory creation lazily, but we can pre-create them
    dirs = ["specs"]
    for d in dirs:
        (manager.specify_dir / d).mkdir(parents=True, exist_ok=True)

    return f"SDD project structure initialized at {manager.specify_dir}"


async def export_sdd_to_markdown(
    ctx: RunContext[AgentDeps], feature_id: str, artifact_type: str = "tasks"
) -> str:
    """Export a structured SDD artifact to a human-readable Markdown file.

    artifact_type: 'spec' or 'tasks'.
    Maintains spec-kit parity by mirroring state in the workspace root.
    (Note: artifacts are now natively stored as Markdown in .specify/)
    """
    manager = SDDManager(ctx.deps.workspace_path)
    model_type: Any = Spec if artifact_type == "spec" else Tasks
    path = manager._get_path(model_type, feature_id)
    return f"Artifact natively available as Markdown at {path}"


async def import_sdd_from_markdown(
    ctx: RunContext[AgentDeps], feature_id: str, markdown_path: str
) -> str:
    """Import an SDD task list from a Markdown file.

    Maintains spec-kit parity by parsing [P] markers and structured metadata.
    """
    manager = SDDManager(ctx.deps.workspace_path)
    tasks = manager.import_from_markdown(markdown_path, feature_id)
    manager.save(tasks, feature_id=feature_id)
    return f"Imported tasks for '{feature_id}' from {markdown_path} into structured storage."


sdd_tools = [
    save_constitution,
    load_constitution,
    save_spec,
    load_spec,
    save_implementation_plan,
    load_implementation_plan,
    save_tasks,
    load_tasks,
    get_sdd_parallel_batches,
    setup_sdd,
    export_sdd_to_markdown,
    import_sdd_from_markdown,
]
