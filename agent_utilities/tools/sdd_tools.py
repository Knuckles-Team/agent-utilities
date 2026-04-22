"""SDD Specialist Tools.

This module provides tools for Spec-Driven Development, including
specification management, task planning, and technical approach generation.
"""

from typing import Any

from pydantic_ai import RunContext

from ..models import AgentDeps, Spec, Tasks
from ..sdd import SDDManager


async def get_project_context(ctx: RunContext[AgentDeps]) -> str:
    """Retrieve the high-level project constitution and context."""
    manager = SDDManager(ctx.deps.workspace_path)
    constitution = manager.get_constitution()
    if not constitution:
        return "No project constitution found. Use setup_sdd to initialize."
    return f"Project: {constitution.get('metadata', {}).get('project_name', 'Unknown')}\nStack: {constitution.get('tech_stack')}\nVision: {constitution.get('vision')}"


async def setup_sdd(ctx: RunContext[AgentDeps], project_name: str) -> str:
    """Initialize the SDD environment in the workspace."""
    manager = SDDManager(ctx.deps.workspace_path)
    manager.initialize(project_name)
    return f"SDD environment initialized for '{project_name}' at .specify/"


async def save_spec(ctx: RunContext[AgentDeps], feature_id: str, content: str) -> str:
    """Save a feature specification to the SDD storage."""
    manager = SDDManager(ctx.deps.workspace_path)
    # Using Pydantic Spec model
    spec = Spec(feature_id=feature_id, title=content[:100], user_stories=[])
    manager.save(spec, feature_id=feature_id)
    return f"Specification for '{feature_id}' saved successfully."


async def save_tasks(
    ctx: RunContext[AgentDeps], feature_id: str, task_list: list[str]
) -> str:
    """Save a list of tasks for a feature implementation."""
    from ..models import Task

    manager = SDDManager(ctx.deps.workspace_path)
    tasks = Tasks(
        feature_id=feature_id,
        tasks=[
            Task(id=str(i), title=t, description=t) for i, t in enumerate(task_list)
        ],
    )
    manager.save(tasks, feature_id=feature_id)
    return f"Task list for '{feature_id}' saved successfully."


async def get_sdd_status(ctx: RunContext[AgentDeps], feature_id: str) -> str:
    """Retrieve the current status of an SDD feature."""
    manager = SDDManager(ctx.deps.workspace_path)
    spec = manager.load(Spec, feature_id)
    tasks = manager.load(Tasks, feature_id)

    status = f"Feature: {feature_id}\n"
    status += f"Spec: {'Found' if spec else 'Missing'}\n"
    status += f"Tasks: {len(tasks.tasks) if tasks else 0} tasks found\n"
    return status


async def export_sdd_to_markdown(
    ctx: RunContext[AgentDeps], feature_id: str, artifact_type: str = "spec"
) -> str:
    """Export an SDD artifact to a Markdown file in the workspace root.

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


async def get_sdd_parallel_batches(
    ctx: RunContext[AgentDeps], feature_id: str
) -> list[list[str]]:
    """Identify batches of tasks that can be executed in parallel.

    Returns lists of task IDs grouped by parallel execution waves.
    """
    manager = SDDManager(ctx.deps.workspace_path)
    tasks = manager.load(Tasks, feature_id)
    if not tasks:
        return []
    # Simplified batching: tasks with [P] marker go into first batch
    parallel = [t.id for t in tasks.tasks if t.parallel]
    sequential = [t.id for t in tasks.tasks if not t.parallel]
    return [parallel, sequential] if parallel else [sequential]


async def run_tdd_cycle(
    ctx: RunContext[AgentDeps],
    feature_id: str,
    context: str = "",
) -> str:
    """Run a complete Red-Green-Refactor TDD cycle for a feature.

    Marries SDD requirements with agentic TDD patterns and KG hoarding.
    """
    from ..patterns.tdd import run_tdd_cycle as execute_tdd

    return await execute_tdd(feature_id=feature_id, deps=ctx.deps, goal=context)


sdd_tools = [
    get_project_context,
    setup_sdd,
    save_spec,
    save_tasks,
    get_sdd_status,
    import_sdd_from_markdown,
    export_sdd_to_markdown,
    get_sdd_parallel_batches,
    run_tdd_cycle,
]
