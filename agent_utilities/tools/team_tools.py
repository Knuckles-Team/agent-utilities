#!/usr/bin/python
"""Agent team coordination tools.

Exposes team management, task assignment, and P2P messaging to agents,
backed by the TeamCapability and ACP.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai import RunContext

from ..capabilities.teams import TeamCapability


async def spawn_team(
    ctx: RunContext[Any], team_name: str, member_ids: list[str]
) -> str:
    """Create a new agent team with the specified members.

    Members should be IDs of available agents or spawned agents.
    """
    capability = getattr(ctx, "team_capability", None)
    if not capability:
        # Auto-initialize if missing
        capability = TeamCapability()
        ctx.team_capability = capability

    team_id = await capability.create_team(ctx, team_name, member_ids)
    return f"Team '{team_name}' created with ID: {team_id}. Members: {', '.join(member_ids)}"


async def assign_team_task(
    ctx: RunContext[Any], content: str, assigned_to: str | None = None
) -> str:
    """Assign a task to the team or a specific member.

    The task is persisted in the knowledge graph for shared visibility.
    """
    capability = getattr(ctx, "team_capability", None)
    if not capability:
        return "No active team found. Use spawn_team first."

    task_id = await capability.add_task(ctx, content, assigned_to)
    target = assigned_to if assigned_to else "the whole team"
    return f"Task assigned to {target}. Task ID: {task_id}"


async def message_teammate(ctx: RunContext[Any], member_id: str, message: str) -> str:
    """Send a direct message to a team member via ACP.

    Used for P2P coordination and sharing intermediate results.
    """
    capability = getattr(ctx, "team_capability", None)
    if not capability:
        return "No active team found."

    success = await capability.message_member(ctx, member_id, message)
    if success:
        return f"Message sent to {member_id} via ACP message bus."
    return f"Failed to send message to {member_id}. Ensure ACP session is active."


async def list_team_tasks(ctx: RunContext[Any]) -> str:
    """List all tasks associated with the current team from the knowledge graph."""
    engine = getattr(ctx.deps, "graph_engine", None)
    if not engine:
        return "Knowledge graph not available."

    tasks = []
    for node_id, node_data in engine.graph.nodes(data=True):
        if node_data.get("type") == "task":
            status = node_data.get("status", "pending")
            assigned = node_data.get("assigned_to", "unassigned")
            tasks.append(
                f"- [{status}] {node_data.get('content')} (ID: {node_id}, Assigned: {assigned})"
            )

    if not tasks:
        return "No tasks found for the current team."
    return "Team Tasks:\n" + "\n".join(tasks)


async def discover_teams(ctx: RunContext[Any]) -> str:
    """Discover all active teams from the knowledge graph.

    Returns a formatted list of teams with their IDs, names, and member counts.
    """
    capability = getattr(ctx, "team_capability", None)
    if not capability:
        capability = TeamCapability()

    teams = await capability.discover_teams(ctx)
    if not teams:
        return "No active teams found."

    lines = ["Active Teams:"]
    for t in teams:
        lines.append(
            f"- {t['name']} (ID: {t['team_id']}, Members: {t['member_count']})"
        )
    return "\n".join(lines)


async def update_task_status(ctx: RunContext[Any], task_id: str, status: str) -> str:
    """Update the status of a team task.

    Args:
        ctx: Run context.
        task_id: The ID of the task to update.
        status: New status ('pending', 'in_progress', 'done').
    """
    capability = getattr(ctx, "team_capability", None)
    if not capability:
        return "No active team found. Use spawn_team first."

    success = await capability.update_task_status(ctx, task_id, status)
    if success:
        return f"Task {task_id} updated to '{status}'."
    return f"Failed to update task {task_id}. Task may not exist."


TEAM_TOOLS = [
    spawn_team,
    assign_team_task,
    message_teammate,
    list_team_tasks,
    discover_teams,
    update_task_status,
]
