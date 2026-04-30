#!/usr/bin/python
"""Scheduler Tools Module.

This module provides tools for managing periodic agent tasks, including
scheduling new background jobs, auditing the cron registry, and
viewing execution logs.
"""

import logging
from typing import Any

from pydantic_ai import RunContext

from agent_utilities.core.scheduler import (
    delete_scheduled_task as delete_scheduled_task_util,
)
from agent_utilities.core.scheduler import (
    list_scheduled_tasks as list_scheduled_tasks_util,
)
from agent_utilities.core.scheduler import (
    schedule_task as schedule_task_util,
)

from ..models import CronRegistryModel

logger = logging.getLogger(__name__)


async def schedule_task(
    ctx: RunContext[Any],
    task_id: str,
    name: str,
    interval_minutes: int,
    prompt: str,
) -> str:
    """Schedule a recurring agent task for periodic execution.

    Args:
        ctx: The agent run context.
        task_id: A unique identifier for the scheduled job.
        name: A human-readable name for the task.
        interval_minutes: The frequency of execution in minutes.
        prompt: The instruction to be executed by the background agent.

    Returns:
        A confirmation message indicating success.

    """
    return schedule_task_util(task_id, name, interval_minutes, prompt)


async def list_tasks(ctx: RunContext[Any]) -> CronRegistryModel:
    """List all periodically scheduled tasks registered in the workspace.

    Args:
        ctx: The agent run context.

    Returns:
        A model containing the list of active background jobs.

    """
    return list_scheduled_tasks_util()


async def delete_task(ctx: RunContext[Any], task_id: str) -> str:
    """Permanently remove a scheduled task from the background processor.

    Args:
        ctx: The agent run context.
        task_id: The unique ID of the task to be deleted.

    Returns:
        A confirmation message indicating success.

    """
    return delete_scheduled_task_util(task_id)


# New: View Cron Log (Code Puppy Port)
async def view_cron_log(ctx: RunContext[Any], lines: int = 50) -> str:
    """Retrieve the recent execution history and diagnostics for scheduled tasks.

    Args:
        ctx: The agent run context.
        lines: The number of trailing log lines to retrieve.

    Returns:
        A formatted string containing the recent execution log entries.

    """
    from agent_utilities.core.workspace import CORE_FILES, read_md_file

    content = read_md_file(CORE_FILES["CRON_LOG"])
    log_lines = content.splitlines()
    return "\n".join(log_lines[-lines:])


# Tool grouping for registration
scheduler_tools = [
    schedule_task,
    list_tasks,
    delete_task,
    view_cron_log,
]
