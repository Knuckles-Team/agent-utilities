import logging
from typing import Any
from pydantic_ai import RunContext
from ..models import CronRegistryModel
from ..agent_utilities import (
    list_scheduled_tasks as list_scheduled_tasks_util,
    delete_scheduled_task as delete_scheduled_task_util,
    schedule_task as schedule_task_util,
)

logger = logging.getLogger(__name__)


async def schedule_task(
    ctx: RunContext[Any],
    task_id: str,
    name: str,
    interval_minutes: int,
    prompt: str,
) -> str:
    """Schedule a task to run periodically (persists in CRON.md)."""
    return schedule_task_util(task_id, name, interval_minutes, prompt)


async def list_tasks(ctx: RunContext[Any]) -> CronRegistryModel:
    """List all active periodic tasks."""
    return list_scheduled_tasks_util()


async def delete_task(ctx: RunContext[Any], task_id: str) -> str:
    """Permanently remove a scheduled task by ID."""
    return delete_scheduled_task_util(task_id)


# New: View Cron Log (Code Puppy Port)
async def view_cron_log(ctx: RunContext[Any], lines: int = 50) -> str:
    """View the recent execution logs for scheduled tasks."""
    from ..workspace import CORE_FILES
    from ..agent_utilities import read_md_file

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
