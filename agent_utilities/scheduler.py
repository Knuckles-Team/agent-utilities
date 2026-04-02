#!/usr/bin/python

from __future__ import annotations

import logging
import asyncio


from typing import Any, List, Optional, TYPE_CHECKING
from datetime import datetime, timedelta

if TYPE_CHECKING:
    pass


from .config import DEFAULT_MAX_CRON_LOG_ENTRIES
from .workspace import (
    CORE_FILES,
    parse_cron_registry,
    serialize_cron_registry,
    parse_cron_log,
    serialize_cron_log,
    load_workspace_file,
    get_workspace_path,
    initialize_workspace,
)
from .prompt_builder import resolve_prompt
from .chat_persistence import save_chat_to_disk


from .models import PeriodicTask, CronRegistryModel, CronTaskModel, CronLogModel

tasks: List[PeriodicTask] = []
lock = asyncio.Lock()


logger = logging.getLogger(__name__)

lock = asyncio.Lock()


def get_cron_tasks_from_md() -> CronRegistryModel:
    """Parse CRON.md and return CronRegistryModel."""
    content = load_workspace_file(CORE_FILES["CRON"])
    if not content:
        return CronRegistryModel()
    return parse_cron_registry(content)


def get_cron_logs_from_md() -> CronLogModel:
    """Parse CRON_LOG.md and return CronLogModel."""
    content = load_workspace_file(CORE_FILES["CRON_LOG"])
    if not content:
        return CronLogModel()
    return parse_cron_log(content)


def update_cron_task_in_cron_md(
    task_id: str, name: str, interval_min: int, prompt: str
):
    """Update/add a task in CRON.md registry."""
    registry = get_cron_tasks_from_md()

    updated = False
    for t in registry.tasks:
        if t.id == task_id:
            t.name = name
            t.interval_minutes = interval_min
            t.prompt = prompt
            t.last_run = datetime.now().strftime("%H:%M")
            updated = True
            break

    if not updated:
        registry.tasks.append(
            CronTaskModel(
                id=task_id,
                name=name,
                interval_minutes=interval_min,
                prompt=prompt,
                last_run=datetime.now().strftime("%H:%M"),
            )
        )

    content = serialize_cron_registry(registry)
    path = get_workspace_path(CORE_FILES["CRON"])
    path.write_text(content, encoding="utf-8")


def schedule_task(task_id: str, name: str, interval_minutes: int, prompt: str) -> str:
    """Consolidated tool to schedule a task persistently."""
    if interval_minutes < 1:
        return "Interval must be ≥ 1 minute"

    update_cron_task_in_cron_md(task_id, name, interval_minutes, prompt)

    global tasks
    found = False
    for t in tasks:
        if t.id == task_id:
            t.name = name
            t.interval_minutes = interval_minutes
            t.prompt = prompt
            t.last_run = datetime.now() - timedelta(minutes=interval_minutes + 1)
            found = True
            break

    if not found:
        tasks.append(
            PeriodicTask(
                id=task_id,
                name=name,
                interval_minutes=interval_minutes,
                prompt=prompt,
                last_run=datetime.now() - timedelta(minutes=interval_minutes + 1),
            )
        )

    return f"✅ Scheduled '{name}' (ID: {task_id}) every {interval_minutes} min"


def delete_scheduled_task(task_id: str) -> str:
    """Remove a task from CRON.md and memory."""
    registry = get_cron_tasks_from_md()
    original_count = len(registry.tasks)
    registry.tasks = [t for t in registry.tasks if t.id != task_id]

    if len(registry.tasks) < original_count:
        content = serialize_cron_registry(registry)
        path = get_workspace_path(CORE_FILES["CRON"])
        path.write_text(content, encoding="utf-8")

    global tasks
    found_in_mem = False
    tasks_to_keep = []
    for t in tasks:
        if t.id == task_id:
            found_in_mem = True
            continue
        tasks_to_keep.append(t)

    tasks[:] = tasks_to_keep

    if found_in_md or found_in_mem:
        return f"✅ Deleted scheduled task '{task_id}'"
    return f"ℹ️ Task '{task_id}' not found."


def list_scheduled_tasks() -> CronRegistryModel:
    """List all active periodic tasks."""
    return get_cron_tasks_from_md()


def append_cron_log(
    task_id: str, task_name: str, output: str, chat_id: Optional[str] = None
):
    """Append a timestamped entry to CRON_LOG.md."""
    path = get_workspace_path("CRON_LOG.md")
    if not path.exists():
        initialize_workspace()

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    chat_info = f" | [View Chat](/{chat_id})" if chat_id else ""
    entry = (
        f"\n### [{ts}] {task_name} (`{task_id}`){chat_info}\n\n"
        f"{output.strip()}\n\n"
        f"---\n"
    )
    with open(path, "a", encoding="utf-8") as f:
        f.write(entry)
    logger.debug(f"Appended cron log entry for {task_id}")


def cleanup_cron_log(max_entries: int = DEFAULT_MAX_CRON_LOG_ENTRIES):
    """Keep only the last `max_entries` log entries in CRON_LOG.md."""
    log_model = get_cron_logs_from_md()
    if len(log_model.entries) <= max_entries:
        return

    log_model.entries = log_model.entries[-max_entries:]
    content = serialize_cron_log(log_model)
    path = get_workspace_path(CORE_FILES["CRON_LOG"])
    path.write_text(content, encoding="utf-8")
    logger.debug(f"Pruned cron log entries, kept {max_entries}")


async def reload_cron_tasks():
    """Reload all tasks from CRON.md.

    Every row in the table becomes a PeriodicTask.  Prompts starting with
    '@' are resolved to workspace file contents at execution time (not here).
    """
    registry = get_cron_tasks_from_md()
    parsed_tasks = [
        PeriodicTask(
            id=t.id,
            name=t.name,
            interval_minutes=t.interval_minutes,
            prompt=t.prompt,
            last_run=datetime.now(),
        )
        for t in registry.tasks
    ]

    async with lock:
        global tasks
        new_list = []
        for pt in parsed_tasks:

            existing = next((t for t in tasks if t.id == pt.id), None)
            if existing and existing.interval_minutes == pt.interval_minutes:
                pt.last_run = existing.last_run
                pt.active = existing.active
            new_list.append(pt)
        tasks = new_list


async def background_processor(agent: Any):
    """Background processor for periodic tasks."""

    logger = logging.getLogger(__name__)
    logger.debug("In-memory periodic processor started (checks every 60 s)")

    while True:
        try:
            await reload_cron_tasks()
        except Exception as e:
            logger.error(f"Error reloading cron tasks: {e}")

        await asyncio.sleep(60)
        now = datetime.now()
        due: list[PeriodicTask] = []
        async with lock:
            for t in tasks:
                if (
                    t.active
                    and (now - t.last_run).total_seconds() / 60 >= t.interval_minutes
                ):
                    due.append(t)
                    t.last_run = now

        for task in due:
            try:

                if task.prompt.startswith("__internal:"):
                    cmd = task.prompt.split(":", 1)[1]
                    if cmd == "cleanup_cron_log":
                        cleanup_cron_log()
                        logger.debug("Cron log cleanup completed")
                    continue

                resolved_prompt = resolve_prompt(task.prompt)

                logger.info(f"Running periodic task → {task.name} (ID: {task.id})")
                result = await agent.run(resolved_prompt)

                output = str(result.output or "")
                if output:
                    logger.info(f"Task result: {output[:200]}...")

                try:
                    chat_id = (
                        f"cron-{task.id}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                    )
                    messages = []

                    messages.append(
                        {
                            "id": "msg-u-1",
                            "role": "user",
                            "content": resolved_prompt,
                            "parts": [{"type": "text", "text": resolved_prompt}],
                        }
                    )

                    messages.append(
                        {
                            "id": "msg-a-1",
                            "role": "assistant",
                            "content": output,
                            "parts": [{"type": "text", "text": output}],
                        }
                    )

                    save_chat_to_disk(chat_id, messages)
                except Exception as e:
                    logger.error(f"Failed to save cron chat: {e}")
                    chat_id = None

                append_cron_log(
                    task_id=task.id,
                    task_name=task.name,
                    output=output or "(no output)",
                    chat_id=chat_id,
                )
            except Exception as e:
                logger.error(f"Error running periodic task {task.id}: {e}")
                append_cron_log(
                    task_id=task.id,
                    task_name=task.name,
                    output=f"❌ ERROR: {e}",
                )

        await asyncio.sleep(60)
