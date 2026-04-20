#!/usr/bin/python
# coding: utf-8
"""Task Scheduler Module.

This module provides a persistent task scheduling system for agents. It
manages periodic tasks defined in CRON.md, maintains an execution history
in CRON_LOG.md, and runs a background processor to execute scheduled tasks
using the agent's core capabilities.
"""

from __future__ import annotations

import logging
import asyncio


from typing import Any, List, Optional, TYPE_CHECKING
from datetime import datetime, timedelta

if TYPE_CHECKING:
    pass


from .config import DEFAULT_MAX_CRON_LOG_ENTRIES
from .prompt_builder import resolve_prompt
from .chat_persistence import save_chat_to_disk


from .models import PeriodicTask, CronRegistryModel, CronTaskModel, CronLogModel

tasks: List[PeriodicTask] = []
lock = asyncio.Lock()


logger = logging.getLogger(__name__)


def get_cron_tasks_from_md() -> CronRegistryModel:
    """Retrieve the current scheduled tasks from the Knowledge Graph.

    Returns:
        A CronRegistryModel object containing the list of active tasks.

    """
    from .knowledge_graph.engine import IntelligenceGraphEngine

    engine = IntelligenceGraphEngine.get_active()
    if not engine or not engine.backend:
        return CronRegistryModel()

    try:
        res = engine.backend.execute(
            "MATCH (j:Job) RETURN j.id, j.name, j.schedule, j.command, j.last_run, j.next_approx"
        )
        tasks = []
        for row in res:
            tasks.append(
                CronTaskModel(
                    id=row.get("j.id", ""),
                    name=row.get("j.name", ""),
                    interval_minutes=int(row.get("j.schedule", 0)),
                    prompt=row.get("j.command", ""),
                    last_run=row.get("j.last_run", "—"),
                    next_approx=row.get("j.next_approx", "—"),
                )
            )
        return CronRegistryModel(tasks=tasks)
    except Exception as e:
        logger.debug(f"Failed to fetch Job nodes: {e}")
        return CronRegistryModel()


def get_cron_logs_from_md() -> CronLogModel:
    """Retrieve the historical task execution logs from the Knowledge Graph.

    Returns:
        A CronLogModel object containing the history of task runs.

    """
    from .knowledge_graph.engine import IntelligenceGraphEngine

    engine = IntelligenceGraphEngine.get_active()
    if not engine or not engine.backend:
        return CronLogModel()

    try:
        res = engine.backend.execute(
            "MATCH (l:Log) RETURN l.timestamp, l.task_id, l.task_name, l.status, l.output, l.chat_id ORDER BY l.timestamp DESC LIMIT 100"
        )
        entries = []
        for row in res:
            entries.append(
                CronLogEntryModel(
                    timestamp=row.get("l.timestamp", ""),
                    task_id=row.get("l.task_id", ""),
                    task_name=row.get("l.task_name", ""),
                    status=row.get("l.status", "success"),
                    message=row.get("l.output", ""),
                    chat_id=row.get("l.chat_id"),
                )
            )
        return CronLogModel(entries=entries)
    except Exception as e:
        logger.debug(f"Failed to fetch Log nodes: {e}")
        return CronLogModel()


def update_cron_task_in_cron_md(
    task_id: str, name: str, interval_min: int, prompt: str
):
    """Update or create a task definition in the Knowledge Graph.

    Args:
        task_id: Unique identifier for the task.
        name: Human-readable task name.
        interval_min: Execution interval in minutes.
        prompt: The prompt or internal command to execute.

    """
    from .knowledge_graph.engine import IntelligenceGraphEngine

    engine = IntelligenceGraphEngine.get_active()
    if not engine or not engine.backend:
        return

    query = """
    MERGE (j:Job {id: $id})
    SET j.name = $name,
        j.schedule = $schedule,
        j.command = $command,
        j.last_run = $last_run
    """
    engine.backend.execute(
        query,
        {
            "id": task_id,
            "name": name,
            "schedule": str(interval_min),
            "command": prompt,
            "last_run": datetime.now().strftime("%H:%M"),
        },
    )


def schedule_task(task_id: str, name: str, interval_minutes: int, prompt: str) -> str:
    """Public helper to schedule a new task and persist it to the graph.

    Args:
        task_id: Unique identifier.
        name: Friendly name.
        interval_minutes: Minutes between runs.
        prompt: Task instructions or internal script identifier.

    Returns:
        A status message indicating success or validation error.

    """
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
    """Permanently remove a scheduled task from both memory and the graph.

    Args:
        task_id: The ID of the task to delete.

    Returns:
        A success or failure status message.

    """
    from .knowledge_graph.engine import IntelligenceGraphEngine

    engine = IntelligenceGraphEngine.get_active()
    if engine and engine.backend:
        engine.backend.execute(
            "MATCH (j:Job {id: $id}) DETACH DELETE j", {"id": task_id}
        )

    global tasks
    tasks[:] = [t for t in tasks if t.id != task_id]

    return f"✅ Deleted scheduled task '{task_id}'"


def list_scheduled_tasks() -> CronRegistryModel:
    """List all currently active scheduled tasks.

    Returns:
        A model containing the list of configured tasks.

    """
    return get_cron_tasks_from_md()


def append_cron_log(
    task_id: str,
    task_name: str,
    output: str,
    status: str = "success",
    chat_id: Optional[str] = None,
):
    """Log the outcome of a periodic task run to the Knowledge Graph.

    Args:
        task_id: ID of the task.
        task_name: Name of the task.
        output: Text output or error message from the run.
        status: Run status ('success' or 'error').
        chat_id: Optional reference to a persistent chat log.

    """
    from .knowledge_graph.engine import IntelligenceGraphEngine

    engine = IntelligenceGraphEngine.get_active()
    if not engine or not engine.backend:
        return

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_id = f"log:{task_id}:{datetime.now().timestamp()}"

    query = """
    CREATE (l:Log {id: $id})
    SET l.timestamp = $ts,
        l.task_id = $task_id,
        l.task_name = $task_name,
        l.status = $status,
        l.output = $output,
        l.chat_id = $chat_id
    WITH l
    MATCH (j:Job {id: $task_id})
    MERGE (j)-[:HAS_LOG]->(l)
    """
    engine.backend.execute(
        query,
        {
            "id": log_id,
            "ts": ts,
            "task_id": task_id,
            "task_name": task_name,
            "status": status,
            "output": output,
            "chat_id": chat_id,
        },
    )
    logger.debug(f"Logged task execution for {task_id}")


def cleanup_cron_log(max_entries: int = DEFAULT_MAX_CRON_LOG_ENTRIES):
    """Prune old Log nodes in the Knowledge Graph.

    Args:
        max_entries: The number of recent log entries to retain.

    """
    from .knowledge_graph.engine import IntelligenceGraphEngine

    engine = IntelligenceGraphEngine.get_active()
    if not engine or not engine.backend:
        return

    # Delete logs older than the last max_entries
    query = """
    MATCH (l:Log)
    WITH l ORDER BY l.timestamp DESC SKIP $skip
    DETACH DELETE l
    """
    engine.backend.execute(query, {"skip": max_entries})
    logger.debug(f"Pruned Knowledge Graph logs, kept {max_entries}")


async def reload_cron_tasks():
    """Synchronize the in-memory task list with the CRON.md registry.

    Ensures the background processor is aware of any changes made to
    the workspace file during runtime.
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
    """The main execution loop for periodic tasks.

    Periodically reloads tasks, monitors run intervals, and executes
    overdue tasks by calling the agent.

    Args:
        agent: The agent instance used to run resolved prompts.

    """
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
