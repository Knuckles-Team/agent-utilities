#!/usr/bin/python

from __future__ import annotations

import re
import logging
import asyncio


from typing import Any, Dict, List, Optional, TYPE_CHECKING
from datetime import datetime, timedelta

if TYPE_CHECKING:
    pass





from .config import *
from .workspace import *


from .models import PeriodicTask

tasks: List[PeriodicTask] = []
lock = asyncio.Lock()




logger = logging.getLogger(__name__)

lock = asyncio.Lock()


def get_cron_tasks_from_md() -> List[Dict[str, Any]]:
    """Parse CRON.md and return active tasks."""
    content = load_workspace_file(CORE_FILES["CRON"])
    tasks = []

    lines = content.split("\n")
    for line in lines:
        if "|" in line and "ID" not in line and "---" not in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= 3:
                tasks.append(
                    {
                        "id": parts[0],
                        "name": parts[1],
                        "schedule": parts[2],
                    }
                )
    return tasks


def get_cron_logs_from_md() -> List[Dict[str, Any]]:
    """Parse CRON_LOG.md and return recent history."""
    content = load_workspace_file(CORE_FILES["CRON_LOG"])
    logs = []

    parts = re.split(r"(?=^### \[)", content, flags=re.MULTILINE)

    for part in parts:
        if not part.strip() or not part.startswith("### ["):
            continue

        try:

            header_match = re.search(
                r"^### \[(.*?)\] (.*?) \(`(.*?)`\)(?: \| \[View Chat\]\((.*?)\))?", part
            )
            if header_match:
                ts = header_match.group(1)
                name = header_match.group(2)
                tid = header_match.group(3)
                cid = header_match.group(4) if header_match.lastindex >= 4 else None

                body = part.split("\n\n", 1)[1] if "\n\n" in part else ""
                output = body.split("\n---")[0].strip()

                logs.append(
                    {
                        "timestamp": ts,
                        "task_id": tid,
                        "task_name": name,
                        "status": "success",
                        "output": output,
                        "chat_id": cid.lstrip("/") if cid else None,
                    }
                )
        except Exception as e:
            logger.debug(f"Error parsing log entry: {e}")

    return logs[::-1]


def update_cron_task_in_cron_md(task: dict):
    """
    Update/add one row in CRON.md table.
    task = {"id": "daily-news", "name": "...", "interval_min": 1440, ...}
    """
    path = get_workspace_path("CRON.md")
    if not path.exists():
        initialize_workspace()

    lines = path.read_text(encoding="utf-8").splitlines()
    table_start = -1
    for i, line in enumerate(lines):
        if "| ID" in line and "| Name" in line:
            table_start = i
            break

    if table_start == -1:

        append_to_file(
            "CRON.md",
            "\n## Active Tasks\n\n| ID | Name | Interval (min) | Prompt starts with | Last run | Next approx |\n|----|------|----------------|--------------------|----------|-------------|",
        )
        lines = path.read_text(encoding="utf-8").splitlines()
        table_start = len(lines) - 2

    new_row = (
        f"| {task.get('id','?')} "
        f"| {task.get('name','?')} "
        f"| {task.get('interval_min','?')} "
        f"| {task.get('prompt','?')[:40]}... "
        f"| {datetime.now().strftime('%H:%M')} "
        f"| — |"
    )

    id_found = False
    for i in range(table_start + 2, len(lines)):
        if (
            lines[i].strip().startswith(f"| {task['id']} ")
            or f"| {task['id']} |" in lines[i]
        ):
            lines[i] = new_row
            id_found = True
            break

    if not id_found:
        lines.insert(table_start + 3, new_row)

    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def schedule_task(task_id: str, name: str, interval_minutes: int, prompt: str) -> str:
    """Consolidated tool to schedule a task persistently."""
    if interval_minutes < 1:
        return "Interval must be ≥ 1 minute"

    task_data = {
        "id": task_id,
        "name": name,
        "interval_min": interval_minutes,
        "prompt": prompt,
    }
    update_cron_task_in_cron_md(task_data)

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
    path = get_workspace_path("CRON.md")
    if not path.exists():
        return "CRON.md not found."

    lines = path.read_text(encoding="utf-8").splitlines()
    new_lines = []
    found_in_md = False
    for line in lines:
        if line.strip().startswith(f"| {task_id} ") or f"| {task_id} |" in line:
            found_in_md = True
            continue
        new_lines.append(line)

    if found_in_md:
        path.write_text("\n".join(new_lines).strip() + "\n", encoding="utf-8")

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


def list_scheduled_tasks() -> str:
    """List all active periodic tasks from memory."""
    global tasks
    if not tasks:
        return "No periodic tasks scheduled."

    lines = ["Active periodic tasks:"]
    now = datetime.now()
    for t in tasks:
        if t.active:
            mins_since = (now - t.last_run).total_seconds() / 60
            next_in = max(0, int(t.interval_minutes - mins_since))
            lines.append(
                f"• {t.id}: {t.name} (every {t.interval_minutes} min, next ≈ {next_in} min)"
            )
    return "\n".join(lines)


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
    path = get_workspace_path("CRON_LOG.md")
    if not path.exists():
        return

    content = path.read_text(encoding="utf-8")

    parts = re.split(r"(?=^### \[)", content, flags=re.MULTILINE)

    header = parts[0] if parts else ""
    entries = [p for p in parts[1:] if p.strip()]

    if len(entries) <= max_entries:
        return

    kept = entries[-max_entries:]
    pruned_count = len(entries) - max_entries
    new_content = header.rstrip() + "\n\n" + "".join(kept)
    path.write_text(new_content.strip() + "\n", encoding="utf-8")
    logger.debug(f"Pruned {pruned_count} old cron log entries, kept {max_entries}")


async def reload_cron_tasks():
    """Reload all tasks from CRON.md.

    Every row in the table becomes a PeriodicTask.  Prompts starting with
    '@' are resolved to workspace file contents at execution time (not here).
    """
    content = load_workspace_file("CRON.md")
    if not content:
        return

    parsed_tasks = []
    lines = content.splitlines()
    in_table = False
    for line in lines:
        if "| ID" in line and "| Name" in line:
            in_table = True
            continue
        if (
            in_table
            and line.strip().startswith("|")
            and not (line.strip().startswith("|---") or "| ID" in line)
        ):
            parts = [p.strip() for p in line.strip("| ").split("|")]
            if len(parts) >= 4:
                try:
                    parsed_tasks.append(
                        PeriodicTask(
                            id=parts[0],
                            name=parts[1],
                            interval_minutes=int(parts[2]),
                            prompt=parts[3],
                            last_run=datetime.now(),
                        )
                    )
                except Exception:
                    continue

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
