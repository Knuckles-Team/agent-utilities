#!/usr/bin/python

from __future__ import annotations

import logging
import asyncio


from typing import List, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    pass





from .config import *
from .workspace import *


from .models import PeriodicTask

tasks: List[PeriodicTask] = []
lock = asyncio.Lock()




logger = logging.getLogger(__name__)


def create_memory(text: str) -> str:
    """
    Save important decisions, outcomes, user preferences, critical
    information, or information the user explicitly requests to long-term memory (MEMORY.md).
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    note = f"- [{timestamp}] {text}"
    append_to_md_file("MEMORY.md", note)
    return "Saved to memory."


def search_memory(query: str) -> str:
    """Search MEMORY.md for a query string."""
    content = load_workspace_file(CORE_FILES["MEMORY"])
    if not content:
        return "Memory is empty."

    lines = content.splitlines()
    results = []

    for i, line in enumerate(lines):
        if query.lower() in line.lower():
            results.append(f"Line {i+1}: {line.strip()}")

    if not results:
        return f"No entries found matching '{query}' in memory."
    return "\n".join(results)


def delete_memory_entry(index: int) -> str:
    """Delete a memory entry by line number (1-indexed)."""
    path = get_workspace_path(CORE_FILES["MEMORY"])
    if not path.exists():
        return "Memory file not found."

    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()

    if index < 1 or index > len(lines):
        return f"❌ Invalid index {index}. Memory has {len(lines)} lines."

    line_to_delete = lines[index - 1].strip()
    if not (
        line_to_delete.startswith("-")
        or line_to_delete.startswith("*")
        or line_to_delete.startswith("|")
    ):
        return f"⚠️ Line {index} does not look like a data entry: '{line_to_delete}'. Deletion aborted for safety."

    deleted_text = lines.pop(index - 1)
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return f"✅ Deleted memory entry {index}: {deleted_text}"


def compress_memory(max_entries: int = 50) -> str:
    """
    Compress MEMORY.md by pruning old entries.
    In a future version this could use an LLM to summarize.
    """
    path = get_workspace_path(CORE_FILES["MEMORY"])
    if not path.exists():
        return "Memory file not found."

    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()

    log_start = -1
    for i, line in enumerate(lines):
        if "## Log of Important Events" in line:
            log_start = i
            break

    if log_start == -1:
        return "❌ Could not find '## Log of Important Events' section in MEMORY.md"

    header = lines[: log_start + 1]
    entries = [line for line in lines[log_start + 1 :] if line.strip()]

    if len(entries) <= max_entries:
        return f"ℹ️ Memory consists of {len(entries)} entries, which is below the limit of {max_entries}. No compression needed."

    pruned = len(entries) - max_entries
    kept_entries = entries[-max_entries:]

    new_content = "\n".join(header).strip() + "\n\n"
    new_content += "> [!NOTE]\n"
    new_content += f"> Memory was compressed on {datetime.now().strftime('%Y-%m-%d')}. {pruned} older entries were pruned.\n\n"
    new_content += "\n".join(kept_entries)

    path.write_text(new_content.strip() + "\n", encoding="utf-8")
    return f"✅ Compressed memory. Pruned {pruned} old entries, kept the most recent {max_entries}."
