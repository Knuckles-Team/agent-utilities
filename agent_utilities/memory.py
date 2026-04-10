#!/usr/bin/python

from __future__ import annotations

import logging


from typing import TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    pass


from .workspace import (
    CORE_FILES,
    parse_memory,
    serialize_memory,
    load_workspace_file,
    write_workspace_file,
)


from .models import MemoryModel, MemoryEntryModel

logger = logging.getLogger(__name__)


def create_memory(text: str) -> str:
    """Save to long-term memory (MEMORY.md)."""
    model = parse_memory(load_workspace_file(CORE_FILES["MEMORY"]))
    model.entries.append(
        MemoryEntryModel(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"), text=text)
    )
    content = serialize_memory(model)
    write_workspace_file(CORE_FILES["MEMORY"], content)
    return "Saved to memory."


def search_memory(query: str) -> MemoryModel:
    """Search MEMORY.md for a query string."""
    model = parse_memory(load_workspace_file(CORE_FILES["MEMORY"]))
    filtered_entries = [e for e in model.entries if query.lower() in e.text.lower()]
    return MemoryModel(entries=filtered_entries)


def delete_memory_entry(index: int) -> str:
    """Delete a memory entry by list index (1-indexed)."""
    model = parse_memory(load_workspace_file(CORE_FILES["MEMORY"]))
    if index < 1 or index > len(model.entries):
        return f"❌ Invalid index {index}. Memory has {len(model.entries)} entries."

    deleted = model.entries.pop(index - 1)
    content = serialize_memory(model)
    write_workspace_file(CORE_FILES["MEMORY"], content)
    return f"✅ Deleted memory entry {index}: {deleted.text}"


def compress_memory(max_entries: int = 50) -> str:
    """Compress MEMORY.md by pruning old entries."""
    model = parse_memory(load_workspace_file(CORE_FILES["MEMORY"]))
    if len(model.entries) <= max_entries:
        return f"ℹ️ Memory consists of {len(model.entries)} entries, no compression needed."

    pruned = len(model.entries) - max_entries
    model.entries = model.entries[-max_entries:]

    content = serialize_memory(model)
    write_workspace_file(CORE_FILES["MEMORY"], content)
    return f"✅ Compressed memory. Pruned {pruned} old entries, kept {max_entries}."
