#!/usr/bin/python
# coding: utf-8
"""
Memory Management Module

This module provides high-level utilities for managing an agent's long-term memory 
stored in the workspace (typically MEMORY.md). It supports creating new memories, 
searching existing ones, deleting entries, and compressing memory to prevent 
context window overflow.
"""

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
    """Save a new entry to the agent's long-term memory (MEMORY.md).

    Args:
        text: The text content to be remembered.

    Returns:
        A confirmation message indicating the memory was saved.
    """
    model = parse_memory(load_workspace_file(CORE_FILES["MEMORY"]))
    model.entries.append(
        MemoryEntryModel(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"), text=text)
    )
    content = serialize_memory(model)
    write_workspace_file(CORE_FILES["MEMORY"], content)
    return "Saved to memory."


def search_memory(query: str) -> MemoryModel:
    """Search the agent's long-term memory for entries matching a query string.

    Args:
        query: The keyword or phrase to search for.

    Returns:
        A MemoryModel containing only the entries that match the query (case-insensitive).
    """
    model = parse_memory(load_workspace_file(CORE_FILES["MEMORY"]))
    filtered_entries = [e for e in model.entries if query.lower() in e.text.lower()]
    return MemoryModel(entries=filtered_entries)


def delete_memory_entry(index: int) -> str:
    """Delete a specific memory entry by its list index.

    Args:
        index: The 1-based index of the memory entry to remove.

    Returns:
        A success message with the deleted content, or an error message if the index is invalid.
    """
    model = parse_memory(load_workspace_file(CORE_FILES["MEMORY"]))
    if index < 1 or index > len(model.entries):
        return f"❌ Invalid index {index}. Memory has {len(model.entries)} entries."

    deleted = model.entries.pop(index - 1)
    content = serialize_memory(model)
    write_workspace_file(CORE_FILES["MEMORY"], content)
    return f"✅ Deleted memory entry {index}: {deleted.text}"


def compress_memory(max_entries: int = 50) -> str:
    """Prune old entries from the agent's memory to maintain a maximum size.

    This is used to prevent long-term memory from consuming too much context window 
    during retrieval. It keeps only the most recent 'max_entries'.

    Args:
        max_entries: The maximum number of recent entries to retain. Defaults to 50.

    Returns:
        A message stating how many entries were pruned or if no compression was needed.
    """
    model = parse_memory(load_workspace_file(CORE_FILES["MEMORY"]))
    if len(model.entries) <= max_entries:
        return f"ℹ️ Memory consists of {len(model.entries)} entries, no compression needed."

    pruned = len(model.entries) - max_entries
    model.entries = model.entries[-max_entries:]

    content = serialize_memory(model)
    write_workspace_file(CORE_FILES["MEMORY"], content)
    return f"✅ Compressed memory. Pruned {pruned} old entries, kept {max_entries}."
