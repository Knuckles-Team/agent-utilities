#!/usr/bin/python
# coding: utf-8
"""Memory Tools Module.

This module provides tools for persisting, searching, managing,
and compressing long-term agent memories in the workspace.
"""

import logging
from typing import Any
from pydantic_ai import RunContext
from ..models import MemoryModel
from ..memory import (
    create_memory as create_memory_util,
    search_memory as search_memory_util,
    delete_memory_entry as delete_memory_entry_util,
    compress_memory as compress_memory_util,
)

logger = logging.getLogger(__name__)


async def create_memory(ctx: RunContext[Any], text: str) -> str:
    """Store high-value information or user preferences into long-term memory.

    Use this to persist decisions, critical context, or explicit user
    instructions into the MEMORY.md file for future session recall.

    Args:
        ctx: The agent run context.
        text: The content to be remembered.

    Returns:
        A confirmation message indicating success.

    """
    return create_memory_util(text)


async def search_memory(ctx: RunContext[Any], query: str) -> MemoryModel:
    """Search for relevant historical entries within long-term memory.

    Args:
        ctx: The agent run context.
        query: Search keywords or semantic intent.

    Returns:
        A model containing matching memory entries.

    """
    return search_memory_util(query)


async def delete_memory_entry(ctx: RunContext[Any], index: int) -> str:
    """Delete a specific memory entry by its line index.

    Args:
        ctx: The agent run context.
        index: The 1-based index found via search_memory.

    Returns:
        A confirmation message indicating success.

    """
    return delete_memory_entry_util(index)


async def compress_memory(ctx: RunContext[Any], max_entries: int = 50) -> str:
    """Optimize or prune long-term memory to maintain relevance and brevity.

    Args:
        ctx: The agent run context.
        max_entries: The maximum number of entries to retain.

    Returns:
        A status message summarizing the compression result.

    """
    return compress_memory_util(max_entries)


# Tool grouping for registration
memory_tools = [
    create_memory,
    search_memory,
    delete_memory_entry,
    compress_memory,
]
