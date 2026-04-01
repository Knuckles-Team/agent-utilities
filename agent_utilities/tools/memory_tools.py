import logging
from typing import Any
from pydantic_ai import RunContext
from ..models import MemoryModel
from ..agent_utilities import (
    create_memory as create_memory_util,
    search_memory as search_memory_util,
    delete_memory_entry as delete_memory_entry_util,
    compress_memory as compress_memory_util,
)

logger = logging.getLogger(__name__)


async def create_memory(ctx: RunContext[Any], text: str) -> str:
    """
    Save important decisions, outcomes, user preferences, critical
    information, or information the user explicitly requests to long-term memory (MEMORY.md).
    """
    return create_memory_util(text)


async def search_memory(ctx: RunContext[Any], query: str) -> MemoryModel:
    """Search through long-term memory (MEMORY.md) for specific information."""
    return search_memory_util(query)


async def delete_memory_entry(ctx: RunContext[Any], index: int) -> str:
    """Delete a specific memory entry by line index (1-based). Use search_memory first to find the index."""
    return delete_memory_entry_util(index)


async def compress_memory(ctx: RunContext[Any], max_entries: int = 50) -> str:
    """Compress or prune long-term memory to keep it concise and relevant."""
    return compress_memory_util(max_entries)


# Tool grouping for registration
memory_tools = [
    create_memory,
    search_memory,
    delete_memory_entry,
    compress_memory,
]
