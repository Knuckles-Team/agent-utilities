import logging
import os
from typing import Any

from pydantic_ai import Agent, RunContext
from .agent_utilities import (
    read_md_file,
    append_to_md_file,
    register_a2a_peer as register_a2a_peer_util,
    create_new_skill,
    load_mcp_config,
    save_mcp_config,
    delete_skill_from_disk,
    write_skill_md,
    search_memory as search_memory_util,
    delete_memory_entry as delete_memory_entry_util,
    compress_memory as compress_memory_util,
    delete_a2a_peer as delete_a2a_peer_util,
    list_a2a_peers as list_a2a_peers_util,
    create_memory as create_memory_util,
    list_scheduled_tasks as list_scheduled_tasks_util,
    delete_scheduled_task as delete_scheduled_task_util,
    schedule_task as schedule_task_util,
    read_skill_md,
)
from .base_utilities import to_boolean

DEFAULT_WORKSPACE_TOOLS = to_boolean(string=os.environ.get("WORKSPACE_TOOLS", "True"))
DEFAULT_A2A_TOOLS = to_boolean(string=os.environ.get("A2A_TOOLS", "True"))
DEFAULT_SCHEDULER_TOOLS = to_boolean(string=os.environ.get("SCHEDULER_TOOLS", "True"))
DEFAULT_DYNAMIC_TOOLS = to_boolean(string=os.environ.get("DYNAMIC_TOOLS", "True"))

logger = logging.getLogger(__name__)

__version__ = "0.2.27"


def register_agent_tools(agent: Agent):
    """Register universal workspace, A2A, and scheduler tools to the agent."""

    # System prompt is now handled purely statically at initialization inside create_agent.

    # --- Workspace Tools ---
    if DEFAULT_WORKSPACE_TOOLS:

        @agent.tool
        async def read_workspace_file(ctx: RunContext[Any], filename: str) -> str:
            """Read content of any .md file in workspace (IDENTITY.md, CRON.md, etc.).
            Read your system prompt, identity, memory, scheduled cron jobs, cron log, and mcp_config.json
            """
            return read_md_file(filename)

        @agent.tool
        async def append_note_to_file(
            ctx: RunContext[Any], filename: str, text: str
        ) -> str:
            """Append a short note or section to a workspace .md file."""
            append_to_md_file(filename, text)
            return f"Appended to {filename}"

        @agent.tool
        async def create_memory(ctx: RunContext[Any], text: str) -> str:
            """
            Save important decisions, outcomes, user preferences, critical
            information, or information the user explicitly requests to long-term memory (MEMORY.md).
            """
            return create_memory_util(text)

        @agent.tool
        async def search_memory(ctx: RunContext[Any], query: str) -> str:
            """Search through long-term memory (MEMORY.md) for specific information."""
            return search_memory_util(query)

        @agent.tool
        async def delete_memory_entry(ctx: RunContext[Any], index: int) -> str:
            """Delete a specific memory entry by line index (1-based). Use search_memory first to find the index."""
            return delete_memory_entry_util(index)

        @agent.tool
        async def compress_memory(ctx: RunContext[Any], max_entries: int = 50) -> str:
            """Compress or prune long-term memory to keep it concise and relevant."""
            return compress_memory_util(max_entries)

    # --- A2A Peer Tools ---
    if DEFAULT_A2A_TOOLS:

        @agent.tool
        async def list_a2a_peers(ctx: RunContext[Any]) -> str:
            """List all known A2A peer agents."""
            return list_a2a_peers_util()

        @agent.tool
        async def register_a2a_peer(
            ctx: RunContext[Any],
            name: str,
            url: str,
            description: str = "",
            capabilities: str = "",
            auth: str = "none",
        ) -> str:
            """Register or update another A2A agent this agent can call."""
            return register_a2a_peer_util(name, url, description, capabilities, auth)

        @agent.tool
        async def delete_a2a_peer(ctx: RunContext[Any], name: str) -> str:
            """Remove an A2A peer agent from the local registry."""
            return delete_a2a_peer_util(name)

    # --- Scheduler Tools ---
    if DEFAULT_SCHEDULER_TOOLS:

        @agent.tool
        async def schedule_task(
            ctx: RunContext[Any],
            task_id: str,
            name: str,
            interval_minutes: int,
            prompt: str,
        ) -> str:
            """Schedule a task to run periodically (persists in CRON.md)."""
            return schedule_task_util(task_id, name, interval_minutes, prompt)

        @agent.tool
        async def list_tasks(ctx: RunContext[Any]) -> str:
            """List all active periodic tasks."""
            return list_scheduled_tasks_util()

        @agent.tool
        async def delete_task(ctx: RunContext[Any], task_id: str) -> str:
            """Permanently remove a scheduled task by ID."""
            return delete_scheduled_task_util(task_id)

    # --- Dynamic Extension Tools ---
    if DEFAULT_DYNAMIC_TOOLS:

        @agent.tool
        async def update_mcp_config(
            ctx: RunContext[Any],
            server_name: str,
            server_url: str,
            transport: str = "sse",
        ) -> str:
            """Add or update an MCP server entry in mcp_config.json (auto-loaded on next run)."""
            config = load_mcp_config()
            config.setdefault("mcpServers", {})[server_name] = {
                "url": server_url,
                "transport": transport,
            }
            save_mcp_config(config)
            return f"✅ MCP server '{server_name}' added/updated. It will be available on next run."

        @agent.tool
        async def create_skill(
            ctx: RunContext[Any],
            name: str,
            description: str,
            when_to_use: str = "",
            how_to_use: str = "",
        ) -> str:
            """Create a brand-new skill folder + SKILL.md that will be auto-loaded on next run."""
            return create_new_skill(name, description, when_to_use, how_to_use)

        @agent.tool
        async def delete_skill(ctx: RunContext[Any], name: str) -> str:
            """Delete a skill folder from the workspace. Only works for workspace skills."""
            return delete_skill_from_disk(name)

        @agent.tool
        async def edit_skill(ctx: RunContext[Any], name: str, new_content: str) -> str:
            """
            Overwrite the SKILL.md of an existing workspace skill.
            Use this to refine a skill's logic, description, or examples.
            """
            return write_skill_md(name, new_content)

        @agent.tool
        async def get_skill_content(ctx: RunContext[Any], name: str) -> str:
            """Read the current SKILL.md of a workspace skill to prepare for editing."""
            return read_skill_md(name)
