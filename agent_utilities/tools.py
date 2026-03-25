import logging
import os
from typing import Any, Optional

from pydantic_ai import Agent, RunContext
from agent_utilities.agent_utilities import (
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
from agent_utilities.base_utilities import to_boolean

DEFAULT_WORKSPACE_TOOLS = to_boolean(string=os.environ.get("WORKSPACE_TOOLS", "True"))
DEFAULT_A2A_TOOLS = to_boolean(string=os.environ.get("A2A_TOOLS", "True"))
DEFAULT_SCHEDULER_TOOLS = to_boolean(string=os.environ.get("SCHEDULER_TOOLS", "True"))
DEFAULT_DYNAMIC_TOOLS = to_boolean(string=os.environ.get("DYNAMIC_TOOLS", "False"))

logger = logging.getLogger(__name__)

__version__ = "0.2.32"


def register_agent_tools(agent: Agent, graph_bundle: Optional[tuple] = None):
    """Register universal workspace, A2A, and scheduler tools to the agent."""

    # System prompt is now handled purely statically at initialization inside create_agent.

    # --- Graph Orchestration ---
    if graph_bundle:
        graph, config = graph_bundle
        from agent_utilities.agent_utilities import run_graph

        @agent.tool
        async def run_graph_flow(ctx: RunContext[Any], prompt: str) -> str:
            """
            Execute a complex query through the graph orchestrator.
            The graph automatically classifies and routes your request to specialized domain nodes.
            Use this for ANY domain-specific operations or multi-step tasks within your expertise.
            """
            logger.info(f"run_graph_flow initiated for prompt: {prompt[:100]}...")
            result = await run_graph(graph, config, prompt)

            from agent_utilities.agent_utilities import get_graph_mermaid

            try:
                mermaid_code = get_graph_mermaid(graph, config)
                res_str = str(result.get("results", result))
                return f"Graph execution complete.\nHere is the graph routing diagram:\n```mermaid\n{mermaid_code}\n```\n\nResults:\n{res_str}"
            except Exception as e:
                logger.error(f"Failed to generate Mermaid diagram: {e}")
                return str(result.get("results", result))

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

        # @agent.tool
        # async def get_mcp_reference(ctx: RunContext[Any], agent_template: str) -> str:
        #     """
        #     Retrieve the documentation for an MCP agent template.
        #     Use this to discover available 'enabled_tools' tags and configuration details.

        #     Args:
        #         agent_template: The name of the agent template. Options: ["adguard-home", "ansible-tower", "archivebox", "arr", "audio-transcriber", "container-manager", "documentdb", "github", "gitlab", "jellyfin", "leanix", "mealie", "media-downloader", "microsoft", "nextcloud", "plane", "portainer", "repository-manager", "searxng", "servicenow", "stirlingpdf", "systems-manager", "tunnel-manager", "vector", "wger"]
        #     """
        #     from universal_skills.skill_utilities import resolve_mcp_reference
        #     from pathlib import Path

        #     # Try direct name, then with -mcp suffix, then with -agent suffix
        #     # resolve_mcp_reference handles extension mapping logic
        #     queries = [
        #         f"{agent_template}.md",
        #         f"{agent_template}-api.md",
        #         f"{agent_template}-mcp.md",
        #         f"{agent_template}-agent.md",
        #     ]

        #     md_path = None
        #     for query in queries:
        #         md_path = resolve_mcp_reference(query)
        #         if md_path:
        #             break

        #     if not md_path:
        #         return f"Error: Reference documentation for '{agent_template}' not found. Tried: {', '.join(queries)}"

        #     path = Path(md_path)
        #     if not path.exists():
        #         return f"Error: Documentation file {md_path} does not exist."

        #     return path.read_text(encoding="utf-8")

        # @agent.tool
        # async def spawn_agent(
        #     ctx: RunContext[Any],
        #     agent_template: Optional[str] = None,
        #     prompt: str = "Please assist.",
        #     mcp_config: Optional[str] = None,
        #     mcp_url: Optional[str] = None,
        #     enabled_tools: Optional[List[str]] = None,
        #     custom_skills_directory: Optional[str] = None,
        #     name: Optional[str] = None,
        #     system_prompt: Optional[str] = None,
        #     provider: Optional[str] = None,
        #     model_id: Optional[str] = None,
        #     base_url: Optional[str] = None,
        #     api_key: Optional[str] = None,
        #     enable_skills: bool = True,
        #     enable_universal_tools: bool = True,
        # ) -> str:
        #     """
        #     Spawn a new Pydantic AI sub-agent dynamically to handle a specific task.
        #     Use this when you need to delegate a task to an agent with a specialized external configuration.

        #     Args:
        #         agent_template: Pre-defined agent configuration from the mcp-client skill. Options: ["adguard-home", "ansible-tower", "archivebox", "arr", "audio-transcriber", "container-manager", "documentdb", "github", "gitlab", "jellyfin", "leanix", "mealie", "media-downloader", "microsoft", "nextcloud", "plane", "portainer", "repository-manager", "searxng", "servicenow", "stirlingpdf", "systems-manager", "tunnel-manager", "vector", "wger"]
        #         prompt: The instruction to send to the spawned agent.
        #         mcp_config: Path to a specific MCP config JSON file or a filename from the mcp-client skill's references.
        #         enabled_tools: Optional list of tool tags to enable (e.g., ["BRANCHESTOOL", "COMMITSTOOL"]). All others will be disabled if the config supports filtering. Use 'get_mcp_reference' to discover available tags for a template.
        #         name: Override the agent's name.
        #         system_prompt: Override the agent's system prompt.
        #         enable_skills: Whether to load general skills (Workspace, A2A, etc.).
        #         enable_universal_tools: Whether to register universal tools (Workspace, A2A, Scheduler).
        #     """
        #     from .agent_utilities import (
        #         create_agent,
        #         get_mcp_config_path,
        #         get_system_prompt_from_reference,
        #     )
        #     from universal_skills.skill_utilities import resolve_mcp_reference
        #     import os

        #     AGENT_TEMPLATES = {
        #         "adguard-home": {
        #             "name": "AdGuard Home Agent",
        #             "mcp_config": "adguard-home-agent.json",
        #         },
        #         "ansible-tower": {
        #             "name": "Ansible Tower Agent",
        #             "mcp_config": "ansible-tower-mcp.json",
        #         },
        #         "archivebox": {
        #             "name": "ArchiveBox Agent",
        #             "mcp_config": "archivebox-api.json",
        #         },
        #         "arr": {
        #             "name": "Arr Agent",
        #             "mcp_config": "arr-mcp.json",
        #         },
        #         "audio-transcriber": {
        #             "name": "Audio Transcriber Agent",
        #             "mcp_config": "audio-transcriber-mcp.json",
        #         },
        #         "container-manager": {
        #             "name": "Container Manager Agent",
        #             "mcp_config": "container-manager-mcp.json",
        #         },
        #         "documentdb": {
        #             "name": "DocumentDB Agent",
        #             "mcp_config": "documentdb-mcp.json",
        #         },
        #         "github": {
        #             "name": "GitHub Agent",
        #             "mcp_config": "github-mcp.json",
        #         },
        #         "gitlab": {
        #             "name": "GitLab Agent",
        #             "mcp_config": "gitlab-api.json",
        #         },
        #         "jellyfin": {
        #             "name": "Jellyfin Agent",
        #             "mcp_config": "jellyfin-mcp.json",
        #         },
        #         "leanix": {
        #             "name": "LeanIX Agent",
        #             "mcp_config": "leanix-agent.json",
        #         },
        #         "mealie": {
        #             "name": "Mealie Agent",
        #             "mcp_config": "mealie-mcp.json",
        #         },
        #         "media-downloader": {
        #             "name": "Media Downloader Agent",
        #             "mcp_config": "media-downloader-mcp.json",
        #         },
        #         "microsoft": {
        #             "name": "Microsoft 365 Agent",
        #             "mcp_config": "microsoft-agent.json",
        #         },
        #         "nextcloud": {
        #             "name": "Nextcloud Agent",
        #             "mcp_config": "nextcloud-agent.json",
        #         },
        #         "plane": {
        #             "name": "Plane Agent",
        #             "mcp_config": "plane-mcp.json",
        #         },
        #         "portainer": {
        #             "name": "Portainer Agent",
        #             "mcp_config": "portainer-agent.json",
        #         },
        #         "repository-manager": {
        #             "name": "Repository Manager Agent",
        #             "mcp_config": "repository-manager.json",
        #         },
        #         "searxng": {
        #             "name": "SearXNG Agent",
        #             "mcp_config": "searxng-mcp.json",
        #         },
        #         "servicenow": {
        #             "name": "ServiceNow Agent",
        #             "mcp_config": "servicenow-api.json",
        #         },
        #         "stirlingpdf": {
        #             "name": "Stirling PDF Agent",
        #             "mcp_config": "stirlingpdf-agent.json",
        #         },
        #         "systems-manager": {
        #             "name": "Systems Manager Agent",
        #             "mcp_config": "systems-manager-mcp.json",
        #         },
        #         "tunnel-manager": {
        #             "name": "Tunnel Manager Agent",
        #             "mcp_config": "tunnel-manager-mcp.json",
        #         },
        #         "vector": {
        #             "name": "Vector Agent",
        #             "mcp_config": "vector-mcp.json",
        #         },
        #         "wger": {
        #             "name": "Wger Fitness Agent",
        #             "mcp_config": "wger-agent.json",
        #         },
        #     }

        #     template_config = (
        #         AGENT_TEMPLATES.get(agent_template, {}) if agent_template else {}
        #     )

        #     # Priority: 1. Explicit arguments 2. Template Database (Dynamic Prompt) 3. OS Fallbacks
        #     final_name = name or template_config.get("name", "SpawnedAgent")

        #     # Dynamic prompt lookup for templates
        #     dynamic_prompt = (
        #         get_system_prompt_from_reference(agent_template)
        #         if agent_template
        #         else None
        #     )
        #     final_system_prompt = (
        #         system_prompt
        #         or dynamic_prompt
        #         or template_config.get("system_prompt", "You are a helpful assistant.")
        #     )

        #     # Resolve MCP configuration
        #     final_mcp_config = mcp_config or template_config.get("mcp_config")
        #     if final_mcp_config:
        #         resolved_path = resolve_mcp_reference(final_mcp_config)
        #         if resolved_path:
        #             final_mcp_config = resolved_path
        #         else:
        #             final_mcp_config = get_mcp_config_path(final_mcp_config)

        #     # Tailored tool loading: Temporarily inject tools to enable into os.environ
        #     # This works because load_mcp_servers expands env vars from the current environment.
        #     original_env = {}
        #     if enabled_tools:
        #         for tool in enabled_tools:
        #             original_env[tool] = os.environ.get(tool)
        #             os.environ[tool] = "True"

        #         # We assume if enabled_tools is provided, we might want to disable others
        #         # that are typically defaulted to True in the JSON configs but support env override.
        #         # However, since we don't know all possible tools, we rely on the user to
        #         # specify what they want. For GitLab specifically, others are defaulted to True
        #         # in the reference JSON, but can be overridden by env vars.
        #         # To truly "only" enable specific ones, the JSON would need ${VAR:-True}
        #         # but if we set the other vars to "False", they will be overridden.

        #     # Create the sub-agent
        #     try:
        #         sub_agent = create_agent(
        #             provider=provider
        #             or template_config.get("provider")
        #             or os.getenv("PROVIDER", "openai"),
        #             model_id=model_id
        #             or template_config.get("model_id")
        #             or os.getenv("MODEL_ID", "gpt-4o"),
        #             base_url=base_url
        #             or template_config.get("base_url")
        #             or os.getenv("LLM_BASE_URL"),
        #             api_key=api_key
        #             or template_config.get("api_key")
        #             or os.getenv("LLM_API_KEY"),
        #             mcp_config=final_mcp_config,
        #             mcp_url=mcp_url
        #             or template_config.get("mcp_url")
        #             or os.getenv("MCP_URL"),
        #             name=final_name,
        #             system_prompt=final_system_prompt,
        #             # elicitation_callback is now handled client-side per agent
        #             debug=True,
        #             enable_skills=enable_skills,
        #             enable_universal_tools=enable_universal_tools,
        #         )
        #     finally:
        #         # Restore environment
        #         for tool, original_val in original_env.items():
        #             if original_val is None:
        #                 del os.environ[tool]
        #             else:
        #                 os.environ[tool] = original_val

        #     # Execute the prompt
        #     logger.info(
        #         f"Sub-agent run started: {agent_template}. Prompt: {prompt[:100]}..."
        #     )

        #     # Retrieve elicitation queue from current context deps if available
        #     elicitation_queue = None
        #     if hasattr(ctx, "deps") and isinstance(ctx.deps, dict):
        #         elicitation_queue = ctx.deps.get("elicitation_queue")

        #     try:
        #         # Pass elicitation_queue to sub-agent to preserve context
        #         result = await sub_agent.run(
        #             prompt, deps={"elicitation_queue": elicitation_queue}
        #         )
        #         logger.info(f"Sub-agent run completed: {agent_template}")

        #         # Safe access to result data
        #         data = getattr(result, "data", None)
        #         if data is None:
        #             # Fallback for some versions or result types
        #             data = getattr(result, "result", result)
        #         return str(data)
        #     except Exception as e:
        #         import traceback

        #         error_msg = f"Error in sub-agent run: {e}\n{traceback.format_exc()}"
        #         logger.error(error_msg)
        #         return error_msg
