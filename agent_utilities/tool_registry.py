#!/usr/bin/python
"""Tool Registry Module.

This module provides a centralized system for registering agent tools. It handles
the aggregation of various domain-specific toolsets (workspace, memory, git, etc.)
and applies environment-based gating to control which tools are exposed to the agent.
"""

import os
from typing import Any

from pydantic_ai import Agent, RunContext

from .models import AgentDeps

__version__ = "0.2.40"


def register_agent_tools(agent: Agent, graph_bundle: tuple | None = None) -> None:
    """Central aggregator for registering all Agent OS tools.

    Groups tools by domain and applies environment-based gating using
    environment variables (e.g., WORKSPACE_TOOLS, GIT_TOOLS). If a graph_bundle
    is provided, the agent is configured as a graph orchestrator, restricting
    it to only use the 'run_graph_flow' tool for strict routing isolation.

    Args:
        agent: The Pydantic AI Agent instance to register tools for.
        graph_bundle: An optional tuple containing (graph, config) used to
            configure the agent as a graph orchestrator. Defaults to None.

    """
    # Late imports to avoid circularity during initialization
    # Default gating via environment variables
    from .base_utilities import to_boolean
    from .tools.a2a_tools import a2a_tools
    from .tools.agent_tools import (
        invoke_specialized_agent,
        list_available_agents,
        share_reasoning,
    )
    from .tools.browser import browser_tools
    from .tools.developer_tools import developer_tools
    from .tools.git_tools import git_tools
    from .tools.mcp_sync_tool import trigger_mcp_sync
    from .tools.onboarding_tools import onboarding_tools
    from .tools.scheduler_tools import scheduler_tools
    from .tools.self_improvement_tools import (
        propose_skills_from_history,
        query_experiment_results,
        run_self_improvement_cycle,
    )
    from .tools.workspace_tools import workspace_tools

    DEFAULT_WORKSPACE_TOOLS = to_boolean(
        string=os.environ.get("WORKSPACE_TOOLS", "True")
    )
    DEFAULT_A2A_TOOLS = to_boolean(string=os.environ.get("A2A_TOOLS", "True"))
    DEFAULT_SCHEDULER_TOOLS = to_boolean(
        string=os.environ.get("SCHEDULER_TOOLS", "True")
    )
    DEFAULT_GIT_TOOLS = to_boolean(string=os.environ.get("GIT_TOOLS", "True"))
    DEFAULT_BROWSER_TOOLS = to_boolean(string=os.environ.get("BROWSER_TOOLS", "True"))
    DEFAULT_DEVELOPER_TOOLS = to_boolean(
        string=os.environ.get("DEVELOPER_TOOLS", "True")
    )

    def _is_tool_registered(name: str) -> bool:
        """Check if a tool with the given name is already registered on the agent."""
        # 1. Check native function tools
        try:
            from pydantic_ai.toolsets.function import FunctionToolset

            for ts in agent.toolsets:
                if isinstance(ts, FunctionToolset):
                    if name in ts.tools:
                        return True
        except (ImportError, AttributeError):
            pass

        # 2. Check internal _function_toolset (legacy/internal fallback)
        try:
            if (
                hasattr(agent, "_function_toolset")
                and name in agent._function_toolset.tools
            ):
                return True
        except (AttributeError, TypeError):
            pass

        return False

    # 1. Graph Flow Tool (Orchestration)
    if graph_bundle:
        graph, config = graph_bundle

        if not _is_tool_registered("run_graph_flow"):

            @agent.tool
            async def run_graph_flow(
                ctx: RunContext[AgentDeps], prompt: str
            ) -> str | Any:
                """Execute a complex query through the graph orchestrator.

                The graph automatically classifies and routes the request to specialized
                domain nodes (e.g., Python Programmer, DevOps Engineer), executes the
                necessary steps in parallel or sequence, and synthesizes a final result.

                Args:
                    ctx: The run context containing agent dependencies.
                    prompt: The user query to be processed by the graph orchestrator.

                Returns:
                    The synthesized output from the graph execution or an error message.

                """
                eq = getattr(ctx.deps, "graph_event_queue", None) if ctx.deps else None
                from .graph_orchestration import run_graph

                # Forward runtime MCP toolsets and LLM config from AgentDeps so the graph uses alread-conencted servers and current credentials instead of None values baked in from the default config
                runtime_toolsets = (
                    getattr(ctx.deps, "mcp_toolsets", None) if ctx.deps else None
                )
                result = await run_graph(
                    graph,
                    config,
                    prompt,
                    eq=eq,
                    mcp_toolsets=runtime_toolsets or config.get("mcp_toolsets"),
                )

                if hasattr(result, "results"):
                    output = result.results.get("output", result.results)
                    if not output or str(output).lower() == "none":
                        return (
                            "The analysis completed, but no specific data was returned for your query. "
                            "This may happen if the target system has no matching resources, if synthesis failed, or if tools were not loaded correctly."
                        )
                    return str(output)
                return str(result)

        # STRICT ISOLATION: If we are a graph orchestrator, we ONLY have run_graph_flow.
        # We skip all other local tools to avoid confusion and force routing.
        return

    # Helper to register tools only if not already present
    def _safe_tool(func):
        name = getattr(func, "name", func.__name__)
        if not _is_tool_registered(name):
            agent.tool(func)

    # 2. Workspace Tools
    if DEFAULT_WORKSPACE_TOOLS:
        for tool in workspace_tools:
            _safe_tool(tool)

    # 4. Git & Worktree Tools
    if DEFAULT_GIT_TOOLS:
        for tool in git_tools:
            _safe_tool(tool)

    # 5. Developer Tools (Ported from code_puppy)
    if DEFAULT_DEVELOPER_TOOLS:
        for tool in developer_tools:
            _safe_tool(tool)

    # 6. A2A & Scheduler Tools
    if DEFAULT_A2A_TOOLS:
        for tool in a2a_tools:
            _safe_tool(tool)

    if DEFAULT_SCHEDULER_TOOLS:
        for tool in scheduler_tools:
            _safe_tool(tool)

    # 7. Browser Tools (Ported from code_puppy)
    if DEFAULT_BROWSER_TOOLS:
        for tool in browser_tools:
            _safe_tool(tool)

    # 8. Specialized Agent Tools
    _safe_tool(invoke_specialized_agent)
    _safe_tool(list_available_agents)
    _safe_tool(share_reasoning)

    # 9. Onboarding Tools
    for tool in onboarding_tools:
        _safe_tool(tool)

    # 10. MCP Management Tools
    _safe_tool(trigger_mcp_sync)

    # 11. Knowledge Graph & KB Tools
    from .tools.knowledge_tools import KB_TOOLS, knowledge_tools

    for tool in knowledge_tools:
        _safe_tool(tool)
    for tool in KB_TOOLS:
        _safe_tool(tool)

    # 12. Team Coordination Tools
    from .tools.team_tools import TEAM_TOOLS

    for tool in TEAM_TOOLS:
        _safe_tool(tool)

    # 13. Output Style Tools
    from .tools.style_tools import list_output_styles, set_output_style

    _safe_tool(set_output_style)
    _safe_tool(list_output_styles)

    from .tool_guard import apply_tool_guard_approvals

    # 14. Self-Improvement & Learning Tools

    _safe_tool(run_self_improvement_cycle)
    _safe_tool(propose_skills_from_history)
    _safe_tool(query_experiment_results)

    # 15. SDD & Pattern Tools
    from .tools.pattern_tools import pattern_tools
    from .tools.sdd_tools import sdd_tools

    for tool in sdd_tools:
        _safe_tool(tool)
    for tool in pattern_tools:
        _safe_tool(tool)

    # 16. Apply Security Guards
    apply_tool_guard_approvals(agent)
