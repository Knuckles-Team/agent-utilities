import os
from typing import Any, Optional, Union
from pydantic_ai import Agent, RunContext

from .models import AgentDeps

__version__ = "0.2.39"


def register_agent_tools(agent: Agent, graph_bundle: Optional[tuple] = None):
    """
    Central aggregator for registering all Agent OS tools.
    Groups tools by domain and applies environment-based gating.
    """
    # Late imports to avoid circularity during initialization
    from .tools.workspace_tools import workspace_tools
    from .tools.memory_tools import memory_tools
    from .tools.scheduler_tools import scheduler_tools
    from .tools.a2a_tools import a2a_tools
    from .tools.git_tools import git_tools
    from .tools.developer_tools import developer_tools
    from .tools.agent_tools import (
        invoke_specialized_agent,
        list_available_agents,
        share_reasoning,
    )
    from .tools.browser import browser_tools
    from .tools.onboarding_tools import onboarding_tools

    # Default gating via environment variables
    from .base_utilities import to_boolean

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
    """
    Central aggregator for registering all Agent OS tools.
    Groups tools by domain and applies environment-based gating.
    """

    # 1. Graph Flow Tool (Orchestration)
    if graph_bundle:
        graph, config = graph_bundle

        @agent.tool
        async def run_graph_flow(
            ctx: RunContext[AgentDeps], prompt: str
        ) -> Union[str, Any]:
            """
            Execute a complex query through the graph orchestrator.
            The graph automatically classifies and routes your request to specialized domain nodes.
            """
            eq = getattr(ctx.deps, "graph_event_queue", None)
            from .graph_orchestration import run_graph

            result = await run_graph(graph, config, prompt, eq=eq)
            return str(result.get("results", result))

    # 2. Workspace Tools
    if DEFAULT_WORKSPACE_TOOLS:
        for tool in workspace_tools:
            agent.tool(tool)

    # 3. Memory Tools
    for tool in memory_tools:
        agent.tool(tool)

    # 4. Git & Worktree Tools
    if DEFAULT_GIT_TOOLS:
        for tool in git_tools:
            agent.tool(tool)

    # 5. Developer Tools (Ported from code_puppy)
    if DEFAULT_DEVELOPER_TOOLS:
        for tool in developer_tools:
            agent.tool(tool)

    # 6. A2A & Scheduler Tools
    if DEFAULT_A2A_TOOLS:
        for tool in a2a_tools:
            agent.tool(tool)

    if DEFAULT_SCHEDULER_TOOLS:
        for tool in scheduler_tools:
            agent.tool(tool)

    # 7. Browser Tools (Ported from code_puppy)
    if DEFAULT_BROWSER_TOOLS:
        for tool in browser_tools:
            agent.tool(tool)

    # 8. Specialized Agent Tools
    agent.tool(invoke_specialized_agent)
    agent.tool(list_available_agents)
    agent.tool(share_reasoning)

    # 9. Onboarding Tools
    for tool in onboarding_tools:
        agent.tool(tool)
