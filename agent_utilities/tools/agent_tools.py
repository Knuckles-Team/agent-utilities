import logging
from typing import Any

from pydantic import BaseModel
from pydantic_ai import RunContext

from ..agent_factory import create_agent
from ..discovery import discover_all_specialists
from ..models import AgentDeps

logger = logging.getLogger(__name__)


class SubAgentResponse(BaseModel):
    """Structured response from a specialized sub-agent execution."""

    id: str
    result: str
    success: bool


async def invoke_specialized_agent(
    ctx: RunContext[AgentDeps],
    agent_name: str,
    prompt: str,
    model: str | None = None,
) -> str:
    """Invoke a specialized sub-agent for a targeted task.

    This tool handles local specialists (Prompts, MCP) and remote A2A peers
    discovered via the Knowledge Graph.

    Args:
        ctx: The agent run context.
        agent_name: The name of the expert node to invoke (e.g., 'python', 'github').
        prompt: The specific instruction or task for the specialist.
        model: Optional model override for the sub-agent.

    Returns:
        The result string from the sub-agent execution.

    """
    from ..a2a import A2AClient

    specialists = discover_all_specialists()
    agent_info = next((a for a in specialists if a.name == agent_name), None)

    if not agent_info:
        # Try fuzzy match
        agent_info = next(
            (
                a
                for a in specialists
                if a.tag == agent_name or a.name.lower() == agent_name.lower()
            ),
            None,
        )

    if not agent_info:
        return f"Error: Specialist '{agent_name}' not found in the Knowledge Graph."

    logger.info(
        f"Invoking specialized agent '{agent_name}' (source: {agent_info.source})"
    )

    if agent_info.source == "a2a":
        # Remote A2A Execution
        client = A2AClient(ssl_verify=getattr(ctx.deps, "ssl_verify", True))
        res = await client.execute_task(agent_info.url, prompt)
        return str(res)

    elif agent_info.source in ["mcp", "prompt", "local_mcp"]:
        # Local Execution (MCP or Prompt)
        # We reuse the existing MCP toolsets from the current context
        mcp_toolsets = getattr(ctx.deps, "mcp_toolsets", [])

        # Filter toolsets for this specific agent if it's an MCP agent
        target_toolsets = mcp_toolsets
        if agent_info.mcp_server:
            target_toolsets = [
                ts
                for ts in mcp_toolsets
                if getattr(ts, "id", getattr(ts, "name", "")) == agent_info.mcp_server
            ]

        sub_agent, _ = create_agent(
            name=agent_info.name,
            model_id=model or ctx.deps.model_id,
            mcp_toolsets=target_toolsets,
            system_prompt=agent_info.description,  # Use description as system prompt if it's a prompt agent
            enable_skills=True,
            enable_universal_tools=False,  # Avoid recursive tool registration
        )

        try:
            res = await sub_agent.run(prompt, deps=ctx.deps)
            return str(res.output)
        except Exception as e:
            return f"Error invoking sub-agent '{agent_name}': {e}"

    return f"Error: Unsupported agent source '{agent_info.source}' for '{agent_name}'"


async def list_available_agents(ctx: RunContext[Any]) -> list[str]:
    """List all specialized expert roles currently registered in the Knowledge Graph.

    Args:
        ctx: The agent run context.

    Returns:
        A list of available specialist names and descriptions.

    """
    specialists = discover_all_specialists()
    return [f"{s.name}: {s.description} (Source: {s.source})" for s in specialists]


async def share_reasoning(ctx: RunContext[Any], reasoning: str) -> str:
    """Explicitly share the agent's internal reasoning or plan for a task.

    This tool is used to provide transparency before a major operation
    or to log complex decision-making steps.

    Args:
        ctx: The agent run context.
        reasoning: The markdown-formatted reasoning string.

    Returns:
        A formatted reasoning log entry.

    """
    # In the future, this could also store a ReasoningTraceNode in the KG
    return f"Reasoning log: {reasoning}"
