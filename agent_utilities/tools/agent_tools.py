"""CONCEPT:ECO-4.0"""

import logging
from typing import Any

from pydantic import BaseModel
from pydantic_ai import RunContext

from agent_utilities.harness.tracing import trace

from ..models import AgentDeps
from .versioning import tool_version

logger = logging.getLogger(__name__)


class SubAgentResponse(BaseModel):
    """Structured response from a specialized sub-agent execution."""

    id: str
    result: str
    success: bool


@trace(name="invoke_specialized_agent", trace_type="TOOL")
@tool_version("1.0.0")
async def invoke_specialized_agent(
    ctx: RunContext[AgentDeps],
    agent_name: str,
    prompt: str,
    model: str | None = None,
) -> str:
    """Invoke a specialized sub-agent for a targeted task.

    This tool handles local adaptive_agent_router (Prompts, MCP) and remote A2A peers
    discovered via the Knowledge Graph.

    Args:
        ctx: The agent run context.
        agent_name: The name of the expert node to invoke (e.g., 'python', 'github').
        prompt: The specific instruction or task for the specialist.
        model: Optional model override for the sub-agent.

    Returns:
        The result string from the sub-agent execution.

    """
    from agent_utilities.agent.discovery import discover_all_specialists
    from agent_utilities.protocols.a2a import A2AClient

    adaptive_agent_router = discover_all_specialists()
    agent_info = next((a for a in adaptive_agent_router if a.name == agent_name), None)

    if not agent_info:
        # Try fuzzy match
        agent_info = next(
            (
                a
                for a in adaptive_agent_router
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

        from agent_utilities.agent.factory import create_agent

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


@trace(name="list_available_agents", trace_type="TOOL")
@tool_version("1.0.0")
async def list_available_agents(ctx: RunContext[Any]) -> list[str]:
    """List all specialized expert roles currently registered in the Knowledge Graph.

    Args:
        ctx: The agent run context.

    Returns:
        A list of available specialist names and descriptions.

    """
    from agent_utilities.agent.discovery import discover_all_specialists

    adaptive_agent_router = discover_all_specialists()
    return [
        f"{s.name}: {s.description} (Source: {s.source})" for s in adaptive_agent_router
    ]


@trace(name="share_reasoning", trace_type="TOOL")
@tool_version("1.0.0")
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


async def reach_user(
    ctx: RunContext[Any],
    message: str,
    wait_for_reply: bool = False,
) -> str:
    """Message the human operator on their last-active channel (e.g. Telegram).

    CONCEPT:ECO-4.53 — Universal agent reach_user tool over the messaging reach service.
    Use it to proactively tell the user something, or to ask a question and optionally wait
    for their reply. Routing follows the user's most-recently-used channel, falling back to
    the configured default; the send is governed by the ActionPolicy gate.

    Args:
        ctx: The agent run context.
        message: The text to send (supports the channel's rich text, e.g. Telegram HTML).
        wait_for_reply: When True, block until the user replies (or a timeout) and return
            their reply; when False, send and return immediately.

    Returns:
        The user's reply (if ``wait_for_reply``), else a short send-status string.
    """
    from agent_utilities.messaging.service import MessagingService

    service = MessagingService.instance()
    if wait_for_reply:
        reply = await service.reach_user_and_wait(
            message, source="agent", reason="agent asked the user"
        )
        return reply if reply else "(no reply received before timeout)"
    result = await service.reach_user(message, source="agent", reason="agent outreach")
    if result.success:
        return f"sent via {result.platform}"
    return f"send failed: {result.error}"
