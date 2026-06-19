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
@tool_version("1.1.0")
async def invoke_specialized_agent(
    ctx: RunContext[AgentDeps],
    agent_name: str,
    prompt: str,
    model: str | None = None,
) -> str:
    """Delegate a targeted task to a specialized sub-agent (local MCP/prompt or remote A2A).

    CONCEPT:ECO-4.77 — ONE delegation surface. This is a thin wrapper over the SAME
    orchestration core as ``graph_orchestrate(action="execute_agent", ...)`` —
    ``Orchestrator.execute_agent`` → ``run_agent`` — which resolves the specialist from the
    Knowledge Graph (server / skill / a2a) and runs it. There is no separate discovery /
    A2A / sub-agent-build path here (No-Legacy); both entrypoints converge on one core.

    Args:
        ctx: The agent run context.
        agent_name: The name of the expert node to invoke (e.g., 'python', 'github').
        prompt: The specific instruction or task for the specialist.
        model: Optional model override (currently advisory; the orchestrator resolves the
            model from the spawned agent's own configuration).

    Returns:
        The result string from the sub-agent execution.
    """
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
    from agent_utilities.orchestration.manager import Orchestrator

    engine = (
        getattr(ctx.deps, "knowledge_engine", None)
        or IntelligenceGraphEngine.get_active()
    )
    if engine is None:
        return "Error: IntelligenceGraphEngine not active for delegation."
    try:
        return await Orchestrator(engine).execute_agent(
            agent_name=agent_name, task=prompt
        )
    except Exception as e:  # noqa: BLE001
        return f"Error invoking sub-agent '{agent_name}': {e}"


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
