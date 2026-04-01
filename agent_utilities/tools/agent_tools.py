import logging
from typing import Any, List, Optional
from pydantic import BaseModel
from pydantic_ai import RunContext

logger = logging.getLogger(__name__)


class SubAgentResponse(BaseModel):
    id: str
    result: str
    success: bool


async def invoke_specialized_agent(
    ctx: RunContext[Any], agent_name: str, prompt: str, model: Optional[str] = None
) -> str:
    """
    Invoke a specialized sub-agent (e.g. 'python', 'security') for a task.
    """

    # Logic to run a specific domain node in our graph
    # (Simplified for now, will integrate with master graph)
    return f"Sub-agent '{agent_name}' result for: {prompt[:50]}..."


async def list_available_agents(ctx: RunContext[Any]) -> List[str]:
    """List all specialized agents currently available in the system."""
    return [
        "python",
        "c",
        "cpp",
        "golang",
        "javascript",
        "typescript",
        "security",
        "qa",
        "mcp",
    ]


async def share_reasoning(ctx: RunContext[Any], reasoning: str) -> str:
    """
    Explicitly share the agent's internal reasoning or plan for a task.
    Useful for 'Thought' logs or before a major operation.
    """
    return f"Reasoning log: {reasoning}"
