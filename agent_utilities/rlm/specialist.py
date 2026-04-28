import logging
from typing import Any

from pydantic_ai import RunContext

from .repl import RLMEnvironment

logger = logging.getLogger(__name__)

# To make this discoverable as a specialist
# we would normally register it in the registry or expose it as a toolset.
# Here we define the specialist tool.


async def recursive_reasoner_tool(
    ctx: RunContext[Any], prompt: str, context_data: str = ""
) -> str:
    """
    A recursive reasoning specialist that can handle massive context and unbounded logical depth.
    Use this when you have huge amounts of data (like full codebase analysis or large DB dumps)
    that would blow up a normal context window.

    Args:
        prompt: The specific analytical task to perform.
        context_data: The massive string or JSON data to reason over.
    """
    logger.info(
        f"RecursiveReasonerSpecialist invoked with {len(context_data)} chars of context."
    )
    env = RLMEnvironment(context=context_data)
    result = await env.run_full_rlm(prompt)
    return result


class RecursiveReasonerSpecialist:
    """
    A wrapper class to allow seamless integration into the pydantic-graph executor.
    """

    def __init__(self, repl: RLMEnvironment | None = None, system_prompt: str = ""):
        self.repl = repl or RLMEnvironment()
        self.system_prompt = system_prompt

    async def run_with_context(self, context_data: Any, task_prompt: str) -> str:
        self.repl.vars["context"] = context_data
        return await self.repl.run_full_rlm(task_prompt)
