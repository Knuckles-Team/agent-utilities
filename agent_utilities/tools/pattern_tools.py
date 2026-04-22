#!/usr/bin/python
"""Pattern-based Agent Tools.

This module provides tools that implement complex engineering patterns
like agentic manual testing and specialized subagent dispatch.
"""

from pydantic_ai import RunContext

from ..models import AgentDeps


async def run_manual_test(
    ctx: RunContext[AgentDeps], verification_goal: str, context: str = ""
) -> str:
    """Perform an agentic manual test to verify a behavior or condition.

    Uses a specialized subagent to execute verification steps (curl, shell, etc.).
    """
    from ..patterns.manual_test import run_agentic_manual_test

    return await run_agentic_manual_test(
        verification_goal=verification_goal, deps=ctx.deps, context=context
    )


pattern_tools = [
    run_manual_test,
]
