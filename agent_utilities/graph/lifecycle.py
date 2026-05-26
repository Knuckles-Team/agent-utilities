#!/usr/bin/python
from __future__ import annotations

"""Graph Lifecycle Steps.

CONCEPT:ORCH-1.0

Session lifecycle, policy enforcement, and human-in-the-loop gates.
Extracted from the monolithic steps.py for maintainability.

- ``usage_guard_step``: Token/cost budget and safety policy check.
- ``onboarding_step``: New-project workspace bootstrap.
- ``approval_gate_step``: Human approval checkpoint for high-risk actions.
- ``_emit_node_lifecycle``: Shared helper for node-level tracing events.
"""


import logging
from typing import Any

from pydantic_ai import Agent
from pydantic_graph import End

try:
    from pydantic_graph.step import StepContext
except ImportError:
    from pydantic_graph.beta import StepContext

from ..models import GraphResponse
from .config_helpers import emit_graph_event, load_specialized_prompts
from .state import GraphDeps, GraphState

logger = logging.getLogger(__name__)

__all__ = [
    "_emit_node_lifecycle",
    "usage_guard_step",
    "onboarding_step",
    "approval_gate_step",
]


def _emit_node_lifecycle(eq, node_name: str, event: str, **kwargs):
    """Emit a node lifecycle event for graph tracing.

    Provides consistent node_start/node_complete events across all step
    functions so the full execution path is visible in both the UI
    sideband stream and server-side structured logs.

    Args:
        eq: The asyncio event queue (may be None).
        node_name: The step function identifier (e.g. 'router', 'verifier').
        event: Either 'node_start' or 'node_complete'.
        **kwargs: Additional metadata (e.g. next_node, duration_ms).

    """
    emit_graph_event(eq, event, node_id=node_name, **kwargs)


async def usage_guard_step(
    ctx: StepContext[GraphState, GraphDeps, Any],
) -> str | None:
    """Evaluate session safety and usage policies.

    Checks the current token usage and estimated cost against safety
    thresholds. Optionally runs a security classifier to ensure the
    user query complies with established safety policies.

    Args:
        ctx: The pydantic-graph step context.

    Returns:
        The next node identifier ('router' or 'error_recovery').

    """
    logger.info(
        f"[LAYER:GRAPH:USAGE_GUARD] Handling query: '{ctx.state.query[:50]}...'"
    )
    _emit_node_lifecycle(ctx.deps.event_queue, "usage_guard", "node_start")
    # Token / cost budget check
    usage = ctx.state.session_usage
    cost_limit = 5.0
    token_limit = 500000

    if usage.estimated_cost_usd > cost_limit or usage.total_tokens > token_limit:
        logger.warning(
            f"UsageGuard: Safety limits reached! Cost: ${usage.estimated_cost_usd:.2f}, Tokens: {usage.total_tokens}"
        )
        emit_graph_event(
            ctx.deps.event_queue,
            event_type="safety_warning",
            message=f"Session usage has exceeded safety limits. Current cost: ${usage.estimated_cost_usd:.2f}",
            usage=usage.model_dump(),
        )

    # Policy enforcement (Optional, based on tool_guard_mode)
    if ctx.deps.tool_guard_mode == "off":
        logger.info("UsageGuard: Tool guard mode is OFF. Bypassing policy check.")
        _emit_node_lifecycle(
            ctx.deps.event_queue, "usage_guard", "node_complete", next_node="router"
        )
        return "router"

    safety_policy = load_specialized_prompts("safety_policy")
    checker = Agent(
        model=ctx.deps.router_model,
        system_prompt=(
            "You are a security guard. Evaluate the user query against the "
            "following safety policy and output 'PASS' if the query is safe, "
            "or a brief error if it violates policy.\n\n"
            f"{safety_policy}"
        ),
    )

    try:
        logger.info("UsageGuard: Starting policy check...")
        res = await checker.run(
            f"Check for policy violations in query: {ctx.state.query}"
        )
        result_text = res.output.upper()
        logger.info(f"UsageGuard: Policy check completed. Result: {result_text}")

        if "PASS" in result_text:
            return "router"

        ctx.state.error = f"Policy violation: {res.output}"
        return "error_recovery"
    except Exception as e:
        logger.error(f"UsageGuard failed: {e}")
        return "router"  # Fail open for now, or route to error if preferred


async def onboarding_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> str | End[GraphResponse]:
    """Initialize a new agent workspace and bootstrap core project files.

    Handles the creation of the standard agent directory structure and
    essential files for a new project.
    a new project.

    Args:
        ctx: The pydantic-graph step context.

    Returns:
        A terminal End state with a GraphResponse containing the onboarding
        status, or 'error_recovery' on failure.

    """
    logger.info("Onboarding: Initializing project...")
    from ..tools.onboarding_tools import bootstrap_project

    try:
        # Wrap bootstrap_project call
        result = await bootstrap_project(ctx)
        return End(
            GraphResponse(
                status="completed",
                results={"onboarding": result},
                metadata={"type": "onboarding"},
            )
        )
    except Exception as e:
        logger.error(f"Onboarding failed: {e}")
        return "error_recovery"


async def approval_gate_step(
    ctx: StepContext[GraphState, GraphDeps, Any],
) -> Any:
    """Implement a human-in-the-loop checkpoint for high-risk actions.

    Pauses graph execution if explicit human intervention, confirmation,
    or plan approval is required by the operational mode or agent state.
    Uses the safety_guard policy to classify pending tool actions.

    Args:
        ctx: The pydantic-graph step context.

    Returns:
        The original inputs to pass to the next node if approved, or
        'router' if redirection feedback is provided.

    """
    if ctx.state.mode != "plan" and not ctx.state.human_approval_required:
        return ctx.inputs

    safety_guard_prompt = load_specialized_prompts("safety_guard")
    logger.info(
        f"Approval Gate: Pausing for user review. "
        f"Safety guard policy loaded ({len(safety_guard_prompt)} chars)."
    )
    ctx.state.human_approval_required = True

    if ctx.state.user_redirect_feedback:
        logger.info(
            "Approval Gate: Captured redirection feedback. Returning to router."
        )
        ctx.state.human_approval_required = False
        return "router"
    return ctx.inputs
