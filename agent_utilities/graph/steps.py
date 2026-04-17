#!/usr/bin/python
# coding: utf-8
"""Graph Steps Module.

This module contains the functional step definitions for the pydantic-graph
orchestrator. It implements the primary node logic for routing, planning,
execution, and synchronization across specialized agent layers.

The orchestration follows a Hierarchical State Machine (HSM) architecture:
- Level 0: Root Graph Orchestration (Router, Planner, Dispatcher, Verifier)
- Level 1: Superstates (Specialist Agents like Python Programmer, DevOps, etc.)
- Level 2: Substates (Agent Internal Loop, multi-turn tool iteration)
- Level 3: Leaf States (Atomic MCP Tool Calls via stdio/HTTP)

Each step function is designed to be atomic and returns the identifier of the
next node to execute, enabling flexible and resilient graph flows.
"""

from __future__ import annotations

import os
import re
import logging
import asyncio
import subprocess
from pathlib import Path

from typing import Any, List


from pydantic_ai import Agent
from pydantic_graph import End


from ..models import (
    ExecutionStep,
    ParallelBatch,
    GraphPlan,
    GraphResponse,
)

from ..workspace import CORE_FILES, load_workspace_file

from .config_helpers import (
    emit_graph_event,
    load_specialized_prompts,
    load_mcp_config,
    load_node_agents_registry,
    get_discovery_registry,
)

from .hsm import (
    assert_state_valid,
    StateInvariantError,
)

from .executor import (
    _execute_specialized_step,
    _execute_domain_logic,
    _execute_dynamic_mcp_agent,
    _execute_agent_package_logic,
)

from .graph_models import ValidationResult

from pydantic_graph.beta import StepContext

from .state import GraphState, GraphDeps

logger = logging.getLogger(__name__)

lock = asyncio.Lock()


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


async def researcher_step(
    ctx: StepContext[GraphState, GraphDeps, ExecutionStep | str],
) -> str:
    """Execute deep discovery across the workspace, codebase, and web.

    This node acts as a unified researcher that gathers the technical
    context required for system architecture or task planning.

    Args:
        ctx: The pydantic-graph step context, potentially containing
            a specific structured research question.

    Returns:
        The ID of the next node to execute ('execution_joiner').

    """
    logger.info("Researcher: Triangulating context...")
    unified_context = await fetch_unified_context()

    # If the dispatcher sent a specific question, use it as the prompt
    step_input = ctx.inputs
    research_query = ctx.state.query
    if isinstance(step_input, ExecutionStep) and step_input.input_data:
        if isinstance(step_input.input_data, dict):
            research_query = step_input.input_data.get("question", research_query)
        elif isinstance(step_input.input_data, str):
            research_query = step_input.input_data

    # Intelligent Researcher Sub-Agents (for internal reasoning)
    researcher_prompt = load_specialized_prompts("researcher")

    from ..tools.developer_tools import project_search
    from ..tools.workspace_tools import read_workspace_file

    researcher = Agent(
        model=ctx.deps.agent_model,
        system_prompt=(f"{researcher_prompt}\n\nWorkspace Context: {unified_context}"),
        tools=[project_search, read_workspace_file],
        toolsets=list(ctx.deps.mcp_toolsets),
    )

    try:
        logger.info(
            f"Researcher: Starting execution. Query length: {len(research_query)}"
        )
        res = await researcher.run(research_query)
        logger.info("Researcher: Execution successfully completed.")
        ctx.state._update_usage(getattr(res, "usage", None))

        # Save to registry for other agents to consume
        node_uid = f"researcher_{ctx.state.step_cursor}"
        ctx.state.results_registry[node_uid] = str(res.output)

        return "execution_joiner"
    except Exception as e:
        logger.error(f"Researcher failed: {e}")
        return "error_recovery"


async def planner_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> GraphPlan | str:
    """Re-plan execution after a verification failure.

    This node is the dedicated re-planning entry point.  It is invoked by
    the verifier when a validation score is very low (< 0.4), indicating
    the *approach* — not just the execution — was wrong.

    Unlike the router (which generates an initial plan from scratch), the
    planner incorporates verification feedback, previous execution results,
    and architectural decisions to craft a corrected plan.

    Args:
        ctx: The pydantic-graph step context.

    Returns:
        ``'dispatcher'`` on successful re-plan, or ``'error_recovery'``
        on failure.

    """
    logger.info(
        f"Planner: Re-planning (attempt {ctx.state.verification_attempts}). "
        f"Feedback: {(ctx.state.validation_feedback or 'none')[:100]}"
    )

    emit_graph_event(
        ctx.deps.event_queue,
        "replanning_started",
        attempt=ctx.state.verification_attempts,
        feedback=ctx.state.validation_feedback or "",
    )

    planner_prompt = load_specialized_prompts("planner")
    unified_context = await fetch_unified_context()

    # Build a rich re-planning context from previous execution
    previous_results = "\n".join(
        f"- {node}: {str(val)[:300]}"
        for node, val in ctx.state.results_registry.items()
    )

    feedback_section = ""
    if ctx.state.validation_feedback:
        feedback_section = (
            f"### VERIFICATION FEEDBACK (CRITICAL)\n"
            f"The previous plan was rejected by the verifier. Address this:\n"
            f"{ctx.state.validation_feedback}\n\n"
        )

    error_section = ""
    if ctx.state.error:
        error_section = f"### PREVIOUS ERROR\n{ctx.state.error}\n\n"

    results_section = ""
    if previous_results:
        results_section = (
            f"### PREVIOUS EXECUTION RESULTS (for context)\n" f"{previous_results}\n\n"
        )

    from .executor import _get_domain_tools

    domain_tools, domain_toolsets = await _get_domain_tools("planner", ctx.deps)

    planner = Agent(
        model=ctx.deps.agent_model,
        output_type=GraphPlan,
        deps_type=GraphDeps,
        tools=domain_tools,
        toolsets=domain_toolsets,
        system_prompt=(
            f"{planner_prompt}\n\n"
            f"{feedback_section}"
            f"{error_section}"
            f"{results_section}"
            f"### ARCHITECTURAL DECISIONS\n{ctx.state.architectural_decisions}\n\n"
            f"### WORKSPACE CONTEXT\n{unified_context}"
        ),
    )

    try:
        res = await planner.run(
            f"Create a CORRECTED execution plan for: {ctx.state.query}\n\n"
            f"The previous approach failed. You MUST use a different strategy.",
            deps=ctx.deps,
        )
        ctx.state.plan = res.output
        ctx.state.step_cursor = 0
        ctx.state.needs_replan = False
        ctx.state.error = None
        logger.info(
            f"Planner: Re-plan generated with {len(ctx.state.plan.steps)} steps."
        )

        emit_graph_event(
            ctx.deps.event_queue,
            "replanning_completed",
            step_count=len(ctx.state.plan.steps),
        )

        return "dispatcher"
    except Exception as e:
        logger.error(f"Re-planning failed: {e}")
        return "error_recovery"


async def fetch_unified_context() -> str:
    """Aggregate essential workspace metadata for agent situational awareness.

    Collects agent registries, historical memory, and VCS state (git status)
    to provide agents with a holistic view of the current repository state
    without overwhelming the context window.

    Returns:
        A formatted markdown string containing truncated registry previews,
        recent memory, and git status.

    """
    mcp_agents = load_workspace_file(CORE_FILES["NODE_AGENTS"])
    if mcp_agents and len(mcp_agents.splitlines()) > 500:
        mcp_agents = "\n".join(mcp_agents.splitlines()[:500]) + "\n\n... (truncated)"

    memory = load_workspace_file(CORE_FILES["MEMORY"])
    # Run git status directly or use our new tool if available
    try:
        git_status = subprocess.check_output(
            ["git", "status", "--short"], text=True
        ).strip()
    except Exception:
        git_status = "Not a git repository or git not installed."

    return (
        f"### PROJECT CONTEXT (Agent OS)\n\n"
        f"**NODE_AGENTS.md (Specialist Registry):**\n{mcp_agents or '(empty)'}\n\n"
        f"**MEMORY.md (Historical Context):**\n{memory or '(empty)'}\n\n"
        f"**Git Status:**\n{git_status or '(clean)'}"
    )


async def router_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> GraphPlan | str | End[GraphResponse]:
    """Analyze the user query and select the optimal execution strategy.

    This is the primary topological decision point. It assesses whether
    the request requires architectural design, deep research, or
    direct specialist execution, and generates the initial GraphPlan using
    a high-level planning model.

    For trivial or conversational queries that don't need specialist
    execution, the router can return a direct response via the fast-path.

    Args:
        ctx: The pydantic-graph step context.

    Returns:
        A GraphPlan instance, a terminal node identifier on failure, or
        End[GraphResponse] for trivial queries that skip the pipeline.

    """
    deps = ctx.deps

    _emit_node_lifecycle(deps.event_queue, "router", "node_start")
    emit_graph_event(
        deps.event_queue,
        "routing_started",
        query=ctx.state.query,
    )
    logger.info(
        f"[LAYER:GRAPH:ROUTER] Routing started for query: '{ctx.state.query[:50]}...'"
    )

    # Track re-planning loops to prevent infinite cycles
    ctx.state.global_research_loops += 1
    if ctx.state.global_research_loops > 3:
        logger.error("Router: Max planning loops exceeded. Aborting.")
        return "error_recovery"

    # Fast-path: skip the full pipeline for trivial/conversational queries
    # that don't require specialist tools (e.g. "hello", "thanks", "what can you do?").
    query_lower = ctx.state.query.strip().lower()
    word_count = len(ctx.state.query.split())
    trivial_prefixes = (
        "hello",
        "hi ",
        "hey ",
        "thanks",
        "thank you",
        "ok",
        "bye",
        "what can you",
    )
    if word_count <= 6 and any(query_lower.startswith(p) for p in trivial_prefixes):
        logger.info(
            f"Router: Fast-path — trivial query detected ('{ctx.state.query[:40]}'). Generating direct response."
        )
        fast_agent = Agent(
            model=deps.router_model,
            system_prompt="You are a helpful assistant. Respond naturally and concisely.",
        )
        try:
            res = await fast_agent.run(ctx.state.query)
            emit_graph_event(
                deps.event_queue,
                "routing_completed",
                plan={},
                reasoning="fast-path: trivial query",
            )
            return End(
                GraphResponse(
                    status="completed",
                    results={"output": str(res.output)},
                    metadata={"fast_path": True, "domain": "conversational"},
                )
            )
        except Exception as e:
            logger.warning(
                f"Router: Fast-path failed ({e}). Falling back to full pipeline."
            )

    # Junction Pseudostate: Try static keyword routing first (saves LLM call)
    # If tag_prompts is empty (e.g. toolset loading failed due to missing env vars),
    # fall back to using the MCP registry directly for keyword-based routing.
    routing_tags = deps.tag_prompts
    if not routing_tags:
        registry = get_discovery_registry()
        routing_tags = {a.name: a.description for a in registry.agents}
        if routing_tags:
            logger.warning(
                f"Router: tag_prompts is empty, falling back to registry tags ({len(routing_tags)} tags)"
            )
    # Reset cursor for the new plan
    ctx.state.step_cursor = 0

    failure_context = ""
    if ctx.state.error:
        failure_context = f"### PREVIOUS FAILURE CONTEXT\nThe last attempt failed with the following error:\n{ctx.state.error}\nUse this information to update your plan. You may need more research or a different approach."

    try:
        unified_context = await fetch_unified_context()

        logger.info("[LAYER:GRAPH:ROUTER] Fetching specialist tags...")
        specialist_tags = deps.tag_prompts
        if not specialist_tags:
            registry = get_discovery_registry()
            specialist_tags = {a.name: a.description for a in registry.agents}
            if specialist_tags:
                logger.info(
                    f"[LAYER:GRAPH:ROUTER] Specialist tags loaded (count: {len(specialist_tags)}). Tags: {list(specialist_tags.keys())}"
                )

        step_info = "\n".join(
            [f"- {tag}: {desc}" for tag, desc in specialist_tags.items()]
        )
        logger.info(
            f"Router: Specialists count: {len(specialist_tags)}, Context length: {len(unified_context)}"
        )

        router_prompt = load_specialized_prompts("router")
        system_prompt_str = (
            f"{router_prompt}\n\n"
            f"### IMPORTANT: PLANNING ONLY MODE\n"
            f"You are a HIGH-LEVEL ARCHITECT. You DO NOT have access to functional tools (e.g. get_stack, Docker tools, etc.).\n"
            f"Your ONLY responsibility is to create the execution plan. DO NOT attempt to fulfill the query yourself.\n\n"
            f"### FAILURE CONTEXT\n{failure_context}\n\n"
            f"### AVAILABLE SPECIALIST NODES\n{step_info}\n\n"
            f"### PROJECT CONTEXT\n{unified_context}"
        )
        router_agent = Agent(
            model=deps.router_model,
            output_type=GraphPlan,
            system_prompt=system_prompt_str,
        )

        logger.info(
            f"[LAYER:GRAPH:ROUTER] Planning for query: '{ctx.state.query}' using model {deps.router_model}"
        )
        try:
            logger.debug(
                f"[LAYER:GRAPH:ROUTER] LLM Call Starting: system_prompt length={len(system_prompt_str)}"
            )
            async with router_agent.run_stream(ctx.state.query) as stream:
                plan_output = await asyncio.wait_for(
                    stream.get_output(), timeout=ctx.deps.router_timeout
                )
            logger.info(
                f"[LAYER:GRAPH:ROUTER] LLM Call Completed. Plan Reasoning: {plan_output.metadata.get('reasoning', 'N/A')}"
            )
            logger.info(
                f"[LAYER:GRAPH:ROUTER] Plan Step Count: {len(plan_output.steps)}"
            )
        except asyncio.TimeoutError:
            logger.warning("Router: LLM planning timed out. Escalating to fallbacks.")
            raise ValueError("LLM planning timed out")

        ctx.state._update_usage(stream.usage())
        ctx.state.plan = plan_output
        ctx.state.step_cursor = 0

        logger.info(f"Router: Generated plan with {len(ctx.state.plan.steps)} steps.")

        if len(ctx.state.plan.steps) == 0:
            logger.warning(
                "Router: LLM generated an empty plan. Escalating to fallbacks."
            )
            raise ValueError("LLM generated an empty plan")

        emit_graph_event(
            deps.event_queue,
            "routing_completed",
            plan=ctx.state.plan.model_dump(),
            reasoning=ctx.state.plan.metadata.get("reasoning", "Optimal dynamic plan"),
        )

        # Bridge: sync the initial plan to ACP native plan state.
        if deps.plan_sync:
            try:
                await deps.plan_sync(
                    "plan_created", ctx.state.plan.to_acp_plan_entries()
                )
            except Exception as sync_err:
                logger.warning(f"ACP plan sync failed: {sync_err}")

        return "dispatcher"
    except Exception as e:
        logger.error(f"Router planning failed: {e}")
        # Detailed logging for debugging
        if "res" in locals():
            logger.debug(f"Router raw response: {res}")

        ctx.state.error = f"Planning failed: {e}"
        return "__end__"


async def dispatcher_step(
    ctx: StepContext[GraphState, GraphDeps, str | list[ExecutionStep] | None],
) -> str | None:
    """Orchestrate the execution flow of a GraphPlan session.

    The dispatcher manages the state machine transitions between plan steps,
    handling state validation, integration of deferred user events, and
    identification of sequential vs parallel execution batches for barrier
    synchronization.

    Args:
        ctx: The pydantic-graph step context containing the current plan.

    Returns:
        The next node identifier (e.g., 'parallel_batch_processor', 'verifier')
        or None if the plan is complete and no synthesis is required.

    """
    logger.info(
        f"[LAYER:GRAPH:DISPATCHER] Transitioning. Current cursor: {ctx.state.step_cursor}"
    )

    # Infinite-loop guard: force-terminate if the graph has exceeded the
    # maximum allowed node transitions.
    ctx.state.node_transitions += 1
    if ctx.state.node_transitions > ctx.state.MAX_NODE_TRANSITIONS:
        logger.error(
            f"Dispatcher: Max node transitions ({ctx.state.MAX_NODE_TRANSITIONS}) exceeded. "
            f"Force-terminating to prevent infinite loop. "
            f"History tail: {ctx.state.node_history[-10:]}"
        )
        emit_graph_event(
            ctx.deps.event_queue,
            event_type="graph_force_terminated",
            reason="max_node_transitions_exceeded",
            transitions=ctx.state.node_transitions,
        )
        ctx.state.error = "Graph terminated: maximum node transitions exceeded."
        return "error_recovery"

    # HSM: State invariant check at transition boundary
    try:
        assert_state_valid(ctx.state, "dispatcher_step")
    except StateInvariantError as e:
        logger.error(f"State invariant violation: {e}")
        ctx.state.error = str(e)
        return "error_recovery"

    # HSM: Process deferred events (user follows-up received mid-execution)
    if ctx.state.deferred_events:
        for event in ctx.state.deferred_events:
            if event.get("type") == "user_followup":
                ctx.state.query += f"\n\nFollow up: {event.get('content', '')[:100]}"
                logger.info(
                    f"Dispatcher: Integrated deferred event: {event.get('content', '')[:100]}"
                )
        ctx.state.deferred_events.clear()

    # Context enrichment: route to memory_selection on the first entry so
    # historical context is available before any plan steps execute.
    if ctx.state.step_cursor == 0 and not ctx.state.exploration_notes:
        logger.info(
            "Dispatcher: First entry — routing to memory_selection for context enrichment."
        )
        return "memory_selection"

    # Phase-ordering guard: ensure research steps precede execution steps.
    # The LLM router may interleave them; we enforce discovery-first so that
    # research results are available to all execution specialists.
    _RESEARCH_NODES = {"researcher", "architect"}
    if ctx.state.step_cursor == 0 and len(ctx.state.plan.steps) > 1:
        research = [s for s in ctx.state.plan.steps if s.node_id in _RESEARCH_NODES]
        execution = [
            s for s in ctx.state.plan.steps if s.node_id not in _RESEARCH_NODES
        ]
        if research and execution:
            reordered = research + execution
            if [s.node_id for s in reordered] != [
                s.node_id for s in ctx.state.plan.steps
            ]:
                logger.info(
                    f"Dispatcher: Reordered plan — {len(research)} research step(s) "
                    f"moved before {len(execution)} execution step(s)."
                )
                ctx.state.plan.steps = reordered

        # Emit plan_created event for UI transparency
        emit_graph_event(
            ctx.deps.event_queue,
            "plan_created",
            steps=[
                {"node_id": s.node_id, "is_parallel": s.is_parallel}
                for s in ctx.state.plan.steps
            ],
            step_count=len(ctx.state.plan.steps),
        )

    logger.info(
        f"Dispatcher: Handling graph execution (Step {ctx.state.step_cursor}/{len(ctx.state.plan.steps)})"
    )
    if ctx.state.step_cursor >= len(ctx.state.plan.steps):
        # All plan steps have been executed.  Mark every step completed
        # and sync to ACP before handing off to the verifier.
        for step in ctx.state.plan.steps:
            step.status = "completed"
        if ctx.deps.plan_sync:
            try:
                await ctx.deps.plan_sync(
                    "step_completed", ctx.state.plan.to_acp_plan_entries()
                )
            except Exception:
                pass

        logger.info(
            f"Dispatcher: Plan completed. Results registry keys: {list(ctx.state.results_registry.keys())}"
        )
        if ctx.state.results_registry or ctx.state.exploration_notes:
            logger.info(
                f"Dispatcher: Results found in registry ({len(ctx.state.results_registry)} items). Routing to Verifier."
            )
            return "verifier"
        logger.warning(
            f"Dispatcher: Plan completed but NO execution results found in registry. State: routed_domain={ctx.state.routed_domain}"
        )
        return None

    # Sequential execution case (default for first step or non-parallel)
    current_step = ctx.state.plan.steps[ctx.state.step_cursor]

    # Internal/Meta nodes should remain as strings for direct routing
    meta_nodes = {
        "router",
        "planner",
        "onboarding",
        "error",
        "usage_guard",
        "memory_selection",
    }

    # Check if this is the start of a parallel batch
    if not current_step.is_parallel:
        ctx.state.step_cursor += 1
        ctx.state.pending_parallel_count = 1

        # If it's a meta-node, return the ID string directly
        if current_step.node_id in meta_nodes:
            logger.info(f"Dispatcher: Routing to meta-node: {current_step.node_id}")
            return current_step.node_id

        logger.info(
            f"Dispatcher: Dispatching sequential expert task: {current_step.node_id}"
        )
        emit_graph_event(
            ctx.deps.event_queue,
            "step_dispatched",
            node_id=current_step.node_id,
            step_index=ctx.state.step_cursor - 1,
            is_parallel=False,
        )

        # Bridge: mark the step as in_progress in ACP plan state.
        if ctx.deps.plan_sync:
            try:
                current_step.status = "in_progress"
                await ctx.deps.plan_sync(
                    "step_started", ctx.state.plan.to_acp_plan_entries()
                )
            except Exception:
                pass

        ctx.state.pending_batch = ParallelBatch(tasks=[current_step])
        return "parallel_batch_processor"

    # Gather all subsequent steps marked for parallel execution
    batch = []
    while (
        ctx.state.step_cursor < len(ctx.state.plan.steps)
        and ctx.state.plan.steps[ctx.state.step_cursor].is_parallel
    ):
        batch.append(ctx.state.plan.steps[ctx.state.step_cursor])
        ctx.state.step_cursor += 1

    # Set the barrier count
    ctx.state.pending_parallel_count = len(batch)
    logger.info(f"Dispatcher: Dispatching parallel batch of {len(batch)} tasks...")

    emit_graph_event(
        ctx.deps.event_queue,
        "batch_dispatched",
        nodes=[s.node_id for s in batch],
        batch_size=len(batch),
    )

    ctx.state.pending_batch = ParallelBatch(tasks=batch)
    return "parallel_batch_processor"


async def parallel_batch_processor(
    ctx: StepContext[GraphState, GraphDeps, Any],
) -> list[ExecutionStep]:
    """Retrieve and unpack a pending parallel execution batch from state.

    This node acts as a functional bridge to prevent passing large plan
    objects directly through graph edges, ensuring state-driven
    parallel dispatch using the pydantic-graph map() primitive.

    Args:
        ctx: The pydantic-graph step context.

    Returns:
        A list of ExecutionStep objects to be processed concurrently.

    """
    batch = ctx.state.pending_batch
    if not batch:
        logger.warning(
            "Parallel Processor: Called but NO pending_batch found in state!"
        )
        return []

    logger.info(f"Parallel Processor: Processing batch with {len(batch.tasks)} tasks.")
    ctx.state.pending_batch = None  # Clear the cache
    return batch.tasks


async def expert_executor_step(
    ctx: StepContext[GraphState, GraphDeps, ExecutionStep],
) -> str:
    """Execute a single specialist task with built-in retry and fallback logic.

    Routes task execution to the appropriate functional handler (e.g.,
    researcher_step, programmer nodes, or dynamic MCP specialists)
    based on the step's node_id. Implements per-node error recovery.

    Args:
        ctx: The pydantic-graph step context containing the targeted step details.

    Returns:
        The identifier of the appropriate joiner node for synchronization.

    """
    step = ctx.inputs
    node_id = step.node_id

    # Reset local retries for this new expert node
    ctx.state.current_node_retries = 0
    max_retries = 2

    while ctx.state.current_node_retries <= max_retries:
        try:
            logger.info(
                f"Expert Execution: Attempt {ctx.state.current_node_retries + 1}/{max_retries + 1} for node '{node_id}'"
            )

            # Check if this is a specialized prompt-based agent defined in the registry
            registry = get_discovery_registry()
            prompt_agents = {a.name for a in registry.agents if a.type == "prompt"}

            if node_id == "researcher":
                await researcher_step(ctx)

            elif node_id in prompt_agents:
                await _execute_specialized_step(ctx, node_id)

            # Generic MCP Step
            elif node_id == "mcp_server":
                domain = ""
                input_data = step.input_data
                if isinstance(input_data, dict):
                    domain = input_data.get("domain", "")
                await _execute_domain_logic(ctx, domain)

            elif node_id == "architect":
                await architect_step(ctx)
            elif node_id == "planner":
                await planner_step(ctx)
            elif node_id == "verifier":
                await verifier_step(ctx)

            # Professional Expert Steps
            elif node_id == "python_programmer":
                await python_programmer_step(ctx)
            elif (
                node_id == "typescript_programmer" or node_id == "javascript_programmer"
            ):
                await typescript_programmer_step(ctx)
            elif node_id == "rust_programmer":
                await rust_programmer_step(ctx)
            elif node_id == "golang_programmer":
                await golang_programmer_step(ctx)
            elif node_id == "security_auditor":
                await security_auditor_step(ctx)
            elif node_id == "qa_expert":
                await qa_expert_step(ctx)
            elif node_id == "debugger_expert" or node_id == "debugger_step":
                await debugger_expert_step(ctx)
            elif node_id == "ui_ux_designer" or node_id == "ui_ux_step":
                await ui_ux_designer_step(ctx)
            elif node_id == "devops_engineer" or node_id == "devops_step":
                await devops_engineer_step(ctx)
            elif node_id == "cloud_architect" or node_id == "cloud_step":
                await cloud_architect_step(ctx)
            elif node_id == "database_expert" or node_id == "database_step":
                await database_expert_step(ctx)
            elif node_id == "java_programmer":
                await java_programmer_step(ctx)
            elif node_id == "data_scientist":
                await data_scientist_step(ctx)
            elif node_id == "document_specialist":
                await document_specialist_step(ctx)
            elif node_id == "mobile_programmer":
                await mobile_programmer_step(ctx)
            elif node_id == "agent_engineer":
                await agent_engineer_step(ctx)
            elif node_id == "project_manager":
                await project_manager_step(ctx)
            elif node_id == "systems_manager":
                await systems_manager_step(ctx)
            elif node_id == "browser_automation":
                await browser_automation_step(ctx)
            elif node_id == "coordinator":
                await coordinator_step(ctx)
            elif node_id == "critique":
                await critique_step(ctx)

            # Dynamic Discovery Execution
            else:
                from .executor import agent_matches_node_id

                registry = load_node_agents_registry()

                mcp_agent = next(
                    (a for a in registry.agents if agent_matches_node_id(a, node_id)),
                    None,
                )

                if mcp_agent:
                    logger.info(
                        f"Expert: Matched node_id='{node_id}' -> agent='{mcp_agent.name}' (tag='{mcp_agent.tag}')"
                    )
                    await _execute_dynamic_mcp_agent(ctx, mcp_agent)
                else:
                    from ..discovery import discover_agents

                    discovered = discover_agents()
                    if node_id in discovered:
                        meta = discovered[node_id]
                        await _execute_agent_package_logic(ctx, node_id, meta)
                    else:
                        avail_tags = sorted(
                            set(a.tag for a in registry.agents if a.tag)
                        )[:15]
                        logger.error(
                            f"Node execution failed: Agent '{node_id}' not found in registry or discovery. "
                            f"Available tags: {avail_tags}..."
                        )
                        ctx.state.error = f"Agent '{node_id}' not found."
                        return "error_recovery"

            # Execution successful, clear error and break retry loop
            ctx.state.error = None
            break

        except Exception as e:
            logger.error(
                f"Execution failed for node '{node_id}' (Attempt {ctx.state.current_node_retries + 1}): {e}"
            )
            ctx.state.error = f"Node {node_id} failed: {e}"
            ctx.state.current_node_retries += 1

            if ctx.state.current_node_retries > max_retries:
                logger.warning(
                    f"Node '{node_id}' exhausted all retries. Escalating to re-planning."
                )
                ctx.state.needs_replan = True
                break

            # Short sleep before local retry
            await asyncio.sleep(1)

            # Return to appropriate joiner for synchronization
            if node_id in ["researcher", "architect", "planner"]:
                return "research_joiner"
            return "execution_joiner"


async def join_step(
    ctx: StepContext[GraphState, GraphDeps, Any],
) -> str | None:
    """Synchronize parallel execution paths using a thread-safe barrier count.

    Monitors the completion of concurrent tasks. Once the pending count
    reaches zero, it triggers a transition back to the dispatcher for the
    subsequent plan phase or to the router if failures require re-planning.

    Args:
        ctx: The pydantic-graph step context.

    Returns:
        The next node identifier ('dispatcher', 'router') or None while waiting.

    """
    async with lock:
        ctx.state.pending_parallel_count -= 1
        count = ctx.state.pending_parallel_count
        logger.debug(f"Join: Remaining parallel tasks = {count}")

        if count <= 0:
            logger.info("Join: All parallel tasks completed.")
            if ctx.state.needs_replan:
                logger.warning(
                    "Join: Re-planning required due to failures. Routing to router_step."
                )
                ctx.state.needs_replan = False  # Reset for the next plan
                return "router_step"
            return "dispatcher"

    # Still waiting for others
    return None


async def architect_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> str:
    """Analyze system requirements and propose high-level design decisions.

    This node uses gathered research findings to define the overall structure,
    technical stack, and implementation approach. It bridges the gap
    between raw research and concrete planning.

    Args:
        ctx: The pydantic-graph step context containing shared state.

    Returns:
        The next node identifier ('planner') or 'error_recovery'.

    """
    logger.info("Architect: Designing system changes...")
    architect_prompt = load_specialized_prompts("architect")

    architect = Agent(
        model=ctx.deps.agent_model,
        system_prompt=architect_prompt + f"\n\nContext:\n{ctx.state.exploration_notes}",
    )

    try:
        res = await architect.run(
            f"Design the architecture for: {ctx.state.query}", deps=ctx.deps
        )
        ctx.state.architectural_decisions = str(res.output)
        return "planner"
    except Exception as e:
        logger.error(f"Architect failed: {e}")
        return "error_recovery"


async def verifier_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> str:
    """Validate execution results and route to synthesis or re-dispatch.

    This node performs a structured quality audit of the accumulated
    execution results.  It does NOT synthesize the final response —
    that is handled by :func:`synthesizer_step`.

    Routing decisions:
    - **score >= 0.7** → ``'synthesizer'`` for final response composition
    - **0.4 <= score < 0.7** → ``'dispatcher'`` for re-execution
    - **score < 0.4** → ``'planner'`` for a fresh approach

    Args:
        ctx: The pydantic-graph step context containing all registry results.

    Returns:
        The next node identifier (``'synthesizer'``, ``'dispatcher'``,
        or ``'planner'``).

    """
    # HSM: State invariant check
    try:
        assert_state_valid(ctx.state, "verifier_step")
    except StateInvariantError as e:
        logger.error(f"Verifier state invariant violation: {e}")

    logger.info(
        f"[LAYER:GRAPH:VERIFIER] Starting (attempt {ctx.state.verification_attempts + 1})..."
    )
    _emit_node_lifecycle(
        ctx.deps.event_queue,
        "verifier",
        "node_start",
        attempt=ctx.state.verification_attempts + 1,
    )

    # Consolidate results for the verifier's context
    results_summary = "\n".join(
        [f"### {node}: {val}" for node, val in ctx.state.results_registry.items()]
    )

    # Structured Validation (quality gate)
    if ctx.state.verification_attempts < 2 and results_summary.strip():
        try:
            from .executor import _get_domain_tools

            domain_tools, domain_toolsets = await _get_domain_tools(
                "verifier", ctx.deps
            )

            validation_agent = Agent(
                model=ctx.deps.agent_model,
                output_type=ValidationResult,
                deps_type=GraphDeps,
                tools=domain_tools,
                toolsets=domain_toolsets,
                system_prompt=(
                    f"You are a quality gate. Evaluate whether the execution results "
                    f"fully and accurately answer the original query with specific data findings.\n\n"
                    f"Original Query: {ctx.state.query}\n\n"
                    f"Execution Results:\n{results_summary}\n\n"
                    f"CRITICAL: If the query asks for a list, status, or specific info, the results MUST contain "
                    f"the actual data records, not just a summary that the task was completed.\n"
                    f"Score 0.0-1.0. If data is missing or results are non-responsive, score < 0.7 and "
                    f"provide EXACT feedback on what is missing (e.g. 'Missing the list of container names')."
                ),
            )
            async with validation_agent.run_stream("Evaluate the results") as stream:
                validation = await asyncio.wait_for(
                    stream.get_output(), timeout=ctx.deps.verifier_timeout
                )

            emit_graph_event(
                ctx.deps.event_queue,
                event_type="verification_result",
                is_valid=validation.is_valid,
                feedback=validation.feedback,
                attempt=ctx.state.verification_attempts + 1,
            )
            if (
                not validation.is_valid
                and validation.score < 0.7
                and validation.feedback
            ):
                ctx.state.verification_attempts += 1
                ctx.state.validation_feedback = validation.feedback

                # Distinguish plan-level failures from execution-level failures.
                # Very low scores (< 0.4) suggest the approach itself was wrong
                # and a fresh plan is needed; moderate scores suggest the right
                # plan was executed poorly and can be re-dispatched.
                if validation.score < 0.4 and ctx.state.verification_attempts <= 2:
                    logger.warning(
                        f"Verifier: Score {validation.score:.2f} < 0.4. "
                        f"Feedback: {validation.feedback[:200]}. "
                        f"Re-planning (attempt {ctx.state.verification_attempts})."
                    )
                    ctx.state.needs_replan = True
                    ctx.state.error = f"Plan-level failure: {validation.feedback[:300]}"
                    return "planner"

                logger.warning(
                    f"Verifier: Score {validation.score:.2f} < 0.7. "
                    f"Feedback: {validation.feedback[:200]}. "
                    f"Re-dispatching (attempt {ctx.state.verification_attempts})."
                )
                ctx.state.step_cursor = 0
                ctx.state.needs_replan = False
                return "dispatcher"
            logger.info(f"Verifier: Validation passed (score: {validation.score:.2f}).")
        except Exception as e:
            logger.warning(
                f"Verifier: Structure validation failed: {e}. Proceeding to synthesis."
            )

    # Validation passed (or was skipped). Route to synthesizer for
    # final response composition.  This separates the quality-gate
    # concern from the response-generation concern.
    _emit_node_lifecycle(
        ctx.deps.event_queue, "verifier", "node_complete", next_node="synthesizer"
    )
    return "synthesizer"


async def synthesizer_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> End[GraphResponse]:
    """Compose the final authoritative response from execution results.

    This node is responsible for synthesizing a cohesive markdown response
    from the disparate specialist findings stored in
    ``ctx.state.results_registry``.  It also persists session metadata to
    MEMORY.md for future context retrieval.

    The synthesizer is intentionally separate from the verifier so that
    synthesis work is never wasted on re-dispatch/re-plan cycles.

    Args:
        ctx: The pydantic-graph step context containing all registry results.

    Returns:
        A terminal ``End`` state with the ``GraphResponse`` instance.

    """
    logger.info("[LAYER:GRAPH:SYNTHESIZER] Composing final response...")
    _emit_node_lifecycle(ctx.deps.event_queue, "synthesizer", "node_start")

    validator_prompt = load_specialized_prompts("verifier")

    results_summary = "\n".join(
        [f"### {node}: {val}" for node, val in ctx.state.results_registry.items()]
    )

    extra_context = ""
    if ctx.state.architectural_decisions:
        extra_context += (
            f"\n### ARCHITECTURAL INTENT\n{ctx.state.architectural_decisions}\n"
        )
    if ctx.state.exploration_notes:
        extra_context += f"\n### EXPLORATION FINDINGS\n{ctx.state.exploration_notes}\n"

    final_system_prompt = (
        f"{validator_prompt}\n"
        f"{extra_context}\n"
        f"### AGENT EXECUTION RESULTS\n{results_summary}\n\n"
        f"### FINAL INSTRUCTION\n"
        f"Synthesize the execution results into a cohesive final answer for the "
        f"user query: '{ctx.state.query}'.\n"
        f"Format data cleanly. Do NOT repeat yourself."
    )

    synthesizer = Agent(
        model=ctx.deps.agent_model,
        system_prompt=final_system_prompt,
    )

    try:
        logger.debug(f"Synthesizer: Prompt summary length: {len(results_summary)}")
        async with synthesizer.run_stream(
            "Consolidate and verify based on provided results. Be concise."
        ) as stream:
            async for chunk in stream.stream_text(delta=True):
                emit_graph_event(
                    ctx.deps.event_queue,
                    "agent_node_delta",
                    content=chunk,
                    node="synthesizer",
                )
            res = await asyncio.wait_for(
                stream.get_output(), timeout=ctx.deps.verifier_timeout
            )
        result_text = str(res) if res else "None"
        if result_text.lower() == "none":
            raise ValueError("Synthesis returned 'None'")

        logger.info(
            f"Synthesizer: Synthesis successful. Output length: {len(result_text)}"
        )
    except Exception as e:
        logger.warning(
            f"Synthesizer: Synthesis failed or timed out: {e}. Falling back to raw results."
        )
        emit_graph_event(
            ctx.deps.event_queue,
            "synthesis_fallback",
            reason=str(e)[:300],
            has_results=bool(results_summary.strip()),
        )
        if results_summary.strip():
            result_text = (
                f"The query was executed, but a final synthesis could not be generated concisely.\n\n"
                f"### RAW EXECUTION RESULTS\n{results_summary}"
            )
        else:
            result_text = (
                "The agent completed its analysis but was unable to find specific data matching your request. "
                "Please verify the query or target system status."
            )

    # Persist session metadata for future context retrieval
    try:
        from ..workspace import append_to_md_file
        from datetime import datetime

        memory_entry = (
            f"\n### Execution {ctx.state.session_id or 'unknown'} "
            f"({datetime.now().isoformat()})\n"
            f"- Query: {ctx.state.query[:200]}\n"
            f"- Plan: {[s.node_id for s in ctx.state.plan.steps]}\n"
            f"- Results: {list(ctx.state.results_registry.keys())}\n"
            f"- Failures: {ctx.state.error or 'none'}\n"
            f"- Tokens: {ctx.state.session_usage.total_tokens}\n"
            f"- Verification attempts: {ctx.state.verification_attempts}\n"
        )
        append_to_md_file("MEMORY.md", memory_entry)
    except Exception as e:
        logger.debug(f"Failed to write execution memory: {e}")

    return End(
        GraphResponse(
            status="completed",
            results={"output": result_text},
            metadata={
                "domain": ctx.state.routed_domain,
                "verification_attempts": ctx.state.verification_attempts,
            },
        )
    )


async def onboarding_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> str | End[GraphResponse]:
    """Initialize a new agent workspace and bootstrap core project files.

    Handles the creation of the standard agent directory structure and
    essential files (e.g., A2A_AGENTS.md, NODE_AGENTS.md, MEMORY.md) for
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


async def memory_selection_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> str | End[Any]:
    """Filter and select long-term project memories relevant to the session.

    Scans the local workspace for relevant documentation and memory files,
    using an LLM-based selector to identify which files contain essential
    context for the current user query.

    When the selector finds no relevant memories and the query appears to
    require background research (more than a simple conversational turn),
    this node routes to the researcher for web/codebase discovery before
    returning to the dispatcher.

    Args:
        ctx: The pydantic-graph step context.

    Returns:
        ``'dispatcher'`` with updated exploration notes, or ``'researcher'``
        when a context gap is detected.

    """
    logger.info("Memory Selection: Identifying relevant context...")
    _emit_node_lifecycle(ctx.deps.event_queue, "memory_selection", "node_start")
    prompt_content = load_specialized_prompts("memory_selection")
    root = ctx.state.project_root or os.getcwd()
    memories = []

    for p in Path(root).rglob("*.md"):
        if ".gemini" in str(p) or "node_modules" in str(p):
            continue
        try:
            content = p.read_text(encoding="utf-8", errors="ignore")
            description = "General project memory"
            if content.startswith("---"):
                match = re.search(r"description:\s*(.*)", content)
                if match:
                    description = match.group(1).strip()
            memories.append(f"- {p.name}: {description}")
        except Exception as e:
            logger.warning(f"Memory selection failed: {e}")

    if not memories:
        logger.info("Memory Selection: No workspace memories found. Skipping.")
        # Context gap detection: non-trivial queries with no memories
        # benefit from a research pass before execution.
        word_count = len(ctx.state.query.split())
        already_researched = ctx.state.global_research_loops > 1
        if (
            word_count > 8
            and not already_researched
            and not ctx.state.exploration_notes
        ):
            logger.info(
                "Memory Selection: Context gap detected — routing to researcher."
            )
            emit_graph_event(
                ctx.deps.event_queue,
                "context_gap_detected",
                reason="no_workspace_memories",
                query_words=word_count,
            )
            _emit_node_lifecycle(
                ctx.deps.event_queue,
                "memory_selection",
                "node_complete",
                next_node="researcher",
            )
            return "researcher"
        _emit_node_lifecycle(
            ctx.deps.event_queue,
            "memory_selection",
            "node_complete",
            next_node="dispatcher",
        )
        return "dispatcher"

    selectors = Agent(
        model=ctx.deps.agent_model,
        system_prompt=prompt_content,
        output_type=dict,
    )
    try:
        res = await selectors.run(
            f"Query: {ctx.state.query}\n\nAvailable memories:\n"
            + "\n".join(memories[:20])
        )
        selected = res.output.get("selected_memories", [])
        logger.info(
            f"Memory Selection: Selected {len(selected)} relevant files: {selected}"
        )

        loaded_context = []
        for filename in selected:
            for p in Path(root).rglob(filename):
                loaded_context.append(
                    f"### {filename}\n{p.read_text(encoding='utf-8', errors='ignore')}"
                )
                break

        ctx.state.exploration_notes += "\n\n### SELECTED MEMORIES\n" + "\n\n".join(
            loaded_context
        )

        # Context gap: memories exist but none are relevant to this query
        already_researched = ctx.state.global_research_loops > 1
        if (
            not selected
            and not already_researched
            and not ctx.state.exploration_notes.strip()
        ):
            logger.info(
                "Memory Selection: No relevant memories matched — routing to researcher."
            )
            emit_graph_event(
                ctx.deps.event_queue,
                "context_gap_detected",
                reason="no_relevant_memories",
                available=len(memories),
                selected=0,
            )
            _emit_node_lifecycle(
                ctx.deps.event_queue,
                "memory_selection",
                "node_complete",
                next_node="researcher",
            )
            return "researcher"

        _emit_node_lifecycle(
            ctx.deps.event_queue,
            "memory_selection",
            "node_complete",
            next_node="dispatcher",
        )
        return "dispatcher"
    except Exception as e:
        logger.error(f"Memory Selection failed: {e}")
        _emit_node_lifecycle(
            ctx.deps.event_queue,
            "memory_selection",
            "node_complete",
            next_node="dispatcher",
        )
        return "dispatcher"


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


async def dynamic_mcp_routing_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> List[str]:
    """Calculate the list of target MCP servers for dynamic tool discovery.

    Parses the local MCP configuration to identify available servers
    that should be probed or delegated to for general-purpose execution.

    Args:
        ctx: The pydantic-graph step context.

    Returns:
        A list of server names to be used as map() inputs for mcp_server_step.

    """
    mcp_config = load_mcp_config()
    servers = list(mcp_config.mcpServers.keys())
    logger.info(f"Dynamic MCP Routing: Routing to {len(servers)} servers: {servers}")
    return servers


async def mcp_server_step(
    ctx: StepContext[GraphState, GraphDeps, str],
) -> str | End[Any]:
    """Execute a query against a specific, dynamically discovered MCP server.

    This node handles direct interaction with target MCP servers,
    attempting to match the server name against registered specialist
    agents or falling back to a generic expert agent with full
    tool access to that server.

    Args:
        ctx: The pydantic-graph step context with the server name as input.

    Returns:
        The identifier of the joiner node ('execution_joiner') on success,
        or 'error_recovery'.

    """
    server_name = ctx.input
    query = ctx.state.query

    logger.info(f"Executing MCP Server Step: {server_name} for query: {query}")

    # Emit node start event
    emit_graph_event(
        ctx.deps.event_queue,
        "node_start",
        node_id="mcp_server_execution",
        server=server_name,
    )

    try:
        # Check if there's a matching dynamic MCP agent in the registry
        registry = load_node_agents_registry()
        matching_agents = [a for a in registry.agents if a.mcp_server == server_name]

        if matching_agents:
            # Execute each matching specialist agent for this server
            for mcp_agent in matching_agents:
                await _execute_dynamic_mcp_agent(ctx, mcp_agent)
        else:
            # Fallback: create ad-hoc agent with all tools from this server
            matched_toolsets = []
            for toolset in ctx.deps.mcp_toolsets:
                server_id = getattr(toolset, "id", getattr(toolset, "name", None))
                if server_id and server_name in str(server_id):
                    matched_toolsets.append(toolset)

            agent = Agent(
                model=ctx.deps.agent_model,
                system_prompt=f"You are a specialist for the '{server_name}' MCP server. Use the available tools to answer queries.",
                toolsets=matched_toolsets,
            )

            async with agent.run_stream(query, deps=ctx.deps) as stream:
                async for chunk in stream.stream_text(delta=True):
                    emit_graph_event(
                        ctx.deps.event_queue,
                        "agent_node_delta",
                        content=chunk,
                        node=ctx.node.name,
                    )
                output = await stream.get_output()
            ctx.state._update_usage(stream.usage())
            ctx.state.results[server_name] = str(output)
            ctx.state.results_registry[f"{server_name}_{ctx.state.step_cursor}"] = str(
                output
            )

            # Stream events to WebUI
            if ctx.deps.event_queue:
                from pydantic_ai.messages import (
                    ModelResponse,
                    ToolCallPart,
                    ToolReturnPart,
                    ModelRequest,
                )

                for msg in stream.all_messages():
                    if isinstance(msg, ModelResponse):
                        for part in msg.parts:
                            if isinstance(part, ToolCallPart):
                                emit_graph_event(
                                    ctx.deps.event_queue,
                                    "expert_tool_call",
                                    domain=server_name,
                                    tool_name=part.tool_name,
                                    args=part.args,
                                )
                    elif isinstance(msg, ModelRequest):
                        for part in msg.parts:
                            if isinstance(part, ToolReturnPart):
                                emit_graph_event(
                                    ctx.deps.event_queue,
                                    event_type="tool_result",
                                    agent=server_name,
                                    tool=part.tool_name,
                                    result=str(part.content)[:500],
                                )

        emit_graph_event(
            ctx.deps.event_queue,
            event_type="node_complete",
            node_id="mcp_server_execution",
            server=server_name,
            result=str(ctx.state.results.get(server_name, ""))[:500],
        )

        return "execution_joiner"
    except Exception as e:
        logger.error(f"MCP Server Step '{server_name}' failed: {e}")
        ctx.state.error = f"MCP Server {server_name} failed: {e}"
        return "error_recovery"


async def error_recovery_step(
    ctx: StepContext[GraphState, GraphDeps, Exception | str | Any],
) -> str | End[dict]:
    """Attempt graceful recovery before terminating the graph.

    Implements a two-tier recovery strategy:

    1. **Recoverable errors** (retry_count < 2 and partial results exist):
       Injects the error as feedback and routes to the planner for a
       fresh strategy, preserving any partial results already gathered.
    2. **Terminal errors** (retries exhausted, policy violations, or
       max-transition overflows): Terminates with a diagnostic report.

    Args:
        ctx: The pydantic-graph step context with the failure details as input.

    Returns:
        ``'planner'`` if recovery is attempted, or a terminal ``End`` state
        with the error summary and partial results.

    """
    error_str = str(ctx.inputs) if ctx.inputs else (ctx.state.error or "Unknown error")
    _emit_node_lifecycle(ctx.deps.event_queue, "error_recovery", "node_start")
    logger.error(f"error_recovery_step: {error_str}")

    # Terminal conditions that should NOT retry
    terminal_keywords = (
        "policy violation",
        "max node transitions",
        "max planning loops",
    )
    is_terminal = any(kw in error_str.lower() for kw in terminal_keywords)

    if not is_terminal and ctx.state.retry_count < 2:
        ctx.state.retry_count += 1
        ctx.state.validation_feedback = (
            f"Previous execution failed with error: {error_str[:500]}. "
            f"Please devise a different strategy to satisfy the query."
        )
        logger.warning(
            f"error_recovery_step: Recoverable failure (attempt {ctx.state.retry_count}). "
            f"Routing to planner for fresh strategy."
        )
        emit_graph_event(
            ctx.deps.event_queue,
            "error_recovery_replan",
            attempt=ctx.state.retry_count,
            error=error_str[:300],
        )
        _emit_node_lifecycle(
            ctx.deps.event_queue,
            "error_recovery",
            "node_complete",
            next_node="planner",
        )
        return "planner"

    logger.error(
        f"error_recovery_step: Terminal failure after {ctx.state.retry_count} retries. "
        f"Error: {error_str[:300]}"
    )
    emit_graph_event(
        ctx.deps.event_queue,
        "error_recovery_terminal",
        error=error_str[:300],
        retries=ctx.state.retry_count,
    )
    _emit_node_lifecycle(
        ctx.deps.event_queue,
        "error_recovery",
        "node_complete",
        next_node="end",
    )
    return End({"error": error_str, "results": ctx.state.results})


async def python_programmer_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> str | End[Any]:
    """Execute the specialized Python engineer role.

    Specializes in Python implementation, refactoring, packaging (Poetry/UV),
    and standalone script generation using domain-specific prompts.

    Args:
        ctx: The pydantic-graph step context.

    Returns:
        The next node identifier or terminal End state.

    """
    return await _execute_specialized_step(ctx, "python_programmer")


async def golang_programmer_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> str | End[Any]:
    """Execute the specialized Go (Golang) engineer role.

    Focuses on cloud-native backends, microservices, and concurrent
    performance optimization using the Go toolkit.

    Args:
        ctx: The pydantic-graph step context.

    Returns:
        The next node identifier or terminal End state.

    """
    return await _execute_specialized_step(ctx, "golang_programmer")


async def typescript_programmer_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> str | End[Any]:
    """Execute the specialized TypeScript engineer role.

    Expertise in modern TypeScript, React, Node.js, and API design.
    Handles type-safe implementation and frontend-backend integration.

    Args:
        ctx: The pydantic-graph step context.

    Returns:
        The next node identifier or terminal End state.

    """
    return await _execute_specialized_step(ctx, "typescript_programmer")


async def rust_programmer_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> str | End[Any]:
    """Execute the specialized Rust engineer role.

    Expertise in memory-safe systems programming, performance critical
    modules, and low-level CLI development using Cargo.

    Args:
        ctx: The pydantic-graph step context.

    Returns:
        The next node identifier or terminal End state.

    """
    return await _execute_specialized_step(ctx, "rust_programmer")


async def security_auditor_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> str | End[Any]:
    """Execute the specialized Security Auditor role.

    Expertise in threat modeling, OWASP standards, dependency vulnerability
    scanning, and secure software development lifecycles.

    Args:
        ctx: The pydantic-graph step context.

    Returns:
        The next node identifier or terminal End state.

    """
    return await _execute_specialized_step(ctx, "security_auditor")


async def javascript_programmer_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> str | End[Any]:
    """Execute the specialized JavaScript engineer role.

    Expertise in general-purpose JS ecosystems, legacy project maintenance,
    and lightweight scripting for web and Node environments.

    Args:
        ctx: The pydantic-graph step context.

    Returns:
        The next node identifier or terminal End state.

    """
    return await _execute_specialized_step(ctx, "javascript_programmer")


async def c_programmer_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> str | End[Any]:
    """Execute the specialized C programmer expert role.

    Args:
        ctx: The pydantic-graph step context.

    Returns:
        The identifier of the next node or terminal state.

    """
    return await _execute_specialized_step(ctx, "c_programmer")


async def cpp_programmer_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> str | End[Any]:
    """Execute the specialized C++ programmer expert role.

    Args:
        ctx: The pydantic-graph step context.

    Returns:
        The identifier of the next node or terminal state.

    """
    return await _execute_specialized_step(ctx, "cpp_programmer")


async def qa_expert_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> str | End[Any]:
    """Execute the specialized Q&A and verification role.

    Specializes in test planning, manual verification strategies,
    regression analysis, and quality assurance gates using QA prompts.

    Args:
        ctx: The pydantic-graph step context.

    Returns:
        The next node identifier or terminal End state.

    """
    return await _execute_specialized_step(ctx, "qa_expert")


async def debugger_expert_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> str | End[Any]:
    """Execute the specialized multi-vector debugging role.

    Expertise in root cause analysis, log forensics, stack trace
    interpretation, and automated repair strategies for complex faults.

    Args:
        ctx: The pydantic-graph step context.

    Returns:
        The next node identifier or terminal End state.

    """
    return await _execute_specialized_step(ctx, "debugger_expert")


async def ui_ux_designer_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> str | End[Any]:
    """Execute the specialized UI/UX design and frontend role.

    Expertise in design systems, accessibility (a11y), responsive layouts,
    user experience auditing, and modern styling (Vanilla CSS, Tailwind).

    Args:
        ctx: The pydantic-graph step context.

    Returns:
        The next node identifier or terminal End state.

    """
    return await _execute_specialized_step(ctx, "ui_ux_designer")


async def devops_engineer_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> str | End[Any]:
    """Execute the specialized DevOps and infrastructure engineer role.

    Focuses on CI/CD pipelines, container orchestration (Docker, K8s),
    IaC (Terraform, Ansible), and environment stabilization.

    Args:
        ctx: The pydantic-graph step context.

    Returns:
        The next node identifier or terminal End state.

    """
    return await _execute_specialized_step(ctx, "devops_engineer")


async def cloud_architect_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> str | End[Any]:
    """Execute the specialized Cloud Architecture role.

    Specializes in distributed systems design, multi-cloud strategy,
    serverless architectures, and scalability modeling across providers.

    Args:
        ctx: The pydantic-graph step context.

    Returns:
        The next node identifier or terminal End state.

    """
    return await _execute_specialized_step(ctx, "cloud_architect")


async def database_expert_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> str | End[Any]:
    """Execute the specialized Database and storage expert role.

    Expertise in SQL/NoSQL schema design, query optimization, data
    migration strategies, and high-availability storage configurations.

    Args:
        ctx: The pydantic-graph step context.

    Returns:
        The next node identifier or terminal End state.

    """
    return await _execute_specialized_step(ctx, "database_expert")


async def java_programmer_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> str | End[Any]:
    """Execute the specialized Java engineer role.

    Focuses on enterprise application development, Spring Boot models,
    JVM performance tuning, and robust backend implementation.

    Args:
        ctx: The pydantic-graph step context.

    Returns:
        The next node identifier or terminal End state.

    """
    return await _execute_specialized_step(ctx, "java_programmer")


async def data_scientist_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> str | End[Any]:
    """Execute the specialized Data Science and AI role.

    Expertise in data analysis, machine learning modeling, statistical
    inference, and pipeline optimization using the Python data stack.

    Args:
        ctx: The pydantic-graph step context.

    Returns:
        The next node identifier or terminal End state.

    """
    return await _execute_specialized_step(ctx, "data_scientist")


async def document_specialist_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> str | End[Any]:
    """Execute the specialized documentation and technical writing role.

    Specializes in README generation, API documentation, Mermaid
    architecture diagrams, and technical manual synthesis.

    Args:
        ctx: The pydantic-graph step context.

    Returns:
        The next node identifier or terminal End state.

    """
    return await _execute_specialized_step(ctx, "document_specialist")


async def mobile_programmer_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> str | End[Any]:
    """Execute the specialized Mobile application engineer role.

    Focuses on cross-platform development (React Native, Flutter) and
    native mobile architectures for iOS and Android.

    Args:
        ctx: The pydantic-graph step context.

    Returns:
        The next node identifier or terminal End state.

    """
    return await _execute_specialized_step(ctx, "mobile_programmer")


async def agent_engineer_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> str | End[Any]:
    """Execute the specialized Agent Engineering role.

    Specializes in building autonomous agents, MCP server construction,
    graph orchestration logic, and LLM tool engineering.

    Args:
        ctx: The pydantic-graph step context.

    Returns:
        The next node identifier or terminal End state.

    """
    return await _execute_specialized_step(ctx, "agent_engineer")


async def project_manager_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> str | End[Any]:
    """Execute the specialized Project Management role.

    Expertise in task decomposition, resource planning, timeline
    estimation, and project health monitoring.

    Args:
        ctx: The pydantic-graph step context.

    Returns:
        The next node identifier or terminal End state.

    """
    return await _execute_specialized_step(ctx, "project_manager")


async def systems_manager_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> str | End[Any]:
    """Execute the specialized Systems Architecture and management role.

    Focuses on codebase-wide structural analysis, dependency mapping,
    and cross-repository integration patterns.

    Args:
        ctx: The pydantic-graph step context.

    Returns:
        The next node identifier or terminal End state.

    """
    return await _execute_specialized_step(ctx, "systems_manager")


async def browser_automation_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> str | End[Any]:
    """Execute the specialized Browser Automation role.

    Specializes in frontend E2E testing, visual regression, deep
    web crawling, and programmatic browser interaction.

    Args:
        ctx: The pydantic-graph step context.

    Returns:
        The next node identifier or terminal End state.

    """
    return await _execute_specialized_step(ctx, "browser_automation")


async def coordinator_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> str | End[Any]:
    """Execute the specialized Coordination expert role.

    Handles high-level task synchronization, multi-agent communication,
    and session state alignment across disparate specialist domains.

    Args:
        ctx: The pydantic-graph step context.

    Returns:
        The next node identifier or terminal End state.

    """
    return await _execute_specialized_step(ctx, "coordinator")


async def critique_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> str | End[Any]:
    """Execute the specialized Critique expert role for peer review.

    Specializes in code review, architectural critique, and logic
    validation. Provides a second pair of eyes to surface potential
    edge cases or design flaws before final verification.

    Args:
        ctx: The pydantic-graph step context.

    Returns:
        The next node identifier or terminal End state.

    """
    return await _execute_specialized_step(ctx, "critique")
