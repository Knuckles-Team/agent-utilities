#!/usr/bin/python

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
    NODE_SKILL_MAP,
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


async def usage_guard_step(
    ctx: StepContext[GraphState, GraphDeps, Any],
) -> str | None:
    """Policy enforcement and token usage monitor. Checks for violations and budget."""
    logger.info(f"[LAYER:GRAPH:USAGE_GUARD] Handling query: '{ctx.state.query[:50]}...'")
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
        return "router"

    from pydantic_ai import Agent

    checker = Agent(
        model=ctx.deps.router_model,
        system_prompt="You are a security guard. Output 'PASS' if the query is safe, or a brief error if it violates policy.",
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
    """Unified discovery agent with Web, Codebase, and Workspace capability."""
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
    # planner_prompt = load_specialized_prompts("planner")
    # architect_prompt = load_specialized_prompts("architect")
    researcher_prompt = load_specialized_prompts("researcher")

    from pydantic_ai import Agent

    # planner = Agent(
    #     model=ctx.deps.agent_model,
    #     system_prompt=(f"{planner_prompt}\n\n" f"Context: {unified_context}"),
    # )
    # architect = Agent(
    #     model=ctx.deps.agent_model,
    #     system_prompt=(f"{architect_prompt}\n\n" f"Context: {unified_context}"),
    # )

    # Main researcher agent with ALL discovery tools
    researcher = Agent(
        model=ctx.deps.agent_model,
        system_prompt=(f"{researcher_prompt}\n\nWorkspace Context: {unified_context}"),
    )
    # Register all discovery tools from universal skills
    from ..tools.developer_tools import project_search

    researcher.tool(project_search)
    from ..tools.workspace_tools import read_workspace_file

    researcher.tool(read_workspace_file)

    # Bind optional MCP research toolsets (web search, etc.)
    for toolset in ctx.deps.mcp_toolsets:
        researcher.toolsets.append(toolset)

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
    """Action step. Converts architectural intent into a concrete execution plan."""
    logger.info("Planner: Formulating task list...")

    planner_prompt = load_specialized_prompts("planner")
    unified_context = await fetch_unified_context()

    from pydantic_ai import Agent

    planner = Agent(
        model=ctx.deps.agent_model,
        output_type=GraphPlan,
        system_prompt=(
            f"{planner_prompt}\n\n"
            f"### ARCHITECTURAL DECISIONS\n{ctx.state.architectural_decisions}\n\n"
            f"### WORKSPACE CONTEXT\n{unified_context}"
        ),
    )

    try:
        res = await planner.run(
            f"Create a specific execution plan for: {ctx.state.query}", deps=ctx.deps
        )
        ctx.state.plan = res.output
        ctx.state.step_cursor = 0  # Reset cursor for the new plan
        logger.info(f"Planner: Generated plan with {len(ctx.state.plan.steps)} steps.")
        return "dispatcher"
    except Exception as e:
        logger.error(f"Planning failed: {e}")
        return "error_recovery"


async def fetch_unified_context() -> str:
    """Fetch A2A_AGENTS.md, NODE_AGENTS.md, MEMORY.md, and git status for unified context."""

    a2a_agents = load_workspace_file(CORE_FILES["A2A_AGENTS"])
    if a2a_agents and len(a2a_agents.splitlines()) > 500:
        a2a_agents = "\n".join(a2a_agents.splitlines()[:500]) + "\n\n... (truncated)"

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
        f"**A2A_AGENTS.md (Peer Registry Preview):**\n{a2a_agents or '(empty)'}\n\n"
        f"**NODE_AGENTS.md (Specialist Registry Preview):**\n{mcp_agents or '(empty)'}\n\n"
        f"**MEMORY.md (Historical Context):**\n{memory or '(empty)'}\n\n"
        f"**Git Status:**\n{git_status or '(clean)'}"
    )


async def router_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> GraphPlan | str:
    """Determines the optimal execution plan for the query."""
    deps = ctx.deps

    emit_graph_event(
        deps.event_queue,
        "routing_started",
        query=ctx.state.query,
    )
    logger.info(f"[LAYER:GRAPH:ROUTER] Routing started for query: '{ctx.state.query[:50]}...'")

    # Track re-planning loops to prevent infinite cycles
    ctx.state.global_research_loops += 1
    if ctx.state.global_research_loops > 3:
        logger.error("Router: Max planning loops exceeded. Aborting.")
        return "error_recovery"

    # HSM Junction Pseudostate: Try static keyword routing first (saves LLM call)
    # If tag_prompts is empty (e.g. toolset loading failed due to missing env vars),
    # fall back to using the MCP registry directly for keyword-based routing.
    routing_tags = deps.tag_prompts
    if not routing_tags:
        registry = load_node_agents_registry()
        routing_tags = {
            a.tag: a.description or a.name for a in registry.agents if a.tag
        }
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
            registry = load_node_agents_registry()
            specialist_tags = {
                a.tag: a.description or a.name for a in registry.agents if a.tag
            }
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
            logger.debug(f"[LAYER:GRAPH:ROUTER] LLM Call Starting: system_prompt length={len(system_prompt_str)}")
            async with router_agent.run_stream(ctx.state.query) as stream:
                res = await asyncio.wait_for(
                    stream.data(), timeout=ctx.deps.router_timeout
                )
            logger.info(f"[LAYER:GRAPH:ROUTER] LLM Call Completed. Plan Reasoning: {res.output.metadata.get('reasoning', 'N/A')}")
            logger.info(f"[LAYER:GRAPH:ROUTER] Plan Step Count: {len(res.output.steps)}")
        except asyncio.TimeoutError:
            logger.warning("Router: LLM planning timed out. Escalating to fallbacks.")
            raise ValueError("LLM planning timed out")

        ctx.state._update_usage(getattr(res, "usage", None))
        ctx.state.plan = res.output
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
    """Manages the execution flow using sequential or parallel steps."""
    logger.info(f"[LAYER:GRAPH:DISPATCHER] Transitioning. Current cursor: {ctx.state.step_cursor}")
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

    logger.info(
        f"Dispatcher: Handling graph execution (Step {ctx.state.step_cursor}/{len(ctx.state.plan.steps)})"
    )
    if ctx.state.step_cursor >= len(ctx.state.plan.steps):
        # If we have gathered research or execution results, route to verifier for final summary
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

    # Set the barrier count
    ctx.state.pending_parallel_count = len(batch)
    logger.info(f"Dispatcher: Dispatching parallel batch of {len(batch)} tasks...")

    ctx.state.pending_batch = ParallelBatch(tasks=batch)
    return "parallel_batch_processor"


async def parallel_batch_processor(
    ctx: StepContext[GraphState, GraphDeps, Any],
) -> list[ExecutionStep]:
    """Retrieves the cached batch from state to avoid direct object transition issues."""
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
    """A generic wrapper for parallel batch execution with local retries."""
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

            # Special Handling for Researcher
            if node_id == "researcher":
                await researcher_step(ctx)

            # Any node_id in NODE_SKILL_MAP -> delegate to shared specialized logic
            elif node_id in NODE_SKILL_MAP:
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

            # # Professional Expert Steps
            # elif node_id == "python_programmer":
            #     await python_step(ctx)
            # elif (
            #         node_id == "typescript_programmer" or node_id == "javascript_programmer"
            # ):
            #     await typescript_step(ctx)
            # elif node_id == "rust_expert":
            #     await rust_step(ctx)
            # elif node_id == "golang_expert":
            #     await golang_step(ctx)
            # elif node_id == "security_auditor":
            #     await security_step(ctx)
            # elif node_id == "qa_expert":
            #     await qa_step(ctx)
            # elif node_id == "debugger_expert" or node_id == "debugger_step":
            #     await debugger_step(ctx)
            # elif node_id == "ui_ux_designer" or node_id == "ui_ux_step":
            #     await ui_ux_step(ctx)
            # elif node_id == "devops_engineer" or node_id == "devops_step":
            #     await devops_step(ctx)
            # elif node_id == "cloud_architect" or node_id == "cloud_step":
            #     await cloud_step(ctx)
            # elif node_id == "database_expert" or node_id == "database_step":
            #     await database_step(ctx)

            # Dynamic Discovery Execution
            else:
                # First check our dynamic MCP registry
                registry = load_node_agents_registry()

                # Normalize the requested node_id for multi-strategy matching
                node_id_norm = node_id.lower().replace("-", "_").replace(" ", "_")

                def _agent_matches(a) -> bool:
                    """Multi-strategy agent name matching handles approximate node IDs from router."""
                    tag = (a.tag or "").lower().replace("-", "_").replace(" ", "_")
                    name = a.name.lower().replace("-", "_").replace(" ", "_")
                    server = (
                        (a.mcp_server or "").lower().replace("-", "_").replace(" ", "_")
                    )
                    desc = (a.description or "").lower()
                    # 1. Exact match on normalized tag, name, or server
                    if (
                        tag == node_id_norm
                        or name == node_id_norm
                        or server == node_id_norm
                    ):
                        return True
                    # 2. Tag/Server substring: node_id contains the tag or server (e.g. 'expert_portainer' ⊇ 'portainer')
                    if (tag and tag in node_id_norm) or (
                        server and server in node_id_norm
                    ):
                        return True
                    # 3. node_id is a prefix/suffix of tag or server
                    if tag and (
                        node_id_norm.startswith(tag) or node_id_norm.endswith(tag)
                    ):
                        return True
                    if server and (
                        node_id_norm.startswith(server) or node_id_norm.endswith(server)
                    ):
                        return True
                    # 4. Keyword intersection: any meaningful word from node_id appears in tag/desc
                    # Handles LLM hallucinations like 'researcher_git_status' → matches 'repository-manager'
                    # because 'git' appears in both the node_id and the agent's description/tag.
                    node_keywords = {
                        w
                        for w in node_id_norm.split("_")
                        if len(w) >= 3
                        and w
                        not in {"researcher", "expert", "agent", "manager", "action"}
                    }
                    if node_keywords:
                        for kw in node_keywords:
                            if kw in tag or kw in desc:
                                logger.debug(
                                    f"Expert: Keyword Match! '{kw}' found in tag/desc of '{a.name}'"
                                )
                                return True
                    return False

                mcp_agent = next(
                    (a for a in registry.agents if _agent_matches(a)),
                    None,
                )

                if mcp_agent:
                    logger.info(
                        f"Expert: Matched node_id='{node_id}' -> agent='{mcp_agent.name}' (tag='{mcp_agent.tag}')"
                    )
                    await _execute_dynamic_mcp_agent(ctx, mcp_agent)
                else:
                    from ..a2a import discover_agents

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
    """Synchronizes parallel executions by decrementing the pending count."""
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
    """
    Design step. Makes high-level architectural decisions.
    """
    logger.info("Architect: Designing system changes...")
    architect_prompt = load_specialized_prompts("architect")

    from pydantic_ai import Agent

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
) -> str | End[GraphResponse]:
    """
    Verification and Synthesis step with feedback loop.

    Phase 1: LLM validates results with structured ValidationResult (score + feedback).
    Phase 2: If score < 0.7 and attempts < 2, inject feedback and re-dispatch
    Phase 3: Synthesize final output (always - even after validation failure)
    """
    # HSM: State invariant check
    try:
        assert_state_valid(ctx.state, "verifier_step")
    except StateInvariantError as e:
        logger.error(f"Verifier state invariant violation: {e}")

    logger.info(
        f"[LAYER:GRAPH:VERIFIER] Starting (attempt {ctx.state.verification_attempts + 1})..."
    )
    validator_prompt = load_specialized_prompts("verifier")

    # Consolidate results for the verifier's context
    results_summary = "\n".join(
        [f"### {node}: {val}" for node, val in ctx.state.results_registry.items()]
    )

    # Optimization: Only include non-empty context blocks to reduce token pressure
    extra_context = ""
    if ctx.state.architectural_decisions:
        extra_context += (
            f"\n### ARCHITECTURAL INTENT\n{ctx.state.architectural_decisions}\n"
        )
    if ctx.state.exploration_notes:
        extra_context += f"\n### EXPLORATION FINDINGS\n{ctx.state.exploration_notes}\n"

    from pydantic_ai import Agent

    # PHase 1: Structured Validation
    if ctx.state.verification_attempts < 2 and results_summary.strip():
        try:
            validation_agent = Agent(
                model=ctx.deps.agent_model,
                output_type=ValidationResult,
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
                val_res = await asyncio.wait_for(
                    stream.data(), timeout=ctx.deps.verifier_timeout
                )
            validation = val_res.output

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

    # Phase 2 Synthesis
    final_system_prompt = (
        f"{validator_prompt}\n"
        f"{extra_context}\n"
        f"### AGENT EXECUTION RESULTS\n{results_summary}\n\n"
        f"### FINAL INSTRUCTION\n"
        f"Synthesize the execution results into a cohesive final answer for the "
        f"user query: '{ctx.state.query}'.\n"
        f"Format data cleanly. Do NOT repeat yourself."
    )

    verifier = Agent(
        model=ctx.deps.agent_model,
        system_prompt=final_system_prompt,
    )

    try:
        logger.debug(f"Verifier: Prompt summary length: {len(results_summary)}")
        # Use a generous timeout for synthesis
        async with verifier.run_stream(
            "Consolidate and verify based on provided results. Be concise."
        ) as stream:
            async for chunk in stream.stream_text(delta=True):
                emit_graph_event(
                    ctx.deps.event_queue,
                    "agent-node-delta",
                    content=chunk,
                    node=ctx.node.name,
                )
            res = await asyncio.wait_for(
                stream.data(), timeout=ctx.deps.verifier_timeout
            )
        result_text = str(res.output) if res.output else "None"
        if result_text.lower() == "none":
            raise ValueError("Synthesis returned 'None'")

        logger.info(
            f"Verifier: Synthesis successful. Output length: {len(result_text)}"
        )
    except Exception as e:
        logger.warning(
            f"Verifier synthesis failed or timed out: {e}. Falling back to raw results."
        )
        # Fallback: Just provide the raw summary if synthesis fails or is empty
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

    try:
        from ..workspace import append_to_md_file
        from datetime import datetime

        memory_entry = (
            f"\n### Execution {ctx.state.session_id or 'unknown'} "
            f"({datetime.now().isoformat()})\n)"
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
    """Handles project initialization and bootstrapping."""
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
    """Identifies relevant memories to load for the current query."""
    logger.info("Memory Selection: Identifying relevant context...")
    prompt_content = load_specialized_prompts("memory_selection")
    root = ctx.state.project_root or os.getcwd()
    memories = []

    for p in Path(root).rglob("*.md"):
        if ".gemini" in str(p) or "node_modules" in str(p):
            continue
        try:
            content = p.read_text(encoding="utf-8", errors="ignore")
            # Extract description from frontmatter if available
            description = "General project memory"
            if content.startswith("---"):
                match = re.search(r"description:\s*(.*)", content)
                if match:
                    description = match.group(1).strip()
            memories.append(f"- {p.name}: {description}")
        except Exception as e:
            logger.warning(f"Memory selection failed: {e}")
            pass

        from pydantic_ai import Agent

        # We use a simple list of memories as context
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

            # Load the selected content into the state for other agents to consume
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
            return ctx.inputs
        except Exception as e:
            logger.error(f"Memory Selection failed: {e}")
            return ctx.inputs


async def approval_gate_step(
    ctx: StepContext[GraphState, GraphDeps, Any],
) -> Any:
    """Pauses for human approval and captures triage redirection feedback."""
    if ctx.state.mode != "plan" and not ctx.state.human_approval_required:
        return ctx.inputs

    logger.info("Approval Gate: Pausing for user review...")
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
    """Dynamically identifies MCP servers to route to based on config."""
    mcp_config = load_mcp_config()
    servers = list(mcp_config.mcpServers.keys())
    logger.info(f"Dynamic MCP Routing: Routing to {len(servers)} servers: {servers}")
    return servers


async def mcp_server_step(
    ctx: StepContext[GraphState, GraphDeps, str],
) -> str | End[Any]:
    """Execute a query against a specific MCP server."""
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
            agent = Agent(
                model=ctx.deps.agent_model,
                system_prompt=f"You are a specialist for the '{server_name}' MCP server. Use the available tools to answer queries.",
            )

            for toolset in ctx.deps.mcp_toolsets:
                server_id = getattr(toolset, "id", getattr(toolset, "name", None))
                if server_id and server_name in str(server_id):
                    agent.toolsets.append(toolset)

            async with agent.run_stream(query, deps=ctx.deps) as stream:
                async for chunk in stream.stream_text(delta=True):
                    emit_graph_event(
                        ctx.deps.event_queue,
                        "agent-node-delta",
                        content=chunk,
                        node=ctx.node.name,
                    )
                res = await stream.data()
            ctx.state._update_usage(getattr(res, "usage", None))
            ctx.state.results[server_name] = str(res.output)
            ctx.state.results_registry[f"{server_name}_{ctx.state.step_cursor}"] = str(
                res.output
            )

            # Stream events to WebUI
            if ctx.deps.event_queue:
                from pydantic_ai.messages import (
                    ModelResponse,
                    ToolCallPart,
                    ToolReturnPart,
                    ModelRequest,
                )

                for msg in res.all_messages():
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
                                    event_type="tool-result",
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
) -> End[dict]:
    """Handles errors by either retrying or ending with error state."""
    logger.error(f"error_recovery_step: {ctx.inputs}")
    return End({"error": str(ctx.inputs), "results": ctx.state.results})


# async def domain_step(
#         ctx: StepContext[GraphState, GraphDeps, str],
# ) -> str | End[Any]:
#     """Executes a single domain's MCP tools or sub-agent."""
#     domain = ctx.inputs
#     ctx.state.routed_domain = domain
#
#     # Logic extracted from DomainNode.execute_domain
#     # (Abbreviated here, I'll need to make sure I get the full implementation)
#     # I'll define a helper for this to avoid duplication.
#     result = await _execute_domain_logic(ctx, domain)
#
#     if isinstance(result, End):
#         return result
#     return domain
#
#
# async def validator_step(
#         ctx: StepContext[GraphState, GraphDeps, str],
# ) -> str | End[dict]:
#     """Validates the output of a domain execution and performs lightweight aggregation."""
#     domain = ctx.inputs
#     result_text = ctx.state.results.get(domain, "")
#     deps = ctx.deps
#
#     # Logic: If this is the last domain to report in a fan-out, we might aggregate.
#     # In Pydantic Graph, each parallel node calls its successor independently.
#     # So we check if we have all the results we expected.
#
#     # 1. Skip if LLM validation is disabled
#     if not deps.enable_llm_validation:
#         return End(
#             {"status": "success", "domain": domain, "results": ctx.state.results}
#         )
#
#     # 2. Lightweight Synthesis: If it's a "list" or "search" query, skip the LLM call
#     read_patterns = ["list", "search", "find", "get", "show", "describe", "where"]
#     is_read_only = any(p in ctx.state.query.lower() for p in read_patterns)
#
#     if is_read_only and len(ctx.state.results) >= 1:
#         logger.info(
#             "validator_step: Read-only query detected. Performing lightweight aggregation."
#         )
#         # We'll let the final caller decide how to join, or just provide the map.
#         return End(
#             {
#                 "status": "success",
#                 "summary": "Aggregated domain results",
#                 "results": ctx.state.results,
#             }
#         )
#
#     # 3. Standard LLM Validation (for complex write-heavy or reasoning tasks)
#     if ctx.state.retry_count < 2:
#         logger.info(f"validator_step: Performing LLM-based validation for '{domain}'")
#         validator_prompt = load_specialized_prompts("validator")
#         validator_agent = Agent(
#             model=deps.router_model,
#             output_type=ValidationResult,
#             system_prompt=(
#                 f"{validator_prompt}\n\n"
#                 f"### CONTEXT\n"
#                 f"Original Query: {ctx.state.query}\n"
#                 f"Agent Result: {result_text}\n"
#             ),
#         )
#         try:
#             val_res = await validator_agent.run("Evaluate the result.")
#             if val_res.output.is_valid:
#                 return End({"status": "success", "results": ctx.state.results})
#             else:
#                 ctx.state.retry_count += 1
#                 ctx.state.validation_feedback = val_res.output.feedback
#                 return domain  # Loop back
#         except Exception:
#             pass
#
#     return End({"status": "success", "results": ctx.state.results})
#
#
# async def python_programmer_step(
#         ctx: StepContext[GraphState, GraphDeps, None],
# ) -> str | End[Any]:
#     """Specialized Python expert step."""
#     return await _execute_specialized_step(ctx, "python_programmer")
#
#
# async def golang_programmer_step(
#         ctx: StepContext[GraphState, GraphDeps, None],
# ) -> str | End[Any]:
#     """Specialized Golang expert step."""
#     return await _execute_specialized_step(ctx, "golang_programmer")
#
#
# async def typescript_programmer_step(
#         ctx: StepContext[GraphState, GraphDeps, None],
# ) -> str | End[Any]:
#     """Specialized TypeScript expert step."""
#     return await _execute_specialized_step(ctx, "typescript_programmer")
#
#
# async def rust_programmer_step(
#         ctx: StepContext[GraphState, GraphDeps, None],
# ) -> str | End[Any]:
#     """Specialized Rust expert step."""
#     return await _execute_specialized_step(ctx, "rust_programmer")
#
#
# async def security_auditor_step(
#         ctx: StepContext[GraphState, GraphDeps, None],
# ) -> str | End[Any]:
#     """Specialized Security expert step."""
#     return await _execute_specialized_step(ctx, "security_auditor")
#
#
# async def javascript_programmer_step(
#         ctx: StepContext[GraphState, GraphDeps, None],
# ) -> str | End[Any]:
#     """Specialized JavaScript expert step."""
#     return await _execute_specialized_step(ctx, "javascript_programmer")
#
#
# async def c_programmer_step(
#         ctx: StepContext[GraphState, GraphDeps, None],
# ) -> str | End[Any]:
#     """Specialized C expert step."""
#     return await _execute_specialized_step(ctx, "c_programmer")
#
#
# async def cpp_programmer_step(
#         ctx: StepContext[GraphState, GraphDeps, None],
# ) -> str | End[Any]:
#     """Specialized C++ expert step."""
#     return await _execute_specialized_step(ctx, "cpp_programmer")
#
#
# async def qa_expert_step(
#         ctx: StepContext[GraphState, GraphDeps, None],
# ) -> str | End[Any]:
#     """Specialized QA expert step."""
#     return await _execute_specialized_step(ctx, "qa_expert")
#
#
# async def debugger_expert_step(
#         ctx: StepContext[GraphState, GraphDeps, None],
# ) -> str | End[Any]:
#     """Specialized Debugging expert step."""
#     return await _execute_specialized_step(ctx, "debugger_expert")
#
#
# async def ui_ux_designer_step(
#         ctx: StepContext[GraphState, GraphDeps, None],
# ) -> str | End[Any]:
#     """Specialized UI/UX expert step."""
#     return await _execute_specialized_step(ctx, "ui_ux_designer")
#
#
# async def devops_engineer_step(
#         ctx: StepContext[GraphState, GraphDeps, None],
# ) -> str | End[Any]:
#     """Specialized DevOps expert step."""
#     return await _execute_specialized_step(ctx, "devops_engineer")
#
#
# async def cloud_architect_step(
#         ctx: StepContext[GraphState, GraphDeps, None],
# ) -> str | End[Any]:
#     """Specialized Cloud expert step."""
#     return await _execute_specialized_step(ctx, "cloud_architect")
#
#
# async def database_expert_step(
#         ctx: StepContext[GraphState, GraphDeps, None],
# ) -> str | End[Any]:
#     """Specialized Database expert step."""
#     return await _execute_specialized_step(ctx, "database_expert")
#


# async def usage_guard_step(
#         ctx: StepContext[GraphState, GraphDeps, Any],
# ) -> Any:
#     """Monitors token usage and cost, emitting warnings if limits are exceeded."""
#     usage = ctx.state.session_usage
#
#     # Defaults: $5.00 limit, 500k tokens
#     cost_limit = 5.0
#     token_limit = 500000
#
#     if usage.estimated_cost_usd > cost_limit or usage.total_tokens > token_limit:
#         logger.warning(
#             f"UsageGuard: Safety limits reached! Cost: ${usage.estimated_cost_usd:.2f}, Tokens: {usage.total_tokens}"
#         )
#         emit_graph_event(
#             ctx.deps.event_queue,
#             "safety_warning",
#             message=f"Session usage has exceeded safety limits. Current cost: ${usage.estimated_cost_usd:.2f}",
#             usage=usage.model_dump(),
#         )
#
#     return ctx.inputs
#
#
# async def approval_gate_step(
#         ctx: StepContext[GraphState, GraphDeps, Any],
# ) -> Any:
#     """Pauses for human approval and captures triage redirection feedback."""
#     if ctx.state.mode != "plan" and not ctx.state.human_approval_required:
#         return ctx.inputs
#
#     logger.info("Approval Gate: Pausing for user review...")
#     ctx.state.human_approval_required = True
#
#     # In a real-time SSE environment, the user would provide feedback via a separate endpoint/call.
#     # If the user provides a "Redirect" command, it will be populated in ctx.state.user_redirect_feedback.
#
#     if ctx.state.user_redirect_feedback:
#         logger.info(
#             "Approval Gate: Captured redirection feedback. Returning to router."
#         )
#         ctx.state.human_approval_required = False
#         return "router"
#
#     return ctx.inputs
#
#
# async def memory_selection_step(
#         ctx: StepContext[GraphState, GraphDeps, Any],
# ) -> str | End[Any]:
#     """Identifies relevant memories to load for the current query."""
#     logger.info("Memory Selection: Identifying relevant context...")
#     prompt_content = load_specialized_prompts("memory_selection")
#
#     # Discovery of local memory files
#     root = ctx.state.project_root or os.getcwd()
#     memories = []
#     for p in Path(root).rglob("*.md"):
#         if ".gemini" in str(p) or "node_modules" in str(p):
#             continue
#         try:
#             content = p.read_text(encoding="utf-8", errors="ignore")
#             # Extract description from frontmatter if available
#             description = "General project memory"
#             if content.startswith("---"):
#                 match = re.search(r"description:\s*(.*)", content)
#                 if match:
#                     description = match.group(1).strip()
#             memories.append(f"- {p.name}: {description}")
#         except Exception:
#             pass
#
#     from pydantic_ai import Agent
#
#     # We use a simple list of memories as context
#     selectors = Agent(
#         model=ctx.deps.agent_model,
#         system_prompt=prompt_content,
#         output_type=dict,  # Simple selected filenames
#     )
#
#     try:
#         res = await selectors.run(
#             f"Query: {ctx.state.query}\n\nAvailable memories:\n"
#             + "\n".join(memories[:20])
#         )
#         selected = res.output.get("selected_memories", [])
#         logger.info(
#             f"Memory Selection: Selected {len(selected)} relevant files: {selected}"
#         )
#
#         # Load the selected content into the state for other agents to consume
#         loaded_context = []
#         for filename in selected:
#             for p in Path(root).rglob(filename):
#                 loaded_context.append(
#                     f"### {filename}\n{p.read_text(encoding='utf-8', errors='ignore')}"
#                 )
#                 break
#
#         ctx.state.exploration_notes += "\n\n### SELECTED MEMORIES\n" + "\n\n".join(
#             loaded_context
#         )
#         return ctx.inputs  # Continue to original intended node
#     except Exception as e:
#         logger.error(f"Memory Selection failed: {e}")
#         return ctx.inputs
#
#
# async def planner_step(
#         ctx: StepContext[GraphState, GraphDeps, MultiDomainChoice | None],
# ) -> TaskList | str | End[Any]:
#     """
#     Consolidated planning step. Handles both project-mode decomposition and
#     simple sequential task list creation.
#     """
#     logger.info("Planner: Analyzing request and creating execution path...")
#     planner_prompt = load_specialized_prompts("planner")
#     memory_instruction = load_specialized_prompts("memory_instruction")
#     unified_context = await fetch_unified_context()
#
#     from pydantic_ai import Agent
#
#     planner = Agent(
#         model=ctx.deps.agent_model,
#         output_type=TaskList,
#         system_prompt=(
#             f"{memory_instruction}\n\n"
#             f"{planner_prompt}\n\n"
#             f"### CONTEXT\n{unified_context}\n\n"
#             f"### FINDINGS\n{ctx.state.exploration_notes}"
#         ),
#     )
#
#     # Integrated Worktree & Development Tools
#     planner.tool(get_git_status)
#     planner.tool(create_worktree)
#     planner.tool(list_worktrees)
#     planner.tool(project_search)
#
#     try:
#         # Determine goal based on input type or state
#         goal = ctx.state.query
#         if isinstance(ctx.inputs, MultiDomainChoice):
#             goal = f"Goal: {goal}\nReasoning: {ctx.inputs.reasoning}"
#
#         res = await planner.run(goal)
#         ctx.state.task_list = res.output
#         ctx.state.sync_to_disk()
#
#         # In 'Plan' mode, we might just end here
#         if ctx.state.mode == "plan":
#             return End({"status": "planned", "plan": ctx.state.task_list.model_dump()})
#
#         # Return the next node in the graph (usually ProjectExecutor or similar)
#         return "ProjectExecutor"
#     except Exception as e:
#         logger.error(f"Planning failed: {e}")
#         return "router_step"  # Retry via re-planning
#
#
# async def project_executor_step(
#         ctx: StepContext[GraphState, GraphDeps, Task],
# ) -> Task:
#     """Executes a single task from a project plan with specialized agent support."""
#     task = ctx.inputs
#     task.status = TaskStatus.IN_PROGRESS
#
#     # Specialist Mapping
#     specialist_map = {
#         "python": "python_programmer",
#         "c": "c_programmer",
#         "cpp": "cpp_programmer",
#         "golang": "golang_programmer",
#         "javascript": "javascript_programmer",
#         "typescript": "typescript_programmer",
#         "rust": "rust_programmer",
#         "security": "security_auditor",
#         "qa": "qa_expert",
#         "debugger": "debugger_expert",
#         "ui_ux": "ui_ux_designer",
#         "devops": "devops_engineer",
#         "cloud": "cloud_architect",
#         "database": "database_expert",
#     }
#
#     prompt_name = None
#     task_type_lower = task.type.lower()
#     for key, name in specialist_map.items():
#         if key in task_type_lower:
#             prompt_name = name
#             break
#
#     if prompt_name:
#         logger.info(
#             f"project_executor_step: Using specialized agent '{prompt_name}' for task '{task.title}'"
#         )
#         special_prompt = load_specialized_prompts(prompt_name)
#         system_prompt = f"Global Goal: {ctx.state.query}\n\nSpecialized Context: {special_prompt}\n\nTarget Task: {task.title}\nDescription: {task.description}"
#     else:
#         system_prompt = f"Task Context: {ctx.state.query}\n\nTask: {task.title}\nDescription: {task.description}"
#
#     from .tools.developer_tools import developer_tools
#
#     agent = Agent(
#         model=ctx.deps.agent_model,
#         system_prompt=system_prompt,
#         tools=developer_tools,
#     )
#     for toolset in ctx.deps.mcp_toolsets:
#         agent.toolsets.append(toolset)
#
#     try:
#         res = await agent.run(f"Execute task: {task.title}")
#         task.result = str(res.data) if hasattr(res, "data") else str(res.output)
#         task.status = TaskStatus.COMPLETED
#     except Exception as e:
#         logger.error(f"Task '{task.title}' failed: {e}")
#         task.status = TaskStatus.FAILED
#         task.result = str(e)
#     return task
# async def explorer_step(
#         ctx: StepContext[GraphState, GraphDeps, None],
# ) -> str:
#     """
#     Discovery step. Uses the explorer prompt to gather context before planning.
#     """
#     logger.info("Explorer: Discovering codebase context...")
#     explorer_prompt = load_specialized_prompts("explorer")
#     unified_context = await fetch_unified_context()
#
#     from pydantic_ai import Agent
#
#     explorer = Agent(
#         model=ctx.deps.agent_model,
#         system_prompt=explorer_prompt + f"\n\n{unified_context}",
#     )
#
#     # Register all developer and git tools for exploration
#     explorer.tool(project_search)
#     explorer.tool(list_files)
#     explorer.tool(get_git_status)
#     explorer.tool(list_worktrees)
#
#     # Add toolsets for additional context if needed
#     for toolset in ctx.deps.mcp_toolsets:
#         explorer.toolsets.append(toolset)
#
#     try:
#         res = await explorer.run(
#             f"Research and map out the context for: {ctx.state.query}"
#         )
#         ctx.state.exploration_notes = str(res.output)
#         return "Coordinator"
#     except Exception as e:
#         logger.error(f"Exploration failed: {e}")
#         return "Error"
#
#
# async def coordinator_step(
#         ctx: StepContext[GraphState, GraphDeps, None],
# ) -> str:
#     """
#     Strategy step. Synthesizes findings into a specific implementation plan.
#     """
#     logger.info("Coordinator: Formulating strategy...")
#     coordinator_prompt = load_specialized_prompts("coordinator")
#
#     from pydantic_ai import Agent
#
#     coordinator = Agent(
#         model=ctx.deps.agent_model,
#         system_prompt=coordinator_prompt
#                       + f"\n\nExploration Findings:\n{ctx.state.exploration_notes}",
#     )
#
#     prompt = f"Goal: {ctx.state.query}\n\nBased on exploration, determine if we need architectural design ('Architect') or can go straight to task planning ('Planner')."
#
#     try:
#         # For now, we'll use a simple classification or just default to Architect if complex
#         if (
#                 "architect" in ctx.state.query.lower()
#                 or len(ctx.state.exploration_notes) > 1000
#         ):
#             return "Architect"
#         return "Planner"
#     except Exception as e:
#         logger.error(f"Coordination failed: {e}")
#         return "Error"


# async def critique_step(
#         ctx: StepContext[GraphState, GraphDeps, None],
# ) -> str:
#     """
#     Self-correction step. Analyzes verification failures.
#     """
#     logger.info("Critique: Analyzing failures...")
#     critique_prompt = load_specialized_prompts("critique")
#
#     # Return to planner for fix
#     return "Planner"
