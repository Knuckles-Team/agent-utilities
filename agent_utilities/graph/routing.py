#!/usr/bin/python
"""Graph Routing Steps.

Core orchestration: routing, dispatching, parallel execution, and MCP routing.
Extracted from the monolithic steps.py for maintainability.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
from typing import Any

from pydantic_ai import Agent
from pydantic_graph import End
from pydantic_graph.beta import StepContext

from ..models import (
    ExecutionStep,
    GraphPlan,
    GraphResponse,
    ParallelBatch,
)
from .config_helpers import (
    emit_graph_event,
    get_discovery_registry,
    get_relevant_specialists,
    load_mcp_config,
    load_node_agents_registry,
    load_specialized_prompts,
)
from .executor import (
    _execute_domain_logic,
    _execute_dynamic_mcp_agent,
)
from .hierarchical_planner import fetch_unified_context
from .hsm import StateInvariantError, assert_state_valid
from .lifecycle import _emit_node_lifecycle
from .state import GraphDeps, GraphState

logger = logging.getLogger(__name__)

__all__ = [
    "router_step",
    "dispatcher_step",
    "parallel_batch_processor",
    "expert_executor_step",
    "dynamic_mcp_routing_step",
    "mcp_server_step",
    "council_step",
]


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
        # CONCEPT:KG-2.1 — Adaptive Model Routing (Fast Path)
        import os

        from ..core.model_factory import create_model

        lightweight_model = create_model(
            model_id=os.environ.get("LIGHTWEIGHT_MODEL", "gpt-4o-mini")
        )

        fast_agent = Agent(
            model=lightweight_model,
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

    # Topological Pre-Routing: Check the Knowledge Graph for direct tool matches and context
    discovery_context = ""
    if deps.knowledge_engine:
        logger.info(
            "[LAYER:GRAPH:ROUTER] Performing topological and hybrid discovery..."
        )

        # 1. Direct tool lookup (keyword based)
        words = re.findall(r"\b[a-z0-9_]{3,}\b", ctx.state.query.lower())
        matched_agents = set()
        for word in words:
            agents = deps.knowledge_engine.find_agent_for_tool(word)
            if agents:
                matched_agents.update(agents)

        # 2. Hybrid Search (Semantic + Keyword)
        hybrid_results = deps.knowledge_engine.search_hybrid(ctx.state.query, top_k=5)

        # 3. Policy and Process Discovery
        relevant_policies = deps.knowledge_engine.find_relevant_policies(
            ctx.state.query
        )
        relevant_processes = deps.knowledge_engine.find_relevant_processes(
            ctx.state.query
        )

        discovery_sections = []
        if relevant_policies:
            discovery_sections.append(
                "### APPLICABLE POLICIES (Governance)\n"
                + "\n".join(
                    [f"- {p['name']}: {p['description']}" for p in relevant_policies]
                )
            )

        if relevant_processes:
            discovery_sections.append(
                "### MATCHING PROCESS FLOWS (SOPs)\n"
                + "\n".join(
                    [f"- {f['name']}: Goal={f['goal']}" for f in relevant_processes]
                )
            )
        if matched_agents:
            discovery_sections.append(
                f"The following agents are confirmed to provide tools matching keywords in the query:\n"
                f"- {', '.join(matched_agents)}"
            )

        if hybrid_results:
            results_text = []
            for res in hybrid_results:
                rtype = res.get("type", "node").upper()
                name = res.get("name", res.get("id"))
                results_text.append(
                    f"- [{rtype}] {name}: {res.get('description', '')[:150]}..."
                )

            discovery_sections.append(
                "Knowledge Graph search found the following relevant entities:\n"
                + "\n".join(results_text)
            )

        if discovery_sections:
            discovery_context = (
                "### KNOWLEDGE GRAPH DISCOVERY\n"
                + "\n\n".join(discovery_sections)
                + "\n\nPRIORITIZE using these agents or referencing this context in your plan.\n\n"
            )
            logger.info(
                f"Router: Knowledge Graph discovery found {len(matched_agents)} tool-matched agents and {len(hybrid_results)} hybrid results."
            )

        # CONCEPT:AHE-3.3 — Check for matching TeamConfig before LLM planning
        if deps.knowledge_engine:
            try:
                from ..knowledge_graph.core.engine_registry import RegistryMixin

                if isinstance(deps.knowledge_engine, RegistryMixin) and hasattr(
                    deps.knowledge_engine, "find_matching_team_config"
                ):
                    matching_teams = deps.knowledge_engine.find_matching_team_config(
                        ctx.state.query, top_k=1
                    )
                    if (
                        matching_teams
                        and matching_teams[0].success_rate
                        > matching_teams[0].reuse_threshold
                    ):
                        team = matching_teams[0]
                        logger.info(
                            f"Router: Reusing TeamConfig '{team.task_pattern}' "
                            f"(success_rate={team.success_rate:.0%}, usage={team.usage_count})"
                        )
                        steps = [
                            ExecutionStep(node_id=sid, input_data=ctx.state.query)
                            for sid in team.specialist_ids
                        ]
                        plan = GraphPlan(
                            steps=steps,
                            metadata={
                                "reasoning": f"Reused proven TeamConfig: {team.task_pattern}",
                                "team_config_id": team.id,
                            },
                        )
                        ctx.state.plan = plan
                        emit_graph_event(
                            deps.event_queue,
                            "routing_completed",
                            plan=plan.model_dump(),
                            reasoning=f"TeamConfig reuse: {team.task_pattern}",
                        )
                        _emit_node_lifecycle(
                            deps.event_queue, "router", "node_complete"
                        )
                        return "dispatcher"
            except Exception as e:
                logger.debug(
                    f"TeamConfig lookup failed, continuing with LLM planning: {e}"
                )

        # CONCEPT:KG-2.1 — Inject Self-Model proficiency into discovery context
        if deps.knowledge_engine:
            try:
                from ..knowledge_graph.retrieval.memory_retriever import MemoryRetriever

                memory_retriever = MemoryRetriever(deps.knowledge_engine)
                current = memory_retriever.get_current()
                if current and current.domain_success_rates:
                    proficiency_lines = [
                        f"- {domain}: {rate:.0%} success rate"
                        for domain, rate in sorted(
                            current.domain_success_rates.items(),
                            key=lambda x: x[1],
                            reverse=True,
                        )[:5]
                    ]
                    discovery_context += (
                        "\n### YOUR DOMAIN PROFICIENCY (from Self-Model)\n"
                        + "\n".join(proficiency_lines)
                        + "\nPrefer routing to domains where you have proven competence.\n\n"
                    )

                    # CONCEPT:KG-2.1 — Inject ACO pheromone trail affinities
                    if current.pheromone_trails:
                        trail_lines = []
                        for specialist_id, domains in sorted(
                            current.pheromone_trails.items(),
                            key=lambda x: max(x[1].values()) if x[1] else 0,
                            reverse=True,
                        )[:7]:
                            top_domain = max(domains, key=domains.get)  # type: ignore[arg-type]
                            trail_lines.append(
                                f"- {specialist_id} → {top_domain} (affinity: {domains[top_domain]:.0%})"
                            )
                        if trail_lines:
                            discovery_context += (
                                "### SPECIALIST AFFINITIES (ACO Pheromone Trails)\n"
                                + "\n".join(trail_lines)
                                + "\nThese specialists have proven track records for these domains.\n\n"
                            )
            except Exception as e:
                logger.debug(f"Self-Model proficiency injection failed: {e}")
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

        # CONCEPT:ORCH-1.2 — Filtered specialist injection for prompt bloat reduction
        relevant = get_relevant_specialists(
            ctx.state.query, engine=deps.knowledge_engine, top_n=7
        )

        # CONCEPT:KG-2.1 — Reward-Driven Routing Optimization
        # Leverage existing ACO pheromone trails (hidden value-add) to filter out low-performing specialists
        try:
            from ..knowledge_graph.retrieval.memory_retriever import MemoryRetriever

            memory_retriever = MemoryRetriever(deps.knowledge_engine)
            current = memory_retriever.get_current()
            if current and current.pheromone_trails and relevant:
                optimized_relevant = []
                for a in relevant:
                    trails = current.pheromone_trails.get(a.name, {})
                    if trails:
                        avg_affinity = sum(trails.values()) / len(trails)
                        if (
                            avg_affinity < 0.1
                        ):  # Downweight historically poor performers
                            logger.info(
                                f"Router: Reward-Driven Optimization — Dropping '{a.name}' due to low historical affinity ({avg_affinity:.2f})"
                            )
                            continue
                    optimized_relevant.append(a)
                relevant = optimized_relevant
        except Exception as e:
            logger.debug(f"Reward-driven routing optimization failed: {e}")
        if relevant:
            step_info = "\n".join([f"- {a.name}: {a.description}" for a in relevant])
            # Append compact fallback list of OTHER specialists (name-only)
            relevant_names = {a.name.lower() for a in relevant}
            other_names = [
                tag for tag in specialist_tags if tag.lower() not in relevant_names
            ]
            if other_names:
                step_info += (
                    f"\n\nOther available specialists (request if needed): "
                    f"{', '.join(other_names)}"
                )
        else:
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
            f"### SUBTASK SPECIFICATION (CONCEPT:ORCH-1.1)\n"
            f"For EACH step in your plan, include a 'refined_subtask' — a focused, "
            f"specific instruction tailored for that specialist. Do NOT just repeat the "
            f"user query. Instead, decompose it into a targeted sub-goal. Example: if the "
            f"user asks 'build a REST API with auth', the python_programmer step should "
            f"get refined_subtask='Implement a FastAPI REST API with JWT authentication "
            f"middleware' — not the raw query.\n"
            f"You may also specify 'access_list' per step to control which prior step "
            f"results are visible. Use ['all'] for full context, specific node_ids for "
            f"selective injection, or leave empty for no prior context.\n\n"
            f"### WIDE-SEARCH ORCHESTRATION (CONCEPT:ORCH-1.1)\n"
            f"If the query requests extracting a large table of data across many entities "
            f"(e.g., 'Web2WideSearch' or 'Wide-Search'), you MUST decompose the extraction "
            f"into discrete batches. Emit multiple parallel ExecutionSteps assigned to an "
            f"extraction specialist (e.g., 'researcher' or 'web_researcher'), each targeting "
            f"a specific partition of the data in its 'refined_subtask' (e.g., 'Extract rows "
            f"for entities A-D'). Set 'is_parallel=True' and configure 'access_list' to "
            f"share the shared workboard context if necessary.\n\n"
            f"### FAILURE CONTEXT\n{failure_context}\n\n"
            f"{discovery_context}"
            f"### AVAILABLE SPECIALIST NODES\n{step_info}\n\n"
            f"### PROJECT CONTEXT\n{unified_context}"
        )
        from ..rlm.config import RLMConfig
        from ..rlm.repl import RLMEnvironment

        rlm_config = RLMConfig()
        use_rlm = (
            rlm_config.enabled
            or len(unified_context) > rlm_config.max_context_threshold
        )

        if use_rlm:
            logger.info(
                "[LAYER:GRAPH:ROUTER] Running in RLM (Recursive Language Model) mode."
            )
            env = RLMEnvironment(
                context=f"SYSTEM_PROMPT:\n{system_prompt_str}\n\nPROJECT_CONTEXT:\n{unified_context}",
                config=rlm_config,
                graph_deps=ctx.deps,
            )
            # Instruct the RLM to generate a GraphPlan
            rlm_result = await env.run_full_rlm(
                f"Create a high-level execution plan for the query: {ctx.state.query}\n\n"
                "Use the REPL to analyze available specialists and the project context. "
                "Decompose the goal into steps that can be handled by the specialists. "
                "You MUST output a valid JSON representation of a GraphPlan. "
                "The GraphPlan should have 'steps' (list of {node_id, input_data}) and 'metadata' ({reasoning}). "
                "Use FINAL_VAR('plan', <json_string>)."
            )

            import json

            try:
                plan_data = json.loads(rlm_result)
                plan_output = GraphPlan.model_validate(plan_data)
            except Exception as parse_e:
                logger.warning(
                    f"RLM output was not valid GraphPlan JSON: {parse_e}. Running fallback parser."
                )
                router_agent = Agent(
                    model=deps.router_model,
                    output_type=GraphPlan,
                    system_prompt="Parse the following text into a valid GraphPlan JSON structure.",
                )
                res = await router_agent.run(f"Text to parse:\n{rlm_result}")
                plan_output = res.output
        else:
            # CONCEPT:KG-2.1 — Adaptive Model Routing (Planner Path)
            # CONCEPT:AHE-3.24 — KG-Native Agentic Task Detection
            # CONCEPT:AHE-3.25 — Topological Reasoning Detection
            import os

            query_length = len(ctx.state.query.split())
            is_complex = False
            requires_reasoning = False

            # Text heuristics (fallback)
            if (
                "complex" in ctx.state.query.lower()
                or "architect" in ctx.state.query.lower()
                or len(relevant) > 3
            ):
                is_complex = True

            if (
                "step by step" in ctx.state.query.lower()
                or "think through" in ctx.state.query.lower()
            ):
                requires_reasoning = True

            # KG-Native Topological overrides
            if deps.knowledge_engine:
                try:
                    # CONCEPT:AHE-3.24 Agentic detection
                    task_topologies = deps.knowledge_engine.search_hybrid(
                        ctx.state.query + " TradingPipeline RiskScoringOntology",
                        top_k=2,
                    )
                    if any(
                        "Trading" in t.get("name", "") or "Risk" in t.get("name", "")
                        for t in task_topologies
                    ):
                        is_complex = True
                        logger.info(
                            "Router: CONCEPT:AHE-3.24 — Detected complex topological subgraphs. Escalate to complex model."
                        )

                    # CONCEPT:AHE-3.25 Reasoning detection
                    math_topologies = deps.knowledge_engine.search_hybrid(
                        ctx.state.query
                        + " MathematicalFoundationNode vectorized topologies OWL Almgren-Chriss",
                        top_k=2,
                    )
                    if any(
                        "Math" in t.get("name", "")
                        or "Quant" in t.get("name", "")
                        or "Almgren" in t.get("name", "")
                        for t in math_topologies
                    ):
                        requires_reasoning = True
                        logger.info(
                            "Router: CONCEPT:AHE-3.25 — Detected mathematical/quantitative topology. Escalate to reasoning model."
                        )
                except Exception as e:
                    logger.warning(f"Topological routing detection failed: {e}")

            # CONCEPT:OS-5.19 — Topological Session Persistence
            if ctx.state.pinned_model_id:
                from ..core.model_factory import create_model

                adaptive_model = create_model(model_id=ctx.state.pinned_model_id)
                logger.info(
                    f"[LAYER:GRAPH:ROUTER] OS-5.19: Reusing pinned session model: {ctx.state.pinned_model_id}"
                )
            elif requires_reasoning:
                from ..core.model_factory import create_model

                reasoning_model_id = os.environ.get("REASONING_MODEL", "o3-mini")
                adaptive_model = create_model(model_id=reasoning_model_id)
                ctx.state.pinned_model_id = reasoning_model_id
                logger.info(
                    f"[LAYER:GRAPH:ROUTER] Selected Reasoning Model: {reasoning_model_id}"
                )
            elif query_length < 20 and not is_complex:
                from ..core.model_factory import create_model

                adaptive_model = create_model(
                    model_id=os.environ.get("LIGHTWEIGHT_MODEL", "gpt-4o-mini")
                )
                logger.info(
                    "[LAYER:GRAPH:ROUTER] Adaptive Routing: Selected lightweight model for simple task."
                )
            else:
                adaptive_model = deps.router_model
                # Only pin if it has a string name we can recover later
                if hasattr(deps.router_model, "model_name"):
                    ctx.state.pinned_model_id = deps.router_model.model_name

            router_agent = Agent(
                model=adaptive_model,
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
            except TimeoutError:
                logger.warning(
                    "Router: LLM planning timed out. Escalating to fallbacks."
                )
                raise ValueError("LLM planning timed out") from None

            usage = stream.usage()
            if asyncio.iscoroutine(usage):
                usage = await usage
            ctx.state._update_usage(usage)
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

    # CONCEPT:ORCH-1.3 — Execution Budget (Cost Governor) enforcement
    import time

    budget = ctx.state.execution_budget

    if (
        budget.max_node_transitions
        and ctx.state.node_transitions > budget.max_node_transitions
    ):
        logger.error(
            f"Dispatcher: Execution budget exceeded for node transitions ({ctx.state.node_transitions} > {budget.max_node_transitions})"
        )
        ctx.state.error = "Execution budget exceeded: max node transitions."
        return "error_recovery"

    if (
        budget.max_total_tokens
        and ctx.state.session_usage.total_tokens > budget.max_total_tokens
    ):
        logger.error(
            f"Dispatcher: Execution budget exceeded for tokens ({ctx.state.session_usage.total_tokens} > {budget.max_total_tokens})"
        )
        ctx.state.error = "Execution budget exceeded: max total tokens."
        return "error_recovery"

    if (
        budget.max_cost_usd
        and ctx.state.session_usage.estimated_cost_usd > budget.max_cost_usd
    ):
        logger.error(
            f"Dispatcher: Execution budget exceeded for cost (${ctx.state.session_usage.estimated_cost_usd} > ${budget.max_cost_usd})"
        )
        ctx.state.error = "Execution budget exceeded: max cost USD."
        return "error_recovery"

    if budget.start_time and budget.max_duration_seconds:
        elapsed = time.time() - budget.start_time
        if elapsed > budget.max_duration_seconds:
            logger.error(
                f"Dispatcher: Execution budget exceeded for duration ({elapsed}s > {budget.max_duration_seconds}s)"
            )
            ctx.state.error = "Execution budget exceeded: max duration."
            return "error_recovery"

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

    # CONCEPT:OS-5.18 — Doom Loop Detection at transition boundary
    try:
        from ..security.doom_loop_detector import DoomLoopDetector

        detector = DoomLoopDetector(session_id=ctx.state.session_id)
        # Feed node history as tool calls for pattern detection
        history = ctx.state.node_history[-20:] if ctx.state.node_history else []
        for node_name in history:
            detector.record_call(node_name)
        incident = detector.check()
        if incident is not None:
            logger.error("Dispatcher: Doom loop detected: %s", incident.name)
            ctx.state.error = f"Doom loop detected: {incident.name}"
            return "error_recovery"
    except ImportError:
        pass
    except Exception as e:
        logger.debug("Doom loop detection skipped: %s", e)

    # CONCEPT:ORCH-1.16 — State checkpoint at transition boundary
    try:
        from .state_checkpoint import StateCheckpointer

        if hasattr(ctx.deps, "knowledge_engine") and ctx.deps.knowledge_engine:
            checkpointer = StateCheckpointer(engine=ctx.deps.knowledge_engine)
            checkpointer.checkpoint(ctx.state, session_id=ctx.state.session_id)
    except ImportError:
        pass
    except Exception as e:
        logger.debug("State checkpointing skipped: %s", e)

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
    if (
        ctx.state.step_cursor == 0
        and hasattr(ctx.state.plan, "steps")
        and len(ctx.state.plan.steps) > 1
    ):
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

    plan_len = len(ctx.state.plan.steps) if hasattr(ctx.state.plan, "steps") else 0
    logger.info(
        f"Dispatcher: Handling graph execution (Step {ctx.state.step_cursor}/{plan_len})"
    )
    if not hasattr(ctx.state.plan, "steps") or ctx.state.step_cursor >= len(
        ctx.state.plan.steps
    ):
        # All plan steps have been executed.  Mark every step completed
        # and sync to ACP before handing off to the verifier.
        if hasattr(ctx.state.plan, "steps"):
            for step in ctx.state.plan.steps:
                step.status = "completed"
        if ctx.deps.plan_sync:
            with contextlib.suppress(Exception):
                await ctx.deps.plan_sync(
                    "step_completed", ctx.state.plan.to_acp_plan_entries()
                )

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
    if not hasattr(ctx.state.plan, "steps"):
        logger.error("Dispatcher: Plan is not a valid GraphPlan object.")
        return "error_recovery"

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

        # CONCEPT:ORCH-1.0 — Stigmergy Signal Board injection
        # If prior specialists left signals, emit them so downstream
        # specialists and the UI are aware of cross-node observations.
        if ctx.state.signal_board:
            signal_summary = "; ".join(
                f"{sig_type}: {', '.join(msgs[:3])}"
                for sig_type, msgs in ctx.state.signal_board.items()
            )
            emit_graph_event(
                ctx.deps.event_queue,
                "signal_board_context",
                node_id=current_step.node_id,
                signals=dict(ctx.state.signal_board),
                summary=signal_summary[:500],
            )

        # Bridge: mark the step as in_progress in ACP plan state.
        if ctx.deps.plan_sync:
            with contextlib.suppress(Exception):
                current_step.status = "in_progress"
                await ctx.deps.plan_sync(
                    "step_started", ctx.state.plan.to_acp_plan_entries()
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

            # CORE ARCHITECTURE STEPS (Preserved for pipeline stability)
            # Lazy imports to avoid circular dependencies between submodules
            from .hierarchical_planner import (
                architect_step,
                planner_step,
                researcher_step,
            )
            from .verification import verifier_step

            if node_id == "researcher":
                await researcher_step(ctx)
            elif node_id == "architect":
                await architect_step(ctx)
            elif node_id == "planner":
                await planner_step(ctx)
            elif node_id == "verifier":
                await verifier_step(ctx)
            elif node_id == "mcp_server":
                domain = ""
                input_data = step.input_data
                if isinstance(input_data, dict):
                    domain = input_data.get("domain", "")
                await _execute_domain_logic(ctx, domain)

            # DYNAMIC GRAPH-NATIVE AGENT SPAWNING
            else:
                logger.info(
                    f"Expert Execution: Spawning dynamic agent for task '{node_id}'"
                )

                # 1. Query Knowledge Graph for best tools & prompts
                engine = ctx.deps.knowledge_engine
                system_prompt = (
                    f"You are a specialized agent handling the task: {node_id}."
                )
                tools_to_inject = []

                if engine:
                    # Check for explicit prompt node
                    prompt_res = engine.query_cypher(
                        "MATCH (p:Prompt) WHERE toLower(p.name) CONTAINS toLower($name) RETURN p.system_prompt AS sp LIMIT 1",
                        {"name": node_id},
                    )
                    if prompt_res and "sp" in prompt_res[0]:
                        system_prompt = prompt_res[0]["sp"]

                    # Find relevant tools (by tag or semantic overlap if we had embeddings, using tag heuristic for now)
                    tool_res = engine.query_cypher(
                        "MATCH (t:Tool) WHERE any(tag IN t.tags WHERE toLower(tag) CONTAINS toLower($name)) OR toLower(t.name) CONTAINS toLower($name) RETURN t.name AS name, t.mcp_server AS server ORDER BY t.relevance_score DESC LIMIT 5",
                        {"name": node_id},
                    )
                    tools_to_inject = [t["name"] for t in tool_res]

                logger.info(
                    f"Dynamic Agent '{node_id}': Injecting {len(tools_to_inject)} tools from Knowledge Graph."
                )

                # 2. Execute Dynamic Agent
                from .executor import _get_domain_tools

                domain_tools, domain_toolsets = await _get_domain_tools(
                    "mcp_server_execution", ctx.deps
                )

                # Filter down to the exact tools
                if tools_to_inject:
                    filtered_tools = [
                        t for t in domain_tools if t.__name__ in tools_to_inject
                    ]
                    if filtered_tools:
                        domain_tools = filtered_tools

                dynamic_agent = Agent(
                    model=ctx.deps.agent_model,
                    system_prompt=system_prompt,
                    tools=domain_tools,
                    toolsets=domain_toolsets,
                )

                async with dynamic_agent.run_stream(
                    f"Task context: {step.input_data}"
                ) as stream:
                    res = await asyncio.wait_for(
                        stream.get_output(), timeout=ctx.deps.verifier_timeout
                    )

                ctx.state.results_registry[node_id] = str(res)

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


async def dynamic_mcp_routing_step(
    ctx: StepContext[GraphState, GraphDeps, None],
) -> list[str]:
    """Calculate the list of target resources for dynamic tool discovery.

    Queries the Knowledge Graph for CallableResourceNode types (MCP servers,
    A2A agents, skills) that should be probed for general-purpose execution.

    Args:
        ctx: The pydantic-graph step context.

    Returns:
        A list of resource names to be used as map() inputs for execution.
    """
    engine = ctx.deps.knowledge_engine
    targets = []

    if engine and hasattr(engine, "discover_callable_resources"):
        resources = engine.discover_callable_resources()
        if resources:
            targets = [res.name for res in resources]

    if not targets:
        # Fallback to local config if KG is empty or disabled
        mcp_config = load_mcp_config()
        targets = list(mcp_config.mcpServers.keys())

    logger.info(
        f"Dynamic Resource Routing: Routing to {len(targets)} targets: {targets}"
    )
    return targets


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
        engine = ctx.deps.knowledge_engine
        resource_node = None
        if engine and hasattr(engine, "ogm"):
            from ..models.knowledge_graph import CallableResourceNode

            nodes = engine.ogm.find(
                CallableResourceNode, properties={"name": server_name}
            )
            if nodes:
                resource_node = nodes[0]

        # Check if there's a matching dynamic MCP agent in the registry
        registry = load_node_agents_registry()
        matching_agents = [a for a in registry.agents if a.mcp_server == server_name]

        if matching_agents:
            # Execute each matching specialist agent for this server
            for mcp_agent in matching_agents:
                await _execute_dynamic_mcp_agent(ctx, mcp_agent)
        else:
            # Use unified resource metadata if available, otherwise fallback
            system_prompt = f"You are a specialist for the '{server_name}' resource. Use the available tools to answer queries."
            if resource_node and resource_node.description:
                system_prompt += f" Context: {resource_node.description}"

            # Fallback: create ad-hoc agent with all tools from this server
            matched_toolsets = []
            for toolset in ctx.deps.mcp_toolsets:
                server_id = getattr(toolset, "id", getattr(toolset, "name", None))
                if server_id and server_name in str(server_id):
                    matched_toolsets.append(toolset)

            agent = Agent(
                model=ctx.deps.agent_model,
                system_prompt=system_prompt,
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
            usage = stream.usage()
            if asyncio.iscoroutine(usage):
                usage = await usage
            ctx.state._update_usage(usage)
            ctx.state.results[server_name] = str(output)
            ctx.state.results_registry[f"{server_name}_{ctx.state.step_cursor}"] = str(
                output
            )

            # Stream events to WebUI
            if ctx.deps.event_queue:
                from pydantic_ai.messages import (
                    ModelRequest,
                    ModelResponse,
                    ToolCallPart,
                    ToolReturnPart,
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


async def council_step(
    ctx: StepContext[GraphState, GraphDeps, ExecutionStep | str],
) -> str:
    """Run a multi-perspective council deliberation on the user's query.

    Dispatches 5 advisors (Contrarian, First Principles, Expansionist,
    Outsider, Executor) in parallel, anonymizes their responses for bias
    prevention, runs peer reviewers, and produces a chairman-synthesized
    :class:`CouncilVerdict`.

    The council is invoked by the dispatcher when the router selects
    ``council`` as a plan step, or when a specialist calls the council
    as a tool for high-stakes decisions.

    Args:
        ctx: The pydantic-graph step context, potentially containing
            a specific structured question via ``ExecutionStep``.

    Returns:
        The ID of the next node to execute (``'execution_joiner'``).

    """
    from .council import run_council_deliberation

    _emit_node_lifecycle(ctx.deps.event_queue, "council", "node_start")

    # If the dispatcher sent a specific question, use it
    step_input = ctx.inputs
    council_query = ctx.state.query
    if isinstance(step_input, ExecutionStep) and step_input.input_data:
        if isinstance(step_input.input_data, dict):
            council_query = step_input.input_data.get("question", council_query)
        elif isinstance(step_input.input_data, str):
            council_query = step_input.input_data

    try:
        verdict, transcript = await run_council_deliberation(
            query=council_query,
            ctx_deps=ctx.deps,
            ctx_state=ctx.state,
        )

        # Store structured verdict in results registry
        node_uid = f"council_{ctx.state.step_cursor}"
        ctx.state.results_registry[node_uid] = verdict.model_dump_json()
        ctx.state.results["council"] = verdict.model_dump_json()

        # Store markdown transcript for human inspection
        ctx.state.results_registry[f"{node_uid}_transcript"] = transcript.to_markdown()

        # Persist to Knowledge Graph as a DecisionNode (if available)
        if ctx.deps.knowledge_engine:
            try:
                ctx.deps.knowledge_engine.add_node(
                    node_id=f"council_verdict_{ctx.state.step_cursor}",
                    node_type="Decision",
                    properties={
                        "name": f"Council Verdict: {council_query[:80]}",
                        "description": verdict.final_recommendation,
                        "rationale": "; ".join(verdict.key_insights),
                        "confidence": verdict.confidence / 10.0,
                    },
                )
            except Exception as e:
                logger.debug(f"Failed to persist council verdict to KG: {e}")

        logger.info(
            f"[COUNCIL] Verdict stored. Confidence: {verdict.confidence}/10. "
            f"Key insights: {len(verdict.key_insights)}"
        )

        _emit_node_lifecycle(
            ctx.deps.event_queue,
            "council",
            "node_complete",
            next_node="execution_joiner",
        )
        return "execution_joiner"

    except Exception as e:
        logger.error(f"Council deliberation failed: {e}")
        ctx.state.results_registry[f"council_{ctx.state.step_cursor}"] = (
            f"Council failed: {e}"
        )
        _emit_node_lifecycle(
            ctx.deps.event_queue,
            "council",
            "node_complete",
            next_node="execution_joiner",
        )
        return "execution_joiner"
