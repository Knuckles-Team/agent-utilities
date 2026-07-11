#!/usr/bin/python
from __future__ import annotations

"""Graph Routing Steps.

Core orchestration: routing, dispatching, parallel execution, and MCP routing.
Extracted from the monolithic steps.py for maintainability.
"""


import asyncio
import contextlib
import logging
import re
from typing import Any

from pydantic_ai import Agent
from pydantic_graph import End

from agent_utilities.core.config import setting

try:
    from pydantic_graph.step import StepContext
except ImportError:
    from pydantic_graph.beta import StepContext

from agent_utilities.core.config import (
    config,
    emit_graph_event,
    get_discovery_registry,
    get_relevant_specialists,
    load_mcp_config,
    load_node_agents_registry,
    load_specialized_prompts,
)

from ..models import (
    ExecutionStep,
    GraphPlan,
    GraphResponse,
    ParallelBatch,
)
from .executor import (
    _execute_domain_logic,
    _execute_dynamic_mcp_agent,
    apply_tool_scope,
    invoker_context_section,
    spawn_usage_limits,
)
from .hsm import StateInvariantError, assert_state_valid
from .lifecycle import _emit_node_lifecycle

logger = logging.getLogger(__name__)

__all__ = [
    "router_step",
    "dispatcher_step",
    "parallel_batch_processor",
    "expert_executor_step",
    "dynamic_mcp_routing_step",
    "mcp_server_step",
]


async def router_step(
    ctx: StepContext,
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

    # CONCEPT:AU-ORCH.execution.direct-completion-shape — a direct-completion / lean turn is answered OUTSIDE this graph by
    # ``agent_runner._run_direct_completion`` (the planner's ``direct_complete`` shape, or the
    # structural classifier for a shape-less caller, short-circuits _execute_graph before the
    # graph is even built). The router therefore only ever runs for a real multi-step turn and
    # has a SINGLE outgoing edge to the dispatcher — it must NOT return ``End`` here, because a
    # second router edge to the end node makes pydantic-graph broadcast-fork the router output
    # to BOTH end and dispatcher, terminating every full-graph turn.

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

    # Topological Pre-Routing: Check the Knowledge Graph for direct tool matches and context.
    # CONCEPT:AU-ORCH.execution.direct-completion-shape — run this several-round-trip discovery bundle only when the job's
    # shape calls for it; a lean shape skips it.
    _shape = getattr(deps, "execution_shape", None)
    discovery_context = ""
    if deps.knowledge_engine and (
        _shape is None or getattr(_shape, "run_discovery", True)
    ):
        logger.info(
            "[LAYER:GRAPH:ROUTER] Performing topological and hybrid discovery..."
        )

        # CONCEPT:AU-ORCH.routing.offload-sync-roundtrip — the pre-LLM discovery below is several SYNCHRONOUS engine
        # round-trips (tool lookup, hybrid search, policy/process discovery). Running them
        # directly on the event loop stalled the async reply path. Run the whole bundle ONCE
        # in a worker thread via ``to_thread`` so the loop stays free. The keyword tool lookup
        # was also an N+1 — ``find_agent_for_tool`` once PER query word — now collapsed to a
        # single de-duplicated pass over the unique keyword set.
        #
        # NOTE (CONCEPT:AU-ORCH.execution.chat-profile-timeouts P2, future optimization): this whole bundle could collapse to
        # a single engine ``discover(query, k)`` round-trip (see
        # docs/architecture/non-blocking-execution.md §8) returning matched agents + hybrid hits
        # + policy/process matches in one Rust call, so the router's pre-LLM discovery is one
        # async hop instead of a thread-offloaded fan-out. Until the engine surfaces
        # ``discover()``, this dedupe/batch + ``to_thread`` is the Python-side mitigation (which
        # the current dedupe/batch implementation is complete and correct today).
        def _run_discovery() -> dict[str, Any]:
            ke = deps.knowledge_engine
            # 1. Direct tool lookup — ONE pass over the unique keyword set (was N+1).
            words = set(re.findall(r"\b[a-z0-9_]{3,}\b", ctx.state.query.lower()))
            _matched: set[str] = set()
            for word in words:
                agents = ke.find_agent_for_tool(word)
                if agents:
                    _matched.update(agents)
            # 1b. CONCEPT:AU-KG.memory.tiered-memory-caching — KG-driven designation (ANN capability index).
            try:
                from .routing.enrichers.capability_designation import (
                    designate_specialists,
                )

                designated = designate_specialists(ke, ctx.state.query, k=5)
                if designated:
                    _matched.update(designated)
            except Exception as e:  # noqa: BLE001
                logger.debug("Router: capability designation skipped: %s", e)
            # 2. Hybrid Search (Semantic + Keyword)
            _hybrid = ke.search_hybrid(ctx.state.query, top_k=5)
            # 3. Policy and Process Discovery
            _policies = ke.find_relevant_policies(ctx.state.query)
            _processes = ke.find_relevant_processes(ctx.state.query)
            return {
                "matched": _matched,
                "hybrid": _hybrid,
                "policies": _policies,
                "processes": _processes,
            }

        _discovery = await asyncio.to_thread(_run_discovery)
        matched_agents = _discovery["matched"]
        hybrid_results = _discovery["hybrid"]
        relevant_policies = _discovery["policies"]
        relevant_processes = _discovery["processes"]

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

        # CONCEPT:AU-AHE.harness.team-config-precheck — Check for matching TeamConfig before LLM planning
        if deps.knowledge_engine:
            try:
                from ..core.registry.kg_adapter import RegistryMixin

                if isinstance(deps.knowledge_engine, RegistryMixin) and hasattr(
                    deps.knowledge_engine, "find_matching_team_config"
                ):
                    # CONCEPT:AU-ORCH.routing.offload-sync-roundtrip — sync KG round-trip; run off the event loop.
                    matching_teams = await asyncio.to_thread(
                        deps.knowledge_engine.find_matching_team_config,
                        ctx.state.query,
                        1,
                    )
                    # R2 (CONCEPT:AU-AHE.harness.team-config-precheck): reuse decision owned by the team_reuse
                    # strategy (single source of truth).
                    from .routing.strategies.team_reuse import select_reusable_team

                    team = select_reusable_team(matching_teams)
                    if team:
                        logger.info(
                            f"Router: Reusing TeamConfig '{team.task_pattern}' "
                            f"(success_rate={team.success_rate:.0%}, usage={team.usage_count})"
                        )
                        steps = [
                            ExecutionStep(id=sid, description=ctx.state.query)
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

        # CONCEPT:AU-ORCH.adapter.kg-graph-materialization — KG-Driven Graph Materialization
        # Check for AgentTemplate nodes before falling back to LLM planning
        if deps.knowledge_engine:
            try:
                from .kg_graph_factory import build_pydantic_graph_from_kg

                kg_result = build_pydantic_graph_from_kg(
                    query=ctx.state.query,
                    engine=deps.knowledge_engine,
                    deps=deps,
                    top_k=7,
                )
                if kg_result.specialist_configs:
                    logger.info(
                        "[CONCEPT:AU-ORCH.adapter.kg-graph-materialization] KG graph materialized with %d steps. "
                        "Using KG-driven topology.",
                        len(kg_result.specialist_configs),
                    )
                    # Convert KG result into a standard GraphPlan for the dispatcher
                    steps = [
                        ExecutionStep(
                            id=cfg["agent_id"],
                            description=ctx.state.query,
                        )
                        for cfg in kg_result.specialist_configs.values()
                    ]
                    plan = GraphPlan(
                        steps=steps,
                        metadata={
                            "reasoning": f"KG-driven graph materialization ({len(steps)} templates)",
                            "kg_topology_id": kg_result.topology_id,
                            "kg_provenance": kg_result.kg_provenance,
                        },
                    )
                    ctx.state.plan = plan

                    # Store KG provenance in state for observability
                    if hasattr(ctx.state, "output_data") and isinstance(
                        ctx.state.output_data, dict
                    ):
                        ctx.state.output_data["kg_provenance"] = kg_result.kg_provenance
                        ctx.state.output_data[
                            "kg_specialist_configs"
                        ] = kg_result.specialist_configs

                    emit_graph_event(
                        deps.event_queue,
                        "routing_completed",
                        plan=plan.model_dump(),
                        reasoning="KG AgentTemplate materialization",
                    )
                    _emit_node_lifecycle(deps.event_queue, "router", "node_complete")
                    return "dispatcher"
            except Exception as e:
                logger.debug(
                    f"KG AgentTemplate routing failed, continuing with LLM planning: {e}"
                )

        # R4 (CONCEPT:AU-KG.memory.tiered-memory-caching) Self-Model proficiency + R5 ACO pheromone affinities —
        # context formatting owned by the self_model enricher (single source).
        if deps.knowledge_engine:
            try:
                from ..knowledge_graph.retrieval.memory_retriever import MemoryRetriever
                from .routing.enrichers.self_model import self_model_context

                memory_retriever = MemoryRetriever(deps.knowledge_engine)
                current = memory_retriever.get_current()
                discovery_context += self_model_context(current)
            except Exception as e:
                logger.debug(f"Self-Model proficiency injection failed: {e}")
    # CONCEPT:AU-ORCH.execution.orchestration-flow-mermaid (perf) — DIRECT-DISPATCH FAST PATH.
    # When the task resolves to a single connected MCP server (the common single-server
    # deployment), skip the planner + memory_selection + verifier entirely: build an agent
    # with just that server's toolset and run it ONCE. Collapses the ~5-call
    # plan→execute→verify loop (which, with both models at supports_json=false, also churns
    # on empty plans) down to a single execution call. First attempt only — re-plans keep
    # the full pipeline. Disable with GRAPH_DIRECT_DISPATCH=false.
    _direct_ok = (
        setting("GRAPH_DIRECT_DISPATCH", True)
        and not ctx.state.error
        and ctx.state.verification_attempts == 0
        and len(deps.mcp_toolsets) == 1
    )
    if _direct_ok:
        try:
            from .executor import agent_deps_from_graph

            _ts = deps.mcp_toolsets
            _srv = getattr(_ts[0], "id", getattr(_ts[0], "name", "mcp-server"))
            logger.info(
                "[LAYER:GRAPH:ROUTER] Direct-dispatch fast-path: single server '%s' — "
                "skipping planner/verifier.",
                _srv,
            )
            _, _scoped_ts = apply_tool_scope(
                ctx.state, [], _ts
            )  # CONCEPT:AU-ORCH.session.invoker-agent-handoff
            _direct_agent = Agent(
                model=deps.agent_model,
                system_prompt=(
                    f"You are operating the '{_srv}' MCP server. Use its tools to satisfy "
                    f"the user's request directly and return exactly the data requested."
                    f"{invoker_context_section(ctx.state)}"  # CONCEPT:AU-ORCH.session.invoker-agent-handoff
                ),
                toolsets=_scoped_ts,
            )
            _direct_deps = agent_deps_from_graph(deps, _ts, state=ctx.state)
            _direct_res = await _direct_agent.run(
                ctx.state.query,
                deps=_direct_deps,
                usage_limits=spawn_usage_limits(
                    ctx.state
                ),  # CONCEPT:AU-ORCH.session.invoker-agent-handoff budget
            )
            emit_graph_event(
                deps.event_queue,
                "routing_completed",
                plan={},
                reasoning=f"direct-dispatch: single server '{_srv}'",
            )
            return End(
                GraphResponse(
                    status="completed",
                    results={"output": str(_direct_res.output)},
                    metadata={
                        "direct_dispatch": True,
                        "server": _srv,
                        "domain": _srv,
                    },
                )
            )
        except Exception as e:  # noqa: BLE001 — fall back to full planning on any failure
            logger.warning(
                "[LAYER:GRAPH:ROUTER] Direct-dispatch failed (%s); "
                "falling back to full planning.",
                e,
            )

    # Reset cursor for the new plan
    ctx.state.step_cursor = 0

    failure_context = ""
    if ctx.state.error:
        failure_context = f"### PREVIOUS FAILURE CONTEXT\nThe last attempt failed with the following error:\n{ctx.state.error}\nUse this information to update your plan. You may need more research or a different approach."

    try:
        # Phase 4 Edge-Computed Scopes: Check if the workflow scope was already computed
        # and signed at the JWT edge layer, bypassing the persistent graph hit.
        if ctx.state.workflow_context and ctx.state.workflow_context.get(
            "edge_computed", False
        ):
            logger.info(
                "Using edge-computed JWT workflow context; bypassing persistent graph."
            )
            from .routing.strategies.workflow_context import ShieldedResult

            # Ensure workflow_id is present for Pydantic validation
            payload = dict(ctx.state.workflow_context)
            if "workflow_id" not in payload:
                payload["workflow_id"] = "jwt_edge_computed"
            workflow_context = ShieldedResult(**payload)
        else:
            from .routing.strategies.workflow_context import WorkflowContextRouter

            router = WorkflowContextRouter(deps.knowledge_engine)
            workflow_context = await router.route_context(ctx.state.query)
            ctx.state.workflow_context = workflow_context.model_dump()

        agent_context = workflow_context.to_prompt_string()

        logger.info("[LAYER:GRAPH:ROUTER] Fetching specialist tags...")
        specialist_tags = deps.tag_prompts
        if not specialist_tags:
            registry = get_discovery_registry()
            specialist_tags = {a.name: a.description for a in registry.agents}
            if specialist_tags:
                logger.info(
                    f"[LAYER:GRAPH:ROUTER] Specialist tags loaded (count: {len(specialist_tags)}). Tags: {list(specialist_tags.keys())}"
                )

        # CONCEPT:AU-ORCH.routing.filtered-specialist-injection — Filtered specialist injection for prompt bloat reduction
        relevant = get_relevant_specialists(
            ctx.state.query, engine=deps.knowledge_engine, top_n=7
        )

        # R7 (CONCEPT:AU-KG.memory.tiered-memory-caching) — Reward-Driven Optimization (pheromone filtering) and
        # R8 (CONCEPT:AU-AHE.optimization.telemetry-optimization) — Telemetry-Driven Optimization (anomaly pruning).
        # Both filters are owned by the optimization strategy (single source).
        from .routing.strategies.optimization import (
            filter_by_pheromone,
            format_specialist_step_info,
            prune_by_telemetry,
        )

        try:
            if deps.knowledge_engine:
                from ..knowledge_graph.retrieval.memory_retriever import MemoryRetriever

                memory_retriever = MemoryRetriever(deps.knowledge_engine)
                current = memory_retriever.get_current()
                if current and current.pheromone_trails and relevant:
                    relevant = filter_by_pheromone(relevant, current.pheromone_trails)
        except Exception as e:
            logger.debug(f"Reward-driven routing optimization failed: {e}")

        try:
            if deps.knowledge_engine:
                anomaly_results = deps.knowledge_engine.query_cypher(
                    "MATCH (a:Agent)-[:CAUSED]->(p:PerformanceAnomaly) "
                    "RETURN a.id AS agent_name, count(p) AS anomaly_count"
                )
                if anomaly_results:
                    anomaly_map = {
                        r.get("agent_name"): r.get("anomaly_count", 0)
                        for r in anomaly_results
                        if r.get("agent_name")
                    }
                    relevant = prune_by_telemetry(relevant, anomaly_map)
        except Exception as e:
            logger.debug(f"Telemetry-driven routing optimization failed: {e}")

        # R6 (CONCEPT:AU-ORCH.routing.filtered-specialist-injection) — filtered specialist injection (prompt-bloat reduction).
        step_info = format_specialist_step_info(relevant, specialist_tags)
        logger.info(
            f"Router: Specialists count: {len(specialist_tags)}, Context length: {len(agent_context)}"
        )

        # R9 (CONCEPT:AU-ORCH.routing.transition-state-checkpoint): subtask-spec + wide-search instructions — owned by
        # the llm_planner strategy (single source of truth).
        from .routing.strategies.llm_planner import (
            subtask_and_widesearch_instructions,
        )

        router_prompt = load_specialized_prompts("router")
        system_prompt_str = (
            f"{router_prompt}\n\n"
            f"### IMPORTANT: PLANNING ONLY MODE\n"
            f"You are a HIGH-LEVEL ARCHITECT. You DO NOT have access to functional tools (e.g. get_stack, Docker tools, etc.).\n"
            f"Your ONLY responsibility is to create the execution plan. DO NOT attempt to fulfill the query yourself.\n\n"
            f"{subtask_and_widesearch_instructions()}"
            f"### FAILURE CONTEXT\n{failure_context}\n\n"
            f"{discovery_context}"
            f"### AVAILABLE SPECIALIST NODES\n{step_info}\n\n"
            f"### PROJECT CONTEXT\n{agent_context}"
        )
        from ..rlm.config import RLMConfig
        from ..rlm.repl import RLMEnvironment

        rlm_config = RLMConfig()
        use_rlm = (
            rlm_config.enabled or len(agent_context) > rlm_config.max_context_threshold
        )

        if use_rlm:
            logger.info(
                "[LAYER:GRAPH:ROUTER] Running in RLM (Recursive Language Model) mode."
            )
            env = RLMEnvironment(
                context=f"SYSTEM_PROMPT:\n{system_prompt_str}\n\nPROJECT_CONTEXT:\n{agent_context}",
                config=rlm_config,
                graph_deps=ctx.deps,
            )
            # R10 (CONCEPT:AU-ORCH.execution.predict-rlm-runtime) — RLM planning + fallback parser. The
            # instruction text and JSON->GraphPlan parse are owned by the
            # llm_planner strategy (single source); the async RLM run + re-parse
            # agent stay here.
            from .routing.strategies.llm_planner import (
                parse_rlm_plan,
                rlm_plan_instruction,
            )

            rlm_result = await env.run_full_rlm(rlm_plan_instruction(ctx.state.query))

            plan_output = parse_rlm_plan(rlm_result, GraphPlan)
            if plan_output is None:
                logger.warning(
                    "RLM output was not valid GraphPlan JSON. Running fallback parser."
                )
                router_agent = Agent(
                    model=deps.router_model,
                    output_type=GraphPlan,
                    system_prompt="Parse the following text into a valid GraphPlan JSON structure.",
                )
                parse_res = await router_agent.run(f"Text to parse:\n{rlm_result}")
                plan_output = parse_res.output
        else:
            # CONCEPT:AU-KG.memory.tiered-memory-caching — Adaptive Model Routing (Planner Path)
            # CONCEPT:AU-AHE.evaluation.backtest-harness — KG-Native Agentic Task Detection
            # CONCEPT:AU-AHE.evaluation.backtest-harness — Topological Reasoning Detection

            query_length = len(ctx.state.query.split())
            is_complex = False
            requires_reasoning = False

            # R11 (CONCEPT:AU-AHE.evaluation.backtest-harness) — text-heuristic complexity detection, owned by
            # the llm_planner strategy (single source). Topology/quant escalation
            # below stays in the router (depends on live KG state).
            from .routing.strategies.llm_planner import is_complex_query

            if is_complex_query(ctx.state.query, len(relevant)):
                is_complex = True

            if (
                "step by step" in ctx.state.query.lower()
                or "think through" in ctx.state.query.lower()
            ):
                requires_reasoning = True

            # KG-Native Topological overrides
            if deps.knowledge_engine:
                try:
                    # CONCEPT:AU-AHE.evaluation.backtest-harness Agentic detection
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
                            "Router: CONCEPT:AU-AHE.evaluation.backtest-harness — Detected complex topological subgraphs. Escalate to complex model."
                        )

                    # CONCEPT:AU-AHE.evaluation.backtest-harness Reasoning detection
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
                            "Router: CONCEPT:AU-AHE.evaluation.backtest-harness — Detected mathematical/quantitative topology. Escalate to reasoning model."
                        )
                except Exception as e:
                    logger.warning(f"Topological routing detection failed: {e}")

            # CONCEPT:AU-OS.safety.doom-loop-detection — Topological Session Persistence
            if ctx.state.pinned_model_id:
                from ..core.model_factory import create_model

                adaptive_model = create_model(model_id=ctx.state.pinned_model_id)
                logger.info(
                    f"[LAYER:GRAPH:ROUTER] OS-5.19: Reusing pinned session model: {ctx.state.pinned_model_id}"
                )
            elif requires_reasoning:
                from ..core.model_factory import create_model

                _super = config.super_chat_model
                # CONCEPT:AU-ORCH.execution.direct-completion-shape — no hard-coded remote model; an unset super-model falls
                # back to the local default (``create_model(None)``), never an unreachable
                # ``o3-mini`` the homelab cannot serve.
                reasoning_model_id = _super.id if _super else None
                logger.debug(
                    "Router: pinning reasoning model %s",
                    reasoning_model_id or "local-default",
                )
                adaptive_model = create_model(model_id=reasoning_model_id)
                if reasoning_model_id:
                    ctx.state.pinned_model_id = reasoning_model_id
                logger.info(
                    f"[LAYER:GRAPH:ROUTER] Selected Reasoning Model: {reasoning_model_id}"
                )
            elif query_length < 20 and not is_complex:
                from ..core.model_factory import create_model

                _lite2 = config.lite_chat_model
                # CONCEPT:AU-ORCH.execution.direct-completion-shape — local default when no lite model is configured.
                adaptive_model = create_model(model_id=_lite2.id if _lite2 else None)
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
                if plan_output is None:
                    raise ValueError("LLM planning returned no plan")
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

            # v1/v2 compat shim: see agent_utilities/graph/executor.py's identical block.
            usage: Any = stream.usage  # v2: property (was a method in v1)
            if callable(usage):
                usage = usage()
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
        logger.error(f"Router planning failed: {e}. Attempting unstructured fallback.")
        try:
            # R13 (multi-level fallback chain) — unstructured natural-language
            # extraction. The prompt + specialist name-matching are owned by the
            # fallback strategy (single source).
            from .routing.strategies.fallback import (
                match_specialists_in_text,
                unstructured_fallback_prompt,
            )

            fallback_agent = Agent(
                model=adaptive_model,
                system_prompt=unstructured_fallback_prompt(system_prompt_str),
            )
            fallback_res = await fallback_agent.run(ctx.state.query)

            raw_text = str(
                getattr(fallback_res, "data", getattr(fallback_res, "output", ""))
            )
            available = (
                list(specialist_tags.keys()) if "specialist_tags" in locals() else []
            )
            if not available and hasattr(deps, "tag_prompts"):
                available = list(deps.tag_prompts.keys())

            steps = [
                ExecutionStep(id=spec, description=ctx.state.query)
                for spec in match_specialists_in_text(raw_text, available)
            ]

            if steps:
                logger.info(
                    f"Router Fallback: Extracted {len(steps)} steps from text: {[s.id for s in steps]}"
                )
                ctx.state.plan = GraphPlan(
                    steps=steps,
                    metadata={"reasoning": "Fallback natural language extraction"},
                )
                ctx.state.step_cursor = 0
                return "dispatcher"
            else:
                logger.warning(
                    f"Router Fallback: No known specialists found in text. Available: {available}. Raw text: {raw_text}"
                )
        except Exception as fallback_e:
            logger.error(f"Router fallback also failed: {fallback_e}")

        # Detailed logging for debugging
        if "res" in locals():
            logger.debug(f"Router raw response: {res}")

        ctx.state.error = f"Planning failed: {e}"
        return "__end__"


async def dispatcher_step(
    ctx: StepContext,
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

    # CONCEPT:AU-ORCH.execution.execution-budget-caps — Execution Budget (Cost Governor) enforcement
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

    # CONCEPT:AU-OS.safety.doom-loop-detection — Doom Loop Detection at transition boundary
    try:
        from ..security.execution_stability_engine import DoomLoopDetector

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

    # CONCEPT:AU-ORCH.routing.transition-state-checkpoint — State checkpoint at transition boundary.
    # Routes through the consolidated CheckpointManager (KG backend). The old
    # graph/state_checkpoint.StateCheckpointer was merged into core/checkpoint
    # (Plan 03 Step 8); the prior import silently failed, dropping this
    # capability — restored here.
    try:
        from ..core.checkpoint.manager import CheckpointManager

        if hasattr(ctx.deps, "knowledge_engine") and ctx.deps.knowledge_engine:
            checkpointer = CheckpointManager.create(
                persistence_type="kg", engine=ctx.deps.knowledge_engine
            )
            checkpointer.save(ctx.state, session_id=ctx.state.session_id)
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

    # Context enrichment: route to memory_selection on the first entry so historical context
    # is available before any plan steps execute — UNLESS the job's shape says this is a lean
    # turn that does not need pre-LLM context gathering (CONCEPT:AU-ORCH.execution.direct-completion-shape). memory_selection
    # gathers workspace/KG context; ``run_discovery`` is the shape's "gather context for this
    # job" signal, so a job the planner shaped as not needing it skips the node entirely.
    _shape = getattr(ctx.deps, "execution_shape", None)
    _want_context = _shape is None or getattr(_shape, "run_discovery", True)
    if ctx.state.step_cursor == 0 and not ctx.state.exploration_notes and _want_context:
        logger.info(
            "Dispatcher: First entry — routing to memory_selection for context enrichment."
        )
        return "memory_selection"

    # Phase-ordering guard: ensure research steps precede execution steps.
    # The LLM router may interleave them; we enforce discovery-first so that
    # research results are available to all execution adaptive_agent_router.
    _RESEARCH_NODES = {"researcher", "architect"}
    if (
        ctx.state.step_cursor == 0
        and hasattr(ctx.state.plan, "steps")
        and len(ctx.state.plan.steps) > 1
    ):
        research = [s for s in ctx.state.plan.steps if s.id in _RESEARCH_NODES]
        execution = [s for s in ctx.state.plan.steps if s.id not in _RESEARCH_NODES]
        if research and execution:
            reordered = research + execution
            if [s.id for s in reordered] != [s.id for s in ctx.state.plan.steps]:
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
                {"node_id": s.id, "is_parallel": s.parallel}
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
    if not current_step.parallel:
        ctx.state.step_cursor += 1
        ctx.state.pending_parallel_count = 1

        # If it's a meta-node, return the ID string directly
        if current_step.id in meta_nodes:
            logger.info(f"Dispatcher: Routing to meta-node: {current_step.id}")
            return current_step.id

        logger.info(
            f"Dispatcher: Dispatching sequential expert task: {current_step.id}"
        )
        emit_graph_event(
            ctx.deps.event_queue,
            "step_dispatched",
            id=current_step.id,
            step_index=ctx.state.step_cursor - 1,
            parallel=False,
        )

        # CONCEPT:AU-ORCH.execution.inject-signal-board-observations — Stigmergy Signal Board injection
        # If prior adaptive_agent_router left signals, emit them so downstream
        # adaptive_agent_router and the UI are aware of cross-node observations.
        if ctx.state.signal_board:
            signal_summary = "; ".join(
                f"{sig_type}: {', '.join(msgs[:3])}"
                for sig_type, msgs in ctx.state.signal_board.items()
            )
            emit_graph_event(
                ctx.deps.event_queue,
                "signal_board_context",
                id=current_step.id,
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
        and ctx.state.plan.steps[ctx.state.step_cursor].parallel
    ):
        batch.append(ctx.state.plan.steps[ctx.state.step_cursor])
        ctx.state.step_cursor += 1

    # Set the barrier count
    ctx.state.pending_parallel_count = len(batch)
    logger.info(f"Dispatcher: Dispatching parallel batch of {len(batch)} tasks...")

    emit_graph_event(
        ctx.deps.event_queue,
        "batch_dispatched",
        nodes=[s.id for s in batch],
        batch_size=len(batch),
    )

    ctx.state.pending_batch = ParallelBatch(tasks=batch)
    return "parallel_batch_processor"


async def parallel_batch_processor(
    ctx: StepContext,
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
    ctx: StepContext,
) -> str:
    """Execute a single specialist task with built-in retry and fallback logic.

    Routes task execution to the appropriate functional handler (e.g.,
    researcher_step, programmer nodes, or dynamic MCP adaptive_agent_router)
    based on the step's node_id. Implements per-node error recovery.

    Args:
        ctx: The pydantic-graph step context containing the targeted step details.

    Returns:
        The identifier of the appropriate joiner node for synchronization.

    """
    step = ctx.inputs
    node_id = step.id

    # Reset local retries for this new expert node
    ctx.state.current_node_retries = 0
    max_retries = 2

    while ctx.state.current_node_retries <= max_retries:
        try:
            logger.info(
                f"Expert Execution: Attempt {ctx.state.current_node_retries + 1}/{max_retries + 1} for node '{node_id}'"
            )

            # Declarative Pre-condition Contract Check (CONCEPT: OS-5.3 / AHE-3.7)
            try:
                from ..harness.contract_validator import ContractValidator

                validator = ContractValidator.instance()
                state_context = {
                    "query": ctx.state.query,
                    "results_registry": ctx.state.results_registry,
                    "step": step.model_dump()
                    if hasattr(step, "model_dump")
                    else str(step),
                }
                if not validator.validate_pre(node_id, state_context):
                    logger.error(
                        f"Contract: Pre-condition check failed for node '{node_id}'"
                    )
                    raise ValueError(
                        f"Pre-condition contract validation failed for node '{node_id}'"
                    )
                logger.info(
                    f"Contract: Pre-condition check passed for node '{node_id}'"
                )
            except Exception as ce:
                if "validation failed" in str(ce):
                    raise
                logger.debug(f"Contract pre-validation skipped: {ce}")

            # Transactional State Forking (CONCEPT: AHE-3.7)
            from ..harness.distributed_state_manager import BranchMergeStateLocker

            locker = BranchMergeStateLocker()
            base_key = f"execution_state:{ctx.state.query[:30]}"
            branch_name = f"branch_{node_id}"
            locker.fork_state(base_key, branch_name)
            locker.update_branch_state(
                base_key,
                branch_name,
                {
                    "node_id": node_id,
                    "input_data": step.description,
                    "results_registry": dict(ctx.state.results_registry),
                },
            )

            # CORE ARCHITECTURE STEPS (Preserved for pipeline stability)
            # Lazy imports to avoid circular dependencies between submodules
            from typing import Any, cast

            from .hierarchical_planner import (
                architect_step,
                planner_step,
                researcher_step,
            )
            from .verification import verifier_step

            if node_id == "researcher":
                await researcher_step(cast(Any, ctx))
            elif node_id == "architect":
                await architect_step(cast(Any, ctx))
            elif node_id == "planner":
                await planner_step(cast(Any, ctx))
            elif node_id == "verifier":
                await verifier_step(cast(Any, ctx))
            elif node_id == "mcp_server":
                domain = ""
                input_data = step.description
                if isinstance(input_data, dict):
                    domain = input_data.get("domain", "")
                await _execute_domain_logic(cast(Any, ctx), domain)

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

                (
                    domain_tools,
                    domain_toolsets,
                ) = apply_tool_scope(  # CONCEPT:AU-ORCH.session.invoker-agent-handoff
                    ctx.state, domain_tools, domain_toolsets
                )
                dynamic_agent = Agent(
                    model=ctx.deps.agent_model,
                    system_prompt=system_prompt + invoker_context_section(ctx.state),
                    tools=domain_tools,
                    toolsets=domain_toolsets,
                )

                # The injected developer_tools/sdd_tools are RunContext[AgentDeps]-typed and
                # read ctx.deps.workspace_path; the graph context is GraphDeps (no
                # workspace_path). Running without deps left ctx.deps=None →
                # "'NoneType' object has no attribute 'workspace_path'". Adapt the graph
                # context into a valid AgentDeps so injected tools AND MCP toolsets work.
                from .executor import agent_deps_from_graph

                _agent_deps = agent_deps_from_graph(
                    ctx.deps, domain_toolsets, state=ctx.state
                )

                # CONCEPT:AU-ORCH.execution.orchestration-flow-mermaid/1.38 — bound requests + enforce the invoker's token budget.
                async with dynamic_agent.run_stream(
                    f"Task context: {step.description}",
                    deps=_agent_deps,
                    usage_limits=spawn_usage_limits(ctx.state),
                ) as stream:
                    res = await asyncio.wait_for(
                        stream.get_output(), timeout=ctx.deps.verifier_timeout
                    )

                ctx.state.results_registry[node_id] = str(res)

            # Update branched state with execution output
            node_result = ctx.state.results_registry.get(node_id, {})
            if not isinstance(node_result, dict):
                node_result = {"output": node_result}

            locker.update_branch_state(
                base_key,
                branch_name,
                {
                    "node_id": node_id,
                    "input_data": step.description,
                    "output": node_result,
                    "results_registry": dict(ctx.state.results_registry),
                },
            )

            # Declarative Post-condition Contract Check (CONCEPT: OS-5.3 / AHE-3.7)
            try:
                if not validator.validate_post(node_id, node_result):
                    logger.error(
                        f"Contract: Post-condition check failed for node '{node_id}'"
                    )
                    raise ValueError(
                        f"Post-condition contract validation failed for node '{node_id}'"
                    )
                logger.info(
                    f"Contract: Post-condition check passed for node '{node_id}'"
                )
            except Exception as ce:
                if "validation failed" in str(ce):
                    raise
                logger.debug(f"Contract post-validation skipped: {ce}")

            # Transactional State Merging (CONCEPT: AHE-3.7)
            merge_success = locker.merge_state(base_key, branch_name)
            if merge_success:
                logger.info(
                    f"Transactional State: Successfully merged branch '{branch_name}' back to '{base_key}'"
                )
            else:
                logger.warning(
                    f"Transactional State: Failed to merge branch '{branch_name}' back to '{base_key}' (FF mismatch or lock conflict)"
                )

            # Execution successful, clear error and break retry loop
            ctx.state.error = None
            break

        except Exception as e:
            # CONCEPT:AU-ORCH.routing.mcp-child-error-unwrap — an expert step that fails by calling a remote MCP tool
            # raises an anyio ``BaseExceptionGroup`` whose ``str()`` is the opaque
            # "unhandled errors in a TaskGroup" (or empty). Flatten to the real leaf
            # cause(s) so the node-failure log is actionable (e.g. the portainer 401 /
            # connect error behind a research-step retry storm) instead of blank.
            from agent_utilities.orchestration.agent_runner import (
                _flatten_exception_group,
            )

            detail = _flatten_exception_group(e)
            logger.error(
                f"Execution failed for node '{node_id}' (Attempt {ctx.state.current_node_retries + 1}): {detail}"
            )
            ctx.state.error = f"Node {node_id} failed: {detail}"
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
    ctx: StepContext,
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
    ctx: StepContext,
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
    server_name = ctx.inputs
    query = ctx.state.query

    logger.info(f"Executing MCP Server Step: {server_name} for query: {query}")

    # Emit node start event
    emit_graph_event(
        ctx.deps.event_queue,
        "node_start",
        id="mcp_server_execution",
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
                system_prompt=system_prompt
                + invoker_context_section(ctx.state),  # ORCH-1.39
                toolsets=matched_toolsets,
            )

            async with agent.run_stream(query, deps=ctx.deps) as stream:
                async for chunk in stream.stream_text(delta=True):
                    emit_graph_event(
                        ctx.deps.event_queue,
                        "agent_node_delta",
                        content=chunk,
                        node="mcp_server_execution",
                    )
                output = await stream.get_output()
            # v1/v2 compat shim: see agent_utilities/graph/executor.py's identical block.
            usage: Any = stream.usage  # v2: property (was a method in v1)
            if callable(usage):
                usage = usage()
            if asyncio.iscoroutine(usage):
                usage = await usage
            ctx.state._update_usage(usage)
            ctx.state.results[server_name] = str(output)
            ctx.state.results_registry[f"{server_name}_{ctx.state.step_cursor}"] = str(
                output
            )

            # Accumulate this MCP server's tool calls for :ToolCall provenance on the
            # graph path (CONCEPT:AU-KG.temporal.message-history-read). Unconditional — the WebUI event
            # block below is gated on ``event_queue`` and skipped for headless
            # (MCP/telegram) delegations, which is exactly the MCP-execution path a
            # fleet-server delegation takes.
            try:
                from ..orchestration.tool_provenance import extract_tool_calls

                ctx.state.tool_calls.extend(extract_tool_calls(stream))
            except Exception as _tc_exc:  # noqa: BLE001 — never break a run
                logger.debug("mcp_server tool-call provenance skipped: %s", _tc_exc)

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
                        for req_part in msg.parts:
                            if isinstance(req_part, ToolReturnPart):
                                emit_graph_event(
                                    ctx.deps.event_queue,
                                    event_type="tool_result",
                                    agent=server_name,
                                    tool=req_part.tool_name,
                                    result=str(req_part.content)[:500],
                                )

        emit_graph_event(
            ctx.deps.event_queue,
            event_type="node_complete",
            id="mcp_server_execution",
            server=server_name,
            result=str(ctx.state.results.get(server_name, ""))[:500],
        )

        return "execution_joiner"
    except Exception as e:
        logger.error(f"MCP Server Step '{server_name}' failed: {e}")
        ctx.state.error = f"MCP Server {server_name} failed: {e}"
        return "error_recovery"
