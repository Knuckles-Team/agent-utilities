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
from typing import TYPE_CHECKING, Any

from pydantic_ai import UsageLimitExceeded
from pydantic_graph import End

from agent_utilities.core.config import setting
from agent_utilities.core.contextual_model import create_context_agent

if TYPE_CHECKING:
    from pydantic_graph.step import StepContext
else:
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
    load_specialized_prompts,
)

from ..models import (
    ExecutionStep,
    GraphPlan,
    GraphResponse,
    ParallelBatch,
)
from ..models.tool_score import normalize_legacy_relevance_score
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
    "_rank_tool_rows_by_relevance",
]

# D-CDX-53: how many name-ordered Tool candidates the dynamic-agent tool
# query pulls from the graph before Python-side relevance ranking. Bounded
# so a broad tag/name match can never turn into an unbounded fetch.
_TOOL_CANDIDATE_POOL_LIMIT = 50
# How many top-ranked tools are actually injected into the dynamic agent.
_TOOL_RESULT_LIMIT = 5


def _rank_tool_rows_by_relevance(
    rows: list[dict[str, Any]] | None, *, limit: int = _TOOL_RESULT_LIMIT
) -> list[dict[str, Any]]:
    """Rank Tool query rows by relevance score on ONE canonical scale.

    D-CDX-53: the live graph can hold both legacy ``relevance_score`` floats
    in ``[0, 1]`` and canonical integer points in ``[0, 100]`` on persisted
    ``Tool`` rows at the same time. Sorting the raw stored value (as a
    database-side ``ORDER BY`` would) ranks semantically-equal scores ~100x
    apart and can drop the better legacy-scored tool before it is ever
    normalized. Every row is normalized through the exact same boundary as
    :class:`agent_utilities.models.knowledge_graph.ToolNode` /
    :class:`agent_utilities.models.mcp.MCPToolInfo`
    (:func:`agent_utilities.models.tool_score.normalize_legacy_relevance_score`)
    before comparison, so a legacy ``0.9`` and a canonical ``90`` rank
    identically instead of ~100x apart.

    A row whose ``relevance_score`` normalizes to something outside the
    canonical ``0..100`` domain (negative, >100, non-legacy fractional,
    bool, string, missing) is treated as score ``0`` for ranking purposes
    ONLY — it still appears in the candidate list (never silently dropped),
    just ranked last among valid scores, so a corrupt persisted value can
    never crowd out a well-scored tool but also never disappears outright.
    """
    from ..models.tool_score import is_canonical_relevance_score

    def _ranking_key(row: dict[str, Any]) -> int:
        raw = row.get("relevance_score", 0)
        normalized = normalize_legacy_relevance_score(raw)
        if is_canonical_relevance_score(normalized):
            return normalized
        return 0

    ranked = sorted(rows or [], key=_ranking_key, reverse=True)
    return ranked[:limit]


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

    # CONCEPT:AU-ORCH.execution.direct-completion-shape — a direct-completion / lean turn is answered OUTSIDE this graph by
    # ``agent_runner._run_direct_completion`` (the planner's ``direct_complete`` shape, or the
    # structural classifier for a shape-less caller, short-circuits _execute_graph before the
    # graph is even built). The router therefore only ever runs for a real multi-step turn and
    # has a SINGLE outgoing edge to the dispatcher — it must NOT return ``End`` here, because a
    # second router edge to the end node makes pydantic-graph broadcast-fork the router output
    # to BOTH end and dispatcher, terminating every full-graph turn.

    guard = _router_check_replanning_budget(ctx)
    if guard is not None:
        return guard

    # CONCEPT:AU-ORCH.routing.offload-sync-roundtrip -- static keyword routing tags
    # lookup; result intentionally unused downstream, preserved as-is (see helper
    # docstring + BUGS FOUND).
    await _router_resolve_static_routing_tags(deps)

    discovery_context, early_return = await _router_topological_pre_routing(ctx, deps)
    if early_return is not None:
        return early_return

    direct_result = await _router_try_direct_dispatch(ctx, deps)
    if direct_result is not None:
        return direct_result

    return await _router_plan_and_dispatch(ctx, deps, discovery_context)


def _router_check_replanning_budget(ctx: StepContext) -> str | None:
    """Bump the re-planning loop counter; return a terminal node id if it is exhausted.

    Extracted verbatim from ``router_step`` (pure extract-method, no behaviour change).
    """
    # Track re-planning loops to prevent infinite cycles
    ctx.state.global_research_loops += 1
    if ctx.state.global_research_loops > 3:
        logger.error("Router: Max planning loops exceeded. Aborting.")
        return "error_recovery"
    return None


async def _router_resolve_static_routing_tags(deps: Any) -> dict[str, str]:
    """Static keyword routing tags lookup.

    Extracted verbatim from ``router_step`` (pure extract-method, no behaviour
    change). NOTE: the returned value is unused by the caller in the pre-refactor
    code too (``routing_tags`` was assigned and never read again) -- preserved
    as-is; not a bug introduced by this decomposition. See BUGS FOUND in the lane
    report.
    """
    # Junction Pseudostate: Try static keyword routing first (saves LLM call)
    # If tag_prompts is empty (e.g. toolset loading failed due to missing env vars),
    # fall back to using the MCP registry directly for keyword-based routing.
    routing_tags = deps.tag_prompts
    if not routing_tags:
        # CONCEPT:AU-ORCH.routing.offload-sync-roundtrip — the registry hydration is a
        # synchronous backend round-trip; keep it off the event loop.
        registry = await asyncio.to_thread(get_discovery_registry)
        routing_tags = {a.name: a.description for a in registry.agents}
        if routing_tags:
            logger.warning(
                f"Router: tag_prompts is empty, falling back to registry tags ({len(routing_tags)} tags)"
            )

    return routing_tags


async def _router_run_discovery_bundle(ctx: StepContext, deps: Any) -> dict[str, Any]:
    """Direct tool lookup + hybrid search + policy/process discovery, off the event loop.

    Extracted verbatim from ``_router_topological_pre_routing`` (pure extract-method,
    no behaviour change).

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
    """

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
        except Exception as e:  # noqa: BLE001 — one of several matching mechanisms feeding `_matched` (keyword lookup above, hybrid search / policy / process discovery below); a failure here just leaves out the ANN-designated specialists this pass, the others still populate the router's candidate set
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

    return await asyncio.to_thread(_run_discovery)


def _router_build_discovery_context(discovery: dict[str, Any]) -> str:
    """Format the discovery-bundle dict into the KG-discovery prompt section.

    Extracted verbatim from ``_router_topological_pre_routing`` (pure extract-method,
    no behaviour change). Returns ``""`` when there is nothing to report.
    """
    matched_agents = discovery["matched"]
    hybrid_results = discovery["hybrid"]
    relevant_policies = discovery["policies"]
    relevant_processes = discovery["processes"]

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

    if not discovery_sections:
        return ""

    logger.info(
        f"Router: Knowledge Graph discovery found {len(matched_agents)} tool-matched agents and {len(hybrid_results)} hybrid results."
    )
    return (
        "### KNOWLEDGE GRAPH DISCOVERY\n"
        + "\n\n".join(discovery_sections)
        + "\n\nPRIORITIZE using these agents or referencing this context in your plan.\n\n"
    )


async def _router_try_team_config_reuse(ctx: StepContext, deps: Any) -> str | None:
    """CONCEPT:AU-AHE.harness.team-config-precheck — Check for matching TeamConfig before LLM planning.

    Extracted verbatim from ``_router_topological_pre_routing`` (pure extract-method,
    no behaviour change; the nested ``if deps.knowledge_engine:`` / ``if isinstance(...)
    and hasattr(...):`` / ``if team:`` chain is flattened to early returns). Returns
    "dispatcher" (having already set ``ctx.state.plan`` and emitted events) on a
    successful reuse, else None to fall through to LLM planning.
    """
    if not deps.knowledge_engine:
        return None
    try:
        from ..core.registry.kg_adapter import RegistryMixin

        if not (
            isinstance(deps.knowledge_engine, RegistryMixin)
            and hasattr(deps.knowledge_engine, "find_matching_team_config")
        ):
            return None

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
        if not team:
            return None

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
        _emit_node_lifecycle(deps.event_queue, "router", "node_complete")
        return "dispatcher"
    except Exception as e:  # noqa: BLE001 — 1st of 3 sequential planning strategies; ctx.state.plan is unconditionally reassigned by the LLM planner below if this path doesn't return "dispatcher"
        logger.debug(f"TeamConfig lookup failed, continuing with LLM planning: {e}")
        return None


async def _router_try_kg_graph_materialization(
    ctx: StepContext, deps: Any
) -> str | None:
    """CONCEPT:AU-ORCH.adapter.kg-graph-materialization — KG-Driven Graph Materialization.

    Check for AgentTemplate nodes before falling back to LLM planning. Extracted
    verbatim from ``_router_topological_pre_routing`` (pure extract-method, no
    behaviour change; flattened to early returns). Returns "dispatcher" (having
    already set ``ctx.state.plan`` and emitted events) on success, else None.
    """
    if not deps.knowledge_engine:
        return None
    try:
        from .kg_graph_factory import build_pydantic_graph_from_kg

        # CONCEPT:AU-ORCH.routing.offload-sync-roundtrip — KG AgentTemplate materialization
        # is several SYNCHRONOUS engine round-trips (template search, KGTeamComposer
        # reuse-lookup, TeamConfig/synthesize_team fallback); run off the event loop.
        kg_result = await asyncio.to_thread(
            build_pydantic_graph_from_kg,
            query=ctx.state.query,
            engine=deps.knowledge_engine,
            deps=deps,
            top_k=7,
        )
        if not kg_result.specialist_configs:
            return None

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
            ctx.state.output_data["kg_specialist_configs"] = (
                kg_result.specialist_configs
            )

        emit_graph_event(
            deps.event_queue,
            "routing_completed",
            plan=plan.model_dump(),
            reasoning="KG AgentTemplate materialization",
        )
        _emit_node_lifecycle(deps.event_queue, "router", "node_complete")
        return "dispatcher"
    except Exception as e:  # noqa: BLE001 — same fallback chain as the TeamConfig lookup above; falls through to the LLM planner, which reassigns ctx.state.plan unconditionally
        logger.debug(
            f"KG AgentTemplate routing failed, continuing with LLM planning: {e}"
        )
        return None


async def _router_inject_self_model_context(deps: Any) -> str:
    """R4 (CONCEPT:AU-KG.memory.tiered-memory-caching) Self-Model proficiency + R5 ACO pheromone affinities.

    Extracted verbatim from ``_router_topological_pre_routing`` (pure extract-method,
    no behaviour change). Returns the additional discovery-context text to append
    (context formatting owned by the self_model enricher, single source), or ""
    on failure/absence.
    """
    if not deps.knowledge_engine:
        return ""
    try:
        from ..knowledge_graph.retrieval.memory_retriever import MemoryRetriever
        from .routing.enrichers.self_model import self_model_context

        memory_retriever = MemoryRetriever(deps.knowledge_engine)
        current = await asyncio.to_thread(memory_retriever.get_current)
        return self_model_context(current)
    except Exception as e:  # noqa: BLE001 — pure prompt-context string enrichment (discovery_context +=); failure just omits the extra text, router prompt still built normally
        logger.debug(f"Self-Model proficiency injection failed: {e}")
        return ""


async def _router_topological_pre_routing(
    ctx: StepContext, deps: Any
) -> tuple[str, str | None]:
    """Topological pre-routing: KG discovery, TeamConfig reuse, KG materialization, self-model.

    Extracted verbatim from ``router_step`` (pure extract-method, no behaviour
    change). Returns ``(discovery_context, early_return_node)``; a non-``None``
    ``early_return_node`` means the caller must return it immediately (mirrors the
    original inline ``return "dispatcher"`` short-circuits).
    """
    # Topological Pre-Routing: Check the Knowledge Graph for direct tool matches and context.
    # CONCEPT:AU-ORCH.execution.direct-completion-shape — run this several-round-trip discovery bundle only when the job's
    # shape calls for it; a lean shape skips it.
    _shape = getattr(deps, "execution_shape", None)
    discovery_context = ""
    if not (
        deps.knowledge_engine
        and (_shape is None or getattr(_shape, "run_discovery", True))
    ):
        return discovery_context, None

    logger.info("[LAYER:GRAPH:ROUTER] Performing topological and hybrid discovery...")

    discovery = await _router_run_discovery_bundle(ctx, deps)
    discovery_context = _router_build_discovery_context(discovery)

    team_result = await _router_try_team_config_reuse(ctx, deps)
    if team_result is not None:
        return discovery_context, team_result

    kg_result = await _router_try_kg_graph_materialization(ctx, deps)
    if kg_result is not None:
        return discovery_context, kg_result

    discovery_context += await _router_inject_self_model_context(deps)

    return discovery_context, None


async def _router_try_direct_dispatch(
    ctx: StepContext, deps: Any
) -> End[GraphResponse] | None:
    """Single-connected-MCP-server fast path.

    Extracted verbatim from ``router_step`` (pure extract-method, no behaviour
    change). Returns ``End(...)`` on a successful direct dispatch, or ``None`` to
    fall through to full planning (mirrors the original inline fall-through).
    """
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
            from agent_utilities.security.tool_guard import (
                flag_mcp_tool_definitions,
            )

            _guarded_ts = flag_mcp_tool_definitions(
                _ts,
                permissions_kernel=ctx.deps.permissions_kernel,
                agent_identity=ctx.deps.agent_identity,
                engine=ctx.deps.knowledge_engine,
            )
            _, _scoped_ts = apply_tool_scope(
                ctx.state, [], _guarded_ts
            )  # CONCEPT:AU-ORCH.session.invoker-agent-handoff
            _skill_prompt = str(
                deps.pinned_skill_prompt or deps.tag_prompts.get(str(_srv), "") or ""
            ).strip()
            _direct_agent = create_context_agent(
                model=deps.agent_model,
                permissions_kernel=ctx.deps.permissions_kernel,
                agent_identity=ctx.deps.agent_identity,
                permission_engine=ctx.deps.knowledge_engine,
                system_prompt=(
                    (f"{_skill_prompt}\n\n" if _skill_prompt else "")
                    + f"You are operating the '{_srv}' MCP server. Use its tools to satisfy "
                    f"the user's request directly and return exactly the data requested."
                    f"{invoker_context_section(ctx.state)}"  # CONCEPT:AU-ORCH.session.invoker-agent-handoff
                ),
                toolsets=_scoped_ts,
            )
            _direct_deps = agent_deps_from_graph(deps, _scoped_ts, state=ctx.state)
            _direct_res = await _direct_agent.run(
                ctx.state.query,
                deps=_direct_deps,
                usage_limits=spawn_usage_limits(
                    ctx.state
                ),  # CONCEPT:AU-ORCH.session.invoker-agent-handoff budget
            )
            from ..orchestration.tool_provenance import extract_tool_calls

            _tool_calls = extract_tool_calls(_direct_res)
            ctx.state.tool_calls.extend(_tool_calls)
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
                        "execution_mode": deps.execution_mode,
                        "skill": deps.pinned_skill_name,
                    },
                    tool_calls=_tool_calls,
                )
            )
        except (PermissionError, UsageLimitExceeded):
            # Authority denials and caller-supplied usage limits are terminal
            # contracts. Falling back to the multi-agent planner would either
            # bypass the denial or spend beyond the delegated budget.
            raise
        except Exception as e:  # noqa: BLE001 — recover from routing/model failures
            logger.warning(
                "[LAYER:GRAPH:ROUTER] Direct-dispatch failed (%s); "
                "falling back to full planning.",
                e,
            )
    return None


async def _router_resolve_workflow_context(ctx: StepContext, deps: Any) -> Any:
    """Phase 4 Edge-Computed Scopes: edge-computed JWT scope vs KG-routed workflow context.

    Extracted verbatim from ``_router_plan_and_dispatch`` (pure extract-method, no
    behaviour change). Only the KG-routed branch reassigns
    ``ctx.state.workflow_context`` (mirrors the original — the edge-computed branch
    reads it, never rewrites it).
    """
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
        return ShieldedResult(**payload)

    from .routing.strategies.workflow_context import WorkflowContextRouter

    router = WorkflowContextRouter(deps.knowledge_engine)
    workflow_context = await router.route_context(ctx.state.query)
    ctx.state.workflow_context = workflow_context.model_dump()
    return workflow_context


async def _router_resolve_specialist_tags(deps: Any) -> dict[str, str]:
    """Fetch specialist tags for planning, falling back to the discovery registry.

    Extracted verbatim from ``_router_plan_and_dispatch`` (pure extract-method, no
    behaviour change).
    """
    logger.info("[LAYER:GRAPH:ROUTER] Fetching specialist tags...")
    specialist_tags = deps.tag_prompts
    if not specialist_tags:
        registry = await asyncio.to_thread(get_discovery_registry)
        specialist_tags = {a.name: a.description for a in registry.agents}
        if specialist_tags:
            logger.info(
                f"[LAYER:GRAPH:ROUTER] Specialist tags loaded (count: {len(specialist_tags)}). Tags: {list(specialist_tags.keys())}"
            )
    return specialist_tags


async def _router_apply_pheromone_filter(
    ctx: StepContext, deps: Any, relevant: list[Any]
) -> list[Any]:
    """R7 (CONCEPT:AU-KG.memory.tiered-memory-caching) — Reward-Driven Optimization (pheromone filtering).

    Extracted verbatim from ``_router_filter_relevant_specialists`` (pure
    extract-method, no behaviour change). Owned by the optimization strategy
    (single source).
    """
    from .routing.strategies.optimization import filter_by_pheromone

    try:
        if deps.knowledge_engine:
            from ..knowledge_graph.retrieval.memory_retriever import MemoryRetriever

            memory_retriever = MemoryRetriever(deps.knowledge_engine)
            current = await asyncio.to_thread(memory_retriever.get_current)
            if current and current.pheromone_trails and relevant:
                relevant = filter_by_pheromone(relevant, current.pheromone_trails)
    except Exception as e:  # noqa: BLE001 — `relevant` keeps its pre-filter value on failure, identical shape to the sibling telemetry-pruning block just below (already annotated in this codebase)
        logger.debug(f"Reward-driven routing optimization failed: {e}")
    return relevant


async def _router_apply_telemetry_prune(
    ctx: StepContext, deps: Any, relevant: list[Any]
) -> list[Any]:
    """R8 (CONCEPT:AU-AHE.optimization.telemetry-optimization) — Telemetry-Driven Optimization (anomaly pruning).

    Extracted verbatim from ``_router_filter_relevant_specialists`` (pure
    extract-method, no behaviour change). Owned by the optimization strategy
    (single source).
    """
    from .routing.strategies.optimization import prune_by_telemetry

    try:
        if deps.knowledge_engine:
            anomaly_results = await asyncio.to_thread(
                deps.knowledge_engine.query_cypher,
                "MATCH (a:Agent)-[:CAUSED]->(p:PerformanceAnomaly) "
                "RETURN a.id AS agent_name, count(p) AS anomaly_count",
            )
            if anomaly_results:
                anomaly_map = {
                    r.get("agent_name"): r.get("anomaly_count", 0)
                    for r in anomaly_results
                    if r.get("agent_name")
                }
                relevant = prune_by_telemetry(relevant, anomaly_map)
    except Exception as e:  # noqa: BLE001 — per-request routing refinement; `relevant` just keeps its pre-prune value on failure, this pass falls back to the unpruned specialist set rather than an incorrect or stale one
        logger.debug(f"Telemetry-driven routing optimization failed: {e}")

    return relevant


async def _router_filter_relevant_specialists(
    ctx: StepContext, deps: Any, relevant: list[Any]
) -> list[Any]:
    """R7/R8: pheromone-trail filtering + anomaly-telemetry pruning of the relevant-specialist list.

    Extracted verbatim from ``_router_plan_and_dispatch`` (pure extract-method, no
    behaviour change; the two independent try/except passes are their own helpers,
    ``_router_apply_pheromone_filter`` / ``_router_apply_telemetry_prune``).
    """
    relevant = await _router_apply_pheromone_filter(ctx, deps, relevant)
    return await _router_apply_telemetry_prune(ctx, deps, relevant)


async def _router_build_planning_system_prompt(
    discovery_context: str, failure_context: str, step_info: str, agent_context: str
) -> str:
    """Assemble the router's planning-only system prompt.

    Extracted verbatim from ``_router_plan_and_dispatch`` (pure extract-method, no
    behaviour change).

    # R9 (CONCEPT:AU-ORCH.routing.transition-state-checkpoint): subtask-spec + wide-search instructions — owned by
    # the llm_planner strategy (single source of truth).
    """
    from .routing.strategies.llm_planner import subtask_and_widesearch_instructions

    # CONCEPT:AU-ORCH.routing.offload-sync-roundtrip — prompt load is file I/O + registry
    # lookup; keep it off the event loop.
    router_prompt = await asyncio.to_thread(load_specialized_prompts, "router")
    return (
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


async def _router_run_rlm_planning(
    ctx: StepContext,
    deps: Any,
    system_prompt_str: str,
    agent_context: str,
    rlm_config: Any,
) -> Any:
    """R10 RLM (Recursive Language Model) planning path, with fallback parser.

    Extracted verbatim from ``_router_plan_and_dispatch`` (pure extract-method, no
    behaviour change).
    """
    logger.info("[LAYER:GRAPH:ROUTER] Running in RLM (Recursive Language Model) mode.")
    from ..rlm.repl import RLMEnvironment

    env = RLMEnvironment(
        context=f"SYSTEM_PROMPT:\n{system_prompt_str}\n\nPROJECT_CONTEXT:\n{agent_context}",
        config=rlm_config,
        graph_deps=ctx.deps,
    )
    # R10 (CONCEPT:AU-ORCH.execution.predict-rlm-runtime) — RLM planning + fallback parser. The
    # instruction text and JSON->GraphPlan parse are owned by the
    # llm_planner strategy (single source); the async RLM run + re-parse
    # agent stay here.
    from .routing.strategies.llm_planner import parse_rlm_plan, rlm_plan_instruction

    rlm_result = await env.run_full_rlm(rlm_plan_instruction(ctx.state.query))

    plan_output = parse_rlm_plan(rlm_result, GraphPlan)
    if plan_output is None:
        logger.warning(
            "RLM output was not valid GraphPlan JSON. Running fallback parser."
        )
        router_agent = create_context_agent(
            model=deps.router_model,
            output_type=GraphPlan,
            system_prompt="Parse the following text into a valid GraphPlan JSON structure.",
        )
        parse_res = await router_agent.run(f"Text to parse:\n{rlm_result}")
        plan_output = parse_res.output
    return plan_output


def _router_detect_text_complexity_and_reasoning(
    ctx: StepContext, relevant: list[Any]
) -> tuple[bool, bool]:
    """R11 (CONCEPT:AU-AHE.evaluation.backtest-harness) — text-heuristic complexity/reasoning detection.

    Extracted verbatim from ``_router_detect_complexity_and_reasoning`` (pure
    extract-method, no behaviour change). Owned by the llm_planner strategy
    (single source); topology/quant escalation is a separate, KG-dependent
    helper (``_router_detect_topological_overrides``).
    """
    is_complex = False
    requires_reasoning = False

    from .routing.strategies.llm_planner import is_complex_query

    if is_complex_query(ctx.state.query, len(relevant)):
        is_complex = True

    if (
        "step by step" in ctx.state.query.lower()
        or "think through" in ctx.state.query.lower()
    ):
        requires_reasoning = True

    return is_complex, requires_reasoning


async def _router_detect_topological_overrides(
    ctx: StepContext, deps: Any, is_complex: bool, requires_reasoning: bool
) -> tuple[bool, bool]:
    """CONCEPT:AU-AHE.evaluation.backtest-harness — KG-Native topological/quant escalation overrides.

    Extracted verbatim from ``_router_detect_complexity_and_reasoning`` (pure
    extract-method, no behaviour change). Takes the text-heuristic
    ``(is_complex, requires_reasoning)`` and may escalate either to True based
    on live KG topology signals; never de-escalates.
    """
    if not deps.knowledge_engine:
        return is_complex, requires_reasoning

    try:

        def _read_topology_signals() -> tuple[list[Any], list[Any]]:
            return (
                deps.knowledge_engine.search_hybrid(
                    ctx.state.query + " TradingPipeline RiskScoringOntology",
                    top_k=2,
                ),
                deps.knowledge_engine.search_hybrid(
                    ctx.state.query
                    + " MathematicalFoundationNode vectorized topologies OWL Almgren-Chriss",
                    top_k=2,
                ),
            )

        # CONCEPT:AU-AHE.evaluation.backtest-harness Agentic detection
        task_topologies, math_topologies = await asyncio.to_thread(
            _read_topology_signals
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

    return is_complex, requires_reasoning


async def _router_detect_complexity_and_reasoning(
    ctx: StepContext, deps: Any, relevant: list[Any]
) -> tuple[bool, bool]:
    """R11 text-heuristic complexity/reasoning detection + KG-Native topological overrides.

    Extracted verbatim from ``_router_plan_and_dispatch`` (pure extract-method, no
    behaviour change). Returns ``(is_complex, requires_reasoning)``.

    # CONCEPT:AU-KG.memory.tiered-memory-caching — Adaptive Model Routing (Planner Path)
    # CONCEPT:AU-AHE.evaluation.backtest-harness — KG-Native Agentic Task Detection
    # CONCEPT:AU-AHE.evaluation.backtest-harness — Topological Reasoning Detection
    """
    is_complex, requires_reasoning = _router_detect_text_complexity_and_reasoning(
        ctx, relevant
    )
    return await _router_detect_topological_overrides(
        ctx, deps, is_complex, requires_reasoning
    )


def _router_select_adaptive_model(
    ctx: StepContext, deps: Any, is_complex: bool, requires_reasoning: bool
) -> Any:
    """CONCEPT:AU-OS.safety.doom-loop-detection — Topological Session Persistence + adaptive model routing.

    Extracted verbatim from ``_router_plan_and_dispatch`` (pure extract-method, no
    behaviour change).
    """
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
        # a model identifier the configured deployment cannot serve.
        reasoning_model_id = _super.id if _super else None
        logger.debug(
            "Router: pinning reasoning model %s",
            reasoning_model_id or "local-default",
        )
        # CONCEPT:AU-ORCH.execution.delegation-reasoning-off — this branch IS the
        # opt-in: the router detected the query genuinely needs deliberation
        # (mathematical/quantitative topology, "think through", etc.), so thinking
        # must be explicitly turned ON here — create_model's own default is OFF, so
        # omitting reasoning_effort would silently leave this "reasoning" branch
        # exactly as fast (and as un-reasoned) as every other one.
        adaptive_model = create_model(
            model_id=reasoning_model_id, reasoning_effort="high"
        )
        if reasoning_model_id:
            ctx.state.pinned_model_id = reasoning_model_id
        logger.info(
            f"[LAYER:GRAPH:ROUTER] Selected Reasoning Model: {reasoning_model_id}"
        )
    elif len(ctx.state.query.split()) < 20 and not is_complex:
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

    return adaptive_model


async def _router_call_planning_llm(
    ctx: StepContext, deps: Any, adaptive_model: Any, system_prompt_str: str
) -> Any:
    """Create the planning agent, run it under the router timeout, and unwrap the result.

    Extracted verbatim from ``_router_plan_and_dispatch`` (pure extract-method, no
    behaviour change). Raises ``ValueError`` on timeout or an empty result (mirrors
    the original inline raises); on any raise, ``ctx.state._update_usage`` is
    correctly NOT reached, exactly as in the original (it sits after the
    try/except in both versions).
    """
    router_agent = create_context_agent(
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
        logger.info(f"[LAYER:GRAPH:ROUTER] Plan Step Count: {len(plan_output.steps)}")
    except TimeoutError:
        logger.warning("Router: LLM planning timed out. Escalating to fallbacks.")
        raise ValueError("LLM planning timed out") from None

    ctx.state._update_usage(stream.usage)
    return plan_output


# Sentinel marking "adaptive_model was never assigned" in ``_router_plan_and_dispatch``.
# The original inline code left the ``adaptive_model`` name completely unbound
# whenever the RLM planning branch ran (or a failure hit before the non-RLM
# branch's model-selection line); the unstructured-fallback except block then
# referenced that bare name, so a failure on the RLM path always hit an
# ``UnboundLocalError`` there, silently swallowed by the fallback's own broad
# ``except Exception``.
#
# BUG-CX-061 (fixed): that meant the "multi-level fallback chain" (R13) --
# whose entire purpose is to rescue a turn when structured planning fails --
# could NEVER actually run for a failure on the RLM path (or any failure
# before the non-RLM branch's model-selection line): it hit the sentinel and
# died before attempting a single fallback LLM call. ``adaptive_model`` is
# not optional to the fallback's own logic (only to how planning happened to
# reach it), so ``_router_run_unstructured_fallback_agent`` now falls back to
# the same default ``deps.router_model`` that ``_router_select_adaptive_model``
# itself falls back to when no other selection criterion applies, instead of
# treating "never computed by planning" as fatal.
_ADAPTIVE_MODEL_UNSET = object()


async def _router_run_unstructured_fallback_agent(
    ctx: StepContext,
    deps: Any,
    system_prompt_str: str,
    adaptive_model: Any,
) -> str:
    """Create + run the unstructured-fallback agent, bounded by the router timeout.

    Extracted from ``_router_attempt_unstructured_fallback``. On the
    ``_ADAPTIVE_MODEL_UNSET`` sentinel (see that sentinel's docstring / BUG-CX-061),
    falls back to ``deps.router_model`` so a planning failure that never
    reached the non-RLM branch's model selection can still attempt a real
    fallback call, rather than failing before ever trying.
    """
    from .routing.strategies.fallback import unstructured_fallback_prompt

    if adaptive_model is _ADAPTIVE_MODEL_UNSET:
        adaptive_model = deps.router_model

    fallback_agent = create_context_agent(
        model=adaptive_model,
        system_prompt=unstructured_fallback_prompt(system_prompt_str),
    )
    # D-RTR-1: the structured planning call above is bounded by
    # ``ctx.deps.router_timeout`` (~12s for the ``chat`` profile), but this
    # unstructured fallback previously had NO timeout — a bare ``await`` that
    # hit the very same degraded backend that just stalled the structured
    # call. Two sequential unbounded-then-bounded LLM calls against a
    # degraded provider is how a single turn reached 264s with no answer.
    # Reuse the same profile budget so the two calls together can never
    # exceed roughly 2x the router timeout.
    try:
        fallback_res = await asyncio.wait_for(
            fallback_agent.run(ctx.state.query),
            timeout=ctx.deps.router_timeout,
        )
    except TimeoutError:
        raise ValueError(
            f"Unstructured fallback planning timed out after {ctx.deps.router_timeout}s"
        ) from None

    return str(getattr(fallback_res, "data", getattr(fallback_res, "output", "")))


def _router_extract_fallback_steps(
    ctx: StepContext,
    deps: Any,
    raw_text: str,
    specialist_tags: dict[str, str] | None,
) -> tuple[str | None, str | None]:
    """Match specialist names in the fallback text and build a GraphPlan from them.

    Extracted verbatim from ``_router_attempt_unstructured_fallback`` (pure
    extract-method, no behaviour change). The
    ``"specialist_tags" in locals()`` check from the original is replaced with an
    explicit ``is not None`` check on the threaded-through parameter —
    behaviourally identical (both ask "was this successfully computed before the
    failure"). Returns ``("dispatcher", None)`` on a successful extraction (having
    already set ``ctx.state.plan``), or ``(None, fallback_failure_detail)`` if no
    known specialist matched.
    """
    from .routing.strategies.fallback import match_specialists_in_text

    available = list(specialist_tags.keys()) if specialist_tags is not None else []
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
        return "dispatcher", None

    # D-RTR-4: make the no-match case actionable instead of a silent
    # fall-through — name what the model proposed and what the
    # registry actually has, so the eventual failure message tells the
    # operator why (e.g. the model named a specialist the registry
    # doesn't know about) rather than just "no answer".
    fallback_failure_detail = (
        f"the model proposed '{raw_text.strip()[:200]}' but no known "
        f"specialist matched. Available specialists: {available or 'none registered'}."
    )
    logger.warning(
        f"Router Fallback: No known specialists found in text. Available: {available}. Raw text: {raw_text}"
    )
    return None, fallback_failure_detail


async def _router_attempt_unstructured_fallback(
    ctx: StepContext,
    deps: Any,
    system_prompt_str: str,
    adaptive_model: Any,
    specialist_tags: dict[str, str] | None,
) -> tuple[str | None, str | None]:
    """R13 multi-level fallback chain: unstructured natural-language extraction.

    Extracted from ``_router_plan_and_dispatch``'s outer ``except`` block. See
    the ``_ADAPTIVE_MODEL_UNSET`` sentinel docstring / BUG-CX-061 for why an
    unset ``adaptive_model`` no longer makes this fallback unconditionally
    unreachable.

    Returns ``("dispatcher", None)`` on a successful fallback extraction (having
    already set ``ctx.state.plan``), or ``(None, fallback_failure_detail)`` if the
    whole fallback attempt failed too.
    """
    try:
        # R13 (multi-level fallback chain) — unstructured natural-language
        # extraction. The prompt + specialist name-matching are owned by the
        # fallback strategy (single source).
        raw_text = await _router_run_unstructured_fallback_agent(
            ctx, deps, system_prompt_str, adaptive_model
        )
        return _router_extract_fallback_steps(ctx, deps, raw_text, specialist_tags)
    except Exception as fallback_e:
        logger.error(f"Router fallback also failed: {fallback_e}")
        return None, f"the fallback attempt itself failed: {fallback_e}"


async def _router_plan_and_dispatch(
    ctx: StepContext, deps: Any, discovery_context: str
) -> str:
    """Full LLM planning pipeline + multi-level fallback chain.

    Extracted verbatim from ``router_step`` (pure extract-method, no behaviour
    change). Always returns the ``"dispatcher"`` node id (matching the original,
    which only ever returned that string from this portion of the function).
    """
    # Reset cursor for the new plan
    ctx.state.step_cursor = 0

    failure_context = ""
    if ctx.state.error:
        failure_context = f"### PREVIOUS FAILURE CONTEXT\nThe last attempt failed with the following error:\n{ctx.state.error}\nUse this information to update your plan. You may need more research or a different approach."

    # See ``_router_attempt_unstructured_fallback`` / ``_ADAPTIVE_MODEL_UNSET``:
    # these three are pre-bound to sentinels so the ``except`` block below can
    # be called unconditionally without itself raising on an unbound name.
    adaptive_model: Any = _ADAPTIVE_MODEL_UNSET
    system_prompt_str = ""
    specialist_tags: dict[str, str] | None = None

    try:
        workflow_context = await _router_resolve_workflow_context(ctx, deps)
        agent_context = workflow_context.to_prompt_string()

        specialist_tags = await _router_resolve_specialist_tags(deps)

        # CONCEPT:AU-ORCH.routing.filtered-specialist-injection — Filtered specialist injection for prompt bloat reduction
        relevant = await asyncio.to_thread(
            get_relevant_specialists,
            ctx.state.query,
            engine=deps.knowledge_engine,
            top_n=7,
        )
        relevant = await _router_filter_relevant_specialists(ctx, deps, relevant)

        # R6 (CONCEPT:AU-ORCH.routing.filtered-specialist-injection) — filtered specialist injection (prompt-bloat reduction).
        from .routing.strategies.optimization import format_specialist_step_info

        step_info = format_specialist_step_info(relevant, specialist_tags)
        logger.info(
            f"Router: Specialists count: {len(specialist_tags)}, Context length: {len(agent_context)}"
        )

        system_prompt_str = await _router_build_planning_system_prompt(
            discovery_context, failure_context, step_info, agent_context
        )

        from ..rlm.config import RLMConfig

        rlm_config = RLMConfig()
        use_rlm = (
            rlm_config.enabled or len(agent_context) > rlm_config.max_context_threshold
        )

        if use_rlm:
            plan_output = await _router_run_rlm_planning(
                ctx, deps, system_prompt_str, agent_context, rlm_config
            )
        else:
            (
                is_complex,
                requires_reasoning,
            ) = await _router_detect_complexity_and_reasoning(ctx, deps, relevant)
            adaptive_model = _router_select_adaptive_model(
                ctx, deps, is_complex, requires_reasoning
            )
            plan_output = await _router_call_planning_llm(
                ctx, deps, adaptive_model, system_prompt_str
            )

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
        node_id, fallback_failure_detail = await _router_attempt_unstructured_fallback(
            ctx, deps, system_prompt_str, adaptive_model, specialist_tags
        )
        if node_id is not None:
            return node_id

        # D-RTR-2: this used to ``return "__end__"``, implying the router could
        # terminate the graph run here. It cannot: ``graph/builder.py`` gives the
        # router a SINGLE static outgoing edge to the dispatcher (a second edge to
        # the end node would broadcast-fork pydantic-graph), so whatever this
        # function returns, execution always proceeds to ``dispatcher_step`` next.
        # "__end__" was therefore dead, misleading dead code. The real failure
        # path is explicit instead: record the concrete reason on ``ctx.state.error``
        # (the plan stays the default empty ``GraphPlan`` — never reassigned on this
        # path) so ``dispatcher_step``'s empty-plan branch can surface *why* there is
        # no answer, and return the node id execution actually reaches.
        ctx.state.error = f"Planning failed: {e}" + (
            f" Fallback also failed: {fallback_failure_detail}"
            if fallback_failure_detail
            else ""
        )
        return "dispatcher"


_NOT_DONE = object()


def _check_transitions_calls_tokens_budgets(ctx: StepContext) -> str | None:
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

    if budget.max_tool_calls and len(ctx.state.tool_calls) > budget.max_tool_calls:
        logger.error(
            f"Dispatcher: Execution budget exceeded for tool calls ({len(ctx.state.tool_calls)} > {budget.max_tool_calls})"
        )
        ctx.state.error = "Execution budget exceeded: max tool calls."
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

    return None


def _check_cost_and_duration_budgets(ctx: StepContext) -> str | None:
    import time

    budget = ctx.state.execution_budget

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

    return None


def _check_cost_governor_budgets(ctx: StepContext) -> str | None:
    """CONCEPT:AU-ORCH.execution.execution-budget-caps — Execution Budget (Cost Governor) enforcement."""
    result = _check_transitions_calls_tokens_budgets(ctx)
    if result is not None:
        return result
    return _check_cost_and_duration_budgets(ctx)


def _check_hard_transition_cap(ctx: StepContext) -> str | None:
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
    return None


def _check_execution_budgets(ctx: StepContext) -> str | None:
    """CONCEPT:AU-ORCH.execution.execution-budget-caps — Execution Budget (Cost Governor) enforcement,
    plus the MAX_NODE_TRANSITIONS hard cap. Returns "error_recovery" (with
    ctx.state.error set) if any budget is exceeded, else None."""
    result = _check_cost_governor_budgets(ctx)
    if result is not None:
        return result
    return _check_hard_transition_cap(ctx)


def _check_state_invariant(ctx: StepContext) -> str | None:
    # HSM: State invariant check at transition boundary
    try:
        assert_state_valid(ctx.state, "dispatcher_step")
    except StateInvariantError as e:
        logger.error(f"State invariant violation: {e}")
        ctx.state.error = str(e)
        return "error_recovery"
    return None


def _check_doom_loop(ctx: StepContext) -> str | None:
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
    except ImportError:  # noqa: BLE001 — optional stability detector is not installed
        pass
    except Exception as e:
        # D-DST-4 (CONCEPT:AU-AHE.evaluation.debug-swallow-justification): DoomLoopDetector
        # is a safety control (AU-OS.safety.doom-loop-detection), not best-effort telemetry —
        # a runtime failure here (module present but erroring, unlike the ImportError case
        # above) silently disables loop protection for this transition with no operator-
        # visible signal. Raised to warning so a persistently-failing detector is
        # diagnosable instead of invisible.
        logger.warning("Doom loop detection failed (safety check skipped): %s", e)
    return None


async def _checkpoint_transition_state(ctx: StepContext) -> None:
    # CONCEPT:AU-ORCH.routing.transition-state-checkpoint — State checkpoint at transition boundary.
    # Routes through the consolidated CheckpointManager (KG backend). The old
    # graph/state_checkpoint.StateCheckpointer was merged into core/checkpoint
    # (Plan 03 Step 8); the prior import silently failed, dropping this
    # capability — restored here.
    import time

    try:
        from ..core.checkpoint.manager import CheckpointManager

        if hasattr(ctx.deps, "knowledge_engine") and ctx.deps.knowledge_engine:
            # CONCEPT:AU-ORCH.routing.offload-sync-roundtrip — a KG-backed checkpoint save is a
            # synchronous engine write; run the create+save pair off the event loop.
            def _checkpoint_state() -> Any:
                checkpointer = CheckpointManager.create(
                    persistence_type="kg", engine=ctx.deps.knowledge_engine
                )
                return checkpointer.save(ctx.state, session_id=ctx.state.session_id)

            checkpoint_id = await asyncio.to_thread(_checkpoint_state)
            if isinstance(checkpoint_id, str) and checkpoint_id:
                ctx.state.checkpoint_ids.append(checkpoint_id)
                ctx.state.checkpoint_ts = time.time()
    except Exception as e:  # noqa: BLE001 — ctx.state.checkpoint_ids.append() only executes inside the try after a successful save (guarded by the isinstance check above), so a failure here never records a checkpoint id that doesn't exist; this is best-effort HSM state durability, not the primary transition flow
        logger.debug("State checkpointing skipped: %s", e)


def _integrate_deferred_events(ctx: StepContext) -> None:
    # HSM: Process deferred events (user follows-up received mid-execution)
    if ctx.state.deferred_events:
        for event in ctx.state.deferred_events:
            if event.get("type") == "user_followup":
                ctx.state.query += f"\n\nFollow up: {event.get('content', '')[:100]}"
                logger.info(
                    f"Dispatcher: Integrated deferred event: {event.get('content', '')[:100]}"
                )
        ctx.state.deferred_events.clear()


def _maybe_route_to_memory_selection(ctx: StepContext) -> str | None:
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
    return None


def _reorder_research_steps_first(ctx: StepContext) -> None:
    _RESEARCH_NODES = {"researcher", "architect"}
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


def _reorder_research_before_execution(ctx: StepContext) -> None:
    # Phase-ordering guard: ensure research steps precede execution steps.
    # The LLM router may interleave them; we enforce discovery-first so that
    # research results are available to all execution adaptive_agent_router.
    if (
        ctx.state.step_cursor == 0
        and hasattr(ctx.state.plan, "steps")
        and len(ctx.state.plan.steps) > 1
    ):
        _reorder_research_steps_first(ctx)

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


async def _finish_completed_plan(ctx: StepContext) -> str | None:
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
    # D-RTR-3: this branch used to ``return None`` unconditionally, and
    # ``dispatcher_route``'s ``type(None)`` branch (graph/builder.py) forwards that
    # bare ``None`` straight to ``g.end_node`` with no payload — the exact "graph
    # terminated with no output" case ``orchestration/engine.py`` has to guard
    # against. Verified empirically (a minimal pydantic-graph reproduction) that
    # this function CANNOT instead return ``End(...)`` here: ``dispatcher_step``'s
    # return value is routed through the ``dispatcher_route`` Decision node
    # (graph/builder.py), whose branches are an exhaustive Literal/type match with
    # no ``End``-shaped branch — an unmatched value raises ``RuntimeError: No
    # branch matched inputs End(...) for decision node dispatcher_route`` (a hard
    # crash), so ``None`` via the ``type(None)`` branch is the only value this
    # function can return that reaches ``g.end_node`` without a builder.py change
    # (out of this file's scope — see handoff notes). What stays in scope: make
    # sure the *reason* is not lost. Stamp a concrete, actionable message on
    # ``ctx.state.error`` (preserving one already set by the router on a planning
    # failure) so the failure is diagnosable from graph state, and so that once
    # ``orchestration/engine.py``'s ``result is None`` guard (~line 925) is updated
    # to surface ``state.error`` instead of its current hardcoded generic string,
    # the user sees *why*, not just that the turn produced nothing.
    if not ctx.state.error:
        ctx.state.error = (
            "The orchestration plan completed with no execution results and no "
            "exploration notes to synthesize a response from "
            f"(routed_domain={ctx.state.routed_domain or 'none'})."
        )
    return None


async def _handle_plan_completion_if_done(ctx: StepContext) -> Any:
    """Returns _NOT_DONE if the plan still has steps left to dispatch.
    Otherwise returns the (str | None) value dispatcher_step should return."""
    plan_len = len(ctx.state.plan.steps) if hasattr(ctx.state.plan, "steps") else 0
    logger.info(
        f"Dispatcher: Handling graph execution (Step {ctx.state.step_cursor}/{plan_len})"
    )
    if not hasattr(ctx.state.plan, "steps") or ctx.state.step_cursor >= len(
        ctx.state.plan.steps
    ):
        return await _finish_completed_plan(ctx)
    return _NOT_DONE


async def _dispatch_sequential_step(ctx: StepContext, current_step: Any) -> str:
    ctx.state.step_cursor += 1
    ctx.state.pending_parallel_count = 1

    # Internal/Meta nodes should remain as strings for direct routing
    meta_nodes = {
        "router",
        "planner",
        "onboarding",
        "error",
        "usage_guard",
        "memory_selection",
    }

    # If it's a meta-node, return the ID string directly
    if current_step.id in meta_nodes:
        logger.info(f"Dispatcher: Routing to meta-node: {current_step.id}")
        return current_step.id

    logger.info(f"Dispatcher: Dispatching sequential expert task: {current_step.id}")
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


def _dispatch_parallel_batch(ctx: StepContext) -> str:
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


async def _dispatch_next_step(ctx: StepContext) -> str:
    # Sequential execution case (default for first step or non-parallel)
    if not hasattr(ctx.state.plan, "steps"):
        logger.error("Dispatcher: Plan is not a valid GraphPlan object.")
        return "error_recovery"

    current_step = ctx.state.plan.steps[ctx.state.step_cursor]

    # Check if this is the start of a parallel batch
    if not current_step.parallel:
        return await _dispatch_sequential_step(ctx, current_step)

    return _dispatch_parallel_batch(ctx)


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

    result = _check_execution_budgets(ctx)
    if result is not None:
        return result

    result = _check_state_invariant(ctx)
    if result is not None:
        return result

    result = _check_doom_loop(ctx)
    if result is not None:
        return result

    await _checkpoint_transition_state(ctx)

    _integrate_deferred_events(ctx)

    result = _maybe_route_to_memory_selection(ctx)
    if result is not None:
        return result

    _reorder_research_before_execution(ctx)

    result = await _handle_plan_completion_if_done(ctx)
    if result is not _NOT_DONE:
        return result

    return await _dispatch_next_step(ctx)


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


async def _expert_try_static_handler(ctx: StepContext, node_id: str, step: Any) -> bool:
    """Dispatch to a known static step handler, if ``node_id`` matches one.

    Extracted verbatim from ``_expert_dispatch_step_handler`` (pure extract-method,
    no behaviour change). Returns True if a static handler ran (each handler
    writes its own result into ``ctx.state.results_registry``), or False if the
    caller must fall through to dynamic agent spawning.
    """
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
    else:
        return False
    return True


async def _expert_resolve_dynamic_bindings(
    ctx: StepContext, node_id: str
) -> tuple[str, list[str]]:
    """Query the KG for an explicit prompt-node match + ranked candidate tools.

    Extracted verbatim from ``_expert_dispatch_step_handler`` (pure extract-method,
    no behaviour change; step "1. Query Knowledge Graph for best tools & prompts"
    of the dynamic-agent-spawning fallback). Returns
    ``(system_prompt, tools_to_inject)``.
    """
    engine = ctx.deps.knowledge_engine
    system_prompt = f"You are a specialized agent handling the task: {node_id}."
    tools_to_inject: list[str] = []

    if not engine:
        return system_prompt, tools_to_inject

    def _read_dynamic_bindings(
        kg_engine: Any = engine,
        dynamic_node_id: str = node_id,
    ) -> tuple[list[Any], list[Any]]:
        return (
            kg_engine.query_cypher(
                "MATCH (p:Prompt) WHERE toLower(p.name) CONTAINS toLower($name) RETURN p.system_prompt AS sp LIMIT 1",
                {"name": dynamic_node_id},
            ),
            kg_engine.query_cypher(
                # D-CDX-53: does NOT ``ORDER BY t.relevance_score``
                # in Cypher. The live graph can hold both legacy
                # ``[0, 1]`` float scores and canonical ``[0, 100]``
                # int points on persisted Tool rows at the same
                # time, and ordering raw mixed-scale values in the
                # database ranks semantically-equal scores ~100x
                # apart and can truncate the better legacy tool out
                # of the result before it is ever normalized. A
                # deterministic ``ORDER BY t.name`` plus a bounded
                # candidate pool (``_TOOL_CANDIDATE_POOL_LIMIT``)
                # is used instead, and the caller ranks the
                # candidates in Python via
                # ``_rank_tool_rows_by_relevance`` — which
                # normalizes every row through the SAME canonical
                # boundary as ``ToolNode``/``MCPToolInfo``
                # (``agent_utilities.models.tool_score``) before
                # comparing scores.
                "MATCH (t:Tool) WHERE any(tag IN t.tags WHERE toLower(tag) CONTAINS toLower($name)) OR toLower(t.name) CONTAINS toLower($name) "
                "RETURN t.name AS name, t.mcp_server AS server, t.relevance_score AS relevance_score "
                f"ORDER BY t.name LIMIT {_TOOL_CANDIDATE_POOL_LIMIT}",
                {"name": dynamic_node_id},
            ),
        )

    # Check for explicit prompt node
    prompt_res, tool_res = await asyncio.to_thread(_read_dynamic_bindings)
    if prompt_res and "sp" in prompt_res[0]:
        system_prompt = prompt_res[0]["sp"]

    # Find relevant tools (by tag or semantic overlap if we had embeddings, using tag heuristic for now)
    ranked_tool_rows = _rank_tool_rows_by_relevance(tool_res, limit=_TOOL_RESULT_LIMIT)
    tools_to_inject = [t["name"] for t in ranked_tool_rows]

    return system_prompt, tools_to_inject


async def _expert_prepare_dynamic_toolsets(
    ctx: StepContext, tools_to_inject: list[str]
) -> tuple[list[Any], list[Any]]:
    """Fetch domain tools/toolsets, splice in native GraphOS toolsets, filter to injected tools.

    Extracted verbatim from ``_expert_dispatch_step_handler`` (pure extract-method,
    no behaviour change; step "2. Execute Dynamic Agent" setup of the
    dynamic-agent-spawning fallback, through the ``apply_tool_scope`` call).
    """
    from .executor import _get_domain_tools

    domain_tools, domain_toolsets = await _get_domain_tools(
        "mcp_server_execution", ctx.deps
    )

    # A delegated skill's native GraphOS toolset is scoped to this
    # run and is therefore authoritative even when the planner
    # emits a dynamic node name that does not match a fleet-server
    # tag. Preserve it through the fallback path and apply the
    # same signed identity policy used by specialist execution.
    native_toolsets = [
        toolset
        for toolset in ctx.deps.mcp_toolsets
        if isinstance((metadata := getattr(toolset, "metadata", None)), dict)
        and metadata.get("graphos_native") is True
    ]
    if native_toolsets:
        from agent_utilities.security.tool_guard import flag_mcp_tool_definitions

        domain_toolsets = [
            *flag_mcp_tool_definitions(
                native_toolsets,
                permissions_kernel=ctx.deps.permissions_kernel,
                agent_identity=ctx.deps.agent_identity,
                engine=ctx.deps.knowledge_engine,
            ),
            *domain_toolsets,
        ]

    # Filter down to the exact tools
    if tools_to_inject:
        filtered_tools = [t for t in domain_tools if t.__name__ in tools_to_inject]
        if filtered_tools:
            domain_tools = filtered_tools

    return apply_tool_scope(  # CONCEPT:AU-ORCH.session.invoker-agent-handoff
        ctx.state, domain_tools, domain_toolsets
    )


async def _expert_spawn_dynamic_agent(
    ctx: StepContext, node_id: str, step: Any
) -> None:
    """Dynamic graph-native agent spawning fallback for an expert-execution step.

    Extracted verbatim from ``_expert_dispatch_step_handler`` (pure extract-method,
    no behaviour change). Writes the result into
    ``ctx.state.results_registry[node_id]``.
    """
    logger.info(f"Expert Execution: Spawning dynamic agent for task '{node_id}'")

    system_prompt, tools_to_inject = await _expert_resolve_dynamic_bindings(
        ctx, node_id
    )

    logger.info(
        f"Dynamic Agent '{node_id}': Injecting {len(tools_to_inject)} tools from Knowledge Graph."
    )

    domain_tools, domain_toolsets = await _expert_prepare_dynamic_toolsets(
        ctx, tools_to_inject
    )

    dynamic_agent = create_context_agent(
        model=ctx.deps.agent_model,
        permissions_kernel=ctx.deps.permissions_kernel,
        agent_identity=ctx.deps.agent_identity,
        permission_engine=ctx.deps.knowledge_engine,
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

    _agent_deps = agent_deps_from_graph(ctx.deps, domain_toolsets, state=ctx.state)

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


async def _expert_dispatch_step_handler(
    ctx: StepContext, node_id: str, step: Any
) -> None:
    """Dispatch a single expert-execution step to its handler.

    Extracted verbatim from ``expert_executor_step`` (pure extract-method, no
    behaviour change). Covers the known static handlers (researcher/architect/
    planner/verifier/mcp_server) and the dynamic graph-native agent spawning
    fallback. Writes results into ``ctx.state.results_registry`` exactly as the
    original inline code did; has no return value.
    """
    if await _expert_try_static_handler(ctx, node_id, step):
        return

    # DYNAMIC GRAPH-NATIVE AGENT SPAWNING
    await _expert_spawn_dynamic_agent(ctx, node_id, step)


async def _expert_execute_attempt(
    ctx: StepContext, node_id: str, step: Any, max_retries: int
) -> None:
    """Execute one retry attempt of an expert step: contract checks, state
    fork, dispatch, state merge.

    Extracted from ``expert_executor_step`` -- the body of the per-attempt
    ``try:`` block. ``validator`` is shared across both contract checks
    (matching the pre-refactor scoping); since a genuine crash on the
    pre-condition check now always re-raises (see below, BUG-CX-070), it can
    no longer reach the post-condition check with ``validator`` unassigned.
    Raises on failure; the caller's retry loop catches and handles it.
    """
    logger.info(
        f"Expert Execution: Attempt {ctx.state.current_node_retries + 1}/{max_retries + 1} for node '{node_id}'"
    )

    # Declarative Pre-condition Contract Check (CONCEPT: OS-5.3 / AHE-3.7)
    #
    # BUG-CX-070 (fixed): ``ContractValidator.validate_pre``/``validate_post``
    # (agent_utilities/harness/contract_validator.py) already catch every
    # exception a REGISTERED contract callable can raise internally and turn
    # it into a plain ``False`` return -- so an exception reaching this
    # try/except can only come from the contract-check plumbing itself (e.g.
    # ``ContractValidator.instance()`` raising, or ``state_context``
    # construction raising), never from a "no contract configured for this
    # node" condition (that returns ``True`` here, no exception at all). This
    # used to catch ANY exception here and treat it identically to "not
    # configured" -- logging it at DEBUG as "skipped" and letting execution
    # proceed as though nothing happened, silently masking a real bug in the
    # validation path as an all-clear. Fail closed: only the explicit
    # ValueError this block itself raises for an actual failed validation is
    # expected; anything else is a genuine crash and must propagate.
    from ..harness.contract_validator import ContractValidator

    validator = ContractValidator.instance()
    state_context = {
        "query": ctx.state.query,
        "results_registry": ctx.state.results_registry,
        "step": step.model_dump() if hasattr(step, "model_dump") else str(step),
    }
    if not validator.validate_pre(node_id, state_context):
        logger.error(f"Contract: Pre-condition check failed for node '{node_id}'")
        raise ValueError(
            f"Pre-condition contract validation failed for node '{node_id}'"
        )
    logger.info(f"Contract: Pre-condition check passed for node '{node_id}'")

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

    await _expert_dispatch_step_handler(ctx, node_id, step)

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
            logger.error(f"Contract: Post-condition check failed for node '{node_id}'")
            raise ValueError(
                f"Post-condition contract validation failed for node '{node_id}'"
            )
        logger.info(f"Contract: Post-condition check passed for node '{node_id}'")
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

    # Execution successful, clear error
    ctx.state.error = None


async def _expert_handle_attempt_failure(
    ctx: StepContext, node_id: str, e: Exception, max_retries: int
) -> bool:
    """Log + record one failed attempt; return True if retries are exhausted.

    Extracted verbatim from ``expert_executor_step`` (pure extract-method, no
    behaviour change) -- the body of the outer ``except Exception as e:``
    block, minus the terminal ``break``/``sleep`` (left to the caller, which
    owns the loop).
    """
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
        return True
    return False


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
            await _expert_execute_attempt(ctx, node_id, step, max_retries)
            break
        except Exception as e:
            exhausted = await _expert_handle_attempt_failure(
                ctx, node_id, e, max_retries
            )
            if exhausted:
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
        resources = await asyncio.to_thread(engine.discover_callable_resources)
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


async def _mcp_lookup_resource_node(ctx: StepContext, server_name: str) -> Any | None:
    """``CallableResourceNode`` lookup for this MCP server, if the KG engine supports it.

    Extracted verbatim from ``mcp_server_step`` (pure extract-method, no behaviour
    change).
    """
    engine = ctx.deps.knowledge_engine
    if not (engine and hasattr(engine, "ogm")):
        return None

    from ..models.knowledge_graph import CallableResourceNode

    nodes = await asyncio.to_thread(
        engine.ogm.find,
        CallableResourceNode,
        properties={"name": server_name},
    )
    return nodes[0] if nodes else None


async def _mcp_find_matching_specialist_agents(server_name: str) -> list[Any]:
    """Registry lookup for specialist agents bound to this MCP server.

    Extracted verbatim from ``mcp_server_step`` (pure extract-method, no behaviour
    change).
    """
    registry = await asyncio.to_thread(get_discovery_registry)
    return [a for a in registry.agents if a.mcp_server == server_name]


async def _mcp_execute_matching_specialists(
    ctx: StepContext, matching_agents: list[Any]
) -> None:
    """Execute each matching specialist agent for this server.

    Extracted verbatim from ``mcp_server_step`` (pure extract-method, no behaviour
    change).
    """
    for mcp_agent in matching_agents:
        await _execute_dynamic_mcp_agent(ctx, mcp_agent)


async def _mcp_build_fallback_toolsets(ctx: StepContext, server_name: str) -> list[Any]:
    """Match + guard + scope the toolsets for the ad-hoc fallback agent.

    Extracted verbatim from ``mcp_server_step`` (pure extract-method, no behaviour
    change).
    """
    # Fallback: create ad-hoc agent with all tools from this server
    matched_toolsets = []
    for toolset in ctx.deps.mcp_toolsets:
        server_id = getattr(toolset, "id", getattr(toolset, "name", None))
        if server_id and server_name in str(server_id):
            matched_toolsets.append(toolset)

    from agent_utilities.security.tool_guard import flag_mcp_tool_definitions

    guarded_toolsets = flag_mcp_tool_definitions(
        matched_toolsets,
        permissions_kernel=ctx.deps.permissions_kernel,
        agent_identity=ctx.deps.agent_identity,
        engine=ctx.deps.knowledge_engine,
    )
    _, scoped_toolsets = apply_tool_scope(ctx.state, [], guarded_toolsets)
    return scoped_toolsets


async def _mcp_run_fallback_agent(
    ctx: StepContext,
    server_name: str,
    query: str,
    resource_node: Any | None,
    scoped_toolsets: list[Any],
) -> tuple[str, Any]:
    """Build the ad-hoc fallback agent, run it streaming, and store the result.

    Extracted verbatim from ``mcp_server_step`` (pure extract-method, no behaviour
    change). Returns ``(result_key, stream)`` — ``stream`` stays valid for
    provenance/WebUI-event use after the ``async with`` exits, exactly as the
    original inline code relied on.
    """
    # Use unified resource metadata if available, otherwise fallback
    system_prompt = f"You are a specialist for the '{server_name}' resource. Use the available tools to answer queries."
    if resource_node and resource_node.description:
        system_prompt += f" Context: {resource_node.description}"

    agent = create_context_agent(
        model=ctx.deps.agent_model,
        permissions_kernel=ctx.deps.permissions_kernel,
        agent_identity=ctx.deps.agent_identity,
        permission_engine=ctx.deps.knowledge_engine,
        system_prompt=system_prompt + invoker_context_section(ctx.state),  # ORCH-1.39
        toolsets=scoped_toolsets,
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
    ctx.state._update_usage(stream.usage)
    result_key = f"{server_name}_{ctx.state.step_cursor}"
    ctx.state.results_registry[result_key] = str(output)
    return result_key, stream


def _mcp_record_tool_call_provenance(ctx: StepContext, stream: Any) -> None:
    """Accumulate this MCP server's tool calls for :ToolCall provenance on the graph path.

    (CONCEPT:AU-KG.temporal.message-history-read). Unconditional — the WebUI event
    block below is gated on ``event_queue`` and skipped for headless
    (MCP/telegram) delegations, which is exactly the MCP-execution path a
    fleet-server delegation takes.
    """
    try:
        from ..orchestration.tool_provenance import extract_tool_calls

        ctx.state.tool_calls.extend(extract_tool_calls(stream))
    except Exception as _tc_exc:  # noqa: BLE001 — never break a run
        logger.debug("mcp_server tool-call provenance skipped: %s", _tc_exc)


def _mcp_emit_tool_call_events(ctx: StepContext, server_name: str, msg: Any) -> None:
    """Emit ``expert_tool_call`` events for each ``ToolCallPart`` in a ``ModelResponse``.

    Extracted verbatim from ``mcp_server_step`` (pure extract-method, no behaviour
    change).
    """
    from pydantic_ai.messages import ToolCallPart

    for part in msg.parts:
        if isinstance(part, ToolCallPart):
            emit_graph_event(
                ctx.deps.event_queue,
                "expert_tool_call",
                domain=server_name,
                tool_name=part.tool_name,
                args=part.args,
            )


def _mcp_emit_tool_result_events(ctx: StepContext, server_name: str, msg: Any) -> None:
    """Emit ``tool_result`` events for each ``ToolReturnPart`` in a ``ModelRequest``.

    Extracted verbatim from ``mcp_server_step`` (pure extract-method, no behaviour
    change).
    """
    from pydantic_ai.messages import ToolReturnPart

    for req_part in msg.parts:
        if isinstance(req_part, ToolReturnPart):
            emit_graph_event(
                ctx.deps.event_queue,
                event_type="tool_result",
                agent=server_name,
                tool=req_part.tool_name,
                result=str(req_part.content)[:500],
            )


def _mcp_stream_events_to_webui(
    ctx: StepContext, server_name: str, stream: Any
) -> None:
    """Stream tool-call/tool-result events to the WebUI, if an event queue is present.

    Extracted verbatim from ``mcp_server_step`` (pure extract-method, no behaviour
    change).
    """
    if not ctx.deps.event_queue:
        return

    from pydantic_ai.messages import ModelRequest, ModelResponse

    for msg in stream.all_messages():
        if isinstance(msg, ModelResponse):
            _mcp_emit_tool_call_events(ctx, server_name, msg)
        elif isinstance(msg, ModelRequest):
            _mcp_emit_tool_result_events(ctx, server_name, msg)


async def _mcp_execute_server_path(
    ctx: StepContext,
    server_name: str,
    query: str,
    resource_node: Any | None,
    matching_agents: list[Any],
) -> list[str]:
    """Execute the specialist or fallback path and return created result keys."""
    if matching_agents:
        existing_result_keys = set(ctx.state.results_registry)
        await _mcp_execute_matching_specialists(ctx, matching_agents)
        return [
            key for key in ctx.state.results_registry if key not in existing_result_keys
        ]
    scoped_toolsets = await _mcp_build_fallback_toolsets(ctx, server_name)
    result_key, stream = await _mcp_run_fallback_agent(
        ctx, server_name, query, resource_node, scoped_toolsets
    )
    _mcp_record_tool_call_provenance(ctx, stream)
    _mcp_stream_events_to_webui(ctx, server_name, stream)
    return [result_key]


def _mcp_result_summary(ctx: StepContext, result_keys: list[str]) -> str:
    """Bound the combined result text emitted with server completion."""
    return "\n".join(
        str(ctx.state.results_registry.get(key, "")) for key in result_keys
    )[:500]


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
        resource_node = await _mcp_lookup_resource_node(ctx, server_name)

        # Check if there's a matching dynamic MCP agent in the registry
        matching_agents = await _mcp_find_matching_specialist_agents(server_name)
        result_keys = await _mcp_execute_server_path(
            ctx, server_name, query, resource_node, matching_agents
        )

        emit_graph_event(
            ctx.deps.event_queue,
            event_type="node_complete",
            id="mcp_server_execution",
            server=server_name,
            result=_mcp_result_summary(ctx, result_keys),
        )

        return "execution_joiner"
    except Exception as e:
        logger.error(f"MCP Server Step '{server_name}' failed: {e}")
        ctx.state.error = f"MCP Server {server_name} failed: {e}"
        return "error_recovery"
