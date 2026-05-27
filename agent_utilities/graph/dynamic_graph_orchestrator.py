from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import networkx as nx

from agent_utilities.core.config import (
    DEFAULT_ENABLE_LLM_VALIDATION,
    DEFAULT_GRAPH_PERSISTENCE_PATH,
    DEFAULT_GRAPH_ROUTER_TIMEOUT,
    DEFAULT_GRAPH_VERIFIER_TIMEOUT,
    DEFAULT_LITE_LLM_MODEL_ID,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_ROUTER_MODEL,
    DEFAULT_SSL_VERIFY,
)
from agent_utilities.core.model_factory import create_model

from ..knowledge_graph.core.formal_reasoning_core import (
    chromatic_schedule,
    dag_critical_path,
)
from ..models import GraphResponse
from .config_helpers import (
    DEFAULT_GRAPH_TIMEOUT,
    emit_graph_event,
    load_node_agents_registry,
)
from .coordination import CoordinationLayer
from .mermaid import get_graph_mermaid
from .state import REQUESTED_MODEL_ID_CTX, GraphDeps, GraphState

# --- Merged from dynamic_graph_orchestrator.py ---

#!/usr/bin/python
"""Dynamic Subgraph Orchestrator (CONCEPT:ORCH-1.4).

Dynamically synthesizes pydantic-graph transition logic from the Knowledge
Graph on the fly without using predefined templates. Uses formal graph theory
primitives (KG-2.41) to determine the exact DAG structure, parallel groups,
and execution paths.
"""


if TYPE_CHECKING:
    from ..knowledge_graph.core.engine import IntelligenceGraphEngine
    from ..models.knowledge_graph import TeamComposition

logger = logging.getLogger(__name__)


class DynamicSubgraphOrchestrator:
    """Dynamically synthesizes graph topology.

    CONCEPT:ORCH-1.4 — Dynamic Subgraph Orchestration
    """

    def __init__(self, engine: IntelligenceGraphEngine | None = None):
        self.engine = engine
        # CONCEPT:ORCH-1.3 — Coordination Protocol Layer (Research: 2605.03310v1)
        self.coordination_layer = CoordinationLayer(engine=engine)

    def synthesize_team(
        self,
        query: str,
        domain: str = "general",
        complexity: int = 3,
        available_tools: list[str] | None = None,
        available_agents: list[str] | None = None,
        delegated_authority: str | None = None,
    ) -> TeamComposition:
        """Synthesize a TeamComposition fully dynamically from the KG.

        Flow:
            1. Parse required capabilities from the query.
            2. Extract matching agent nodes from the KG.
            3. Build a dependency DAG based on capability prerequisites.
            4. Use graph theory primitives to find critical paths and parallel groups.
            5. Construct the TeamComposition.
        """
        from ..models.knowledge_graph import TeamComposition

        team_id = f"team:dyn:{uuid.uuid4().hex[:8]}"

        # Step 1 & 2: Get candidate agents for this task
        agents = self._retrieve_candidate_agents(
            query, domain, available_agents, delegated_authority
        )
        if not agents:
            # Fallback to a basic executor if KG lookup fails
            agents = [{"role": "executor", "agent_id": "executor", "tools": []}]

        # Step 3: Build dependency DAG
        dag = self._build_dependency_dag(agents)

        # Step 4: Analyze DAG (Critical path & parallel groups)
        critical_info = {}
        if len(dag) > 1:
            try:
                critical_info = dag_critical_path(dag)
            except Exception as e:
                logger.debug("DAG critical path failed: %s", e)

        # Build conflict graph for parallelization (chromatic scheduling)
        conflict_graph = self._build_conflict_graph(dag)
        coloring = chromatic_schedule(conflict_graph)

        # Group by color (parallel groups)
        parallel_groups_dict: dict[int, list[str]] = {}
        for node, color in coloring.items():
            parallel_groups_dict.setdefault(color, []).append(node)

        parallel_groups = [g for g in parallel_groups_dict.values() if len(g) > 1]

        # Determine execution mode
        if len(agents) == 1:
            mode = "sequential"
        elif parallel_groups:
            if len(parallel_groups) == 1 and len(parallel_groups[0]) == len(agents):
                mode = "parallel"
            else:
                mode = "mixed"
        else:
            mode = "sequential"

        # Build adaptive_agent_router configs
        adaptive_agent_router = []
        for agent in agents:
            # Step 5: Ask KG for tools
            tools = self._discover_tools_for_agent(str(agent["role"]), available_tools)

            # CONCEPT:ORCH-1.2 - Adaptive Model Routing (Assimilated from Pydantic AI)
            model_id = agent.get("model_id", "")
            if self.engine and not model_id:
                try:
                    from .adaptive_agent_router import (
                        RoutingCandidate,
                        RoutingPrimitive,
                        TopologicalRoutingPolicy,
                    )

                    # Generate candidate models dynamically
                    candidates = [
                        RoutingCandidate(
                            model_id="gemini-2.5-pro",
                            primitive=RoutingPrimitive.DECOMPOSE,
                            confidence=0.9,
                        ),
                        RoutingCandidate(
                            model_id="gemini-2.5-flash",
                            primitive=RoutingPrimitive.DIRECT,
                            confidence=0.7,
                        ),
                    ]
                    policy = TopologicalRoutingPolicy(engine=self.engine)
                    decision = policy.route(query, candidates)
                    model_id = decision.selected.model_id
                    logger.debug(
                        f"[ORCH-1.2] Adaptive Model Routing selected {model_id} for role {agent['role']}"
                    )
                except Exception as e:
                    logger.debug(f"Adaptive model routing failed: {e}")

            adaptive_agent_router.append(
                {
                    "role": agent["role"],
                    "agent_id": agent["agent_id"],
                    "tools": tools,
                    "model_id": model_id,
                    "system_prompt": agent.get(
                        "system_prompt",
                        f"You are a dynamically spawned {agent['role']}.",
                    ),
                    "memory_channels": ["episodic", domain],
                }
            )

        composition = TeamComposition(
            team_id=team_id,
            source="dynamic_synthesis",
            topology_template_id=f"topo:dyn:{domain}:{complexity}",
            adaptive_agent_router=adaptive_agent_router,
            execution_mode=mode,
            parallel_groups=parallel_groups,
            memory_channels=["episodic", domain],
            confidence=0.85,
            reasoning=f"Dynamically synthesized topology via ORCH-1.19. Makespan: {critical_info.get('makespan', 1.0)}",
        )

        # CONCEPT:ORCH-1.3 — Select and apply coordination protocol (Research: 2605.03310v1)
        agent_ids: list[str] = [str(a["agent_id"]) for a in agents]
        protocol = self.coordination_layer.select_protocol(
            agent_count=len(agents),
            task_type=domain,
            execution_mode=mode,
        )
        coord_result = self.coordination_layer.apply_protocol(
            protocol=protocol,
            agent_ids=agent_ids,
            task=query,
            task_type=domain,
        )
        # Persist coordination trace to KG
        self.coordination_layer.log_coordination_trace(coord_result)
        # Attach protocol metadata to the composition
        composition.coordination_protocol = {
            "protocol_id": protocol.protocol_id,
            "protocol_type": protocol.protocol_type.value,
            "name": protocol.name,
            "quality_score": coord_result.quality_score,
            "converged": coord_result.converged,
        }

        logger.info(
            "[CONCEPT:ORCH-1.4] Dynamically synthesized subgraph '%s': %d adaptive_agent_router, mode=%s, protocol=%s",
            team_id,
            len(adaptive_agent_router),
            mode,
            protocol.name,
        )

        return composition

    def _retrieve_candidate_agents(
        self,
        query: str,
        domain: str,
        available_agents: list[str] | None,
        delegated_authority: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query the KG for agents capable of handling aspects of the query."""
        if not self.engine or not self.engine.backend:
            return []

        agents = []
        try:
            query_str = "MATCH (a:Agent)-[:HAS_CAPABILITY]->(c:AgentCapability) "

            if delegated_authority:
                query_str += "MATCH (a)-[:HAS_DELEGATED_AUTHORITY_FROM*0..5]->(:Person {id: $delegated_authority}) "

            query_str += (
                "WHERE toLower($query) CONTAINS toLower(c.name) "
                "RETURN a.id AS agent_id, a.name AS name, a.role AS role "
                "LIMIT 5"
            )

            results = self.engine.backend.execute(
                query_str,
                {"query": query, "delegated_authority": delegated_authority},
            )
            for r in results:
                role = r.get("role") or r.get("name") or r.get("agent_id")
                aid = r.get("agent_id")
                if available_agents is None or aid in available_agents:
                    agent_data = {"role": role, "agent_id": aid}
                    model_id = r.get("model_id")
                    if model_id:
                        agent_data["model_id"] = model_id
                    agents.append(agent_data)
        except Exception:
            pass  # nosec

        return agents

    def _build_dependency_dag(self, agents: list[dict[str, Any]]) -> nx.DiGraph:
        """Build a DAG of agent dependencies based on expected data flow."""
        dag = nx.DiGraph()
        for a in agents:
            dag.add_node(a["role"], weight=1.0)

        # In a fully KG-driven system, we'd query REQUIRES_OUTPUT_FROM edges.
        # Here we create a simple sequential DAG if no explicit edges exist.
        roles = [a["role"] for a in agents]
        for i in range(len(roles) - 1):
            dag.add_edge(roles[i], roles[i + 1], weight=1.0)

        return dag

    def _build_conflict_graph(self, dag: nx.DiGraph) -> nx.Graph:
        """Convert DAG into a conflict graph for chromatic scheduling.
        Nodes that share a directed path have a conflict (must run sequentially).
        """
        conflict_graph = nx.Graph()
        conflict_graph.add_nodes_from(dag.nodes())

        # Add edges between any two nodes where one is reachable from the other
        for u in dag.nodes():
            reachable = nx.descendants(dag, u)
            for v in reachable:
                conflict_graph.add_edge(u, v)

        return conflict_graph

    def _discover_tools_for_agent(
        self, role: str, available_tools: list[str] | None
    ) -> list[str]:
        if not self.engine or not self.engine.backend:
            return []
        tools = []
        try:
            results = self.engine.backend.execute(
                "MATCH (a)-[:PROVIDES|HAS_CAPABILITY]->(t:CallableResource) "
                "WHERE toLower(a.name) CONTAINS toLower($role) "
                "RETURN t.name AS tool_name LIMIT 5",
                {"role": role},
            )
            for r in results:
                name = r.get("tool_name", "")
                if name:
                    if available_tools is None or name in available_tools:
                        tools.append(name)
        except Exception:
            pass  # nosec
        return tools


# --- Merged from dynamic_graph_orchestrator.py ---

#!/usr/bin/python
"""KG-Driven Pydantic Graph Engine (CONCEPT:ORCH-1.4).

Shifts control of pydantic-graph workflows from Python logic to Knowledge Graph state transitions.
Every step polls the KG for the next optimal execution node instead of relying on statically compiled transitions.
"""


logger = logging.getLogger(__name__)


class KGDrivenExecutionEngine:
    """Orchestrates dynamic pydantic-graph execution using the KG.

    CONCEPT:ORCH-1.4 - KG-Driven Pydantic Graph Engine
    """

    def __init__(self, engine: IntelligenceGraphEngine):
        self.engine = engine

    def determine_next_node(self, current_node_id: str, context: dict[str, Any]) -> str:
        """Poll the KG to determine the optimal next execution node based on current context.

        Args:
            current_node_id: The identifier of the node that just completed.
            context: The execution context containing findings, errors, and task state.

        Returns:
            The identifier of the next node to execute.
        """
        # If no backend is available, fallback to a simple sequential heuristic or end.
        if not self.engine.backend:
            logger.warning("No backend available. Falling back to default END node.")
            return "END"

        # Query the KG for potential transitions linked to the current state node
        query = """
        MATCH (current:ExecutionStateNode {node_id: $current_id})-[r:TRANSITION_TO]->(next:ExecutionStateNode)
        RETURN next.node_id as next_id, r.condition as condition, r.priority as priority
        ORDER BY r.priority DESC
        """

        results = self.engine.backend.execute(query, {"current_id": current_node_id})

        for row in results:
            condition = row.get("condition")
            # Evaluate the condition dynamically against the context.
            # In a production system, this would be a safe eval or pattern match.
            # For this KG-native engine, we check if the condition text matches context markers.
            if self._evaluate_condition(condition, context):
                next_id = row.get("next_id")
                if next_id:
                    logger.debug(
                        f"[ORCH-1.20] KG routing from {current_node_id} -> {next_id}"
                    )
                    return str(next_id)

        # Fallback to the dynamic orchestrator if no predefined static route matches
        logger.debug("[ORCH-1.20] No static route matched. Using dynamic synthesis.")
        task = context.get("task", "")
        if task:
            # Re-synthesize team or fetch next best from orchestrator capabilities
            # For now, default to END if we reach the edge of the graph.
            pass

        return "END"

    def _evaluate_condition(
        self, condition: str | None, context: dict[str, Any]
    ) -> bool:
        """Evaluate a KG string condition against the current execution context."""
        if not condition or condition.lower() == "always":
            return True
        if condition.lower() == "on_error" and context.get("error"):
            return True
        if condition.lower() == "on_success" and not context.get("error"):
            return True

        # More complex ontological evaluations could be plugged in here
        return False


# --- Merged from runner.py ---

#!/usr/bin/python
"""Graph Execution Module.

This module provides the execution logic for the pydantic-graph orchestrator.
It defines functions to run graphs synchronously or as asynchronous streams (SSE),
handles state persistence, MCP server connectivity during runs, and graph validation.

CONCEPT:ORCH-1.0 Graph Orchestration
"""


try:
    from opentelemetry import trace

    tracer = trace.get_tracer("agent-utilities.graph")
except ImportError:
    tracer = None

logger = logging.getLogger(__name__)


async def run_graph(
    graph,
    config: dict,
    query: str,
    run_id: str | None = None,
    persist: bool = False,
    state_dir: str = DEFAULT_GRAPH_PERSISTENCE_PATH or "graph_state",
    streamdown: bool = True,
    eq: asyncio.Queue[Any] | None = None,
    mode: str = "ask",
    topology: str = "basic",
    mcp_toolsets: list[Any] | None = None,
    query_parts: list[dict[str, Any]] | None = None,
    plan_sync=None,
    requested_model_id: str | None = None,
) -> dict:
    """Execute a query through the graph orchestrator (synchronous/batch).

    This function initializes the execution context, connects to the required
    MCP servers, and executes the graph loop until completion or timeout.

    Args:
        graph: The Graph object created by the builder.
        config: Configuration dictionary containing dependencies and settings.
        query: The user's input query string.
        run_id: Optional unique identifier for this execution session.
        persist: Whether to enable persistent state storage for this run.
        state_dir: Directory path for state persistence files.
        streamdown: Whether to include a mermaid diagram of the graph in the output.
        eq: Optional asyncio.Queue for sideband graph lifecycle events.
        mode: The orchestrator's execution mode (e.g., 'ask', 'research').
        topology: The selected graph topology (e.g., 'basic', 'dynamic').
        mcp_toolsets: Optional override list of MCP toolsets to use.
        query_parts: Optional structural message parts for complex queries.
        requested_model_id: Optional per-turn model id sourced from the
            ``x-agent-model-id`` request header. When valid within the
            attached ``model_registry``, specialist spawning uses it
            verbatim (see :func:`pick_specialist_model`).

    Returns:
        A GraphResponse instance containing the final synthesized output
        and execution metadata.

    """
    if run_id is None:
        run_id = uuid4().hex

    if requested_model_id is None:
        requested_model_id = REQUESTED_MODEL_ID_CTX.get()

    mermaid_prefix = ""
    if streamdown:
        with contextlib.suppress(Exception):
            mermaid_prefix = f"```mermaid\n{get_graph_mermaid(graph, config)}\n```\n\n"

    state = GraphState(
        query=query, query_parts=query_parts or [], mode=mode, topology=topology
    )

    _custom_headers = config.get("custom_headers")
    _ssl_verify = config.get("ssl_verify", DEFAULT_SSL_VERIFY)

    deps = GraphDeps(
        tag_prompts=config.get("tag_prompts", {}),
        tag_env_vars=config.get("tag_env_vars", {}),
        mcp_toolsets=(
            mcp_toolsets if mcp_toolsets is not None else config.get("mcp_toolsets", [])
        ),
        mcp_url=config.get("mcp_url", ""),
        mcp_config=config.get("mcp_config", ""),
        router_model=create_model(
            model_id=config.get("router_model", DEFAULT_ROUTER_MODEL),
            api_key=config.get("api_key"),
            base_url=config.get("base_url"),
            custom_headers=_custom_headers,
            provider=config.get("provider", DEFAULT_LLM_PROVIDER),
            ssl_verify=_ssl_verify,
        ),
        agent_model=create_model(
            model_id=config.get("agent_model", DEFAULT_LITE_LLM_MODEL_ID),
            api_key=config.get("api_key"),
            base_url=config.get("base_url"),
            custom_headers=_custom_headers,
            provider=config.get("provider", DEFAULT_LLM_PROVIDER),
            ssl_verify=_ssl_verify,
        ),
        nodes=config.get("nodes", {}),
        min_confidence=config.get("min_confidence", 0.6),
        sub_agents=config.get("sub_agents", {}),
        provider=config.get("provider", DEFAULT_LLM_PROVIDER),
        base_url=config.get("base_url"),
        api_key=config.get("api_key"),
        ssl_verify=_ssl_verify,
        event_queue=eq,
        router_timeout=config.get("router_timeout", DEFAULT_GRAPH_ROUTER_TIMEOUT),
        verifier_timeout=config.get("verifier_timeout", DEFAULT_GRAPH_VERIFIER_TIMEOUT),
        request_id=config.get("request_id", run_id),
        routing_strategy=config.get("routing_strategy", "hybrid"),
        enable_llm_validation=config.get(
            "enable_llm_validation", DEFAULT_ENABLE_LLM_VALIDATION
        ),
        discovery_metadata=config.get("discovery_metadata", {}),
        plan_sync=plan_sync,
        approval_manager=config.get("approval_manager"),
        model_registry=config.get("model_registry"),
        requested_model_id=requested_model_id,
    )

    state = GraphState(
        query=query,
        query_parts=query_parts or [],
        session_id=run_id,
        mode=mode,
        topology=topology,
    )

    if persist:
        path = Path(state_dir)
        if path.suffix != ".json":
            path = path / f"{run_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        # persistence = FileStatePersistence(json_file=path)

    from contextlib import AsyncExitStack

    import anyio

    # Track which MCP servers fail to connect so we can report them clearly.
    failed_servers: list[tuple[str, str]] = []
    connected_toolsets: list = []

    # Track which toolsets have already been connected by the global
    # lifespan (server.py pre-connects them at startup).  Use object
    # identity instead of mutating toolset attributes.
    _already_connected: set[int] = set()

    async with AsyncExitStack() as stack:
        for ts in deps.mcp_toolsets:
            if not hasattr(ts, "__aenter__"):
                connected_toolsets.append(ts)
                continue
            if id(ts) in _already_connected:
                connected_toolsets.append(ts)
                continue

            srv_id = getattr(ts, "id", None) or getattr(ts, "name", None) or repr(ts)
            try:
                logger.debug(f"run_graph: Connecting to MCP server '{srv_id}'...")
                connected = await stack.enter_async_context(ts)
                _already_connected.add(id(ts))
                connected_toolsets.append(connected)
                logger.info(f"run_graph: ✅ MCP server '{srv_id}' connected")
            except Exception as e:
                import traceback

                err_msg = "".join(
                    traceback.format_exception(type(e), e, e.__traceback__)
                )
                logger.error(
                    f"run_graph: ❌ MCP server '{srv_id}' FAILED to connect: {err_msg}"
                )
                failed_servers.append((srv_id, err_msg))

        from pydantic_ai.toolsets.abstract import AbstractToolset

        class PreconnectedToolsetWrapper(AbstractToolset):
            """Wraps a connected toolset to hide its context manager methods from Pydantic-AI."""

            def __init__(self, ts):
                self._ts = ts

            @property
            def id(self):
                return getattr(self._ts, "id", getattr(self._ts, "name", "unknown"))

            async def get_tools(self, ctx):
                if hasattr(self._ts, "get_tools"):
                    return await self._ts.get_tools(ctx)
                return {}

            async def call_tool(self, name, tool_args, ctx, tool):
                if hasattr(self._ts, "call_tool"):
                    return await self._ts.call_tool(name, tool_args, ctx, tool)
                raise NotImplementedError(f"Toolset {self.id} cannot call tools")

            async def get_instructions(self, ctx):
                if hasattr(self._ts, "get_instructions"):
                    return await self._ts.get_instructions(ctx)
                return None

        deps.mcp_toolsets = [
            PreconnectedToolsetWrapper(ts)
            for ts in connected_toolsets
            if ts is not None
        ]

        if failed_servers:
            logger.warning(
                f"run_graph: {len(failed_servers)} MCP server(s) failed to connect — "
                f"graph will proceed without them:\n"
                + "\n".join(f"  ❌ {sid}: {err}" for sid, err in failed_servers)
            )

        # Standardize tag_prompts from the registry for high-fidelity routing.
        # We merge existing prompts with registry-provided domain tags.
        registry = load_node_agents_registry()
        for agent in registry.agents:
            # Domain tags (like 'git_operations') are the primary routing targets
            # MCPAgent uses 'name' as the primary identifier
            if agent.name and agent.name not in deps.tag_prompts:
                deps.tag_prompts[agent.name] = agent.description or agent.name

            # Legacy node mapping (by name)
            node_id = agent.name.lower().replace(" ", "_")
            if node_id not in deps.tag_prompts:
                deps.tag_prompts[node_id] = agent.description or agent.name

        emit_graph_event(
            deps.event_queue,
            "graph_start",
            run_id=run_id,
            query=query,
            topology=topology,
        )
        logger.info(
            f"run_graph: Starting graph execution for run_id {run_id}. Registered {len(deps.tag_prompts)} adaptive_agent_router."
        )

        # --- CONCEPT:ORCH-1.4 Service Registry Initialization ---
        try:
            from .service_registry import ServiceRegistry

            svc_registry = ServiceRegistry.instance()
            svc_count = svc_registry.initialize()
            logger.debug(
                "run_graph: Service registry initialized with %d services", svc_count
            )

            # --- CONCEPT:ECO-4.0 Dynamic Tool Hydration (Plugin Registry) ---
            from .plugin_registry import PluginRegistry

            plugin_registry = PluginRegistry.instance()
            deps.plugin_registry = plugin_registry
            logger.debug(
                "run_graph: Plugin registry initialized for dynamic tool hydration."
            )
        except Exception as e:
            logger.debug("run_graph: Service registry init skipped: %s", e)

        # --- Telemetry Engine Initialization (OS-5.6, OS-5.7, OS-5.9) ---
        _telemetry = None
        _telemetry_start = time.monotonic()
        try:
            from ..observability import TelemetryEngine

            _telemetry = TelemetryEngine()
            _telemetry.on_graph_start(
                run_id=run_id, agent_id=config.get("agent_id", ""), query=query
            )
        except Exception as e:
            logger.debug("run_graph: Telemetry init skipped: %s", e)

        # --- Security Guard Pre-Flight (OS-5.4, OS-5.5) ---
        try:
            from ..security.threat_defense_engine import PromptInjectionScanner

            scanner = PromptInjectionScanner()
            scan_result = scanner.scan_text(query)
            if scan_result.is_malicious:
                logger.warning(
                    "run_graph: Query blocked by prompt scanner: %s",
                    scan_result.explanation,
                )
                return GraphResponse(
                    status="blocked",
                    error=f"Security: {scan_result.explanation}",
                    metadata={
                        "run_id": run_id,
                        "is_error": True,
                        "security": {
                            "confidence": scan_result.confidence,
                            "finding_id": scan_result.finding_id,
                        },
                    },
                ).model_dump()
            if scan_result.matches:
                logger.info(
                    "run_graph: Prompt scanner warnings: %d patterns below threshold",
                    len(scan_result.matches),
                )
        except ImportError:
            pass  # Scanner not available
        except Exception as e:
            logger.debug("run_graph: Prompt scanning skipped: %s", e)
        result = None
        try:
            if tracer:
                with tracer.start_as_current_span(f"graph_run:{run_id}") as span:
                    span.set_attribute("query", query)
                    span.set_attribute("request_id", deps.request_id)
                    with anyio.move_on_after(DEFAULT_GRAPH_TIMEOUT / 1000.0) as scope:
                        result = await graph.run(state=state, deps=deps)
                    if scope.cancel_called:
                        logger.error(
                            f"run_graph: Graph execution TIMEOUT after {DEFAULT_GRAPH_TIMEOUT}ms"
                        )
                        result = "timeout"
                    span.set_status(
                        trace.Status(
                            trace.StatusCode.OK if result else trace.StatusCode.ERROR
                        )
                    )
            else:
                logger.info("run_graph: Running beta graph.run (no tracer)...")
                with anyio.move_on_after(DEFAULT_GRAPH_TIMEOUT / 1000.0) as scope:
                    result = await graph.run(state=state, deps=deps)
                if scope.cancel_called:
                    logger.error(
                        f"run_graph: Graph execution TIMEOUT after {DEFAULT_GRAPH_TIMEOUT}ms"
                    )
                    result = "timeout"
        except Exception as e:
            logger.error(f"run_graph: CRITICAL ERROR during graph execution: {e}")
            emit_graph_event(
                deps.event_queue, "graph_complete", run_id=run_id, status="error"
            )
            return GraphResponse(
                status="error",
                error=str(e),
                metadata={"run_id": run_id, "is_error": True},
            ).model_dump()

        logger.info(
            f"run_graph: graph.run finished. Result type: {type(result)}, Result: {result}"
        )
        emit_graph_event(
            deps.event_queue,
            "graph_complete",
            run_id=run_id,
            status="success" if result else "timeout",
        )
        logger.info(
            f"run_graph: Final state: routed_domain={state.routed_domain}, "
            f"registry_keys={list(state.results_registry.keys())}"
        )

    # pydantic-graph End wraps the value in .data
    result = getattr(result, "data", result)

    if isinstance(result, GraphResponse):
        result.mermaid = mermaid_prefix if mermaid_prefix else None
        result.metadata.update({"run_id": run_id, "domain": state.routed_domain})
        if _telemetry:
            _telemetry.on_graph_end(
                run_id=run_id,
                status=result.status or "success",
                duration_ms=(time.monotonic() - _telemetry_start) * 1000,
            )
        return result.model_dump()

    # Guard: graph.run() returned a plain string (node label) instead of GraphResponse.
    # This happens when the graph exits without hitting End[GraphResponse] — e.g. when
    # dispatcher returns None with an empty results_registry. Extract the best available
    # result from state before wrapping.
    if isinstance(result, str):
        logger.error(
            f"run_graph: graph.run() returned node label '{result}' instead of GraphResponse. "
            f"This indicates the graph terminated unexpectedly. "
            f"Registry keys: {list(state.results_registry.keys())}, "
            f"Results keys: {list(state.results.keys())}"
        )
        # Priority: results_registry (plan-based) → results (domain-based) → error message
        output = (
            next(iter(state.results_registry.values()), None)
            or next(iter(state.results.values()), None)
            or f"Graph terminated unexpectedly at node '{result}'. No results were generated."
        )
        return GraphResponse(
            status="partial",
            results={"output": output},
            mermaid=mermaid_prefix if mermaid_prefix else None,
            metadata={
                "run_id": run_id,
                "domain": state.routed_domain,
                "terminated_at": result,
            },
        ).model_dump()

    return GraphResponse(
        status="completed",
        results={"output": str(result)},
        mermaid=mermaid_prefix if mermaid_prefix else None,
        metadata={"run_id": run_id, "domain": state.routed_domain},
    ).model_dump()


async def run_graph_stream(
    graph,
    config: dict,
    query: str,
    run_id: str | None = None,
    persist: bool = False,
    state_dir: str = DEFAULT_GRAPH_PERSISTENCE_PATH or "agent_data/graph_state",
    mode: str = "ask",
    topology: str = "basic",
    mcp_toolsets: list[Any] | None = None,
    query_parts: list[dict[str, Any]] | None = None,
    plan_sync=None,
    requested_model_id: str | None = None,
):
    r"""Generator that yields graph events and text output as a stream of SSE events.

    This function handles the concurrent execution of the graph while
    streaming real-time updates (thoughts, tool calls, status) to the
    Agent UI via an asynchronous event queue.

    Args:
        graph: The Graph object created by the builder.
        config: Execution configuration dictionary.
        query: User input query string.
        run_id: Optional identifier for the session; auto-generated if omitted.
        persist: Whether to persist state metadata.
        state_dir: Path to the persistence directory.
        mode: The orchestrator's execution mode.
        topology: The graph topology to use.
        mcp_toolsets: Toolsets to inject during execution.
        query_parts: Structured message parts for the initial prompt.
        requested_model_id: Optional per-turn model id sourced from the
            ``x-agent-model-id`` request header. See :func:`run_graph`.

    Yields:
        SSE-formatted strings ('data: {JSON}\n\n') containing lifecycle events.

    """
    import asyncio
    from pathlib import Path
    from uuid import uuid4

    if run_id is None:
        run_id = uuid4().hex

    if requested_model_id is None:
        requested_model_id = REQUESTED_MODEL_ID_CTX.get()

    eq: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    # Emit graph-start event via the sideband queue
    emit_graph_event(
        eq,
        "graph_start",
        run_id=run_id,
        query=query,
        topology=topology,
    )

    _custom_headers = config.get("custom_headers")
    _ssl_verify = config.get("ssl_verify", DEFAULT_SSL_VERIFY)

    deps = GraphDeps(
        tag_prompts=config.get("tag_prompts", {}),
        tag_env_vars=config.get("tag_env_vars", {}),
        mcp_toolsets=(
            mcp_toolsets if mcp_toolsets is not None else config.get("mcp_toolsets", [])
        ),
        mcp_url=config.get("mcp_url", ""),
        mcp_config=config.get("mcp_config", ""),
        router_model=create_model(
            model_id=config.get("router_model", DEFAULT_ROUTER_MODEL),
            api_key=config.get("api_key"),
            base_url=config.get("base_url"),
            custom_headers=_custom_headers,
            provider=config.get("provider", DEFAULT_LLM_PROVIDER),
            ssl_verify=_ssl_verify,
        ),
        agent_model=create_model(
            model_id=config.get("agent_model", DEFAULT_LITE_LLM_MODEL_ID),
            api_key=config.get("api_key"),
            base_url=config.get("base_url"),
            custom_headers=_custom_headers,
            provider=config.get("provider", DEFAULT_LLM_PROVIDER),
            ssl_verify=_ssl_verify,
        ),
        min_confidence=config.get("min_confidence", 0.6),
        sub_agents=config.get("sub_agents", {}),
        provider=config.get("provider", DEFAULT_LLM_PROVIDER),
        base_url=config.get("base_url"),
        api_key=config.get("api_key"),
        ssl_verify=_ssl_verify,
        event_queue=eq,
        router_timeout=config.get("router_timeout", DEFAULT_GRAPH_ROUTER_TIMEOUT),
        verifier_timeout=config.get("verifier_timeout", DEFAULT_GRAPH_VERIFIER_TIMEOUT),
        request_id=run_id,
        routing_strategy=config.get("routing_strategy", "hybrid"),
        enable_llm_validation=config.get(
            "enable_llm_validation", DEFAULT_ENABLE_LLM_VALIDATION
        ),
        discovery_metadata=config.get("discovery_metadata", {}),
        plan_sync=plan_sync,
        approval_manager=config.get("approval_manager"),
        model_registry=config.get("model_registry"),
        requested_model_id=requested_model_id,
    )

    state = GraphState(
        query=query, query_parts=query_parts or [], mode=mode, topology=topology
    )

    if persist:
        path = Path(state_dir)
        if path.suffix != ".json":
            path = path / f"{run_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        # persistence = FileStatePersistence(json_file=path)

    # Shared container for background tasks to pass graph results to the main loop
    graph_result_holder = {"value": None}

    async def run_in_background() -> None:
        from contextlib import AsyncExitStack

        try:
            async with AsyncExitStack() as stack:
                connected_toolsets = []
                _stream_connected: set[int] = set()
                for ts in deps.mcp_toolsets:
                    if not hasattr(ts, "__aenter__"):
                        connected_toolsets.append(ts)
                        continue
                    if id(ts) in _stream_connected:
                        connected_toolsets.append(ts)
                        continue
                    srv_id = getattr(ts, "id", getattr(ts, "name", repr(ts)))
                    logger.info(
                        f"run_graph_stream_bg: Connecting to MCP server '{srv_id}'..."
                    )

                    try:
                        connected = await stack.enter_async_context(ts)
                        _stream_connected.add(id(ts))
                        connected_toolsets.append(connected)
                    except Exception as e:
                        logger.error(
                            f"run_graph_stream_bg: Failed to connect to MCP server '{srv_id}': {e}"
                        )

                deps.mcp_toolsets = [ts for ts in connected_toolsets if ts is not None]

                result = await asyncio.wait_for(
                    graph.run(state=state, deps=deps),
                    timeout=DEFAULT_GRAPH_TIMEOUT / 1000.0,
                )
                graph_result_holder["value"] = result
        except TimeoutError:
            await eq.put({"type": "error", "error": "Graph execution timed out"})
        except Exception as e:
            await eq.put({"type": "error", "error": str(e)})
        finally:
            # Emit graph-complete event
            from .config_helpers import emit_graph_event

            emit_graph_event(eq, "graph_complete", run_id=run_id, status="success")
            await eq.put({"type": "complete"})

    task = asyncio.create_task(run_in_background())

    while True:
        event = await eq.get()
        if event.get("type") == "complete":
            break

        yield f"data: {json.dumps(event)}\n\n"

    await task

    # Extract the best available output with fallback mechanisms
    # 1. Graph End result (verifier's synthesized GraphResponse)
    # 2. state.results keyed by routed_domain (set by expert executor)
    # 3. First value in results_registry (plan-based, set by any step)
    # 4. First value in state.results (domain-based)
    # 5. Fallback message
    final_output = None
    graph_result = graph_result_holder.get("value")
    if graph_result is not None:
        # pydantic-graph End wraps the value in .data; GraphResponse has .results["Output"]
        result_data = getattr(graph_result, "data", graph_result)
        if hasattr(result_data, "results"):
            final_output = result_data.results.get("output")
        elif isinstance(result_data, dict):
            final_output = result_data.get("output", str(result_data))
        elif result_data:
            final_output = str(result_data)
    if not final_output:
        final_output = (
            state.results.get(state.routed_domain) if state.routed_domain else None
        )
    if not final_output:
        final_output = next(iter(state.results_registry.values()), None)
    if not final_output:
        final_output = next(iter(state.results.values()), None)
    if not final_output:
        final_output = "No output generated."
    yield f"data: {json.dumps({'type': 'final_output', 'content': final_output})}\n\n"


async def run_graph_iter(
    graph,
    config: dict,
    query: str,
    run_id: str | None = None,
    mode: str = "ask",
    topology: str = "basic",
    mcp_toolsets: list[Any] | None = None,
    query_parts: list[dict[str, Any]] | None = None,
    plan_sync=None,
    requested_model_id: str | None = None,
    elicitation_callback=None,
):
    r"""Execute graph step-by-step using ``graph.iter()`` for maximum control.

    Unlike :func:`run_graph` which delegates to ``graph.run()`` (blocking
    until completion), this function uses the beta ``graph.iter()`` API to
    yield per-step execution events.  This enables:

    * **Progress streaming** — each step yields metadata about the active
      node, enabling real-time AG-UI sideband updates.
    * **Pause/resume** — callers can stop iterating and serialize
      ``GraphState`` to disk, resuming later.
    * **Elicitation** — between steps the function checks
      ``state.human_approval_required`` and, if an ``elicitation_callback``
      is provided, pauses for human input before continuing.
    * **State snapshots** — every yielded event includes a lightweight
      snapshot of ``GraphState`` for audit/debugging.

    Args:
        graph: The Graph object created by the builder.
        config: Execution configuration dictionary.
        query: User input query string.
        run_id: Optional unique identifier for the execution session.
        mode: The orchestrator's execution mode.
        topology: The selected graph topology.
        mcp_toolsets: Toolsets to inject during execution.
        query_parts: Structured message parts for the initial prompt.
        plan_sync: Optional async callback for bridging plan state to ACP.
        requested_model_id: Optional per-turn model id override.
        elicitation_callback: Optional async callable invoked when the graph
            requires human approval.  Signature:
            ``async def cb(state: GraphState) -> str | None``.  Return a
            redirect string to override the next node, or ``None`` to
            continue normally.

    Yields:
        Dictionaries with the following ``type`` keys:

        * ``"node_start"`` — a graph node is about to execute
        * ``"node_complete"`` — a graph node has finished
        * ``"elicitation"`` — the graph is pausing for human input
        * ``"state_snapshot"`` — periodic serialization of ``GraphState``
        * ``"graph_complete"`` — the graph has finished executing
        * ``"error"`` — an error occurred during execution

    CONCEPT:ORCH-1.0 Graph Orchestration

    """
    from contextlib import AsyncExitStack

    from pydantic_graph.beta.graph import EndMarker

    if run_id is None:
        run_id = uuid4().hex

    if requested_model_id is None:
        requested_model_id = REQUESTED_MODEL_ID_CTX.get()

    eq: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    emit_graph_event(eq, "graph_start", run_id=run_id, query=query, topology=topology)

    _custom_headers = config.get("custom_headers")
    _ssl_verify = config.get("ssl_verify", DEFAULT_SSL_VERIFY)

    deps = GraphDeps(
        tag_prompts=config.get("tag_prompts", {}),
        tag_env_vars=config.get("tag_env_vars", {}),
        mcp_toolsets=(
            mcp_toolsets if mcp_toolsets is not None else config.get("mcp_toolsets", [])
        ),
        mcp_url=config.get("mcp_url", ""),
        mcp_config=config.get("mcp_config", ""),
        router_model=create_model(
            model_id=config.get("router_model", DEFAULT_ROUTER_MODEL),
            api_key=config.get("api_key"),
            base_url=config.get("base_url"),
            custom_headers=_custom_headers,
            provider=config.get("provider", DEFAULT_LLM_PROVIDER),
            ssl_verify=_ssl_verify,
        ),
        agent_model=create_model(
            model_id=config.get("agent_model", DEFAULT_LITE_LLM_MODEL_ID),
            api_key=config.get("api_key"),
            base_url=config.get("base_url"),
            custom_headers=_custom_headers,
            provider=config.get("provider", DEFAULT_LLM_PROVIDER),
            ssl_verify=_ssl_verify,
        ),
        nodes=config.get("nodes", {}),
        min_confidence=config.get("min_confidence", 0.6),
        sub_agents=config.get("sub_agents", {}),
        provider=config.get("provider", DEFAULT_LLM_PROVIDER),
        base_url=config.get("base_url"),
        api_key=config.get("api_key"),
        ssl_verify=_ssl_verify,
        event_queue=eq,
        router_timeout=config.get("router_timeout", DEFAULT_GRAPH_ROUTER_TIMEOUT),
        verifier_timeout=config.get("verifier_timeout", DEFAULT_GRAPH_VERIFIER_TIMEOUT),
        request_id=run_id,
        routing_strategy=config.get("routing_strategy", "hybrid"),
        enable_llm_validation=config.get(
            "enable_llm_validation", DEFAULT_ENABLE_LLM_VALIDATION
        ),
        discovery_metadata=config.get("discovery_metadata", {}),
        plan_sync=plan_sync,
        approval_manager=config.get("approval_manager"),
        model_registry=config.get("model_registry"),
        requested_model_id=requested_model_id,
    )

    state = GraphState(
        query=query, query_parts=query_parts or [], mode=mode, topology=topology
    )

    # Merge registry tags into deps (same as run_graph)
    registry = load_node_agents_registry()
    for agent in registry.agents:
        if agent.name and agent.name not in deps.tag_prompts:
            deps.tag_prompts[agent.name] = agent.description or agent.name
        node_id = agent.name.lower().replace(" ", "_")
        if node_id not in deps.tag_prompts:
            deps.tag_prompts[node_id] = agent.description or agent.name

    # Connect MCP servers
    failed_servers: list[tuple[str, str]] = []
    connected_toolsets: list = []
    _already_connected: set[int] = set()

    async with AsyncExitStack() as stack:
        for ts in deps.mcp_toolsets:
            if not hasattr(ts, "__aenter__"):
                connected_toolsets.append(ts)
                continue
            if id(ts) in _already_connected:
                connected_toolsets.append(ts)
                continue
            srv_id = getattr(ts, "id", getattr(ts, "name", repr(ts)))
            try:
                connected = await asyncio.wait_for(
                    stack.enter_async_context(ts), timeout=60.0
                )
                _already_connected.add(id(ts))
                connected_toolsets.append(connected)
                logger.info(f"run_graph_iter: ✅ MCP server '{srv_id}' connected")
            except Exception as e:
                import traceback

                err_msg = "".join(
                    traceback.format_exception(type(e), e, e.__traceback__)
                )
                logger.error(
                    f"run_graph_iter: ❌ MCP server '{srv_id}' FAILED: {err_msg}"
                )
                failed_servers.append((srv_id, err_msg))

        deps.mcp_toolsets = [ts for ts in connected_toolsets if ts is not None]

        if failed_servers:
            logger.warning(
                f"run_graph_iter: {len(failed_servers)} MCP server(s) failed:\n"
                + "\n".join(f"  ❌ {sid}: {err}" for sid, err in failed_servers)
            )

        step_count = 0
        try:
            async with graph.iter(state=state, deps=deps) as graph_run:
                async for event in graph_run:
                    if isinstance(event, EndMarker):
                        # Graph completed — yield final result
                        yield {
                            "type": "graph_complete",
                            "run_id": run_id,
                            "output": event.value,
                            "state_snapshot": _build_state_snapshot(state),
                        }
                        break

                    # event is Sequence[GraphTask] — list of active tasks
                    step_count += 1
                    active_nodes = [
                        {"node_id": str(t.node_id), "task_id": str(t.task_id)}
                        for t in event
                    ]

                    yield {
                        "type": "node_transition",
                        "step": step_count,
                        "run_id": run_id,
                        "active_nodes": active_nodes,
                        "state_snapshot": _build_state_snapshot(state),
                    }

                    # Drain sideband events emitted by the step
                    while not eq.empty():
                        sideband_event = eq.get_nowait()
                        yield {"type": "sideband", "event": sideband_event}

                    # Elicitation check: pause for human approval if needed
                    if (
                        state.human_approval_required
                        and elicitation_callback is not None
                    ):
                        yield {
                            "type": "elicitation",
                            "reason": "human_approval_required",
                            "state_snapshot": _build_state_snapshot(state),
                        }
                        redirect = await elicitation_callback(state)
                        state.human_approval_required = False
                        if redirect:
                            logger.info(
                                f"run_graph_iter: Elicitation redirect to '{redirect}'"
                            )
                            # The redirect will be picked up by the next
                            # dispatcher iteration via state.user_redirect_feedback
                            state.user_redirect_feedback = redirect

        except TimeoutError:
            yield {
                "type": "error",
                "run_id": run_id,
                "error": "Graph execution timed out",
            }
        except Exception as e:
            logger.error(f"run_graph_iter: CRITICAL ERROR: {e}")
            yield {"type": "error", "run_id": run_id, "error": str(e)}

        # Drain any remaining sideband events
        while not eq.empty():
            sideband_event = eq.get_nowait()
            yield {"type": "sideband", "event": sideband_event}


def _build_state_snapshot(state: GraphState) -> dict[str, Any]:
    """Build a lightweight serializable snapshot of the current graph state.

    This snapshot is included in every yielded event for observability,
    audit trails, and potential pause/resume support.

    Args:
        state: The current GraphState instance.

    Returns:
        A dictionary with key state fields.

    """
    return {
        "routed_domain": state.routed_domain,
        "step_cursor": state.step_cursor,
        "mode": state.mode,
        "topology": state.topology,
        "node_history": list(state.node_history),
        "node_transitions": state.node_transitions,
        "error": state.error,
        "results_registry_keys": list(state.results_registry.keys()),
        "session_id": state.session_id,
    }


def validate_graph(graph: Any, config: dict) -> dict:
    """Validate the graph topology and report on system readiness.

    Performs a structural and configuration audit to ensure all specialist
    domains, MCP servers, and A2A agents are correctly registered and
    reachable within the orchestration environment.

    Args:
        graph: The Graph object to validate.
        config: Execution configuration dictionary containing registry info.

    Returns:
        A dictionary containing validation metrics: domain count, MCP agent
        availability, discovered A2A agents, edge counts, and a list of
        any critical warnings or errors.

    """
    warnings: list[str] = []
    info: dict[str, Any] = {}

    # Extract tag_prompts (domain adaptive_agent_router)
    tag_prompts = config.get("tag_prompts", {})
    info["domain_count"] = len(tag_prompts)
    info["domain_tags"] = list(tag_prompts.keys())

    # MCP toolsets
    mcp_toolsets = config.get("mcp_toolsets", [])
    info["mcp_toolset_count"] = len(mcp_toolsets)

    # MCP agents from registry
    registry = load_node_agents_registry()
    info["mcp_agent_count"] = len(registry.agents)
    info["mcp_agents"] = [
        {
            "name": a.name,
            "agent_type": a.agent_type,
            "mcp_server": a.mcp_server,
            "tool_count": len(a.tools),
        }
        for a in registry.agents
    ]
    info["mcp_tool_count"] = len(registry.tools)

    # Discovered agents (A2A)
    from agent_utilities.agent.discovery import discover_agents

    discovered = discover_agents()
    info["discovered_agent_count"] = len(discovered)
    info["discovered_agents"] = list(discovered.keys())

    # Graph structure
    if hasattr(graph, "mermaid_code"):
        mermaid = graph.mermaid_code()
        # Count node definitions (rough heuristic)
        [line for line in mermaid.split("\n") if "-->" in line or ":" in line]
        info["graph_edge_count"] = len(
            [line for line in mermaid.split("\n") if "-->" in line]
        )
    else:
        info["graph_edge_count"] = "unknown"

    # Warnings
    if not tag_prompts:
        warnings.append(
            "No domain tags discovered. Graph will have no specialist routing."
        )
    if not mcp_toolsets:
        warnings.append(
            "No MCP toolsets loaded. Specialist agents will have no MCP tools."
        )
    if not registry.agents and not mcp_toolsets:
        warnings.append(
            f"{len(registry.agents)} MCP agents registered but no MCP toolsets loaded."
        )

    info["warnings"] = warnings
    info["valid"] = len(warnings) == 0

    logger.info(
        f"Graph Validation: {info['domain_count']} domains, "
        f"{info['mcp_agent_count']} MCP agents, "
        f"{info['mcp_toolset_count']} MCP toolsets, "
        f"{info['discovered_agent_count']} discovered agents, "
        f"{len(warnings)} warnings"
    )
    return info
