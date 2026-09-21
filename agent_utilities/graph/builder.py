#!/usr/bin/python
from __future__ import annotations

"""Graph Builder Module.

This module provides the factory and registration logic for constructing
pydantic-graph instances. It handles domain discovery, specialist node
registration, and the definition of the graph's dynamic routing topology.
"""


import logging
from typing import Any, Literal, cast

from pydantic_graph import End
from pydantic_graph.graph_builder import Graph, GraphBuilder
from pydantic_graph.step import StepContext

from agent_utilities.agent.discovery import discover_agents, discover_all_specialists
from agent_utilities.agent.registry_builder import ingest_prompts_to_graph
from agent_utilities.core.config import (
    DEFAULT_KNOWLEDGE_GRAPH_SYNC_BACKGROUND,
    DEFAULT_LITE_LLM_MODEL_ID,
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_MCP_CONFIG,
    DEFAULT_MCP_URL,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_ROUTER_MODEL,
    DEFAULT_ROUTING_STRATEGY,
    DEFAULT_VALIDATION_MODE,
    get_discovery_registry,
    load_mcp_servers_from_config,
)
from agent_utilities.core.workspace import get_agent_workspace, resolve_mcp_config_path
from agent_utilities.mcp.agent_manager import should_sync, sync_mcp_agents

from ..base_utilities import (
    is_loopback_url,
)
from ..knowledge_graph.core.engine import IntelligenceGraphEngine
from ..knowledge_graph.pipeline import IntelligencePipeline
from ..models import GraphResponse
from ..models.knowledge_graph import PipelineConfig
from .executor import (
    agent_package_step,
)
from .hierarchical_planner import (
    architect_step,
    memory_selection_step,
    planner_step,
    researcher_step,
)
from .lifecycle import approval_gate_step, onboarding_step, usage_guard_step
from .nodes import (
    load_and_execute_process_flow,
)
from .routing import (
    dispatcher_step,
    dynamic_mcp_routing_step,
    expert_executor_step,
    mcp_server_step,
    parallel_batch_processor,
    router_step,
)
from .state import GraphDeps, GraphState
from .verification import (
    error_recovery_step,
    join_step,
    synthesizer_step,
    verifier_step,
    wide_search_joiner_step,
)

logger = logging.getLogger(__name__)


class _BuiltGraphCache:
    """Process-local bounded LRU of structural graph builds (CONCEPT:AU-ORCH.routing.structural-build-reuse).

    ``create_graph_agent`` rebuilt the entire topology + ``discover_agents()`` on EVERY
    turn. The topology is a pure function of (tag_prompts, models, routing strategy, sub
    agents), so we memoize the structural build keyed by a hash of that
    config and reuse a warm graph. Toolset connections stay per-run (built outside the
    cache). Small cap — distinct routing configs are few — and thread-safe for concurrent
    gateway turns.
    """

    def __init__(self, max_entries: int = 64) -> None:
        import threading
        from collections import OrderedDict

        self._max = max_entries
        self._lock = threading.Lock()
        self._store: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is not None:
                self._store.move_to_end(key)
            return entry

    def put(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._store[key] = value
            self._store.move_to_end(key)
            while len(self._store) > self._max:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


_GRAPH_CACHE = _BuiltGraphCache()


def _graph_cache_key(
    *,
    name: str,
    tag_prompts: dict[str, str],
    router_model: str | None,
    agent_model: str | None,
    routing_strategy: str,
    sub_agents: dict[str, Any] | None,
) -> str:
    """A stable hash of the STRUCTURAL graph inputs (CONCEPT:AU-ORCH.routing.structural-build-reuse).

    Keys on what changes the built graph's identity: the graph ``name`` (which the
    builder stamps onto the returned graph, so two same-topology graphs with
    different names are distinct objects), the set of routing tags, the models, the
        routing strategy and the sub-agent tags. A
    change in discovery (new agent registered) changes the tag/sub-agent set and so
    invalidates the key naturally. Excludes per-run values (toolsets, timeouts, api
    keys, the query).
    """
    import hashlib

    parts = [
        str(name),
        "|".join(sorted(tag_prompts.keys())),
        str(router_model),
        str(agent_model),
        str(routing_strategy),
        "|".join(sorted((sub_agents or {}).keys())),
    ]
    return hashlib.sha256("\x1e".join(parts).encode("utf-8")).hexdigest()


def build_tag_env_map(tag_names: list[str]) -> dict[str, str]:
    """Build a tag→env_var mapping following the standard convention.

    The standard convention maps a domain tag (e.g., "incidents") to an
    environment variable (e.g., "INCIDENTSTOOL") used to gate access to
    that specific specialist domain.

    Args:
        tag_names: List of domain tag names to map.

    Returns:
        A dictionary mapping lowercase tag names to upper-cased environment
        variable names with the 'TOOL' suffix.

    """
    result = {}
    for tag in tag_names:
        env_var = tag.upper().replace("-", "_") + "TOOL"
        result[tag] = env_var
    return result


def _get_running_loop() -> Any:
    """Return the active event loop, if initialization runs asynchronously."""
    import asyncio

    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def _run_or_schedule(
    operation: Any,
    *,
    loop: Any,
    label: str,
    **kwargs: Any,
) -> bool:
    """Run a sync operation now or schedule it without blocking a live loop.

    Returns ``True`` only when the operation completed synchronously.  A caller
    can therefore distinguish a completed startup sync from a scheduled or
    deliberately backgrounded one when deciding which status to log.
    """
    if loop is not None and loop.is_running():
        loop.create_task(operation(**kwargs))
        return False
    if DEFAULT_KNOWLEDGE_GRAPH_SYNC_BACKGROUND:
        logger.info("Backgrounding %s...", label)
        return False
    import asyncio

    asyncio.run(operation(**kwargs))
    return True


def _ingest_prompts_if_enabled(loop: Any) -> None:
    """Start prompt ingestion when a configured MCP workspace is available."""
    if DEFAULT_VALIDATION_MODE:
        return
    try:
        _run_or_schedule(
            ingest_prompts_to_graph,
            loop=loop,
            label="prompt ingestion",
        )
    except Exception as exc:  # noqa: BLE001 — ingestion owns its durable retry checkpoint
        logger.debug("Registry rebuild failed: %s", exc)


def _sync_mcp_agents_if_needed(config_path: Any, loop: Any) -> None:
    """Synchronize MCP agents when the registry is stale."""
    needs_sync = should_sync(config_path)
    if not needs_sync or DEFAULT_VALIDATION_MODE:
        logger.debug(
            "Initializing Graph: Valid registry found. Skipping live extraction."
        )
        return
    try:
        synced = _run_or_schedule(
            sync_mcp_agents,
            loop=loop,
            label="MCP agent sync",
            config_path=config_path,
        )
        if synced:
            logger.info("Initializing Graph: MCP agents synced successfully.")
    except Exception as exc:  # noqa: BLE001 — a failed sync is retried from its durable timestamp
        logger.debug("Sync skip/fail: %s", exc)


def _build_discovery_metadata(config_path: Any, loop: Any) -> dict[str, Any]:
    """Synchronize configured MCP agents and build their registry metadata."""
    _ingest_prompts_if_enabled(loop)
    try:
        _sync_mcp_agents_if_needed(config_path, loop)

        from collections import defaultdict

        registry = get_discovery_registry()
        tools_by_server: dict[str | None, list[Any]] = defaultdict(list)
        for agent in registry.agents:
            for tool in agent.tools:
                tools_by_server[agent.mcp_server].append(tool)
        # Empty agents never create buckets; remove the unbound sentinel.
        tools_by_server.pop(None, None)

        discovery_metadata = cast(dict[str, Any], dict(tools_by_server))
        logger.info(
            "Initializing Graph: Verified %s servers from registry.",
            len(discovery_metadata),
        )
        return discovery_metadata
    except Exception as exc:  # noqa: BLE001 — registry discovery falls back to an empty metadata map
        logger.warning("Failed to load MCP discovery metadata: %s", exc)
        return {}


def _sync_a2a_agents_if_enabled(config_path: Any, loop: Any) -> None:
    """Synchronize A2A agents without blocking an active event loop."""
    if not config_path or DEFAULT_VALIDATION_MODE:
        return
    try:
        from agent_utilities.protocols.a2a_config import sync_a2a_agents

        synced = _run_or_schedule(
            sync_a2a_agents,
            loop=loop,
            label="A2A agent sync",
            config_path=config_path,
        )
        if synced:
            logger.info("Initializing Graph: A2A agents synced successfully.")
    except Exception as exc:  # noqa: BLE001 — discovery retries A2A sync on the next initialization
        logger.debug("A2A agent sync skip/fail: %s", exc)


def _discover_tag_prompts() -> dict[str, str]:
    """Return specialist prompts, using a deterministic validation roster."""
    if DEFAULT_VALIDATION_MODE:
        return {"validation": "dummy"}

    all_specialists = discover_all_specialists()
    tag_prompts = {
        specialist.tag: specialist.description for specialist in all_specialists
    }
    if not tag_prompts:
        raise RuntimeError("no specialist metadata is available")
    return tag_prompts


def initialize_graph_from_workspace(
    mcp_config: str | None = "mcp_config.json",
    a2a_config: str | None = None,
    router_model: str | None = None,
    agent_model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    workspace: str | None = None,
    custom_headers: dict[str, Any] | None = None,
    router_timeout: float | None = None,
    verifier_timeout: float | None = None,
) -> tuple[Graph, dict]:
    """Initialize a graph bundle by discovering domains in the current workspace.

    This utility handles MCP agent synchronization, A2A agent discovery,
    domain tag discovery, and graph construction.

    Args:
        mcp_config: Filename or path to the MCP configuration file.
        a2a_config: Filename or path to the A2A agent configuration file (CONCEPT:AU-ECO.interop.a2a-agent-sync).
        router_model: Optional override for the router's LLM model ID.
        agent_model: Optional override for the specialist agents' LLM model ID.
        api_key: Optional API key for the LLM provider.
        base_url: Optional override for the LLM base URL.
        workspace: Optional explicit path to the agent workspace.
        custom_headers: Optional HTTP headers for provider requests.
        router_timeout: Per-request timeout for the router node.
        verifier_timeout: Per-request timeout for the verifier node.

    Returns:
        A tuple containing the initialized pydantic-graph Graph instance and
        its configuration dictionary (GraphDeps source).

    """
    logger.info("Initializing graph from configured workspace")

    if workspace:
        from ..core import workspace as _ws_mod

        _ws_mod.WORKSPACE_DIR = workspace
        logger.info("Initializing graph with pinned workspace")

    _mcp_cfg_path = resolve_mcp_config_path(mcp_config) if mcp_config else None
    discovery_metadata = {}
    loop = _get_running_loop() if _mcp_cfg_path else None
    if _mcp_cfg_path:
        discovery_metadata = _build_discovery_metadata(_mcp_cfg_path, loop)

    # --- CONCEPT:AU-ECO.interop.a2a-agent-sync: A2A Agent Sync ---
    from agent_utilities.core.config import config as app_config

    _a2a_config = a2a_config or app_config.a2a_config
    _sync_a2a_agents_if_enabled(_a2a_config, loop)

    # Unified Discovery: merge MCP, A2A, and prompt sources into a single roster
    logger.info("Initializing Graph: Discovering domain tags and agents...")

    tag_prompts = _discover_tag_prompts()

    logger.info(f"Initializing Graph: Discovered {len(tag_prompts)} domain tags.")

    tag_env_vars = build_tag_env_map(list(tag_prompts.keys()))

    # Initialize Graph & Deps
    logger.info("Initializing Graph: Building graph topology...")
    graph, config = create_graph_agent(
        tag_prompts=tag_prompts,
        tag_env_vars=tag_env_vars,
        mcp_url=DEFAULT_MCP_URL,
        mcp_config=mcp_config,
        router_timeout=router_timeout,
        verifier_timeout=verifier_timeout,
        discovery_metadata=discovery_metadata,
        router_model=router_model or DEFAULT_ROUTER_MODEL,
        agent_model=agent_model or DEFAULT_LITE_LLM_MODEL_ID,
        api_key=api_key,
        base_url=base_url,
        workspace=workspace,
        custom_headers=custom_headers,
    )
    logger.info("Initializing Graph: Topology built successfully.")

    return graph, config


def create_master_graph(
    name: str = "MasterGraph",
    include_agents: list[str] | None = None,
    exclude_agents: list[str] | None = None,
    skill_agents: dict[str, dict] | None = None,
    **kwargs,
) -> tuple[Graph, dict]:
    """Factory to create a master orchestrator graph for sub-agent routing.

    Discovers available agent packages via the A2A protocol and registers
    them as routable specialist nodes within a unified graph topology.
    This is used for higher-level orchestration across separate agent services.

    Args:
        name: Human-readable name of the master graph.
        include_agents: Optional list of specific agent packages to discover.
        exclude_agents: Optional list of agent packages to ignore.
        skill_agents: Dict of specialized skill definitions to inject.
        **kwargs: Additional configuration parameters for the graph.

    Returns:
        A tuple containing the initialized Graph and its configuration dictionary.

    """

    agents = discover_agents(
        include_packages=include_agents, exclude_packages=exclude_agents
    )

    tag_prompts = {
        name: f"Specialized agent for {meta.get('package', name)}"
        for name, meta in agents.items()
    }

    _skill_agents = skill_agents or {}
    for tag, agent_cfg in _skill_agents.items():
        if tag not in tag_prompts:
            tag_prompts[tag] = agent_cfg.get(
                "description", f"Specialized skill agent for {tag}"
            )

    sub_agents: dict[str, Any] = {
        name: {
            "description": tag_prompts[name],
            "tags": list(package_meta.get("skills") or [name]),
            "package": str(package_meta.get("package", name)),
        }
        for name, package_meta in agents.items()
    }
    for tag, agent_cfg in _skill_agents.items():
        if tag not in sub_agents:
            sub_agents[tag] = {
                "description": tag_prompts[tag],
                "tags": list(agent_cfg.get("tags") or [tag]),
            }

    return create_graph_agent(
        tag_prompts=tag_prompts,
        name=name,
        sub_agents=sub_agents,
        **kwargs,
    )


def _initialize_registry_engine() -> Any:
    """Initialize the optional engine-backed registry graph."""
    knowledge_engine = None
    try:
        if not all([IntelligenceGraphEngine, PipelineConfig, IntelligencePipeline]):
            raise ImportError("Registry Graph dependencies missing")

        if DEFAULT_VALIDATION_MODE:
            logger.info("Registry Graph: Skipping initialization in VALIDATION_MODE.")
        else:
            from agent_utilities.knowledge_graph.backends.base import (
                require_engine_authority_backend,
            )

            ws = get_agent_workspace()
            # Engine-only (CONCEPT:AU-KG.compute.graph-builder): the registry graph persists as
            # nodes/edges ON THE ONE epistemic-graph engine authority — never a
            # local ladybug ``registry_graph.db`` beside it. Resolve the engine
            # backend (the OS-5.63 resolver auto-starts the mandatory full engine
            # artifact in prod; the KG-2.238 fixture provides a real ephemeral one
            # in tests), raising
            # a clear error if the engine is genuinely unreachable.
            active_backend = require_engine_authority_backend(
                "agent registry graph (CONCEPT:AU-KG.compute.graph-builder)"
            )
            reg_config = PipelineConfig(
                workspace_path=str(ws),
                persist_to_ladybug=False,
            )
            reg_pipeline = IntelligencePipeline(reg_config, backend=active_backend)
            logger.debug(
                "Registry Graph: engine-backed via %s",
                type(active_backend).__name__,
            )

            # We run the pipeline synchronously here during initialization
            import asyncio

            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    # We are in a running loop (e.g. during a request).
                    # We can't block. We'll skip sync and hope the DB is ready.
                    logger.debug(
                        "Registry Graph: Skipping blocking sync in running loop."
                    )
            except RuntimeError:
                # No running loop, safe to run blocking
                try:
                    logger.info("Running IntelligencePipeline sync...")
                    asyncio.run(reg_pipeline.run())
                    knowledge_engine = IntelligenceGraphEngine.get_or_create(
                        backend=active_backend
                    )
                except Exception as e:  # noqa: BLE001 — knowledge_engine defaults to None (line ~503) and every consumer graph-wide already gates on `if deps.knowledge_engine:`; this is the supported "KG disabled" path, not a false-success state
                    logger.debug(f"Knowledge engine initialization failed: {e}")
    except ImportError:
        logger.debug("Registry Graph subpackage not found or dependencies missing.")
    return knowledge_engine


def _make_agent_step(tag: str) -> Any:
    """Build the specialist step wrapper for a discovered agent tag."""

    async def agent_specific_step(ctx: StepContext) -> str | End[Any]:
        return await agent_package_step(ctx, node_id=tag)

    agent_specific_step.__name__ = f"agent_{tag}_step"
    return agent_specific_step


def _register_graph_steps(
    graph_builder: GraphBuilder,
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    """Register static and discovered specialist nodes on a graph builder."""
    router = graph_builder.step(router_step, node_id="router")
    planner = graph_builder.step(planner_step, node_id="planner")
    onboarding = graph_builder.step(onboarding_step, node_id="onboarding")
    error = graph_builder.step(error_recovery_step, node_id="error_recovery")
    process_executor = graph_builder.step(
        load_and_execute_process_flow, node_id="process_executor"
    )
    dispatcher = graph_builder.step(dispatcher_step, node_id="dispatcher")
    parallel_batch_step = graph_builder.step(
        parallel_batch_processor, node_id="parallel_batch_processor"
    )
    expert_executor = graph_builder.step(
        expert_executor_step, node_id="expert_executor"
    )
    research_joiner = graph_builder.step(join_step, node_id="research_joiner")
    execution_joiner = graph_builder.step(join_step, node_id="execution_joiner")
    wide_search_joiner = graph_builder.step(
        wide_search_joiner_step, node_id="wide_search_joiner"
    )
    architect = graph_builder.step(architect_step, node_id="architect")
    verifier = graph_builder.step(verifier_step, node_id="verifier")
    synthesizer = graph_builder.step(synthesizer_step, node_id="synthesizer")
    researcher = graph_builder.step(researcher_step, node_id="researcher")

    dedicated_nodes = {
        "researcher",
        "architect",
        "planner",
        "verifier",
        "python_programmer",
        "c_programmer",
        "cpp_programmer",
        "golang_programmer",
        "javascript_programmer",
        "typescript_programmer",
        "security_auditor",
        "qa_expert",
        "debugger_expert",
        "ui_ux_designer",
        "devops_engineer",
        "cloud_architect",
        "database_expert",
        "rust_programmer",
        "java_programmer",
        "data_scientist",
        "document_specialist",
        "mobile_programmer",
        "agent_engineer",
        "project_manager",
        "systems_manager",
        "browser_automation",
        "coordinator",
        "critique",
    }
    memory_selection = graph_builder.step(
        memory_selection_step, node_id="memory_selection"
    )
    mcp_router = graph_builder.step(dynamic_mcp_routing_step, node_id="mcp_router")
    mcp_server = graph_builder.step(mcp_server_step, node_id="mcp_server_execution")

    # Error and Onboarding
    error = graph_builder.step(error_recovery_step, node_id="error_recovery")
    onboarding = graph_builder.step(onboarding_step, node_id="onboarding")

    _approval = graph_builder.step(approval_gate_step, node_id="approval_gate")
    usage_guard = graph_builder.step(usage_guard_step, node_id="usage_guard")

    specialist_node_configs = {
        tag: _make_agent_step(tag)
        for tag in discover_agents()
        if tag not in dedicated_nodes and tag not in {"onboarding", "error_recovery"}
    }
    expert_nodes = {
        node_id: graph_builder.step(step_func, node_id=node_id)
        for node_id, step_func in specialist_node_configs.items()
    }
    for node_id in expert_nodes:
        logger.debug("Registered graph specialist node: %s", node_id)

    nodes_registry = {
        "router": router,
        "planner": planner,
        "error_recovery": error,
        "onboarding": onboarding,
        "dispatcher": dispatcher,
        "parallel_batch_processor": parallel_batch_step,
        "expert_executor": expert_executor,
        "research_joiner": research_joiner,
        "execution_joiner": execution_joiner,
        "wide_search_joiner": wide_search_joiner,
        "architect": architect,
        "verifier": verifier,
        "synthesizer": synthesizer,
        "researcher": researcher,
        "memory_selection": memory_selection,
        "mcp_router": mcp_router,
        "mcp_server_execution": mcp_server,
        "process_executor": process_executor,
        **expert_nodes,
    }
    return nodes_registry, expert_nodes, usage_guard


def _wire_graph_routes(
    graph_builder: GraphBuilder,
    nodes_registry: dict[str, Any],
    expert_nodes: dict[str, Any],
    usage_guard: Any,
) -> None:
    """Add the explicit dispatcher, joiner, and lifecycle graph routes."""
    dispatcher_route = graph_builder.decision(node_id="dispatcher_route")
    dispatcher_route.branches.append(
        graph_builder.match(Literal["parallel_batch_processor"]).to(
            nodes_registry["parallel_batch_processor"]
        )  # type: ignore[arg-type]
    )
    sequential_routes = [
        "researcher",
        "architect",
        "planner",
        "verifier",
        "synthesizer",
        "wide_search_joiner",
        "mcp_router",
        "error_recovery",
        "onboarding",
        "expert_executor",
        "memory_selection",
        "process_executor",
    ]
    for node_id in sequential_routes:
        dispatcher_route.branches.append(
            graph_builder.match(Literal[node_id]).to(nodes_registry[node_id])  # type: ignore[arg-type]
        )
    for node_id in ("dispatcher", "error", "error_recovery"):
        target = "error_recovery" if node_id != "dispatcher" else "dispatcher"
        dispatcher_route.branches.append(
            graph_builder.match(Literal[node_id]).to(nodes_registry[target])  # type: ignore[arg-type]
        )
    for node_id, node in expert_nodes.items():
        dispatcher_route.branches.append(
            graph_builder.match(Literal[node_id]).to(node)  # type: ignore[arg-type]
        )
    dispatcher_route.branches.append(
        graph_builder.match(type(None)).to(graph_builder.end_node)
    )

    research_joiner_route = graph_builder.decision(node_id="research_joiner_route")
    research_joiner_route.branches.append(
        graph_builder.match(Literal["dispatcher"]).to(nodes_registry["dispatcher"])  # type: ignore[arg-type]
    )
    research_joiner_route.branches.append(
        graph_builder.match(type(None)).to(graph_builder.end_node)
    )

    execution_joiner_route = graph_builder.decision(node_id="execution_joiner_route")
    execution_joiner_route.branches.append(
        graph_builder.match(Literal["dispatcher"]).to(nodes_registry["dispatcher"])  # type: ignore[arg-type]
    )
    execution_joiner_route.branches.append(
        graph_builder.match(Literal["verifier"]).to(nodes_registry["verifier"])  # type: ignore[arg-type]
    )
    execution_joiner_route.branches.append(
        graph_builder.match(type(None)).to(graph_builder.end_node)
    )

    memory_selection_route = graph_builder.decision(node_id="memory_selection_route")
    memory_selection_route.branches.append(
        graph_builder.match(Literal["dispatcher"]).to(nodes_registry["dispatcher"])  # type: ignore[arg-type]
    )
    memory_selection_route.branches.append(
        graph_builder.match(Literal["researcher"]).to(nodes_registry["researcher"])  # type: ignore[arg-type]
    )

    verifier_route = graph_builder.decision(node_id="verifier_route")
    for node_id in ("synthesizer", "dispatcher", "planner"):
        verifier_route.branches.append(
            graph_builder.match(Literal[node_id]).to(nodes_registry[node_id])  # type: ignore[arg-type]
        )

    # The second execution-joiner decision is the effective route; retain its
    # existing node id and branch order for pydantic-graph compatibility.
    execution_joiner_route = graph_builder.decision(node_id="execution_joiner_route")
    for node_id in ("dispatcher", "router_step", "router", "wide_search_joiner"):
        target_id = "router" if node_id == "router_step" else node_id
        execution_joiner_route.branches.append(
            graph_builder.match(Literal[node_id]).to(nodes_registry[target_id])  # type: ignore[arg-type]
        )
    execution_joiner_route.branches.append(
        graph_builder.match(type(None)).to(graph_builder.end_node)
    )

    graph_builder.add(
        graph_builder.edge_from(graph_builder.start_node)
        .label("Query")
        .to(usage_guard),
        graph_builder.edge_from(usage_guard)
        .label("Policy OK")
        .to(nodes_registry["router"]),
        graph_builder.edge_from(nodes_registry["router"])
        .label("Plan")
        .to(nodes_registry["dispatcher"]),
        # CONCEPT:AU-ORCH.routing.single-router-edge — the router has a SINGLE outgoing edge (→ dispatcher). It must NOT
        # have a second edge to the end node: pydantic-graph turns two edges from one node into
        # a BROADCAST FORK (router → {end, dispatcher}), which terminated every full-graph turn
        # via the end branch. A direct-completion turn never reaches the router — it is answered
        # outside the graph by ``_run_direct_completion`` (agent_runner) — so the router never
        # needs to end the run itself.
        graph_builder.edge_from(nodes_registry["dispatcher"]).to(dispatcher_route),
        graph_builder.edge_from(nodes_registry["planner"]).to(
            nodes_registry["dispatcher"]
        ),
        graph_builder.edge_from(nodes_registry["process_executor"]).to(
            nodes_registry["dispatcher"]
        ),
        graph_builder.edge_from(nodes_registry["memory_selection"]).to(
            memory_selection_route
        ),
        graph_builder.edge_from(nodes_registry["parallel_batch_processor"])
        .map()
        .to(nodes_registry["expert_executor"]),
        graph_builder.edge_from(nodes_registry["researcher"])
        .label("Research Done")
        .to(nodes_registry["research_joiner"]),
        graph_builder.edge_from(nodes_registry["architect"])
        .label("Design Done")
        .to(nodes_registry["research_joiner"]),
        *(
            graph_builder.edge_from(node).to(nodes_registry["execution_joiner"])
            for node in expert_nodes.values()
        ),
        graph_builder.edge_from(nodes_registry["expert_executor"]).to(
            nodes_registry["execution_joiner"]
        ),
        graph_builder.edge_from(nodes_registry["mcp_router"])
        .map()
        .to(nodes_registry["mcp_server_execution"]),
        graph_builder.edge_from(nodes_registry["mcp_server_execution"]).to(
            nodes_registry["execution_joiner"]
        ),
        graph_builder.edge_from(nodes_registry["research_joiner"]).to(
            research_joiner_route
        ),
        graph_builder.edge_from(nodes_registry["execution_joiner"]).to(
            execution_joiner_route
        ),
        graph_builder.edge_from(nodes_registry["wide_search_joiner"]).to(
            dispatcher_route
        ),
        graph_builder.edge_from(nodes_registry["error_recovery"]).to(
            nodes_registry["planner"]
        ),
        graph_builder.edge_from(nodes_registry["verifier"]).to(verifier_route),
        graph_builder.edge_from(nodes_registry["synthesizer"]).to(
            graph_builder.end_node
        ),
        graph_builder.edge_from(nodes_registry["onboarding"]).to(
            graph_builder.end_node
        ),
    )


def _build_mcp_toolsets(
    mcp_toolsets: list[Any] | None,
    mcp_url: str | None,
    mcp_config: str | None,
    kwargs: dict[str, Any],
) -> list[Any]:
    """Build per-run MCP toolsets without caching connection objects."""
    toolsets = list(mcp_toolsets) if mcp_toolsets else []
    if DEFAULT_VALIDATION_MODE:
        return toolsets
    if mcp_url:
        from agent_utilities.mcp.toolset_factory import build_http_toolset

        if not is_loopback_url(
            mcp_url, kwargs.get("current_host"), kwargs.get("current_port")
        ):
            toolsets.append(build_http_toolset(mcp_url, timeout=60))
    if mcp_config:
        config_path = resolve_mcp_config_path(mcp_config)
        if config_path:
            # Load MCP servers individually so that a single undefined env-var
            # does not prevent the rest of the toolsets from loading.
            # The canonical config loader validates commands, expands
            # environment references, and constructs each MCP toolset.
            toolsets = load_mcp_servers_from_config(config_path)
            for toolset in toolsets:
                server_id = getattr(toolset, "id", getattr(toolset, "name", "unknown"))
                logger.info("MCP Startup: Registered server '%s'", server_id)
        else:
            logger.warning("MCP config %s not found", mcp_config)
    return toolsets


def create_graph_agent(
    tag_prompts: dict[str, str],
    tag_env_vars: dict[str, str] | None = None,
    mcp_url: str | None = DEFAULT_MCP_URL,
    mcp_config: str | None = DEFAULT_MCP_CONFIG,
    name: str = "GraphAgent",
    router_model: str | None = DEFAULT_ROUTER_MODEL,
    agent_model: str | None = DEFAULT_LITE_LLM_MODEL_ID,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    sub_agents: dict[str, str | Any] | None = None,
    mcp_toolsets: list[Any] | None = None,
    routing_strategy: str = DEFAULT_ROUTING_STRATEGY,
    router_timeout: float | None = None,
    verifier_timeout: float | None = None,
    workspace: str | None = None,
    **kwargs,
) -> tuple[Graph, dict]:
    """Factory to create a router-led graph assistant using pydantic-graph.

    This function defines the end-to-end graph topology, including
    onboarding, policy guarding, multi-step routing, parallel execution
    batches, and final output verification.

    Args:
        tag_prompts: Mapping of domain tags to specialist persona descriptions.
        tag_env_vars: Mapping of domain tags to environment variable names for gating.
        mcp_url: Base URL for a standalone MCP server.
        mcp_config: Path to the mcp_config.json for tool discovery.
        name: Internal name for the graph instance.
        router_model: LLM model ID used by the routing and planning nodes.
        agent_model: Default LLM model ID used by specialist expert nodes.
        min_confidence: Confidence score threshold for valid routing.
        sub_agents: Mapping of tags to agent packages or pre-built Agent instances.
        mcp_toolsets: List of pre-initialized toolsets to inject into the graph.
        routing_strategy: The logic used for routing ('hybrid', 'llm', 'rules').
        router_timeout: Per-node timeout for the router (seconds).
        verifier_timeout: Per-node timeout for the verifier (seconds).
        workspace: Path to the persistent storage directory.
        **kwargs: Additional low-level configuration overrides.

    Returns:
        A tuple containing the configured Graph and its execution dictionary.

    """
    if tag_env_vars is None:
        tag_env_vars = build_tag_env_map(list(tag_prompts.keys()))

    # CONCEPT:AU-ORCH.routing.structural-build-reuse — cache the built graph TOPOLOGY per routing-config. The topology +
    # ``discover_agents()`` are a pure function of (tag_prompts, models, routing strategy,
    # sub-agents), rebuilt on EVERY turn before this. We memoize the structural
    # build keyed by a hash of that config, so a turn reuses a warm graph. Toolset *connections*
    # (mcp_url/mcp_config/mcp_toolsets) stay per-run and are built fresh below — so we only
    # serve the cache when the structure is toolset-free (the messaging chat default). When a
    # run binds toolsets, we build fresh (correctness over the micro-optimisation).
    _cache_key = _graph_cache_key(
        name=name,
        tag_prompts=tag_prompts,
        router_model=router_model,
        agent_model=agent_model,
        routing_strategy=routing_strategy,
        sub_agents=sub_agents,
    )
    # Cache only when there are no per-run toolset connections.
    _cacheable = not mcp_url and not mcp_config and not mcp_toolsets
    _toolset_free = _cacheable
    _cached = _GRAPH_CACHE.get(_cache_key) if _cacheable else None
    if _cached is not None:
        # Warm graph hit (CONCEPT:AU-ORCH.routing.structural-build-reuse): reuse the structural topology + node registry
        # + registry engine; build only the cheap per-run config. No toolsets on this path
        # (toolset-free by the cache guard above), so connections never get reused.
        logger.debug(
            "create_graph_agent: reusing cached graph topology (key=%s)", _cache_key
        )
        return _cached["graph"], _build_graph_config(
            graph_nodes=_cached["nodes_registry"],
            knowledge_engine=_cached["knowledge_engine"],
            agent_subject=name,
            mcp_toolsets=[],
            tag_prompts=tag_prompts,
            tag_env_vars=tag_env_vars,
            mcp_url=mcp_url,
            mcp_config=mcp_config,
            router_model=router_model,
            agent_model=agent_model,
            router_timeout=router_timeout,
            verifier_timeout=verifier_timeout,
            min_confidence=min_confidence,
            sub_agents=sub_agents,
            routing_strategy=routing_strategy,
            kwargs=kwargs,
        )

    knowledge_engine = _initialize_registry_engine()

    # Initialize GraphBuilder

    g = GraphBuilder(
        name=name,
        state_type=GraphState,
        deps_type=GraphDeps,
        output_type=GraphResponse,
    )
    nodes_registry, expert_nodes, usage_guard = _register_graph_steps(g)
    _wire_graph_routes(g, nodes_registry, expert_nodes, usage_guard)
    graph = g.build()

    _mcp_toolsets = _build_mcp_toolsets(
        mcp_toolsets,
        mcp_url,
        mcp_config,
        kwargs,
    )
    config = _build_graph_config(
        graph_nodes=nodes_registry,
        knowledge_engine=knowledge_engine,
        agent_subject=name,
        mcp_toolsets=_mcp_toolsets,
        tag_prompts=tag_prompts,
        tag_env_vars=tag_env_vars,
        mcp_url=mcp_url,
        mcp_config=mcp_config,
        router_model=router_model,
        agent_model=agent_model,
        router_timeout=router_timeout,
        verifier_timeout=verifier_timeout,
        min_confidence=min_confidence,
        sub_agents=sub_agents,
        routing_strategy=routing_strategy,
        kwargs=kwargs,
    )

    logger.debug(
        "create_graph_agent: returning config with mcp_toolsets of len: "
        f"{len(config['mcp_toolsets'])}"
    )

    # CONCEPT:AU-ORCH.routing.structural-build-reuse — store the toolset-free structural build for reuse next turn.
    if _toolset_free:
        _GRAPH_CACHE.put(
            _cache_key,
            {
                "graph": graph,
                "nodes_registry": nodes_registry,
                "knowledge_engine": knowledge_engine,
            },
        )

    return graph, config


def _build_graph_config(
    *,
    graph_nodes: dict[str, Any],
    knowledge_engine: Any,
    agent_subject: str,
    mcp_toolsets: list[Any],
    tag_prompts: dict[str, str],
    tag_env_vars: dict[str, str],
    mcp_url: str | None,
    mcp_config: str | None,
    router_model: str | None,
    agent_model: str | None,
    router_timeout: float | None,
    verifier_timeout: float | None,
    min_confidence: float,
    sub_agents: dict[str, str | Any] | None,
    routing_strategy: str,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Build the per-run execution config dict (CONCEPT:AU-ORCH.routing.structural-build-reuse).

    The config is cheap and per-run (carries the run's toolsets / models / timeouts); only
    the graph TOPOLOGY is cached. Extracted so the cache-hit and cache-miss paths build the
    same config shape from a (possibly cached) ``graph_nodes`` registry and registry engine.
    """
    from agent_utilities.core.config import (
        DEFAULT_GRAPH_ROUTER_TIMEOUT,
        DEFAULT_GRAPH_VERIFIER_TIMEOUT,
    )
    from agent_utilities.core.config import (
        config as agent_config,
    )
    from agent_utilities.security.permissions_kernel import (
        resolve_permission_context,
    )
    from agent_utilities.security.tool_guard import is_identity_governed_toolset

    _api_key = kwargs.get("api_key")
    _base_url = kwargs.get("base_url")
    permission_context = resolve_permission_context(
        agent_config,
        permissions_kernel=kwargs.get("permissions_kernel"),
        agent_identity=kwargs.get("agent_identity"),
        required=any(is_identity_governed_toolset(toolset) for toolset in mcp_toolsets),
        engine=knowledge_engine,
        agent_subject=agent_subject,
        capabilities=kwargs.get("capabilities") or (),
    )
    return {
        "tag_prompts": tag_prompts,
        "tag_env_vars": tag_env_vars,
        "mcp_url": mcp_url,
        "mcp_config": mcp_config,
        "mcp_toolsets": mcp_toolsets,
        "router_model": router_model,
        "agent_model": agent_model,
        "router_timeout": (
            router_timeout
            if router_timeout is not None
            else DEFAULT_GRAPH_ROUTER_TIMEOUT
        ),
        "verifier_timeout": (
            verifier_timeout
            if verifier_timeout is not None
            else DEFAULT_GRAPH_VERIFIER_TIMEOUT
        ),
        "min_confidence": min_confidence,
        "valid_domains": tuple(tag_prompts.keys()),
        "provider": kwargs.get("provider") or DEFAULT_LLM_PROVIDER,
        "base_url": _base_url if _base_url is not None else DEFAULT_LLM_BASE_URL,
        "api_key": _api_key if _api_key is not None else DEFAULT_LLM_API_KEY,
        "custom_headers": kwargs.get("custom_headers"),
        "sub_agents": sub_agents or {},
        "routing_strategy": routing_strategy,
        "nodes": graph_nodes,
        "discovery_metadata": kwargs.get("discovery_metadata") or {},
        "knowledge_engine": knowledge_engine,
        "permissions_kernel": (
            permission_context.kernel if permission_context is not None else None
        ),
        "agent_identity": (
            permission_context.identity if permission_context is not None else None
        ),
    }
