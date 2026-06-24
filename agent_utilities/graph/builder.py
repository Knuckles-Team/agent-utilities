#!/usr/bin/python
from __future__ import annotations

"""Graph Builder Module.

This module provides the factory and registration logic for constructing
pydantic-graph instances. It handles domain discovery, specialist node
registration, and the definition of the graph's dynamic routing topology.
"""


import logging
import os
from typing import Any, Literal

from pydantic_ai import Agent
from pydantic_graph import End

from agent_utilities.core.config import setting

try:
    from pydantic_graph.graph_builder import Graph, GraphBuilder
    from pydantic_graph.step import StepContext
except ImportError:
    from pydantic_graph.beta import Graph, GraphBuilder, StepContext

from agent_utilities.agent.discovery import discover_agents, discover_all_specialists
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
    DEFAULT_SSL_VERIFY,
    DEFAULT_VALIDATION_MODE,
    get_discovery_registry,
)
from agent_utilities.core.workspace import get_agent_workspace, resolve_mcp_config_path
from agent_utilities.mcp.agent_manager import should_sync, sync_mcp_agents
from agent_utilities.mcp_utilities import load_mcp_config

from ..base_utilities import (
    is_loopback_url,
)
from ..models import GraphResponse
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

try:
    from ..knowledge_graph.core.engine import IntelligenceGraphEngine
    from ..knowledge_graph.pipeline import RegistryPipeline
    from ..models.knowledge_graph import PipelineConfig
except ImportError:
    # These might be missing if the extra is not installed, but we want them at top level for patching
    IntelligenceGraphEngine = None  # type: ignore
    PipelineConfig = None  # type: ignore
    RegistryPipeline = None  # type: ignore
from agent_utilities.agent.registry_builder import ingest_prompts_to_graph

_PYDANTIC_GRAPH_AVAILABLE = True

logger = logging.getLogger(__name__)


class _BuiltGraphCache:
    """Process-local bounded LRU of structural graph builds (CONCEPT:ORCH-1.64).

    ``create_graph_agent`` rebuilt the entire topology + ``discover_agents()`` on EVERY
    turn. The topology is a pure function of (tag_prompts, models, routing strategy, sub
    agents, custom nodes), so we memoize the structural build keyed by a hash of that
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
    custom_nodes: list[Any] | None,
) -> str:
    """A stable hash of the STRUCTURAL graph inputs (CONCEPT:ORCH-1.64).

    Keys on what changes the built graph's identity: the graph ``name`` (which the
    builder stamps onto the returned graph, so two same-topology graphs with
    different names are distinct objects), the set of routing tags, the models, the
    routing strategy, the sub-agent tags, and whether custom nodes are present. A
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
        # custom nodes are uncacheable-shaped; only their presence matters to the topology.
        str(bool(custom_nodes)),
    ]
    return hashlib.sha1(
        "\x1e".join(parts).encode("utf-8"), usedforsecurity=False
    ).hexdigest()


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


def initialize_graph_from_workspace(
    mcp_config: str | None = "mcp_config.json",
    a2a_config: str | None = None,
    router_model: str | None = None,
    agent_model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    workspace: str | None = None,
    custom_headers: dict[str, Any] | None = None,
    ssl_verify: bool | None = True,
    router_timeout: float | None = None,
    verifier_timeout: float | None = None,
) -> tuple[Graph, dict]:
    """Initialize a graph bundle by discovering domains in the current workspace.

    This utility handles MCP agent synchronization, A2A agent discovery,
    domain tag discovery, and graph construction.

    Args:
        mcp_config: Filename or path to the MCP configuration file.
        a2a_config: Filename or path to the A2A agent configuration file (CONCEPT:ECO-4.0).
        router_model: Optional override for the router's LLM model ID.
        agent_model: Optional override for the specialist agents' LLM model ID.
        api_key: Optional API key for the LLM provider.
        base_url: Optional override for the LLM base URL.
        workspace: Optional explicit path to the agent workspace.
        custom_headers: Optional HTTP headers for provider requests.
        ssl_verify: Whether to enforce SSL certificate verification.
        router_timeout: Per-request timeout for the router node.
        verifier_timeout: Per-request timeout for the verifier node.

    Returns:
        A tuple containing the initialized pydantic-graph Graph instance and
        its configuration dictionary (GraphDeps source).

    """
    logger.info(f"Initializing graph from workspace: {os.getcwd()}")

    if workspace:
        from ..core import workspace as _ws_mod

        _ws_mod.WORKSPACE_DIR = workspace
        logger.info(f"Initializing Graph: workspace pinned to {workspace}")

    # load_node_agents_registry is in this module
    # build_tag_env_map is in this module

    _mcp_cfg_path = resolve_mcp_config_path(mcp_config) if mcp_config else None
    discovery_metadata = {}
    loop = None
    if _mcp_cfg_path:
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            if loop.is_running() and not DEFAULT_VALIDATION_MODE:
                # Already in a loop, create a task
                loop.create_task(ingest_prompts_to_graph())
        except RuntimeError:
            # No running loop, safe to run if not in a server startup context that needs fast health checks.
            # However, during server startup, this is often called before the loop starts.
            # We'll use a thread or just let it be for now, but we'll optimize the functions themselves.
            try:
                # We'll skip the blocking run if we are likely in a server startup
                if not DEFAULT_VALIDATION_MODE:
                    if DEFAULT_KNOWLEDGE_GRAPH_SYNC_BACKGROUND:
                        logger.info("Backgrounding prompt ingestion...")
                        # We can't easily background without a loop here, but we can optimize the call.
                        # For now, let's just ensure it's not called twice.
                        pass
                    else:
                        asyncio.run(ingest_prompts_to_graph())
            except Exception as e:
                logger.debug(f"Registry rebuild failed: {e}")

        try:
            # Check if sync is required first (querying graph last_sync vs mcp_config mtime)
            needs_sync = should_sync(_mcp_cfg_path)
            if needs_sync and not DEFAULT_VALIDATION_MODE:
                try:
                    if loop and loop.is_running():
                        loop.create_task(sync_mcp_agents(config_path=_mcp_cfg_path))
                    else:
                        if DEFAULT_KNOWLEDGE_GRAPH_SYNC_BACKGROUND:
                            logger.info("Backgrounding MCP agent sync...")
                        else:
                            asyncio.run(sync_mcp_agents(config_path=_mcp_cfg_path))
                            logger.info(
                                "Initializing Graph: MCP agents synced successfully."
                            )
                except Exception as e:
                    logger.debug(f"Sync skip/fail: {e}")
            else:
                logger.debug(
                    "Initializing Graph: Valid registry found. Skipping live extraction."
                )

            # Build discovery_metadata directly from the registry
            from collections import defaultdict

            registry = get_discovery_registry()

            tools_by_server = defaultdict(list)
            for agent in registry.agents:
                for tool in agent.tools:
                    tools_by_server[agent.mcp_server].append(tool)

            discovery_metadata = dict(tools_by_server)

            logger.info(
                f"Initializing Graph: Verified {len(discovery_metadata)} servers from registry."
            )
        except Exception as e:
            logger.warning(f"Failed to load MCP discovery metadata: {e}")

    # --- CONCEPT:ECO-4.0: A2A Agent Sync ---
    from agent_utilities.core.config import config as app_config

    _a2a_config = a2a_config or app_config.a2a_config
    if _a2a_config and not DEFAULT_VALIDATION_MODE:
        try:
            from agent_utilities.protocols.a2a_config import sync_a2a_agents

            if loop and loop.is_running():
                loop.create_task(sync_a2a_agents(config_path=_a2a_config))
            else:
                sync_bg = DEFAULT_KNOWLEDGE_GRAPH_SYNC_BACKGROUND
                if sync_bg:
                    logger.info("Backgrounding A2A agent sync...")
                else:
                    import asyncio as _aio

                    _aio.run(sync_a2a_agents(config_path=_a2a_config))
                    logger.info("Initializing Graph: A2A agents synced successfully.")
        except Exception as e:
            logger.debug(f"A2A agent sync skip/fail: {e}")

    # Unified Discovery: merge MCP, A2A, and prompt sources into a single roster
    logger.info("Initializing Graph: Discovering domain tags and agents...")

    if not DEFAULT_VALIDATION_MODE:
        all_specialists = discover_all_specialists()
        tag_prompts = {s.tag: s.description for s in all_specialists}

        if not tag_prompts:
            from agent_utilities.agent.discovery import discover_agents

            logger.debug(
                "No domain tags discovered, falling back to legacy agent discovery..."
            )
            discovered_legacy = discover_agents()
            tag_prompts = {
                tag: meta.get("description", "A2A Specialist")
                for tag, meta in discovered_legacy.items()
            }
    else:
        tag_prompts = {"validation": "dummy"}

    logger.info(f"Initializing Graph: Discovered {len(tag_prompts)} domain tags.")

    tag_env_vars = build_tag_env_map(list(tag_prompts.keys()))

    # Initialize Graph & Deps
    logger.info("Initializing Graph: Building graph topology...")
    graph, config = create_agent(
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
        ssl_verify=ssl_verify,
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

    sub_agents: dict[str, str | Agent] = {
        name: str(package_meta.get("package", name))
        for name, package_meta in agents.items()
    }
    for tag in _skill_agents.keys():
        if tag not in sub_agents:
            sub_agents[tag] = tag

    return create_agent(
        tag_prompts=tag_prompts,
        name=name,
        sub_agents=sub_agents,
        **kwargs,
    )


def create_agent(
    tag_prompts: dict[str, str],
    tag_env_vars: dict[str, str] | None = None,
    mcp_url: str | None = DEFAULT_MCP_URL,
    mcp_config: str | None = DEFAULT_MCP_CONFIG,
    name: str = "GraphAgent",
    router_model: str | None = DEFAULT_ROUTER_MODEL,
    agent_model: str | None = DEFAULT_LITE_LLM_MODEL_ID,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    sub_agents: dict[str, str | Agent] | None = None,
    mcp_toolsets: list[Any] | None = None,
    routing_strategy: str = DEFAULT_ROUTING_STRATEGY,
    router_timeout: float | None = None,
    verifier_timeout: float | None = None,
    custom_nodes: list[Any] | None = None,
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
        custom_nodes: Optional list of additional Starlette/FastAPI nodes to register.
        workspace: Path to the persistent storage directory.
        **kwargs: Additional low-level configuration overrides.

    Returns:
        A tuple containing the configured Graph and its execution dictionary.

    """
    if not _PYDANTIC_GRAPH_AVAILABLE:
        raise ImportError("pydantic-graph is required for graph agents.")

    if tag_env_vars is None:
        tag_env_vars = build_tag_env_map(list(tag_prompts.keys()))

    # CONCEPT:ORCH-1.64 — cache the built graph TOPOLOGY per routing-config. The topology +
    # ``discover_agents()`` are a pure function of (tag_prompts, models, routing strategy,
    # sub-agents, custom nodes), rebuilt on EVERY turn before this. We memoize the structural
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
        custom_nodes=custom_nodes,
    )
    # Cache only when there are no per-run toolset connections AND no custom nodes (whose
    # identity the structural key only approximates by presence). The messaging chat default
    # — the hot path this targets — is exactly toolset-free with no custom nodes.
    _cacheable = (
        not mcp_url and not mcp_config and not mcp_toolsets and not custom_nodes
    )
    _toolset_free = _cacheable
    _cached = _GRAPH_CACHE.get(_cache_key) if _cacheable else None
    if _cached is not None:
        # Warm graph hit (CONCEPT:ORCH-1.64): reuse the structural topology + node registry
        # + registry engine; build only the cheap per-run config. No toolsets on this path
        # (toolset-free by the cache guard above), so connections never get reused.
        logger.debug("create_agent: reusing cached graph topology (key=%s)", _cache_key)
        return _cached["graph"], _build_graph_config(
            graph_nodes=_cached["nodes_registry"],
            knowledge_engine=_cached["knowledge_engine"],
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

    knowledge_engine = None
    try:
        if not all([IntelligenceGraphEngine, PipelineConfig, RegistryPipeline]):
            raise ImportError("Registry Graph dependencies missing")

        if DEFAULT_VALIDATION_MODE:
            logger.info("Registry Graph: Skipping initialization in VALIDATION_MODE.")
        else:
            from agent_utilities.core.paths import data_dir

            ws = get_agent_workspace()
            if setting("AGENT_UTILITIES_TESTING"):
                registry_db = ws / ".agent_utilities_test" / "kg" / "registry_graph.db"
            else:
                registry_db = data_dir() / "kg" / "registry_graph.db"

            registry_db.parent.mkdir(parents=True, exist_ok=True)
            reg_config = PipelineConfig(
                workspace_path=str(ws),
                persist_to_ladybug=True,
                ladybug_path=str(registry_db),
            )
            reg_pipeline = RegistryPipeline(reg_config)
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
                    logger.info("Running RegistryPipeline sync...")
                    asyncio.run(reg_pipeline.run())
                    knowledge_engine = IntelligenceGraphEngine(
                        graph=reg_pipeline.graph, db_path=reg_config.ladybug_path
                    )
                except Exception as e:
                    logger.debug(f"Knowledge engine initialization failed: {e}")
    except ImportError:
        logger.debug("Registry Graph subpackage not found or dependencies missing.")

    # Initialize GraphBuilder

    g = GraphBuilder(
        name=name,
        state_type=GraphState,
        deps_type=GraphDeps,
        output_type=GraphResponse,
    )

    # Register Steps
    _router = g.step(router_step, node_id="router")
    _planner = g.step(planner_step, node_id="planner")
    _onboarding = g.step(onboarding_step, node_id="onboarding")
    _error = g.step(error_recovery_step, node_id="error_recovery")
    _process_executor = g.step(
        load_and_execute_process_flow, node_id="process_executor"
    )

    # Dynamic Dispatcher Nodes
    _dispatcher = g.step(dispatcher_step, node_id="dispatcher")
    _parallel_batch_processor = g.step(
        parallel_batch_processor, node_id="parallel_batch_processor"
    )
    _expert_executor = g.step(expert_executor_step, node_id="expert_executor")

    # Dual Joiners for Phase Separation
    _research_joiner = g.step(join_step, node_id="research_joiner")
    _execution_joiner = g.step(join_step, node_id="execution_joiner")
    _wide_search_joiner = g.step(wide_search_joiner_step, node_id="wide_search_joiner")
    _architect = g.step(architect_step, node_id="architect")
    _verifier = g.step(verifier_step, node_id="verifier")
    _synthesizer = g.step(synthesizer_step, node_id="synthesizer")

    # Native Developer Steps
    _researcher = g.step(researcher_step, node_id="researcher")

    _dedicated_nodes = {
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

    # --- Step Configuration Registry ---
    # We will consolidate all specialized nodes (Skills, Graphs, A2A, Specialist MCP Agents)
    # to ensure each unique node_id (tag) is only registered once in the graph.
    specialist_node_configs = {}

    # Steps for dedicated expert personas are removed.
    # They are now instantiated dynamically via Knowledge Graph context in expert_executor_step.

    _memory_selection = g.step(memory_selection_step, node_id="memory_selection")

    _mcp_router = g.step(dynamic_mcp_routing_step, node_id="mcp_router")
    _mcp_server = g.step(mcp_server_step, node_id="mcp_server_execution")

    # Error and Onboarding
    _error = g.step(error_recovery_step, node_id="error_recovery")
    _onboarding = g.step(onboarding_step, node_id="onboarding")

    # Approval Gate
    _approval = g.step(approval_gate_step, node_id="approval_gate")

    # Usage Guard
    _usage_guard = g.step(usage_guard_step, node_id="usage_guard")

    # --- Dynamic Agent Package & Specialist Registration ---

    discovered_agents_map = discover_agents()

    # 2. Expert Specialist Agents (prioritized over raw skills)
    for tag, meta in discovered_agents_map.items():
        if tag in _dedicated_nodes or tag == "onboarding" or tag == "error_recovery":
            continue

        def make_agent_step(t):
            async def agent_specific_step(
                ctx: StepContext,
            ) -> str | End[Any]:
                return await agent_package_step(ctx, node_id=t)

            agent_specific_step.__name__ = f"agent_{t}_step"
            return agent_specific_step

        # Overwrite or Add the specialist agent configuration
        specialist_node_configs[tag] = make_agent_step(tag)

    # 3. Final Step Registration
    # Now we register all collected configurations exactly once.
    expert_nodes = {}
    for nid, step_func in specialist_node_configs.items():
        expert_nodes[nid] = g.step(step_func, node_id=nid)
        logger.debug(f"Registered graph specialist node: {nid}")

    # --- Node Registry for Explicit Transitions ---
    # We populate a registry of all steps so that step functions can return explicit transitions (StepNodes)
    # This bypasses ambiguity in pydantic-graph Beta's implicit type-matching.
    nodes_registry = {
        "router": _router,
        "planner": _planner,
        "error_recovery": _error,
        "onboarding": _onboarding,
        "dispatcher": _dispatcher,
        "parallel_batch_processor": _parallel_batch_processor,
        "expert_executor": _expert_executor,
        "research_joiner": _research_joiner,
        "execution_joiner": _execution_joiner,
        "wide_search_joiner": _wide_search_joiner,
        "architect": _architect,
        "verifier": _verifier,
        "synthesizer": _synthesizer,
        "researcher": _researcher,
        "memory_selection": _memory_selection,
        "mcp_router": _mcp_router,
        "mcp_server_execution": _mcp_server,
        "process_executor": _process_executor,
        **{nid: step for nid, step in expert_nodes.items()},
    }

    # Dispatcher: The Main Dynamic Branching Logic
    # In pydantic-graph Beta, branching MUST use a Decision node.
    _dispatcher_route = g.decision(node_id="dispatcher_route")

    # 1. Parallel Batch Route (using the state-based caching pattern)
    _dispatcher_route.branches.append(
        g.match(Literal["parallel_batch_processor"]).to(_parallel_batch_processor)  # type: ignore[arg-type]
    )

    # 2. Sequential/Expert Routes (Literal matching on string return value)
    _sequential_routes = [
        ("researcher", _researcher),
        ("architect", _architect),
        ("planner", _planner),
        ("verifier", _verifier),
        ("synthesizer", _synthesizer),
        ("wide_search_joiner", _wide_search_joiner),
        ("mcp_router", _mcp_router),
        ("error_recovery", _error),
        ("onboarding", _onboarding),
        ("expert_executor", _expert_executor),
        ("memory_selection", _memory_selection),
        ("process_executor", _process_executor),
    ]
    for nid, node in _sequential_routes:
        _dispatcher_route.branches.append(g.match(Literal[nid]).to(node))  # type: ignore[arg-type]

    # Explicit dispatcher routing if returned (e.g. by verifier)
    _dispatcher_route.branches.append(g.match(Literal["dispatcher"]).to(_dispatcher))  # type: ignore[arg-type]
    _dispatcher_route.branches.append(g.match(Literal["error"]).to(_error))  # type: ignore[arg-type]
    _dispatcher_route.branches.append(g.match(Literal["error_recovery"]).to(_error))  # type: ignore[arg-type]

    # Skill/Agent Nodes
    for nid, node in expert_nodes.items():
        _dispatcher_route.branches.append(g.match(Literal[nid]).to(node))  # type: ignore[arg-type]

    # 3. Termination Route (returns None)
    _dispatcher_route.branches.append(g.match(type(None)).to(g.end_node))

    # Joiner Routes
    _research_joiner_route = g.decision(node_id="research_joiner_route")
    _research_joiner_route.branches.append(
        g.match(Literal["dispatcher"]).to(_dispatcher)  # type: ignore[arg-type]
    )
    _research_joiner_route.branches.append(g.match(type(None)).to(g.end_node))

    _execution_joiner_route = g.decision(node_id="execution_joiner_route")
    _execution_joiner_route.branches.append(
        g.match(Literal["dispatcher"]).to(_dispatcher)  # type: ignore[arg-type]
    )
    _execution_joiner_route.branches.append(g.match(Literal["verifier"]).to(_verifier))  # type: ignore[arg-type]
    _execution_joiner_route.branches.append(g.match(type(None)).to(g.end_node))

    _memory_selection_route = g.decision(node_id="memory_selection_route")
    _memory_selection_route.branches.append(
        g.match(Literal["dispatcher"]).to(_dispatcher)  # type: ignore[arg-type]
    )
    _memory_selection_route.branches.append(
        g.match(Literal["researcher"]).to(_researcher)  # type: ignore[arg-type]
    )

    _verifier_route = g.decision(node_id="verifier_route")
    _verifier_route.branches.append(g.match(Literal["synthesizer"]).to(_synthesizer))  # type: ignore[arg-type]
    _verifier_route.branches.append(g.match(Literal["dispatcher"]).to(_dispatcher))  # type: ignore[arg-type]
    _verifier_route.branches.append(g.match(Literal["planner"]).to(_planner))  # type: ignore[arg-type]

    _execution_joiner_route = g.decision(node_id="execution_joiner_route")
    _execution_joiner_route.branches.append(
        g.match(Literal["dispatcher"]).to(_dispatcher)  # type: ignore[arg-type]
    )
    _execution_joiner_route.branches.append(g.match(Literal["router_step"]).to(_router))  # type: ignore[arg-type]
    _execution_joiner_route.branches.append(g.match(Literal["router"]).to(_router))  # type: ignore[arg-type]
    _execution_joiner_route.branches.append(
        g.match(Literal["wide_search_joiner"]).to(_wide_search_joiner)  # type: ignore[arg-type]
    )
    _execution_joiner_route.branches.append(g.match(type(None)).to(g.end_node))

    # Register the decision node and edges
    g.add(
        # Start -> UsageGuard -> Router -> Dispatcher
        g.edge_from(g.start_node).label("Query").to(_usage_guard),
        g.edge_from(_usage_guard).label("Policy OK").to(_router),
        g.edge_from(_router).label("Plan").to(_dispatcher),
        # CONCEPT:ORCH-1.68 — the router has a SINGLE outgoing edge (→ dispatcher). It must NOT
        # have a second edge to the end node: pydantic-graph turns two edges from one node into
        # a BROADCAST FORK (router → {end, dispatcher}), which terminated every full-graph turn
        # via the end branch. A direct-completion turn never reaches the router — it is answered
        # outside the graph by ``_run_direct_completion`` (agent_runner) — so the router never
        # needs to end the run itself.
        # Edge to the decision node for experts
        g.edge_from(_dispatcher).to(_dispatcher_route),
        # Dead-end elimination for unused but registered nodes
        g.edge_from(_planner).to(_dispatcher),
        g.edge_from(_process_executor).to(_dispatcher),
        g.edge_from(_memory_selection).to(_memory_selection_route),
        # Rest of the graph
        g.edge_from(_parallel_batch_processor).map().to(_expert_executor),
        # Expert Nodes: Return to Joiner for synchronization
        g.edge_from(_researcher).label("Research Done").to(_research_joiner),
        g.edge_from(_architect).label("Design Done").to(_research_joiner),
        # adaptive_agent_router returning strings are handled by dispatcher_route if they loop
        *(g.edge_from(node).to(_execution_joiner) for node in expert_nodes.values()),
        g.edge_from(_expert_executor).to(_execution_joiner),
        # Special Handling for MCP Parallel Flow
        g.edge_from(_mcp_router).map().to(_mcp_server),
        g.edge_from(_mcp_server).to(_execution_joiner),
        # Joiners: Return control to Dispatcher or designated node
        g.edge_from(_research_joiner).to(_research_joiner_route),
        g.edge_from(_execution_joiner).to(_execution_joiner_route),
        g.edge_from(_wide_search_joiner).to(_dispatcher_route),
        # Error handling and Finalization
        g.edge_from(_error).to(_planner),
        g.edge_from(_verifier).to(_verifier_route),
        g.edge_from(_synthesizer).to(g.end_node),
        g.edge_from(_onboarding).to(g.end_node),
    )

    # Add custom nodes if provided (legacy support via wrapping)
    if custom_nodes:
        for node in custom_nodes:
            logger.warning(
                f"Custom node {node} detected. Functional steps are preferred in Beta API."
            )
            # We could wrap them here if needed, but for now we prioritize functional steps.

    graph = g.build()

    # MCP Setup (same as before)
    _mcp_toolsets = list(mcp_toolsets) if mcp_toolsets else []
    # (Loading toolsets logic preserved...)
    if not DEFAULT_VALIDATION_MODE:
        if mcp_url:
            from agent_utilities.mcp.toolset_factory import build_http_toolset

            if is_loopback_url(
                mcp_url, kwargs.get("current_host"), kwargs.get("current_port")
            ):
                pass
            else:
                _mcp_toolsets.append(
                    build_http_toolset(
                        mcp_url,
                        verify=kwargs.get("ssl_verify", DEFAULT_SSL_VERIFY),
                        timeout=60,
                    )
                )

        if mcp_config:
            _mcp_cfg_path = resolve_mcp_config_path(mcp_config)
            if _mcp_cfg_path:
                # Load MCP servers individually so that a single undefined env-var
                # does not prevent the rest of the toolsets from loading.
                # Standardized loading via mcp_utilities (handles parallel probing and robust expansion)
                _mcp_toolsets = load_mcp_config(_mcp_cfg_path)
                for ts in _mcp_toolsets:
                    srv_id = getattr(ts, "id", getattr(ts, "name", "unknown"))
                    logger.info(f"MCP Startup: Registered server '{srv_id}'")
            else:
                logger.warning(f"MCP config {mcp_config} not found")

    config = _build_graph_config(
        graph_nodes=nodes_registry,
        knowledge_engine=knowledge_engine,
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
        f"create_agent: returning config with mcp_toolsets of len: {len(config['mcp_toolsets'])}"
    )

    # CONCEPT:ORCH-1.64 — store the toolset-free structural build for reuse next turn.
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
    sub_agents: dict[str, str | Agent] | None,
    routing_strategy: str,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Build the per-run execution config dict (CONCEPT:ORCH-1.64).

    The config is cheap and per-run (carries the run's toolsets / models / timeouts); only
    the graph TOPOLOGY is cached. Extracted so the cache-hit and cache-miss paths build the
    same config shape from a (possibly cached) ``graph_nodes`` registry and registry engine.
    """
    from agent_utilities.core.config import (
        DEFAULT_GRAPH_ROUTER_TIMEOUT,
        DEFAULT_GRAPH_VERIFIER_TIMEOUT,
    )

    _api_key = kwargs.get("api_key")
    _base_url = kwargs.get("base_url")
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
        "ssl_verify": kwargs.get("ssl_verify", DEFAULT_SSL_VERIFY),
        "custom_headers": kwargs.get("custom_headers"),
        "sub_agents": sub_agents or {},
        "routing_strategy": routing_strategy,
        "nodes": graph_nodes,
        "discovery_metadata": kwargs.get("discovery_metadata") or {},
        "knowledge_engine": knowledge_engine,
    }


# Alias for backward compatibility
create_graph_agent = create_agent
