#!/usr/bin/python
"""Graph Builder Module.

This module provides the factory and registration logic for constructing
pydantic-graph instances. It handles domain discovery, specialist node
registration, and the definition of the graph's dynamic routing topology.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

from pydantic_ai import Agent
from pydantic_graph import End, Graph
from pydantic_graph.beta import GraphBuilder, StepContext

from ..base_utilities import (
    is_loopback_url,
)
from ..config import (
    DEFAULT_GRAPH_AGENT_MODEL,
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_MCP_CONFIG,
    DEFAULT_MCP_URL,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_PROVIDER,
    DEFAULT_ROUTER_MODEL,
    DEFAULT_ROUTING_STRATEGY,
    DEFAULT_SSL_VERIFY,
    DEFAULT_VALIDATION_MODE,
)
from ..discovery import discover_agents, discover_all_specialists
from ..mcp_agent_manager import should_sync, sync_mcp_agents
from ..mcp_utilities import load_mcp_config
from ..models import GraphResponse
from ..workspace import get_agent_workspace, resolve_mcp_config_path
from .config_helpers import (
    get_discovery_registry,
)
from .executor import (
    agent_package_step,
)
from .nodes import (
    LoadAndExecuteProcessFlow,
)
from .state import GraphDeps, GraphState
from .steps import (
    approval_gate_step,
    architect_step,
    council_step,
    dispatcher_step,
    dynamic_mcp_routing_step,
    error_recovery_step,
    expert_executor_step,
    join_step,
    mcp_server_step,
    memory_selection_step,
    onboarding_step,
    parallel_batch_processor,
    planner_step,
    researcher_step,
    router_step,
    synthesizer_step,
    usage_guard_step,
    verifier_step,
)

try:
    from ..knowledge_graph.engine import RegistryGraphEngine
    from ..knowledge_graph.models import PipelineConfig
    from ..knowledge_graph.pipeline import RegistryPipeline
except ImportError:
    # These might be missing if the extra is not installed, but we want them at top level for patching
    RegistryGraphEngine = None  # type: ignore
    PipelineConfig = None  # type: ignore
    RegistryPipeline = None  # type: ignore
from ..agent_registry_builder import ingest_prompts_to_graph

_PYDANTIC_GRAPH_AVAILABLE = True

logger = logging.getLogger(__name__)


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

    This utility handles MCP agent synchronization, domain tag discovery,
    and graph construction. It matches the local workspace directory against
    the provided MCP configuration to build a functional orchestrator.

    Args:
        mcp_config: Filename or path to the MCP configuration file.
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
        from .. import workspace as _ws_mod

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
            if loop.is_running():
                # Already in a loop, create a task
                loop.create_task(ingest_prompts_to_graph())
        except RuntimeError:
            # No running loop, safe to run if not in a server startup context that needs fast health checks.
            # However, during server startup, this is often called before the loop starts.
            # We'll use a thread or just let it be for now, but we'll optimize the functions themselves.
            try:
                # We'll skip the blocking run if we are likely in a server startup
                if (
                    os.getenv("KNOWLEDGE_GRAPH_SYNC_BACKGROUND", "true").lower()
                    == "true"
                ):
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
            if needs_sync:
                try:
                    if loop and loop.is_running():
                        loop.create_task(sync_mcp_agents(config_path=_mcp_cfg_path))
                    else:
                        if (
                            os.getenv("KNOWLEDGE_GRAPH_SYNC_BACKGROUND", "true").lower()
                            == "true"
                        ):
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

    # Unified Discovery: merge MCP and A2A sources into a single roster
    logger.info("Initializing Graph: Discovering domain tags and agents...")

    all_specialists = discover_all_specialists()
    tag_prompts = {s.tag: s.description for s in all_specialists}

    if not tag_prompts:
        from ..discovery import discover_agents

        discovered = discover_agents()
        tag_prompts = {
            tag: meta.get("description", "A2A Specialist")
            for tag, meta in discovered.items()
        }
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
        agent_model=agent_model or DEFAULT_GRAPH_AGENT_MODEL,
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

    return create_graph_agent(
        tag_prompts=tag_prompts,
        name=name,
        sub_agents=sub_agents,
        **kwargs,
    )


def create_graph_agent(
    tag_prompts: dict[str, str],
    tag_env_vars: dict[str, str] | None = None,
    mcp_url: str | None = DEFAULT_MCP_URL,
    mcp_config: str | None = DEFAULT_MCP_CONFIG,
    name: str = "GraphAgent",
    router_model: str | None = DEFAULT_ROUTER_MODEL,
    agent_model: str | None = DEFAULT_GRAPH_AGENT_MODEL,
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

    knowledge_engine = None
    try:
        if not all([RegistryGraphEngine, PipelineConfig, RegistryPipeline]):
            raise ImportError("Registry Graph dependencies missing")

        if DEFAULT_VALIDATION_MODE:
            logger.info("Registry Graph: Skipping initialization in VALIDATION_MODE.")
        else:
            ws = get_agent_workspace()
            reg_config = PipelineConfig(
                workspace_path=str(ws),
                persist_to_ladybug=True,
                ladybug_path=str(ws / "registry_graph.db"),
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
                    from filelock import FileLock, Timeout

                    lock_path = ws / ".registry.sync.lock"
                    lock = FileLock(str(lock_path), timeout=0)

                    # Only run the pipeline if we can acquire the lock (non-blocking)
                    # or if we are forced to run it.
                    sync_background = (
                        os.getenv("KNOWLEDGE_GRAPH_SYNC_BACKGROUND", "true").lower()
                        == "true"
                    )

                    try:
                        with lock.acquire(timeout=0 if sync_background else 60):
                            logger.info("Running RegistryPipeline sync...")
                            asyncio.run(reg_pipeline.run())
                    except Timeout:
                        if sync_background:
                            logger.info(
                                "Another process is syncing the Registry Graph. Skipping to avoid memory exhaustion."
                            )
                        else:
                            logger.warning(
                                "Timed out waiting for Registry Graph sync lock."
                            )
                    knowledge_engine = RegistryGraphEngine(
                        reg_pipeline.graph, db_path=reg_config.ladybug_path
                    )
                except Exception as e:
                    logger.debug(f"Knowledge engine initialization failed: {e}")
    except ImportError:
        logger.debug("Registry Graph subpackage not found or dependencies missing.")

    # Initialize GraphBuilder

    g = GraphBuilder(
        state_type=GraphState,
        deps_type=GraphDeps,
        output_type=GraphResponse,
    )

    # Register Steps
    _router = g.step(router_step, node_id="router")
    _planner = g.step(planner_step, node_id="planner")
    _onboarding = g.step(onboarding_step, node_id="onboarding")
    _error = g.step(error_recovery_step, node_id="error_recovery")
    _process_executor = g.step(LoadAndExecuteProcessFlow, node_id="process_executor")

    # Dynamic Dispatcher Nodes
    _dispatcher = g.step(dispatcher_step, node_id="dispatcher")
    _parallel_batch_processor = g.step(
        parallel_batch_processor, node_id="parallel_batch_processor"
    )
    _expert_executor = g.step(expert_executor_step, node_id="expert_executor")

    # Dual Joiners for Phase Separation
    _research_joiner = g.step(join_step, node_id="research_joiner")
    _execution_joiner = g.step(join_step, node_id="execution_joiner")
    _architect = g.step(architect_step, node_id="architect")
    _verifier = g.step(verifier_step, node_id="verifier")
    _synthesizer = g.step(synthesizer_step, node_id="synthesizer")

    # Native Developer Steps
    _researcher = g.step(researcher_step, node_id="researcher")
    _council = g.step(council_step, node_id="council")

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
        "council",
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
                ctx: StepContext[GraphState, GraphDeps, Any],
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
        g.match(Literal["parallel_batch_processor"]).to(_parallel_batch_processor)
    )

    # 2. Sequential/Expert Routes (Literal matching on string return value)
    _sequential_routes = [
        ("researcher", _researcher),
        ("architect", _architect),
        ("planner", _planner),
        ("verifier", _verifier),
        ("synthesizer", _synthesizer),
        ("council", _council),
        ("mcp_router", _mcp_router),
        ("error_recovery", _error),
        ("onboarding", _onboarding),
        ("expert_executor", _expert_executor),
        ("memory_selection", _memory_selection),
        ("process_executor", _process_executor),
    ]
    for nid, node in _sequential_routes:
        _dispatcher_route.branches.append(g.match(Literal[nid]).to(node))

    # Explicit dispatcher routing if returned (e.g. by verifier)
    _dispatcher_route.branches.append(g.match(Literal["dispatcher"]).to(_dispatcher))
    _dispatcher_route.branches.append(g.match(Literal["error"]).to(_error))
    _dispatcher_route.branches.append(g.match(Literal["error_recovery"]).to(_error))

    # Skill/Agent Nodes
    for nid, node in expert_nodes.items():
        _dispatcher_route.branches.append(g.match(Literal[nid]).to(node))

    # 3. Termination Route (returns None)
    _dispatcher_route.branches.append(g.match(type(None)).to(g.end_node))

    # Register the decision node and edges
    g.add(
        # Start -> UsageGuard -> Router -> Dispatcher
        g.edge_from(g.start_node).label("Query").to(_usage_guard),
        g.edge_from(_usage_guard).label("Policy OK").to(_router),
        g.edge_from(_router).label("Plan").to(_dispatcher),
        # Edge to the decision node for experts
        g.edge_from(_dispatcher).to(_dispatcher_route),
        # Dead-end elimination for unused but registered nodes
        g.edge_from(_planner).to(_dispatcher),
        g.edge_from(_process_executor).to(_dispatcher),
        g.edge_from(_memory_selection).to(_dispatcher),
        g.edge_from(_memory_selection).label("Context Gap").to(_researcher),
        # Rest of the graph
        g.edge_from(_parallel_batch_processor).map().to(_expert_executor),
        # Expert Nodes: Return to Joiner for synchronization
        g.edge_from(_researcher).label("Research Done").to(_research_joiner),
        g.edge_from(_architect).label("Design Done").to(_research_joiner),
        # specialists returning strings are handled by dispatcher_route if they loop
        *(g.edge_from(node).to(_execution_joiner) for node in expert_nodes.values()),
        g.edge_from(_expert_executor).to(_execution_joiner),
        g.edge_from(_council).to(_execution_joiner),
        # Special Handling for MCP Parallel Flow
        g.edge_from(_mcp_router).map().to(_mcp_server),
        g.edge_from(_mcp_server).to(_execution_joiner),
        # Joiners: Return control to Dispatcher
        g.edge_from(_research_joiner).to(_dispatcher),
        g.edge_from(_execution_joiner).to(_dispatcher),
        # Error handling and Finalization
        g.edge_from(_error).label("error_recovery").to(g.end_node),
        g.edge_from(_error).label("planner").to(_planner),
        g.edge_from(_verifier).label("synthesizer").to(_synthesizer),
        g.edge_from(_verifier).label("dispatcher").to(_dispatcher),
        g.edge_from(_verifier).label("planner").to(_planner),
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
            import httpx
            from pydantic_ai.mcp import MCPServerSSE, MCPServerStreamableHTTP

            if is_loopback_url(
                mcp_url, kwargs.get("current_host"), kwargs.get("current_port")
            ):
                pass
            elif mcp_url.lower().endswith("/sse"):
                _mcp_toolsets.append(
                    MCPServerSSE(
                        mcp_url,
                        http_client=httpx.AsyncClient(
                            verify=kwargs.get("ssl_verify", DEFAULT_SSL_VERIFY),
                            timeout=60,
                        ),
                    )
                )
            else:
                _mcp_toolsets.append(
                    MCPServerStreamableHTTP(
                        mcp_url,
                        http_client=httpx.AsyncClient(
                            verify=kwargs.get("ssl_verify", DEFAULT_SSL_VERIFY),
                            timeout=60,
                        ),
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

    _api_key = kwargs.get("api_key")
    _base_url = kwargs.get("base_url")

    from ..config import (
        DEFAULT_GRAPH_ROUTER_TIMEOUT,
        DEFAULT_GRAPH_VERIFIER_TIMEOUT,
    )

    config = {
        "tag_prompts": tag_prompts,
        "tag_env_vars": tag_env_vars,
        "mcp_url": mcp_url,
        "mcp_config": mcp_config,
        "mcp_toolsets": _mcp_toolsets,
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
        "provider": kwargs.get("provider") or DEFAULT_PROVIDER,
        "base_url": _base_url if _base_url is not None else DEFAULT_LLM_BASE_URL,
        "api_key": _api_key if _api_key is not None else DEFAULT_LLM_API_KEY,
        "ssl_verify": kwargs.get("ssl_verify", DEFAULT_SSL_VERIFY),
        "custom_headers": kwargs.get("custom_headers"),
        "sub_agents": sub_agents or {},
        "routing_strategy": routing_strategy,
        "nodes": nodes_registry,
        "discovery_metadata": kwargs.get("discovery_metadata") or {},
        "knowledge_engine": knowledge_engine,
    }

    logger.debug(
        f"create_graph_agent: returning config with mcp_toolsets of len: {len(config['mcp_toolsets'])}"
    )

    return graph, config
