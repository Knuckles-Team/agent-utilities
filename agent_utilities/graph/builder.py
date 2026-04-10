#!/usr/bin/python

from __future__ import annotations

import os
import logging


from typing import Any, Optional, Literal


from pydantic_ai import Agent


from ..config import (
    DEFAULT_PROVIDER,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_API_KEY,
    DEFAULT_SSL_VERIFY,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_ROUTING_STRATEGY,
    DEFAULT_VALIDATION_MODE,
    DEFAULT_ROUTER_MODEL,
    DEFAULT_GRAPH_AGENT_MODEL,
    DEFAULT_MCP_URL,
    DEFAULT_MCP_CONFIG,
)
from ..workspace import get_workspace_path, CORE_FILES
from ..base_utilities import (
    is_loopback_url,
)
from ..mcp_utilities import load_mcp_config
from ..models import GraphResponse
from .config_helpers import (
    load_node_agents_registry,
    NODE_SKILL_MAP,
)

from .state import GraphState, GraphDeps

from .executor import (
    _execute_specialized_step,
    agent_package_step,
)

from .steps import (
    usage_guard_step,
    router_step,
    planner_step,
    researcher_step,
    dispatcher_step,
    parallel_batch_processor,
    expert_executor_step,
    join_step,
    architect_step,
    verifier_step,
    onboarding_step,
    memory_selection_step,
    approval_gate_step,
    dynamic_mcp_routing_step,
    mcp_server_step,
    error_recovery_step,
)

from pydantic_graph import End, Graph
from pydantic_graph.beta import GraphBuilder, StepContext

_PYDANTIC_GRAPH_AVAILABLE = True

logger = logging.getLogger(__name__)


def build_tag_env_map(tag_names: list[str]) -> dict[str, str]:
    """Build a tag→env_var mapping following the standard convention.

    Standard convention: tag "incidents" → env var "INCIDENTSTOOL"
    (upper-cased tag + "TOOL" suffix).

    Args:
        tag_names: List of domain tag names.

    Returns:
        Dict mapping tag name → env var name.
    """
    result = {}
    for tag in tag_names:
        env_var = tag.upper().replace("-", "_") + "TOOL"
        result[tag] = env_var
    return result


def initialize_graph_from_workspace(
    mcp_config: Optional[str] = "mcp_config.json",
    router_model: Optional[str] = None,
    agent_model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    workspace: Optional[str] = None,
) -> tuple[Graph, GraphDeps]:
    """
    Centralized utility to discover domains and initialize a graph bundle from the current workspace.
    Used for both server startup and standalone testing.
    """
    logger.info(f"Initializing graph from workspace: {os.getcwd()}")

    if workspace:
        from .. import workspace as _ws_mod

        _ws_mod.WORKSPACE_DIR = workspace
        logger.info(f"Initializing Graph: workspace pinned to {workspace}")

    # load_node_agents_registry is in this module
    from ..a2a import discover_agents
    from ..mcp_agent_manager import sync_mcp_agents, should_sync

    # build_tag_env_map is in this module
    from ..config import (
        DEFAULT_ROUTER_MODEL,
        DEFAULT_GRAPH_AGENT_MODEL,
        DEFAULT_MCP_URL,
    )

    from ..workspace import resolve_mcp_config_path

    _mcp_cfg_path = resolve_mcp_config_path(mcp_config) if mcp_config else None
    discovery_metadata = {}
    if _mcp_cfg_path:
        agents_path = get_workspace_path(CORE_FILES["NODE_AGENTS"])
        try:
            # Check if sync is required first
            needs_sync = should_sync(_mcp_cfg_path, agents_path)

            if needs_sync:
                logger.info(
                    "Initializing Graph: Registry out of sync. Standardizing MCP agents..."
                )
                from ..mcp_agent_manager import sync_mcp_agents

                try:
                    import asyncio

                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        logger.warning(
                            "Initializing Graph: Skip standardization (loop already running)."
                        )
                    else:
                        loop.run_until_complete(sync_mcp_agents(_mcp_cfg_path))
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

            registry = load_node_agents_registry()

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

    # Discovery Logic
    logger.info("Initializing Graph: Discovering domain tags and agents...")
    registry = load_node_agents_registry()
    tag_prompts = {a.tag: a.description for a in registry.agents if a.tag}
    if not tag_prompts:
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
        discovery_metadata=discovery_metadata,
        router_model=router_model or DEFAULT_ROUTER_MODEL,
        agent_model=agent_model or DEFAULT_GRAPH_AGENT_MODEL,
        api_key=api_key,
        base_url=base_url,
        workspace=workspace,
    )
    logger.info("Initializing Graph: Topology built successfully.")

    return graph, config


def create_master_graph(
    name: str = "MasterGraph",
    include_agents: list[str] | None = None,
    exclude_agents: list[str] | None = None,
    skill_agents: dict[str, dict] | None = None,
    **kwargs,
) -> tuple["Graph", dict]:
    """Factory to create a master orchestrator graph that discovers and routes to sub-agents.

    Args:
        name: Name of the master graph.
        include_agents: Optional list of specific agent package names to include.
        exclude_agents: Optional list of agent package names to ignore.
        skill_agents: Optional dict of specialized skill agents (Tag -> Config).
        **kwargs: Forwarded to create_graph_agent.

    Returns:
        tuple: (Graph, config_dict)
    """
    from ..a2a import discover_agents

    agents = discover_agents(
        include_packages=include_agents, exclude_packages=exclude_agents
    )

    tag_prompts = {
        name: f"Specialized agent for {package_name}"
        for name, package_name in agents.items()
    }

    _skill_agents = skill_agents or {}
    for tag, agent_cfg in _skill_agents.items():
        if tag not in tag_prompts:
            tag_prompts[tag] = agent_cfg.get(
                "description", f"Specialized skill agent for {tag}"
            )

    sub_agents = {name: package_name for name, package_name in agents.items()}
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
    router_model: str = DEFAULT_ROUTER_MODEL,
    agent_model: str = DEFAULT_GRAPH_AGENT_MODEL,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    sub_agents: dict[str, str | Agent] | None = None,
    mcp_toolsets: list[Any] | None = None,
    routing_strategy: str = DEFAULT_ROUTING_STRATEGY,
    custom_nodes: list[Any] | None = None,
    workspace: str | None = None,
    **kwargs,
) -> tuple["Graph", dict]:
    """Factory to create a router-led graph assistant using Pydantic Graph Beta.

    Args:
        tag_prompts: Dict of domain tags → intent description prompts.
        tag_env_vars: Dict of domain tags → env var gating names.
        mcp_url: Optional base MCP URL for all nodes.
        mcp_config: Optional path to JSON MCP config.
        name: Name of the graph.
        router_model: Model for the router node.
        agent_model: Model for the domain nodes.
        min_confidence: Confidence threshold for routing.
        sub_agents: Dict of domain tags → sub-agent package name or instance.
        mcp_toolsets: Optional list of pre-instantiated toolsets (e.g. FastMCP).
        routing_strategy: Strategy for routing (hybrid, rules, llm).

    Returns:
        Graph and config dict.
    """
    if not _PYDANTIC_GRAPH_AVAILABLE:
        raise ImportError("pydantic-graph is required for graph agents.")

    if tag_env_vars is None:
        tag_env_vars = build_tag_env_map(list(tag_prompts.keys()))

    # Initialize GraphBuilder

    g = GraphBuilder(
        state_type=GraphState,
        deps_type=GraphDeps,
        output_type=GraphResponse,
    )

    # Register Steps
    _router = g.step(router_step, node_id="router")
    # _validator = g.step(validator_step, node_id="validator")
    _planner = g.step(planner_step, node_id="planner")
    # _executor = g.step(project_executor_step, node_id="project_executor")
    _error = g.step(error_recovery_step, node_id="error_recovery")
    _onboarding = g.step(onboarding_step, node_id="onboarding")

    # Dynamic Dispatcher Nodes
    _dispatcher = g.step(dispatcher_step, node_id="dispatcher")
    _parallel_batch_processor = g.step(
        parallel_batch_processor, node_id="parallel_batch_processor"
    )
    _expert_executor = g.step(expert_executor_step, node_id="expert_executor")

    # Dual Joiners for Phase Separation
    _research_joiner = g.step(join_step, node_id="research_joiner")
    _execution_joiner = g.step(join_step, node_id="execution_joiner")
    # _joiner = g.step(join_step, node_id="joiner")  # Legacy fallback

    # _coordinator = g.step(coordinator_step, node_id="coordinator")
    _architect = g.step(architect_step, node_id="architect")
    _verifier = g.step(verifier_step, node_id="verifier")
    # _critique = g.step(critique_step, node_id="critique")

    # Native Developer Steps
    _researcher = g.step(researcher_step, node_id="researcher")

    _dedicated_nodes = {"researcher", "architect", "planner", "verifier"}

    # --- Step Configuration Registry ---
    # We will consolidate all specialized nodes (Skills, Graphs, A2A, Specialist MCP Agents)
    # to ensure each unique node_id (tag) is only registered once in the graph.
    specialist_node_configs = {}

    # 1. Base Skill-based Specialized Steps (from NODE_SKILL_MAP)
    for _nid in NODE_SKILL_MAP:
        if _nid in _dedicated_nodes:
            continue

        def _make_skill_step(nid: str):
            async def _step(
                ctx: StepContext[GraphState, GraphDeps, None],
            ) -> str | End[Any]:
                return await _execute_specialized_step(ctx, nid)

            _step.__name__ = f"{nid}_step"
            return _step

        specialist_node_configs[_nid] = _make_skill_step(_nid)

    # _python = g.step(python_programmer_step, node_id="python_programmer")
    # _c = g.step(c_programmer_step, node_id="c_programmer")
    # _cpp = g.step(cpp_programmer_step, node_id="cpp_programmer")
    # _golang = g.step(golang_programmer_step, node_id="golang_programmer")
    # _javascript = g.step(javascript_programmer_step, node_id="javascript_programmer")
    # _typescript = g.step(typescript_programmer_step, node_id="typescript_programmer")
    # _security = g.step(security_auditor_step, node_id="security_auditor")
    # _qa = g.step(qa_expert_step, node_id="qa_expert")
    # _debugger = g.step(debugger_expert_step, node_id="debugger_expert")
    # _ui_ux = g.step(ui_ux_designer_step, node_id="ui_ux_designer")
    # _devops = g.step(devops_engineer_step, node_id="devops_engineer")
    # _cloud = g.step(cloud_architect_step, node_id="cloud_architect")
    # _database = g.step(database_expert_step, node_id="database_expert")
    # _rust = g.step(rust_programmer_step, node_id="rust_programmer")
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
    from ..a2a import discover_agents

    discovered_agents_map = discover_agents()

    # 2. Expert Specialist Agents (prioritized over raw skills)
    for tag, meta in discovered_agents_map.items():

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
        "researcher": _researcher,
        "memory_selection": _memory_selection,
        "mcp_router": _mcp_router,
        "mcp_server_execution": _mcp_server,
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
        ("mcp_router", _mcp_router),
        ("error_recovery", _error),
        ("onboarding", _onboarding),
        ("expert_executor", _expert_executor),
        ("memory_selection", _memory_selection),
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
        g.edge_from(_memory_selection).to(_dispatcher),
        # Rest of the graph
        g.edge_from(_parallel_batch_processor).map().to(_expert_executor),
        # Expert Nodes: Return to Joiner for synchronization
        g.edge_from(_researcher).label("Research Done").to(_research_joiner),
        g.edge_from(_architect).label("Design Done").to(_research_joiner),
        # specialists returning strings are handled by dispatcher_route if they loop
        *(g.edge_from(node).to(_execution_joiner) for node in expert_nodes.values()),
        g.edge_from(_expert_executor).to(_execution_joiner),
        # Special Handling for MCP Parallel Flow
        g.edge_from(_mcp_router).map().to(_mcp_server),
        g.edge_from(_mcp_server).to(_execution_joiner),
        # Joiners: Return control to Dispatcher
        g.edge_from(_research_joiner).to(_dispatcher),
        g.edge_from(_execution_joiner).to(_dispatcher),
        # Error handling and Finalization
        g.edge_from(_error).label("Error").to(g.end_node),
        g.edge_from(_verifier).label("Verified").to(g.end_node),
        g.edge_from(_verifier).label("Self-Correction").to(_dispatcher),
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
            from ..workspace import resolve_mcp_config_path

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

    config = {
        "tag_prompts": tag_prompts,
        "tag_env_vars": tag_env_vars,
        "mcp_url": mcp_url,
        "mcp_config": mcp_config,
        "mcp_toolsets": _mcp_toolsets,
        "router_model": router_model,
        "agent_model": agent_model,
        "min_confidence": min_confidence,
        "valid_domains": tuple(tag_prompts.keys()),
        "provider": kwargs.get("provider", DEFAULT_PROVIDER),
        "base_url": kwargs.get("base_url", DEFAULT_LLM_BASE_URL),
        "api_key": kwargs.get("api_key", DEFAULT_LLM_API_KEY),
        "ssl_verify": kwargs.get("ssl_verify", DEFAULT_SSL_VERIFY),
        "sub_agents": sub_agents or {},
        "routing_strategy": routing_strategy,
        "nodes": nodes_registry,
    }

    import sys

    print(
        f"DEBUG: create_graph_agent returning config with mcp_toolsets of len: {len(config['mcp_toolsets'])}",
        file=sys.stderr,
    )

    return graph, config
