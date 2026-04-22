#!/usr/bin/python
"""Graph Execution Module.

This module provides the execution logic for the pydantic-graph orchestrator.
It defines functions to run graphs synchronously or as asynchronous streams (SSE),
handles state persistence, MCP server connectivity during runs, and graph validation.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..config import (
    DEFAULT_ENABLE_LLM_VALIDATION,
    DEFAULT_GRAPH_AGENT_MODEL,
    DEFAULT_GRAPH_PERSISTENCE_PATH,
    DEFAULT_GRAPH_ROUTER_TIMEOUT,
    DEFAULT_GRAPH_VERIFIER_TIMEOUT,
    DEFAULT_PROVIDER,
    DEFAULT_ROUTER_MODEL,
    DEFAULT_SSL_VERIFY,
)
from ..model_factory import create_model
from ..models import GraphResponse
from .config_helpers import (
    DEFAULT_GRAPH_TIMEOUT,
    emit_graph_event,
    load_node_agents_registry,
)
from .mermaid import get_graph_mermaid
from .state import GraphDeps, GraphState

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

    Returns:
        A GraphResponse instance containing the final synthesized output
        and execution metadata.

    """
    if run_id is None:
        run_id = uuid4().hex

    mermaid_prefix = ""
    if streamdown:
        try:
            mermaid_prefix = f"```mermaid\n{get_graph_mermaid(graph, config)}\n```\n\n"
        except Exception:
            pass

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
            provider=config.get("provider", DEFAULT_PROVIDER),
            ssl_verify=_ssl_verify,
        ),
        agent_model=create_model(
            model_id=config.get("agent_model", DEFAULT_GRAPH_AGENT_MODEL),
            api_key=config.get("api_key"),
            base_url=config.get("base_url"),
            custom_headers=_custom_headers,
            provider=config.get("provider", DEFAULT_PROVIDER),
            ssl_verify=_ssl_verify,
        ),
        nodes=config.get("nodes", {}),
        min_confidence=config.get("min_confidence", 0.6),
        sub_agents=config.get("sub_agents", {}),
        provider=config.get("provider", DEFAULT_PROVIDER),
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

            srv_id = getattr(ts, "id", getattr(ts, "name", repr(ts)))
            try:
                logger.debug(f"run_graph: Connecting to MCP server '{srv_id}'...")
                connected = await asyncio.wait_for(
                    stack.enter_async_context(ts), timeout=60.0
                )
                _already_connected.add(id(ts))
                connected_toolsets.append(connected)
                logger.info(f"run_graph: ✅ MCP server '{srv_id}' connected")
            except Exception as e:
                err_msg = str(e)
                logger.error(
                    f"run_graph: ❌ MCP server '{srv_id}' FAILED to connect: {err_msg}"
                )
                failed_servers.append((srv_id, err_msg))

        deps.mcp_toolsets = [ts for ts in connected_toolsets if ts is not None]

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
            f"run_graph: Starting graph execution for run_id {run_id}. Registered {len(deps.tag_prompts)} specialists."
        )
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

    if isinstance(result, GraphResponse):
        result.mermaid = mermaid_prefix if mermaid_prefix else None
        result.metadata.update({"run_id": run_id, "domain": state.routed_domain})
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

    Yields:
        SSE-formatted strings ('data: {JSON}\n\n') containing lifecycle events.

    """
    import asyncio
    from pathlib import Path
    from uuid import uuid4

    if run_id is None:
        run_id = uuid4().hex

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
            provider=config.get("provider", DEFAULT_PROVIDER),
            ssl_verify=_ssl_verify,
        ),
        agent_model=create_model(
            model_id=config.get("agent_model", DEFAULT_GRAPH_AGENT_MODEL),
            api_key=config.get("api_key"),
            base_url=config.get("base_url"),
            custom_headers=_custom_headers,
            provider=config.get("provider", DEFAULT_PROVIDER),
            ssl_verify=_ssl_verify,
        ),
        min_confidence=config.get("min_confidence", 0.6),
        sub_agents=config.get("sub_agents", {}),
        provider=config.get("provider", DEFAULT_PROVIDER),
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

    # Extract tag_prompts (domain specialists)
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
    from ..discovery import discover_agents

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
