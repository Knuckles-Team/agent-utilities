"""Agent Runner — CONCEPT:ORCH-1.21 KG-to-LLM Execution Bridge.

Bridges the ``graph_orchestrate(action='execute_agent')`` MCP tool to the
pydantic-graph execution infrastructure. Resolves the agent name against
the Knowledge Graph, materializes a graph with appropriate toolsets, and
executes it against the configured LLM provider (typically LM Studio via
OpenAI-compatible API).

This module provides deep KG integration rather than a simple passthrough:

1. **KG-Driven Agent Resolution**: Queries the KG for AgentTemplate,
   CallableResource, and Server nodes matching the requested agent name.
2. **Dynamic Tool Binding**: Discovers MCP servers registered for the
   agent and binds them as toolsets in the execution graph.
3. **Capability-Aware Routing**: Uses KG-stored capabilities to select
   the optimal graph topology (basic, dynamic, research).
4. **Provenance Tracking**: Logs execution results back to the KG as
   execution trace nodes for auditability.
5. **Fallback Strategies**: If KG resolution fails, falls back to
   workspace-based discovery via ``initialize_graph_from_workspace()``.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)


async def run_agent(
    agent_name: str,
    task: str,
    max_steps: int = 30,
    engine: IntelligenceGraphEngine | None = None,
    return_mermaid: bool = False,
    context: str | None = None,
    budget_tokens: int | None = None,
    context_ref: str | None = None,
    allowed_tools: list[str] | None = None,
) -> str:
    """Execute a named agent using the KG-backed pydantic-graph pipeline.

    CONCEPT:ORCH-1.21 — KG-to-LLM Execution Bridge

    This is the primary entry point for ``graph_orchestrate(action='execute_agent')``.
    It provides deep KG integration by:

    1. Resolving the agent against KG nodes (Server, CallableResource, AgentTemplate).
    2. Materializing a pydantic-graph with the agent's tool bindings.
    3. Executing against the configured LLM (LM Studio by default).
    4. Recording execution provenance in the KG.

    Args:
        agent_name: Name of the agent to execute (e.g., ``portainer-agent``).
            Matched against KG Server nodes, A2A agents, and skill nodes.
        task: The task description / user query to execute.
        max_steps: Maximum graph execution steps (guards against loops).
        engine: Optional pre-initialized IntelligenceGraphEngine instance.
            If not provided, one will be created from the environment.
        return_mermaid: CONCEPT:ORCH-1.37 — when True and the routed graph produced a
            Mermaid diagram, return a JSON string ``{"output", "mermaid"}`` instead of the
            bare output string. Default False preserves the bare-string contract relied on
            by internal callers (e.g. the dynamic-workflow fan-out in
            ``engine.execute_workflow``, which filters on ``isinstance(r, str)``).

    Returns:
        The synthesized result string from the graph execution, or (when
        ``return_mermaid`` is set and a diagram exists) a JSON string with ``output`` and
        ``mermaid`` keys.

    """
    run_id = f"run:{uuid.uuid4().hex[:8]}"
    start_time = time.monotonic()
    logger.info(
        "[ORCH-1.21] Starting agent execution: agent=%s, run_id=%s, task=%.100s...",
        agent_name,
        run_id,
        task,
    )

    # Step 1: Resolve engine
    engine = engine or _get_or_create_engine()

    if agent_name.lower() == "enterprise":
        from agent_utilities.graph.manifest_generators import manifest_for_enterprise
        from agent_utilities.graph.parallel_engine import ParallelEngine

        logger.info(
            "[ORCH-1.9] Executing full Enterprise Autonomous Company orchestration"
        )
        manifest = manifest_for_enterprise(task, engine)
        pe = ParallelEngine(engine=engine)

        try:
            pe_result = await pe.execute(manifest)
            duration_ms = (time.monotonic() - start_time) * 1000
            _record_execution_trace(
                engine,
                run_id,
                "enterprise",
                task,
                status="completed",
                duration_ms=duration_ms,
                result_preview=str(pe_result)[:500],
            )
            return str(pe_result)
        except Exception as e:
            logger.error("[ORCH-1.9] Enterprise execution failed: %s", e)
            _record_execution_trace(
                engine, run_id, "enterprise", task, status="failed", error=str(e)
            )
            return f"Enterprise execution failed: {e}"

    # Step 1b: Check if agent_name maps to a native ServiceRegistry capability (e.g. trading_swarm)
    try:
        import inspect
        import json

        from agent_utilities.core.registry.service_adapter import ServiceRegistry

        registry = ServiceRegistry.instance()
        svc = registry.get(agent_name)
        if svc:
            logger.info(
                "[ORCH-1.21] Routing to ServiceRegistry capability: %s", agent_name
            )
            cls = svc.get_class()
            if cls:
                # Instantiate capability
                sig = inspect.signature(cls)
                if "engine" in sig.parameters:
                    instance = cls(engine=engine)
                elif "config" in sig.parameters:
                    instance = cls(config=None)
                else:
                    instance = cls()

                # Execute capability
                if hasattr(instance, "analyze"):
                    # Specifically for TradingSwarm
                    try:
                        task_data = json.loads(task)
                    except Exception:
                        task_data = {"raw_task": task}

                    result = (
                        await instance.analyze(task_data)
                        if getattr(instance.analyze, "__iscoroutinefunction__", False)
                        else instance.analyze(task_data)
                    )
                    return str(result)
                elif hasattr(instance, "select_pattern"):
                    # Specifically for SubagentPatternRouter
                    result = (
                        await instance.select_pattern(needs_collaboration=True)
                        if getattr(
                            instance.select_pattern, "__iscoroutinefunction__", False
                        )
                        else instance.select_pattern(needs_collaboration=True)
                    )
                    return str(result)
                elif hasattr(instance, "run"):
                    result = (
                        await instance.run(task)
                        if getattr(instance.run, "__iscoroutinefunction__", False)
                        else instance.run(task)
                    )
                    return str(result)
                elif hasattr(instance, "execute"):
                    result = (
                        await instance.execute(task)
                        if getattr(instance.execute, "__iscoroutinefunction__", False)
                        else instance.execute(task)
                    )
                    return str(result)
    except Exception as e:
        logger.warning(
            "[ORCH-1.21] ServiceRegistry execution failed for %s, falling back: %s",
            agent_name,
            e,
        )

    # Step 2: Query KG for agent metadata
    agent_meta = _resolve_agent_from_kg(engine, agent_name)

    # Step 3: Build execution config from KG metadata
    config = _build_execution_config(engine, agent_name, agent_meta)
    # CONCEPT:ORCH-1.38 — carry the invoker's curated context + token budget into the spawn.
    # context_ref resolves a persisted ContextBlob (cross-process handoff): fetch its content
    # from the epistemic-graph and link it to this run's RunTrace for provenance.
    if context_ref and not context:
        try:
            _rows = engine.query_cypher(
                "MATCH (c:ContextBlob) WHERE c.id = $id RETURN c.content AS content",
                {"id": context_ref},
            )
            if _rows and _rows[0].get("content"):
                context = str(_rows[0]["content"])
        except Exception as _ctx_exc:  # noqa: BLE001
            logger.warning("context_ref %s resolution failed: %s", context_ref, _ctx_exc)
    if context:
        config["invoker_context"] = context
    if budget_tokens:
        config["invoker_budget_tokens"] = int(budget_tokens)
    if allowed_tools:
        config["invoker_allowed_tools"] = list(allowed_tools)

    # Step 4: Materialize and run the graph
    try:
        result = await _execute_graph(
            config=config,
            query=task,
            run_id=run_id,
            max_steps=max_steps,
            agent_meta=agent_meta,
        )
    except Exception as e:
        logger.error(
            "[ORCH-1.21] Agent execution failed: agent=%s, error=%s",
            agent_name,
            e,
        )
        # Record failure provenance
        _record_execution_trace(
            engine, run_id, agent_name, task, status="failed", error=str(e)
        )
        return f"Agent execution failed: {e}"

    # Step 5: Record success provenance
    duration_ms = (time.monotonic() - start_time) * 1000
    _record_execution_trace(
        engine,
        run_id,
        agent_name,
        task,
        status="completed",
        duration_ms=duration_ms,
        result_preview=str(result)[:500],
    )

    logger.info(
        "[ORCH-1.21] Agent execution complete: agent=%s, run_id=%s, duration=%.0fms",
        agent_name,
        run_id,
        duration_ms,
    )

    # Extract the output string from the GraphResponse
    if isinstance(result, dict):
        # GraphResponse.model_dump() shape
        results = result.get("results", {})
        output = results.get("output", "")
        if output:
            output_str = str(output)
        elif results:
            # Fallback to full results dict
            output_str = str(results)
        else:
            output_str = str(result)
        # CONCEPT:ORCH-1.37 — surface the routed-graph diagram when requested.
        mermaid = result.get("mermaid")
        if return_mermaid and mermaid:
            return json.dumps({"output": output_str, "mermaid": mermaid}, default=str)
        return output_str

    return str(result)


# ---------------------------------------------------------------------------
# Internal: KG Resolution
# ---------------------------------------------------------------------------


def _get_or_create_engine() -> IntelligenceGraphEngine:
    """Get the active engine or create one from environment config."""
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    # Check for singleton
    if IntelligenceGraphEngine._ACTIVE_ENGINE is not None:
        return IntelligenceGraphEngine._ACTIVE_ENGINE

    # Create from environment
    from agent_utilities.knowledge_graph.backends import create_backend

    db_path = os.environ.get("GRAPH_PERSISTENCE_PATH", "")
    backend = create_backend(db_path=db_path) if db_path else None

    engine = IntelligenceGraphEngine(backend=backend, db_path=db_path)
    return engine


def _resolve_agent_from_kg(
    engine: IntelligenceGraphEngine,
    agent_name: str,
) -> dict[str, Any]:
    """Query the KG for metadata about a named agent.

    Searches across Server nodes, CallableResource nodes, and AgentTemplate
    nodes to build a comprehensive capability profile.

    Returns:
        Dict with keys: ``type`` (server/skill/a2a/unknown), ``server_id``,
        ``tools`` (list of tool names), ``capabilities``, ``mcp_command``,
        ``url`` (for A2A agents), ``system_prompt``.

    """
    meta: dict[str, Any] = {
        "type": "unknown",
        "server_id": "",
        "tools": [],
        "capabilities": [],
        "mcp_command": "",
        "url": "",
        "system_prompt": "",
    }

    if not engine or not engine.backend:
        logger.warning("[ORCH-1.21] No KG backend — using empty agent metadata")
        return meta

    # --- Search 1: Server nodes (MCP servers) ---
    try:
        server_rows = engine.backend.execute(
            "MATCH (s:Server) WHERE s.name = $name OR s.id = $sid "
            "RETURN s.id AS sid, s.name AS name, s.url AS url, "
            "s.command AS cmd, s.args AS args, s.tool_count AS tc, s.env AS env",
            {"name": agent_name, "sid": f"srv:{agent_name}"},
        )
        if server_rows:
            row = server_rows[0]
            meta["type"] = "server"
            meta["server_id"] = row.get("sid", "")
            meta["mcp_command"] = row.get("cmd", "")
            meta["url"] = row.get("url", "")
            meta["env"] = row.get("env", "")

            # Fetch tools provided by this server
            tool_rows = engine.backend.execute(
                "MATCH (s:Server {id: $sid})-[:PROVIDES]->(r:CallableResource) "
                "RETURN r.name AS name, r.description AS description",
                {"sid": meta["server_id"]},
            )
            meta["tools"] = [
                {"name": r.get("name", ""), "description": r.get("description", "")}
                for r in tool_rows
            ]
            logger.info(
                "[ORCH-1.21] Resolved '%s' as MCP server with %d tools",
                agent_name,
                len(meta["tools"]),
            )
            return meta
    except Exception as e:
        logger.debug("Server lookup failed for '%s': %s", agent_name, e)

    # --- Search 2: CallableResource nodes (skills, A2A agents) ---
    try:
        resource_rows = engine.backend.execute(
            "MATCH (r:CallableResource) WHERE r.name = $name "
            "RETURN r.id AS rid, r.resource_type AS rtype, r.description AS description, "
            "r.skill_code_path AS skill_path",
            {"name": agent_name},
        )
        if resource_rows:
            row = resource_rows[0]
            rtype = row.get("rtype", "")
            if rtype == "A2A_AGENT":
                meta["type"] = "a2a"
                meta["url"] = row.get(
                    "description", ""
                )  # URL stored in description for A2A
            elif rtype == "AGENT_SKILL":
                meta["type"] = "skill"
            else:
                meta["type"] = "resource"
            logger.info(
                "[ORCH-1.21] Resolved '%s' as %s",
                agent_name,
                meta["type"],
            )
            return meta
    except Exception as e:
        logger.debug("Resource lookup failed for '%s': %s", agent_name, e)

    # --- Search 3: Hybrid semantic search ---
    try:
        results = engine.search_hybrid(agent_name, top_k=3)
        if results:
            best = results[0]
            meta["capabilities"] = [best.get("name", "")]
            logger.info(
                "[ORCH-1.21] Resolved '%s' via semantic search: %s",
                agent_name,
                best.get("name", ""),
            )
    except Exception as e:
        logger.debug("Semantic search failed for '%s': %s", agent_name, e)

    return meta


# ---------------------------------------------------------------------------
# Internal: Config Construction
# ---------------------------------------------------------------------------


def _build_execution_config(
    engine: IntelligenceGraphEngine,
    agent_name: str,
    agent_meta: dict[str, Any],
) -> dict[str, Any]:
    """Build a graph execution config dict from KG-resolved agent metadata.

    This produces the same config shape that ``create_graph_agent()`` and
    ``run_graph()`` expect, but tailored to the specific agent being executed.
    """
    from agent_utilities.core.config import (
        DEFAULT_GRAPH_ROUTER_TIMEOUT,
        DEFAULT_GRAPH_VERIFIER_TIMEOUT,
        DEFAULT_LITE_LLM_MODEL_ID,
        DEFAULT_LLM_API_KEY,
        DEFAULT_LLM_BASE_URL,
        DEFAULT_LLM_PROVIDER,
        DEFAULT_MIN_CONFIDENCE,
        DEFAULT_ROUTER_MODEL,
        DEFAULT_SSL_VERIFY,
    )

    # Tag prompts: the agent itself + any capabilities
    tag_prompts = {
        agent_name: f"Specialized agent: {agent_name}",
    }
    for cap in agent_meta.get("capabilities", []):
        if cap and cap != agent_name:
            tag_prompts[cap] = f"Capability: {cap}"

    # Fetch recent Mementos to build the sawtooth context
    try:
        from agent_utilities.knowledge_graph.memory import (
            get_recent_mementos,
        )

        recent_mementos = get_recent_mementos(engine, source=agent_name, limit=3)
        if recent_mementos:
            memento_text = "\n\n---\n\n".join(recent_mementos)
            tag_prompts[
                "mementos"
            ] = f"Past Context Mementos (Compressed State):\n{memento_text}"
    except Exception as e:
        logger.debug("Failed to fetch Mementos for context: %s", e)

    # Tool descriptions from KG
    for tool in agent_meta.get("tools", []):
        tool_name = tool.get("name", "")
        if tool_name:
            tag_prompts[tool_name] = tool.get("description", tool_name)

    config: dict[str, Any] = {
        "tag_prompts": tag_prompts,
        "tag_env_vars": {},
        "mcp_url": "",
        "mcp_config": "",
        "mcp_toolsets": [],
        "router_model": DEFAULT_ROUTER_MODEL,
        "agent_model": DEFAULT_LITE_LLM_MODEL_ID,
        "router_timeout": DEFAULT_GRAPH_ROUTER_TIMEOUT,
        "verifier_timeout": DEFAULT_GRAPH_VERIFIER_TIMEOUT,
        "min_confidence": DEFAULT_MIN_CONFIDENCE,
        "valid_domains": tuple(tag_prompts.keys()),
        "provider": DEFAULT_LLM_PROVIDER,
        "base_url": DEFAULT_LLM_BASE_URL,
        "api_key": DEFAULT_LLM_API_KEY,
        "ssl_verify": DEFAULT_SSL_VERIFY,
        "nodes": {},
        "sub_agents": {},
        "routing_strategy": "hybrid",
        "enable_llm_validation": False,
        "discovery_metadata": {},
    }

    # If the agent has MCP server URL/command, add it as a toolset
    if agent_meta.get("type") == "server" and agent_meta.get("url"):
        try:
            url = agent_meta["url"]
            if url.startswith("stdio://"):
                # Stdio-based MCP server — need to parse command
                from pydantic_ai.mcp import MCPServerStdio

                parts = url.replace("stdio://", "").split(" ", 1)
                command = parts[0]
                args = parts[1].split() if len(parts) > 1 else []

                env_dict = None
                env_str = agent_meta.get("env", "")
                if env_str and env_str != "{}":
                    import json

                    try:
                        env_dict = json.loads(env_str)
                    except Exception:  # nosec B110
                        pass

                if env_dict is not None:
                    import os

                    # Start with OS environment and ensure VIRTUAL_ENV is present if applicable
                    env_vars = os.environ.copy()

                    # Force VIRTUAL_ENV injection
                    venv_path = os.environ.get("VIRTUAL_ENV")
                    if venv_path:
                        env_vars["VIRTUAL_ENV"] = venv_path

                    # Update with specific tool environment variables
                    if env_dict:
                        env_vars.update(env_dict)

                    # Silence FastMCP startup output to prevent stdout pollution
                    env_vars["FASTMCP_SHOW_SERVER_BANNER"] = "false"
                    env_vars["FASTMCP_LOG_LEVEL"] = "WARNING"

                    # Prevent spawned MCP servers from acquiring a write lock on the knowledge graph
                    env_vars["LADYBUG_DB_READ_ONLY"] = "1"

                    # Ensure PATH is correct (uv uses PATH)
                    if "PATH" in os.environ:
                        env_vars["PATH"] = os.environ["PATH"]

                    server = MCPServerStdio(
                        command=command, args=args, env=env_vars, timeout=30.0
                    )
                    print(
                        f"DEBUG [agent_runner]: Created MCPServerStdio with command='{command}', args={args}, env_keys={list(env_vars.keys())}"
                    )
                    config["mcp_toolsets"].append(server)
                else:
                    import os

                    merged_env = os.environ.copy()
                    venv_path = os.environ.get("VIRTUAL_ENV")
                    if venv_path:
                        merged_env["VIRTUAL_ENV"] = venv_path
                    merged_env["FASTMCP_SHOW_SERVER_BANNER"] = "false"
                    merged_env["FASTMCP_LOG_LEVEL"] = "WARNING"
                    merged_env["LADYBUG_DB_READ_ONLY"] = "1"
                    for k in ["TERM", "COLORTERM", "FORCE_COLOR"]:
                        merged_env.pop(k, None)
                    config["mcp_toolsets"].append(
                        MCPServerStdio(
                            command=command, args=args, env=merged_env, timeout=30.0
                        )
                    )
            elif url.lower().endswith("/sse"):
                import httpx
                from pydantic_ai.mcp import MCPServerSSE

                config["mcp_toolsets"].append(
                    MCPServerSSE(
                        url,
                        http_client=httpx.AsyncClient(
                            verify=DEFAULT_SSL_VERIFY,
                            timeout=60,
                        ),
                    )
                )
            else:
                import httpx
                from pydantic_ai.mcp import MCPServerStreamableHTTP

                config["mcp_toolsets"].append(
                    MCPServerStreamableHTTP(
                        url,
                        http_client=httpx.AsyncClient(
                            verify=DEFAULT_SSL_VERIFY,
                            timeout=60,
                        ),
                    )
                )
        except Exception as e:
            logger.warning(
                "[ORCH-1.21] Failed to bind MCP toolset for '%s': %s", agent_name, e
            )

    return config


# ---------------------------------------------------------------------------
# Internal: Graph Execution
# ---------------------------------------------------------------------------


async def _execute_graph(
    config: dict[str, Any],
    query: str,
    run_id: str,
    max_steps: int,
    agent_meta: dict[str, Any],
) -> dict[str, Any]:
    """Materialize a pydantic-graph and execute it.

    Uses ``create_graph_agent()`` for graph construction and
    ``run_graph()`` for execution — the same pipeline used by
    the A2A agent and the main server.
    """
    from agent_utilities.graph.builder import create_graph_agent
    from agent_utilities.orchestration.engine import AgentOrchestrationEngine

    # Build graph from config
    graph, full_config = create_graph_agent(
        tag_prompts=config["tag_prompts"],
        tag_env_vars=config.get("tag_env_vars", {}),
        mcp_config=config.get("mcp_config"),
        mcp_url=config.get("mcp_url"),
        router_model=config.get("router_model"),
        agent_model=config.get("agent_model"),
        mcp_toolsets=config.get("mcp_toolsets"),
        routing_strategy=config.get("routing_strategy", "hybrid"),
        router_timeout=config.get("router_timeout"),
        verifier_timeout=config.get("verifier_timeout"),
    )

    # Merge any additional config keys
    full_config.update(
        {k: v for k, v in config.items() if k not in full_config and v is not None}
    )

    # Execute the graph
    # CONCEPT:ORCH-1.37 — streamdown=True populates GraphResponse.mermaid so the
    # routed-graph diagram can be surfaced to the MCP caller (see run_agent return_mermaid).
    result = await AgentOrchestrationEngine().execute_graph(
        graph=graph,
        config=full_config,
        query=query,
        run_id=run_id,
        persist=False,
        mode="ask",
        topology="basic",
        streamdown=True,
        mcp_toolsets=config.get("mcp_toolsets"),
    )

    return result


# ---------------------------------------------------------------------------
# Internal: Provenance Tracking
# ---------------------------------------------------------------------------


def _record_execution_trace(
    engine: IntelligenceGraphEngine,
    run_id: str,
    agent_name: str,
    task: str,
    status: str,
    error: str | None = None,
    duration_ms: float | None = None,
    result_preview: str | None = None,
) -> None:
    """Record an execution trace in the KG for auditability.

    CONCEPT:ORCH-1.21 — Execution provenance tracking.

    Creates a ``RunTrace`` node linked to the agent's Server/Resource node,
    enabling full audit trail of agent invocations.
    """
    if not engine:
        return

    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    trace_id = f"trace:{run_id}"

    props: dict[str, Any] = {
        "agent_name": agent_name,
        "task": task[:500],
        "status": status,
        "timestamp": ts,
    }
    if error:
        props["error"] = error[:500]
    if duration_ms is not None:
        props["duration_ms"] = round(duration_ms, 1)
    if result_preview:
        props["result_preview"] = result_preview[:500]

    try:
        engine.add_node(trace_id, "RunTrace", properties=props)

        # Link to the agent's server node if it exists
        server_id = f"srv:{agent_name}"
        if engine.backend:
            engine.backend.execute(
                "MATCH (s:Server {id: $sid}), (t:RunTrace {id: $tid}) "
                "MERGE (t)-[:EXECUTED_ON]->(s)",
                {"sid": server_id, "tid": trace_id},
            )
    except Exception as e:
        logger.debug("Failed to record execution trace: %s", e)
