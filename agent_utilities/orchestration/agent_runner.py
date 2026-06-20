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

import asyncio
import contextlib
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

from agent_utilities.core.config import setting

if TYPE_CHECKING:
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)

# CONCEPT:ORCH-1.72 — prompt-only universal entrypoints that must flow through the full
# multi-agent graph AS THEMSELVES, never resolved to a KG specialist. Resolving one is pure
# waste (a multi-second semantic search) and actively wrong (it mis-binds the universal
# messaging assistant to an unrelated tag). Keep this to genuine pass-through identities.
_PASSTHROUGH_AGENTS = frozenset({"messaging-assistant"})


def _flatten_exception_group(exc: BaseException) -> str:
    """Flatten a (possibly nested) ExceptionGroup into an actionable message.

    CONCEPT:ORCH-1.21 — when a remote MCP child fails, anyio wraps the real cause
    in a ``BaseExceptionGroup`` whose ``str()`` is the opaque "unhandled errors in
    a TaskGroup (N sub-exceptions)". This recursively collects the LEAF exceptions
    so the caller sees the actual error message(s) (and, where the leaf carries it,
    which child server/URL failed) instead of a sub-exception count.

    Returns a single ``"; "``-joined string of de-duplicated leaf messages. For a
    non-group exception it returns ``str(exc)`` unchanged.
    """
    leaves: list[str] = []
    seen: set[str] = set()

    def _walk(e: BaseException) -> None:
        if isinstance(e, BaseExceptionGroup):
            for sub in e.exceptions:
                _walk(sub)
            return
        # Prefer "<ExcType>: <msg>" so the failure kind is visible even when the
        # message is empty (e.g. a bare ConnectError).
        msg = str(e).strip()
        label = type(e).__name__
        rendered = f"{label}: {msg}" if msg else label
        if rendered not in seen:
            seen.add(rendered)
            leaves.append(rendered)

    _walk(exc)
    if not leaves:
        # No leaf errors (e.g. an empty group) — fall back to the group's own repr.
        return str(exc).strip() or type(exc).__name__
    return "; ".join(leaves)


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
    cred_ref: str | None = None,
    session_id: str | None = None,
    open_channel: bool = False,
    memento_source: str | None = None,
    execution_profile: str | None = None,
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

    # CONCEPT:ORCH-1.67 — construct the execution shape for THIS job ONCE, up front. The
    # escalating planner decides how much graph the job needs from cheap signals; a trivial
    # turn gets a lean shape that skips KG agent resolution, the usage-guard LLM round,
    # discovery, and the verifier (CONCEPT:ORCH-1.68), so the heavy apparatus never runs for a
    # simple chat reply.
    from agent_utilities.orchestration.execution_profile import plan_execution_shape

    shape = plan_execution_shape(task, profile_hint=execution_profile, engine=engine)

    # Step 2: Query KG for agent metadata — ONLY when the shape targets a specific specialist.
    # CONCEPT:ORCH-1.65 — ``_resolve_agent_from_kg`` runs synchronous backend round-trips;
    # run them OFF the event loop via ``to_thread`` so they never stall the async reply path.
    # CONCEPT:ORCH-1.68 — a direct-completion / generic chat turn does not target a named
    # specialist, so we skip the resolution entirely (it is a multi-second semantic-search
    # round-trip that mis-resolves a prompt-only agent like ``messaging-assistant`` anyway).
    # CONCEPT:ORCH-1.72 — and a PASS-THROUGH identity (the universal messaging assistant) is
    # resolution-exempt regardless of the shape: it is a prompt-only universal entrypoint that
    # is MEANT to flow through the full multi-agent graph as itself, and resolving it both
    # wastes a ~21 s semantic search and mis-binds it to an unrelated tag (``prepare_messages``).
    if shape.resolve_agent and agent_name.strip().lower() not in _PASSTHROUGH_AGENTS:
        agent_meta = await asyncio.to_thread(_resolve_agent_from_kg, engine, agent_name)
    else:
        agent_meta = _unresolved_agent_meta()

    # Step 2b: Prime the recent compressed mementos for this run OFF the event loop.
    # CONCEPT:KG-2.131 — read the per-session memento cache (zero I/O); only on a cold
    # miss do we fetch via ``to_thread`` so the synchronous backend round-trip never
    # blocks the async reply path (the priming used to run inline in
    # ``_build_execution_config``). The background ``_persist_and_enrich`` pass refreshes
    # the cache after each turn, so turn N+1 reads turn N's memento from memory.
    recent_mementos = await _prime_recent_mementos(engine, memento_source or agent_name)

    # Step 3: Build execution config from KG metadata.
    # CONCEPT:ORCH-1.62/1.67 — the constructed shape (already planned above) selects the
    # per-node timeout budget and the dynamic graph shape; pass it through so the config
    # carries it to the graph deps (ExecutionProfile instances are accepted as-is).
    config = _build_execution_config(
        engine,
        agent_name,
        agent_meta,
        memento_source=memento_source,
        execution_profile=shape,
        recent_mementos=recent_mementos,
    )
    # CONCEPT:ORCH-1.39 — carry the invoker's curated context + token budget into the spawn.
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
                # Provenance: link this run to the context it consumed (CONCEPT:ORCH-1.39).
                _add_edge = getattr(engine, "add_edge", None)
                if callable(_add_edge):
                    with contextlib.suppress(Exception):
                        _add_edge(f"trace:{run_id}", context_ref, "HAS_CONTEXT")
        except Exception as _ctx_exc:  # noqa: BLE001
            logger.warning(
                "context_ref %s resolution failed: %s", context_ref, _ctx_exc
            )
    if context:
        config["invoker_context"] = context
    if budget_tokens:
        config["invoker_budget_tokens"] = int(budget_tokens)
    if allowed_tools:
        config["invoker_allowed_tools"] = list(allowed_tools)
    if cred_ref:
        config["invoker_cred_ref"] = cred_ref
    # CONCEPT:ORCH-1.40 — open the invoker↔spawned native message channel for this run when
    # requested (or when an explicit session_id is given). The id is stamped into config so
    # GraphState/AgentDeps carry it to the spawned agent, and echoed back in the JSON wrapper
    # so the invoker knows where to send/receive.
    channel_id: str | None = None
    if open_channel or session_id:
        from agent_utilities.messaging import agent_channel

        channel_id = agent_channel.open_channel(engine, session_id or run_id, run_id)
        if channel_id:
            config["message_channel_id"] = channel_id

    # Step 4: Execute. A resolved single MCP-server agent runs a DETERMINISTIC
    # direct tool loop (bind only that server's toolset, no router); anything else
    # goes through the full multi-agent orchestration graph. Routing a one-server
    # task through the graph let the LLM router/dispatcher mis-route it (e.g. to a
    # verifier that ran on empty results), so the server's tools were never called.
    try:
        if getattr(shape, "tool_servers", ()):
            # CONCEPT:ORCH-1.74 — FOCUSED-TOOLS altitude: the lexical gate named concrete fleet
            # server(s), so bind exactly those toolsets and run ONE direct agent loop (parallel
            # tool calls) instead of the planning graph, which over-decomposes a named-tool ask
            # into a multi-step plan + expert fan-out. A failure (e.g. a server unreachable)
            # falls through to the full graph rather than erroring the turn.
            try:
                result = await _execute_focused_tools(
                    task=task,
                    shape=shape,
                    config=config,
                    agent_name=agent_name,
                    max_steps=max_steps,
                )
            except Exception as e:  # noqa: BLE001 — degrade to the graph, never drop the turn
                logger.warning(
                    "[ORCH-1.74] focused-tools path failed (%s); falling through to the full graph.",
                    _flatten_exception_group(e),
                )
                result = await _execute_graph(
                    config=config,
                    query=task,
                    run_id=run_id,
                    max_steps=max_steps,
                    agent_meta=agent_meta,
                )
        elif _is_single_server_agent(agent_meta, config):
            result = await _execute_single_server(
                config=config,
                task=task,
                max_steps=max_steps,
                agent_meta=agent_meta,
                agent_name=agent_name,
            )
        else:
            result = await _execute_graph(
                config=config,
                query=task,
                run_id=run_id,
                max_steps=max_steps,
                agent_meta=agent_meta,
            )
    except BaseException as e:  # noqa: BLE001 — see _flatten_exception_group
        # A remote MCP child (streamable-http/sse) that fails to connect or errors
        # mid-call surfaces through anyio as a BaseExceptionGroup ("unhandled errors
        # in a TaskGroup (1 sub-exception)") — an opaque message that hides WHICH
        # child failed and WHY. BaseExceptionGroup is a BaseException (not always an
        # Exception), so we catch BaseException and flatten the group to surface the
        # real underlying error(s); a bare KeyboardInterrupt/SystemExit (no leaf
        # errors) is re-raised untouched.
        # A cooperative cancellation — e.g. an outer ``asyncio.wait_for`` wall-clock
        # timeout in ``_run_agent_bounded`` — MUST propagate so the timeout surfaces
        # as a clean "timed out" result, not be flattened into "Agent execution
        # failed: CancelledError". CancelledError is a bare BaseException here (not a
        # group), so re-raise it before the flatten path.
        if isinstance(e, asyncio.CancelledError):
            raise
        if isinstance(e, KeyboardInterrupt | SystemExit) and not isinstance(
            e, BaseExceptionGroup
        ):
            raise
        err_msg = _flatten_exception_group(e)
        logger.error(
            "[ORCH-1.21] Agent execution failed: agent=%s, error=%s",
            agent_name,
            err_msg,
        )
        # Record failure provenance
        _record_execution_trace(
            engine, run_id, agent_name, task, status="failed", error=err_msg
        )
        # ARPO read-back (CONCEPT:AHE-3.15): failed runs carry step credit too
        # (a correct step in a failed trajectory must not be penalized).
        _write_step_credit(engine, run_id, agent_name, None, success=False)
        # CONCEPT:ORCH-1.70/1.71 — fold the failure back into the planner: evict this job's
        # cached recipe AND teach the shape policy (this archetype failed for this task-class).
        from agent_utilities.orchestration.execution_profile import record_shape_outcome

        record_shape_outcome(
            task,
            execution_profile,
            success=False,
            latency_s=time.monotonic() - start_time,
            shape=shape,
        )
        return f"Agent execution failed: {err_msg}"

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
    # ARPO read-back (CONCEPT:AHE-3.15): credit the intermediate agent-steps of
    # this run into the capability reward-EMA so routing learns from the steps,
    # not only the final answer. Guarded — never breaks the run path.
    _write_step_credit(engine, run_id, agent_name, result, success=True)
    # CONCEPT:ORCH-1.71 — teach the shape policy this archetype SUCCEEDED for this task-class,
    # rewarded by speed (success × how little of the budget it spent).
    from agent_utilities.orchestration.execution_profile import record_shape_outcome

    record_shape_outcome(
        task,
        execution_profile,
        success=True,
        latency_s=duration_ms / 1000.0,
        shape=shape,
    )
    # CONCEPT:ORCH-1.40 — anchor this run to its Session (id-addressable) so "list runs by
    # session" is a reliable single-hop traversal, mirroring HAS_CONTEXT/HAS_MESSAGE.
    if session_id:
        snode = f"session:{session_id}"
        with contextlib.suppress(Exception):
            engine.add_node(
                snode, "Session", properties={"id": snode, "session_id": session_id}
            )
            engine.add_edge(snode, f"trace:{run_id}", "HAS_RUN")

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
        # CONCEPT:ORCH-1.40 — surface the message channel id when one was opened.
        mermaid = result.get("mermaid")
        if (return_mermaid and mermaid) or channel_id:
            wrapper: dict[str, Any] = {"output": output_str}
            if return_mermaid and mermaid:
                wrapper["mermaid"] = mermaid
            if channel_id:
                wrapper["channel_id"] = channel_id
            return json.dumps(wrapper, default=str)
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

    db_path = setting("GRAPH_PERSISTENCE_PATH", "")
    backend = create_backend(db_path=db_path) if db_path else None

    engine = IntelligenceGraphEngine(backend=backend, db_path=db_path)
    return engine


def _unresolved_agent_meta() -> dict[str, Any]:
    """The empty agent-metadata shape used when KG resolution is skipped (CONCEPT:ORCH-1.68).

    A direct-completion / generic chat turn does not target a named specialist, so we skip the
    (multi-second) semantic-search resolution and run with this neutral metadata. It matches
    the default :func:`_resolve_agent_from_kg` returns on a miss, so every downstream consumer
    (``_build_execution_config``, ``_is_single_server_agent``) behaves identically to a miss.
    """
    return {
        "type": "unknown",
        "server_id": "",
        "tools": [],
        "capabilities": [],
        "mcp_command": "",
        "url": "",
        "system_prompt": "",
    }


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


def _spawn_auth_headers() -> dict[str, str]:
    """Outbound service-account auth for a spawned agent's REMOTE MCP toolsets.

    CONCEPT:ORCH-1.21 / OS-5.32 — a spawned agent that binds a jwt-protected
    fleet server (``*.arpa``) over SSE/streamable-HTTP must carry the same
    service-account bearer the multiplexer attaches to its children, or the
    call is rejected ``401`` (the toolset connected unauthenticated). Reuses the
    one minting path (``client_credentials.bearer_header``): opt-in via
    ``MCP_CLIENT_AUTH=oidc-client-credentials``, an inert ``{}`` otherwise, and
    a mint failure degrades to no header (unchanged behaviour when disabled).
    """
    try:
        from agent_utilities.mcp.client_credentials import bearer_header

        return bearer_header(None)
    except Exception:  # noqa: BLE001 — auth is best-effort; never block a spawn
        return {}


async def _prime_recent_mementos(
    engine: IntelligenceGraphEngine,
    source: str,
    limit: int = 3,
) -> list[str]:
    """Return the recent compressed mementos for ``source`` WITHOUT blocking the loop.

    CONCEPT:KG-2.131 — reads the per-session memento cache first (zero I/O). The cache
    is refreshed by the background ``_persist_and_enrich`` pass after each turn, so the
    common case (turn N+1 of a live conversation) is a pure in-memory read. On a cold
    miss we fetch once via ``to_thread`` (off the event loop) and populate the cache, so
    even the first turn never stalls the async reply path on a synchronous backend query.
    """
    try:
        from agent_utilities.knowledge_graph.memory.memento_compressor import (
            get_recent_mementos,
        )
        from agent_utilities.knowledge_graph.memory.session_memento_cache import (
            SessionMementoCache,
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("Memento priming unavailable: %s", e)
        return []

    cache = SessionMementoCache.instance()
    cached = cache.get(source)
    if cached is not None:
        return cached

    try:
        mementos = await asyncio.to_thread(get_recent_mementos, engine, source, limit)
    except Exception as e:  # noqa: BLE001
        logger.debug("Failed to prime Mementos for %s: %s", source, e)
        return []
    cache.put(source, mementos)
    return mementos


def _build_execution_config(
    engine: IntelligenceGraphEngine,
    agent_name: str,
    agent_meta: dict[str, Any],
    memento_source: str | None = None,
    execution_profile: str | None = None,
    recent_mementos: list[str] | None = None,
) -> dict[str, Any]:
    """Build a graph execution config dict from KG-resolved agent metadata.

    This produces the same config shape that ``create_graph_agent()`` and
    ``run_graph()`` expect, but tailored to the specific agent being executed.

    CONCEPT:ECO-4.78 — ``memento_source`` selects WHICH stream of compressed
    mementos primes the run's context. It defaults to ``agent_name`` (an agent's
    own past runs), but a session-scoped caller (e.g. a chat channel) passes its
    session key so successive turns share continuity through the core memory: the
    prior turns of THAT conversation are recalled as mementos, not via a bespoke
    per-surface history query.

    CONCEPT:KG-2.131 — ``recent_mementos`` is the already-primed memento list (read
    off the event loop by :func:`_prime_recent_mementos`). When ``None`` (the legacy
    direct callers / tests), we fall back to a synchronous fetch here so the function
    stays self-contained, but the hot reply path always passes the primed list so no
    blocking backend round-trip runs on the loop.

    CONCEPT:ORCH-1.62 — ``execution_profile`` ("chat" vs default "task") selects the
    per-node timeout budget: the chat profile bounds router/verifier to tens of seconds
    so a degraded backend fails fast inside the chat budget rather than at 300 s.
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
    from agent_utilities.orchestration.execution_profile import (
        resolve_execution_profile,
    )

    profile = resolve_execution_profile(execution_profile)

    # Tag prompts: the agent itself + any capabilities
    tag_prompts = {
        agent_name: f"Specialized agent: {agent_name}",
    }
    for cap in agent_meta.get("capabilities", []):
        if cap and cap != agent_name:
            tag_prompts[cap] = f"Capability: {cap}"

    # Prime recent Mementos into the sawtooth context. The hot reply path supplies them
    # already (read off the loop, CONCEPT:KG-2.131); only a direct caller that passed
    # nothing falls back to a synchronous fetch here.
    if recent_mementos is None:
        try:
            from agent_utilities.knowledge_graph.memory import get_recent_mementos

            recent_mementos = get_recent_mementos(
                engine, source=memento_source or agent_name, limit=3
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("Failed to fetch Mementos for context: %s", e)
            recent_mementos = []
    if recent_mementos:
        memento_text = "\n\n---\n\n".join(recent_mementos)
        tag_prompts[
            "mementos"
        ] = f"Past Context Mementos (Compressed State):\n{memento_text}"

    # Tool descriptions from KG
    for tool in agent_meta.get("tools", []):
        tool_name = tool.get("name", "")
        if tool_name:
            tag_prompts[tool_name] = tool.get("description", tool_name)

    # CONCEPT:ORCH-1.62 — chat profile bounds node timeouts to the chat budget; the task
    # profile keeps the long defaults.
    router_timeout = (
        profile.router_timeout
        if profile.router_timeout is not None
        else DEFAULT_GRAPH_ROUTER_TIMEOUT
    )
    verifier_timeout = (
        profile.verifier_timeout
        if profile.verifier_timeout is not None
        else DEFAULT_GRAPH_VERIFIER_TIMEOUT
    )

    config: dict[str, Any] = {
        "tag_prompts": tag_prompts,
        "tag_env_vars": {},
        "mcp_url": "",
        "mcp_config": "",
        "mcp_toolsets": [],
        "router_model": DEFAULT_ROUTER_MODEL,
        "agent_model": DEFAULT_LITE_LLM_MODEL_ID,
        "router_timeout": router_timeout,
        "verifier_timeout": verifier_timeout,
        "execution_profile": profile.name,
        # CONCEPT:ORCH-1.67/1.68 — carry the constructed shape to the graph deps so each node
        # can decide whether to run its work or pass through for this job.
        "execution_shape": profile,
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
                    venv_path = setting("VIRTUAL_ENV")
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
                        env_vars["PATH"] = setting("PATH")

                    server = MCPServerStdio(
                        command=command, args=args, env=env_vars, timeout=30.0
                    )
                    logger.debug(
                        "[agent_runner] Created MCPServerStdio command=%s args=%s",
                        command,
                        args,
                    )
                    config["mcp_toolsets"].append(server)
                else:
                    import os

                    merged_env = os.environ.copy()
                    venv_path = setting("VIRTUAL_ENV")
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
                            headers=_spawn_auth_headers() or None,
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
                            headers=_spawn_auth_headers() or None,
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


def _is_single_server_agent(agent_meta: dict[str, Any], config: dict[str, Any]) -> bool:
    """True when the resolved agent is exactly one MCP server with a bound toolset.

    Such an agent is eligible for the deterministic direct-execution path: it has a
    concrete toolset to call, so there is nothing for the multi-agent router to plan.
    """
    return bool(agent_meta.get("type") == "server" and config.get("mcp_toolsets"))


async def _execute_single_server(
    config: dict[str, Any],
    task: str,
    max_steps: int,
    agent_meta: dict[str, Any],
    agent_name: str,
) -> dict[str, Any]:
    """Run a single-MCP-server agent directly against its bound toolset.

    CONCEPT:ORCH-1.21 — a named MCP-server agent must actually USE that server's
    tools. Sending a one-server task through the full orchestration graph let the
    LLM router/dispatcher mis-route it (e.g. to a verifier that runs on empty
    results) so the tool was never invoked. This binds ONLY the resolved server's
    toolset (server-granular least privilege), applies the ORCH-1.39 invoker
    controls threaded onto ``config`` (allowed_tools / budget_tokens / context),
    and runs a direct agent loop — deterministic tool use, no LLM-router dependency.
    Returns the GraphResponse-compatible ``{"results": {"output": ...}}`` shape.
    """
    from agent_utilities.agent.factory import create_agent

    toolsets = list(config.get("mcp_toolsets") or [])

    # ORCH-1.39 least-privilege: restrict to the invoker's tool allow-list.
    # The filtered toolset MUST reach the agent as a real callable toolset (it is
    # passed through to ``create_agent(mcp_toolsets=...)`` → ``Agent(toolsets=...)``,
    # not merely described in the prompt). A ``.filtered()`` failure must NOT be
    # swallowed into an agent with zero bound tools that then hallucinates a tool
    # call — fail loudly instead (CONCEPT:ORCH-1.39).
    allowed = config.get("invoker_allowed_tools")
    if allowed:
        allow_set = {str(t).strip() for t in allowed if str(t).strip()}
        filtered: list[Any] = []
        for ts in toolsets:
            _filter = getattr(ts, "filtered", None)
            if not callable(_filter):
                raise RuntimeError(
                    f"toolset {type(ts).__name__!r} does not support tool filtering; "
                    f"cannot enforce allowed_tools={sorted(allow_set)} for agent "
                    f"'{agent_name}'"
                )
            filtered.append(_filter(lambda _ctx, td, _a=allow_set: td.name in _a))
        toolsets = filtered

    # An agent resolved as a single MCP server but left with no toolset would have
    # nothing to call and would fabricate tool calls. Surface that clearly rather
    # than producing a zero-tool agent (CONCEPT:ORCH-1.21).
    if not toolsets:
        raise RuntimeError(
            f"agent '{agent_name}' resolved to a single MCP server but has no bound "
            f"toolset to invoke"
            + (
                f" (allowed_tools={sorted({str(t).strip() for t in allowed if str(t).strip()})})"
                if allowed
                else ""
            )
        )

    system_prompt = agent_meta.get("system_prompt") or (
        f"You are the '{agent_name}' agent with direct access to its MCP server "
        "tools. Choose and call the appropriate tool(s) to fulfil the user's "
        "request, then return a concise, direct answer grounded in the tool results."
    )
    agent, _initialized = create_agent(
        provider=config.get("provider"),
        model_id=config.get("agent_model"),
        base_url=config.get("base_url"),
        api_key=config.get("api_key"),
        ssl_verify=config.get("ssl_verify", True),
        mcp_toolsets=toolsets,
        enable_skills=False,
        enable_universal_tools=False,
        name=agent_name,
        system_prompt=system_prompt,
    )

    prompt = task
    ctx_blob = config.get("invoker_context")
    if ctx_blob:
        prompt = f"Context:\n{ctx_blob}\n\nTask:\n{task}"

    run_kwargs: dict[str, Any] = {"message_history": []}
    limit_kwargs: dict[str, Any] = {}
    budget = config.get("invoker_budget_tokens")
    if budget:
        limit_kwargs["total_tokens_limit"] = int(budget)
    # max_steps bounds model round-trips; keep headroom for tool call/response turns.
    if max_steps:
        limit_kwargs["request_limit"] = max(int(max_steps) * 2, 10)
    if limit_kwargs:
        from pydantic_ai.usage import UsageLimits

        run_kwargs["usage_limits"] = UsageLimits(**limit_kwargs)

    result = await agent.run(prompt, **run_kwargs)
    output = (
        getattr(result, "output", None)
        if getattr(result, "output", None) is not None
        else getattr(result, "data", None) or getattr(result, "content", None) or result
    )
    return {"results": {"output": str(output)}}


# Generic suffixes stripped to show the product name in the focused-tools prompt.
_FLEET_PRODUCT_SUFFIXES = ("-mcp", "_mcp", "-agent", "_agent", "-api", "_api")


def _fleet_product(server: str) -> str:
    """Human product name for a fleet server (``portainer-mcp`` → ``portainer``)."""
    s = (server or "").strip()
    for suf in _FLEET_PRODUCT_SUFFIXES:
        if s.endswith(suf):
            return s[: -len(suf)]
    return s


def _fleet_server_url(server: str) -> str:
    """Served MCP URL for a fleet server. Homelab convention is
    ``http://<server>.<domain>/mcp`` (streamable-http); the domain is overridable for
    other deployments via ``FLEET_MCP_DOMAIN`` (CONCEPT:ORCH-1.74)."""
    domain = (setting("FLEET_MCP_DOMAIN", "arpa") or "arpa").strip().strip(".")
    return f"http://{server}.{domain}/mcp"


def _focused_tools_prompt(servers: list[str], config: dict[str, Any]) -> str:
    """System prompt for the focused-tools agent: keep any conversational persona the
    config carries, name the bound capabilities, and BIAS toward parallel tool calls
    (CONCEPT:ORCH-1.74)."""
    products = ", ".join(_fleet_product(s) for s in servers) or "the bound"
    persona = str(config.get("system_prompt") or "").strip()
    directive = (
        f"You have direct access to the {products} tools. Call the appropriate tool(s) to "
        "fulfil the user's request. When the request involves SEVERAL independent tools or "
        "services, call them IN PARALLEL — emit all the independent tool calls together in a "
        "single step rather than one after another — then give a concise, natural, friendly "
        "answer grounded in the tool results."
    )
    return f"{persona}\n\n{directive}".strip() if persona else directive


async def _execute_focused_tools(
    *,
    task: str,
    shape: Any,
    config: dict[str, Any],
    agent_name: str,
    max_steps: int,
) -> dict[str, Any]:
    """FOCUSED-TOOLS altitude (CONCEPT:ORCH-1.74): the ontology lexical gate named concrete
    fleet server(s), so bind ONLY those servers' toolsets (least privilege) and run ONE direct
    agent loop — no planner, no usage_guard / memory_selection / expert fan-out / verifier. The
    agent is biased to call independent tools in parallel; ActionPolicy still governs each call.
    Reuses :func:`_execute_single_server` (which binds a LIST of toolsets) for the loop itself.
    """
    import httpx
    from pydantic_ai.mcp import MCPServerSSE, MCPServerStreamableHTTP

    from agent_utilities.core.config import DEFAULT_SSL_VERIFY

    servers = [s for s in (getattr(shape, "tool_servers", ()) or ()) if s]
    if not servers:
        raise RuntimeError("focused-tools shape carried no servers")

    toolsets: list[Any] = []
    for srv in servers:
        url = _fleet_server_url(srv)
        client = httpx.AsyncClient(
            verify=DEFAULT_SSL_VERIFY,
            timeout=60,
            headers=_spawn_auth_headers() or None,
        )
        toolsets.append(
            MCPServerSSE(url, http_client=client)
            if url.lower().endswith("/sse")
            else MCPServerStreamableHTTP(url, http_client=client)
        )

    focused_config = dict(config)
    focused_config["mcp_toolsets"] = toolsets
    agent_meta = {
        "type": "server",
        "system_prompt": _focused_tools_prompt(servers, config),
    }
    logger.info(
        "[ORCH-1.74] focused-tools: binding %d server(s) %s for a direct parallel tool loop",
        len(servers),
        servers,
    )
    return await _execute_single_server(
        config=focused_config,
        task=task,
        max_steps=max_steps,
        agent_meta=agent_meta,
        agent_name=agent_name,
    )


async def _run_direct_completion(query: str, shape: Any) -> dict[str, Any]:
    """Answer a lean turn with ONE local-model round, OUTSIDE the multi-agent graph
    (CONCEPT:ORCH-1.68). A ``direct_complete`` shape must NOT enter the graph: a functional
    router step cannot terminate the graph mid-flow without an extra edge that pydantic-graph
    turns into a BROADCAST FORK (router → {end, dispatcher}), which silently killed every
    full-graph / tool task. So the lean answer is produced here and the graph is reserved for
    real multi-step work. Reasoning is off by default (fast); the model/timeout come from the
    shape. Returns a GraphResponse-shaped dict.
    """
    from pydantic_ai import Agent, ModelSettings

    from agent_utilities.core.config import DEFAULT_EXTRA_BODY
    from agent_utilities.core.model_factory import create_model

    model_id = getattr(shape, "model_id", None) if shape is not None else None
    reason_on = (
        bool(getattr(shape, "enable_reasoning", False)) if shape is not None else False
    )
    budget = getattr(shape, "router_timeout", None) if shape is not None else None
    extra = dict(DEFAULT_EXTRA_BODY or {})
    ctk = dict(extra.get("chat_template_kwargs") or {})
    ctk["enable_thinking"] = reason_on
    extra["chat_template_kwargs"] = ctk
    agent = Agent(
        model=create_model(model_id=model_id),
        system_prompt="You are a helpful assistant. Respond naturally and concisely.",
        model_settings=ModelSettings(
            extra_body=extra, max_tokens=1024, timeout=budget or 30.0
        ),
    )
    res = await agent.run(query)
    return {
        "status": "completed",
        "results": {"output": str(res.output)},
        "metadata": {"direct_complete": True, "domain": "conversational"},
    }


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

    # CONCEPT:ORCH-1.68 — a direct-completion shape answers with one lean local-model round and
    # NEVER enters the multi-agent graph (see _run_direct_completion: the in-graph router
    # variant created a broadcast fork that broke full-graph tool tasks). Decide once, here; a
    # genuine failure falls through to the full graph.
    _shape = config.get("execution_shape")
    _direct = (
        bool(getattr(_shape, "direct_complete", False)) if _shape is not None else False
    )
    if _shape is None:
        from agent_utilities.graph.routing.strategies.fast_path import is_trivial_query

        _direct = is_trivial_query(query)
    if _direct:
        try:
            return await _run_direct_completion(query, _shape)
        except Exception as e:  # noqa: BLE001 — a failed lean answer falls through to the graph
            logger.warning(
                "[ORCH-1.68] direct completion failed (%s); falling through to the graph.",
                e,
            )

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
    # CONCEPT:ORCH-1.37 — Orchestration execution-flow mermaid-diagram surfacing in graph_orchestrate responses
    # streamdown=True populates GraphResponse.mermaid so the routed-graph diagram can be
    # surfaced to the MCP caller (see run_agent return_mermaid).
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


def _stamp_run_identity(props: dict[str, Any]) -> None:
    """Add tenant_id/actor_id + correlation_id to an audit record in place.

    Best-effort: identity and correlation are ambient context, so any failure
    (no actor in scope, observability not wired) leaves the record unstamped
    rather than failing the write.
    """
    try:
        from agent_utilities.security.brain_context import current_actor

        actor = current_actor()
        if actor.actor_id:
            props.setdefault("actor_id", actor.actor_id)
        if actor.tenant_id:
            props.setdefault("tenant_id", actor.tenant_id)
    except Exception as exc:  # pragma: no cover - identity best-effort
        logger.debug("run identity stamp skipped: %s", exc)
    try:
        from agent_utilities.observability.correlation import get_correlation_id

        cid = get_correlation_id()
        if cid:
            props.setdefault("correlation_id", cid)
    except Exception as exc:  # pragma: no cover - correlation best-effort
        logger.debug("correlation stamp skipped: %s", exc)


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

    # Stamp the originating identity + correlation so the audit trail answers
    # "which tenant/actor ran this, and which agents share its run?" as a
    # tenant-scoped graph query (CONCEPT:OS-5.11 + OS-5.14 + KG-2.60).
    _stamp_run_identity(props)

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


# ---------------------------------------------------------------------------
# Internal: ARPO step-credit read-back (CONCEPT:AHE-3.15)
# ---------------------------------------------------------------------------

# Bookkeeping keys in GraphResponse.results that are not per-step outputs.
_NON_STEP_RESULT_KEYS = {"output", "mermaid", "usage", "error", "metadata", "status"}


def _extract_steps(
    result: Any, agent_name: str, success: bool
) -> tuple[list[dict[str, Any]], list[str | None]]:
    """Derive ARPO (step, agent-id) pairs from a GraphResponse-shaped result.

    Each non-bookkeeping key in ``results`` is one completed agent/specialist
    step (the executor stores per-node outputs under the node name); a truthy
    output counts as a locally-successful step. When no per-step structure is
    available, the whole run collapses to a single step credited to the
    invoked agent.
    """
    steps: list[dict[str, Any]] = []
    agent_ids: list[str | None] = []
    results = result.get("results") if isinstance(result, dict) else None
    if isinstance(results, dict):
        for key, value in results.items():
            if not isinstance(key, str) or key in _NON_STEP_RESULT_KEYS:
                continue
            steps.append({"action": key, "success": bool(value)})
            agent_ids.append(key)
    if not steps:
        steps = [{"action": agent_name, "success": success}]
        agent_ids = [agent_name]
    return steps, agent_ids


def _write_step_credit(
    engine: IntelligenceGraphEngine | None,
    run_id: str,
    agent_name: str,
    result: Any,
    success: bool,
) -> int:
    """Write ARPO per-step advantages into the capability reward-EMA.

    CONCEPT:AHE-3.15 — this is the read-back half of agent-step policy
    optimization: :func:`write_back_step_credit` existed but was never invoked
    from the live step lifecycle, so routing only ever learned from final
    answers. Called on every run completion (success AND failure); guarded so
    a credit failure can never break the step path (log-and-continue).
    Returns the number of steps credited (0 when no capability index exists).
    """
    try:
        kg = getattr(engine, "knowledge_graph", None) or getattr(engine, "kg", None)
        capability_index = getattr(kg, "retrieval", None) if kg is not None else None
        if capability_index is None:
            return 0

        from agent_utilities.graph.agent_step_po import write_back_step_credit
        from agent_utilities.graph.reward_decomposition import RewardDecomposer

        steps, agent_ids = _extract_steps(result, agent_name, success)
        decomposer = RewardDecomposer()
        record = decomposer.decompose(run_id, steps, goal_achieved=success)
        advantages = decomposer.step_advantages(record)
        # Group-normalization centers a uniform trajectory at 0 (neutral 0.5
        # reward); shift by the centered trajectory outcome so the final
        # result still moves the EMA even for single-step runs.
        outcome_shift = record.total_reward - 0.5
        advantages = [a + outcome_shift for a in advantages]
        written = write_back_step_credit(capability_index, agent_ids, advantages)
        if written:
            logger.debug(
                "[AHE-3.15] run %s: %d agent-step credits written", run_id, written
            )
        return written
    except Exception as e:  # noqa: BLE001 — credit must never break the run
        logger.debug("step-credit write-back skipped for %s: %s", run_id, e)
        return 0
