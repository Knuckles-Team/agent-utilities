"""Agent Runner — CONCEPT:AU-ORCH.routing.mcp-child-error-unwrap KG-to-LLM Execution Bridge.

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
    from agent_utilities.orchestration.execution_profile import ExecutionProfile

logger = logging.getLogger(__name__)

# CONCEPT:AU-ORCH.execution.passthrough-identity — prompt-only universal entrypoints that must flow through the full
# multi-agent graph AS THEMSELVES, never resolved to a KG specialist. Resolving one is pure
# waste (a multi-second semantic search) and actively wrong (it mis-binds the universal
# messaging assistant to an unrelated tag). Keep this to genuine pass-through identities.
_PASSTHROUGH_AGENTS = frozenset({"messaging-assistant"})


def _flatten_exception_group(exc: BaseException) -> str:
    """Flatten a (possibly nested) ExceptionGroup into an actionable message.

    CONCEPT:AU-ORCH.routing.mcp-child-error-unwrap — when a remote MCP child fails, anyio wraps the real cause
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

    CONCEPT:AU-ORCH.routing.mcp-child-error-unwrap — KG-to-LLM Execution Bridge

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
        return_mermaid: CONCEPT:AU-ORCH.execution.orchestration-flow-mermaid — when True and the routed graph produced a
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

    # CONCEPT:AU-ORCH.execution.per-job-shape-construction — construct the execution shape for THIS job ONCE, up front. The
    # escalating planner decides how much graph the job needs from cheap signals; a trivial
    # turn gets a lean shape that skips KG agent resolution, the usage-guard LLM round,
    # discovery, and the verifier (CONCEPT:AU-ORCH.execution.direct-completion-shape), so the heavy apparatus never runs for a
    # simple chat reply.
    from agent_utilities.orchestration.execution_profile import plan_execution_shape

    shape = plan_execution_shape(task, profile_hint=execution_profile, engine=engine)

    # Step 2: Query KG for agent metadata — ONLY when the shape targets a specific specialist.
    # CONCEPT:AU-ORCH.routing.offload-sync-roundtrip — ``_resolve_agent_from_kg`` runs synchronous backend round-trips;
    # run them OFF the event loop via ``to_thread`` so they never stall the async reply path.
    # CONCEPT:AU-ORCH.execution.direct-completion-shape — a direct-completion / generic chat turn does not target a named
    # specialist, so we skip the resolution entirely (it is a multi-second semantic-search
    # round-trip that mis-resolves a prompt-only agent like ``messaging-assistant`` anyway).
    # CONCEPT:AU-ORCH.execution.passthrough-identity — and a PASS-THROUGH identity (the universal messaging assistant) is
    # resolution-exempt regardless of the shape: it is a prompt-only universal entrypoint that
    # is MEANT to flow through the full multi-agent graph as itself, and resolving it both
    # wastes a ~21 s semantic search and mis-binds it to an unrelated tag (``prepare_messages``).
    if shape.resolve_agent and agent_name.strip().lower() not in _PASSTHROUGH_AGENTS:
        agent_meta = await asyncio.to_thread(_resolve_agent_from_kg, engine, agent_name)
    else:
        agent_meta = _unresolved_agent_meta()

    # Step 2b: Prime the recent compressed mementos for this run OFF the event loop.
    # CONCEPT:AU-KG.memory.refresh-per-session-memento — read the per-session memento cache (zero I/O); only on a cold
    # miss do we fetch via ``to_thread`` so the synchronous backend round-trip never
    # blocks the async reply path (the priming used to run inline in
    # ``_build_execution_config``). The background ``_persist_and_enrich`` pass refreshes
    # the cache after each turn, so turn N+1 reads turn N's memento from memory.
    recent_mementos = await _prime_recent_mementos(engine, memento_source or agent_name)

    # Step 2c: Prime the KG's synthesized view of the task's code area (CONCEPT:AU-KG.retrieval.task-start-kg-priming)
    # — the task-start "query the code KG before you grep" default. Off the loop,
    # best-effort, skipped on the chat profile.
    code_context_prime = await _prime_code_context(
        engine, task, execution_profile=execution_profile
    )

    # Step 3: Build execution config from KG metadata.
    # CONCEPT:AU-ORCH.execution.chat-profile-timeouts/1.67 — the constructed shape (already planned above) selects the
    # per-node timeout budget and the dynamic graph shape; pass it through so the config
    # carries it to the graph deps (ExecutionProfile instances are accepted as-is).
    config = _build_execution_config(
        engine,
        agent_name,
        agent_meta,
        memento_source=memento_source,
        execution_profile=shape,
        recent_mementos=recent_mementos,
        code_context_prime=code_context_prime,
    )
    # CONCEPT:AU-ORCH.session.carry-invoker — carry the invoker's curated context + token budget into the spawn.
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
                # Provenance: link this run to the context it consumed (CONCEPT:AU-ORCH.session.carry-invoker).
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
    # CONCEPT:AU-ORCH.execution.task-aware-tool-selection — a resolved fleet server can expose HUNDREDS
    # of tools; binding every schema to the single-server agent makes the LLM call hang
    # and the run silently degrade to a hallucinating toolless graph. When the caller
    # set no explicit allow-list, bind only the top-K task-relevant tools (KG capability
    # index, bounded; lexical fallback; hard cap). Only for resolved MCP servers.
    if agent_meta.get("type") == "server" and not config.get("invoker_allowed_tools"):
        _selected = await _select_relevant_tool_names(
            engine, task, agent_meta.get("tools") or [], agent_name=agent_name
        )
        if _selected:
            config["invoker_allowed_tools"] = _selected
    # CONCEPT:AU-ORCH.session.session-anchored-collections-native — open the invoker↔spawned native message channel for this run when
    # requested (or when an explicit session_id is given). The id is stamped into config so
    # GraphState/AgentDeps carry it to the spawned agent, and echoed back in the JSON wrapper
    # so the invoker knows where to send/receive.
    channel_id: str | None = None
    if open_channel or session_id:
        from agent_utilities.messaging import agent_channel

        channel_id = agent_channel.open_channel(engine, session_id or run_id, run_id)
        if channel_id:
            config["message_channel_id"] = channel_id

    # CONCEPT:AU-ORCH.execution.skill-utilization-provenance — capture whether a package SKILL drove
    # this run (its SOP is the prompt) and which server's tools it bound (F7), so the
    # RunTrace records skill utilization: bare skill (prompt-only) has type=="skill";
    # a skill bound to its server (F7) carries ``skill_of_server``.
    _skill_used = (
        agent_name
        if (agent_meta.get("type") == "skill" or agent_meta.get("skill_of_server"))
        else ""
    )
    _bound_server = str(agent_meta.get("skill_of_server", "") or "")
    _skill_id = str(agent_meta.get("skill_id", "") or "")

    # Step 4: Execute. A resolved single MCP-server agent runs a DETERMINISTIC
    # direct tool loop (bind only that server's toolset, no router); anything else
    # goes through the full multi-agent orchestration graph. Routing a one-server
    # task through the graph let the LLM router/dispatcher mis-route it (e.g. to a
    # verifier that ran on empty results), so the server's tools were never called.
    try:
        if _is_bound_template_agent(agent_meta, config):
            # CONCEPT:AU-ORCH.adapter.transport-toolset-factory — a KG-bound persona (e.g. agent-utilities-expert)
            # runs a DIRECT grounding loop: its recovered persona prompt drives the
            # run and its now-bound toolsets (graph-os + the fleet) let it query the
            # KG and ground the answer, instead of the prompt-only run that
            # hallucinated. Takes precedence over the generic focused-tools lexical
            # gate because the template DECLARES its own toolsets. A failure falls
            # through to the full graph (never drops the turn).
            try:
                result = await _execute_single_server(
                    config=config,
                    task=task,
                    max_steps=max_steps,
                    agent_meta=agent_meta,
                    agent_name=agent_name,
                )
            except Exception as e:  # noqa: BLE001 — degrade to the graph, never drop the turn
                logger.warning(
                    "[ORCH-1.101] bound-template path failed (%s); falling through to the full graph.",
                    _flatten_exception_group(e),
                )
                result = await _execute_graph(
                    config=config,
                    query=task,
                    run_id=run_id,
                    max_steps=max_steps,
                    agent_meta=agent_meta,
                )
        elif getattr(shape, "tool_servers", ()):
            # CONCEPT:AU-ORCH.execution.focused-tools-altitude — FOCUSED-TOOLS altitude: the lexical gate named concrete fleet
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
            except Exception as e:  # noqa: BLE001
                err = _flatten_exception_group(e)
                if agent_meta.get("type") == "server":
                    # CONCEPT:AU-ORCH.execution.no-silent-hallucination — a delegation that named a SPECIFIC
                    # fleet server must NOT fall through to a toolless graph that fabricates a
                    # plausible answer (the failure this program was built to catch). Surface it
                    # as degraded so it is traced + fed back, never recorded as a clean success.
                    logger.warning(
                        "[ORCH-1.74] focused-tools path failed for fleet server '%s' (%s); "
                        "surfacing degraded instead of hallucinating via the graph.",
                        agent_name,
                        err,
                    )
                    result = _fleet_server_failed_result(agent_name, err)
                else:
                    logger.warning(
                        "[ORCH-1.74] focused-tools path failed (%s); falling through to the full graph.",
                        err,
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
            engine,
            run_id,
            agent_name,
            task,
            status="failed",
            error=err_msg,
            skill_used=_skill_used,
            bound_server=_bound_server,
            skill_id=_skill_id,
        )
        # ARPO read-back (CONCEPT:AU-AHE.reward.this-is-read-back): failed runs carry step credit too
        # (a correct step in a failed trajectory must not be penalized).
        _write_step_credit(engine, run_id, agent_name, None, success=False)
        # CONCEPT:AU-ORCH.execution.planner-failure-feedback/1.71 — fold the failure back into the planner: evict this job's
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

    # Step 5: Record provenance. A delegation that fell through to the graph's "no data"
    # sentinel (or returned an empty answer) is a DEGRADED outcome, not a success —
    # returning a confident-empty "completed" is the failure this fixes. Detect it so the
    # RunTrace status is truthful, the reward/shape learning is not poisoned by a
    # non-answer, and the failure is fed back so routing self-corrects next time
    # (CONCEPT:AU-ORCH.execution.degraded-no-data-outcome; F2/F5).
    degraded = _delegation_degraded(result)
    duration_ms = (time.monotonic() - start_time) * 1000
    _record_execution_trace(
        engine,
        run_id,
        agent_name,
        task,
        status="degraded" if degraded else "completed",
        duration_ms=duration_ms,
        result_preview=str(result)[:500],
        error="delegation produced no usable data (degraded)" if degraded else None,
        skill_used=_skill_used,
        bound_server=_bound_server,
        skill_id=_skill_id,
    )
    # CONCEPT:AU-KG.temporal.message-history-read — persist each tool call the local LLM made as a :ToolCall
    # node on this run's RunTrace, so the delegated action is fully visible over
    # graph-os ("what tools, what args, what result"). Best-effort, never breaks.
    if isinstance(result, dict) and result.get("tool_calls"):
        _persist_tool_calls(
            engine, run_id, agent_name, agent_name, result["tool_calls"]
        )
    # Self-healing (CONCEPT:AU-AHE.evaluation.action-outcome-feedback): a degraded run teaches the
    # reward-EMA that this agent/task-class produced a non-answer, so routing prefers
    # actions that actually achieve the goal. Best-effort; never breaks the run.
    if degraded:
        _record_degraded_feedback(engine, agent_name, task, result)
    # ARPO read-back (CONCEPT:AU-AHE.reward.this-is-read-back): credit the intermediate agent-steps of
    # this run into the capability reward-EMA so routing learns from the steps,
    # not only the final answer. Guarded — never breaks the run path.
    _write_step_credit(engine, run_id, agent_name, result, success=not degraded)
    # CONCEPT:AU-ORCH.execution.shape-policy-learning — teach the shape policy whether this archetype
    # SUCCEEDED for this task-class, rewarded by speed (success × how little of the budget it spent).
    from agent_utilities.orchestration.execution_profile import record_shape_outcome

    record_shape_outcome(
        task,
        execution_profile,
        success=not degraded,
        latency_s=duration_ms / 1000.0,
        shape=shape,
    )
    # CONCEPT:AU-ORCH.session.session-anchored-collections-native — anchor this run to its Session (id-addressable) so "list runs by
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
        # CONCEPT:AU-ORCH.execution.orchestration-flow-mermaid — surface the routed-graph diagram when requested.
        # CONCEPT:AU-ORCH.session.session-anchored-collections-native — surface the message channel id when one was opened.
        # CONCEPT:AU-ORCH.execution.rich-result-wrapper — when the caller opts into the rich wrapper
        # (``return_mermaid``, the MCP execute_agent path), ALWAYS surface the
        # ``run_id`` so a delegation is trackable — the handle to query this run's
        # RunTrace + :ToolCall provenance (KG-2.296) over graph-os, and the
        # prerequisite for async/streaming/steering later. Internal callers
        # (``return_mermaid=False``) keep the bare-string contract bit-for-bit.
        mermaid = result.get("mermaid")
        if return_mermaid or channel_id:
            wrapper: dict[str, Any] = {"output": output_str, "run_id": run_id}
            if return_mermaid and mermaid:
                wrapper["mermaid"] = mermaid
            if channel_id:
                wrapper["channel_id"] = channel_id
            return json.dumps(wrapper, default=str)
        return output_str

    if return_mermaid or channel_id:
        wrapper2: dict[str, Any] = {"output": str(result), "run_id": run_id}
        if channel_id:
            wrapper2["channel_id"] = channel_id
        return json.dumps(wrapper2, default=str)
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
    """The empty agent-metadata shape used when KG resolution is skipped (CONCEPT:AU-ORCH.execution.direct-completion-shape).

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


def _hydrate_skill_runnable(
    engine: IntelligenceGraphEngine,
    meta: dict[str, Any],
    *,
    skill_id: str,
    name: str,
    description: str,
    system_prompt: str,
    skill_code_path: str,
    source_is_resource: bool = True,
) -> None:
    """Populate ``meta`` so an ingested skill runs with its own instructions + tools.

    CONCEPT:AU-ORCH.dispatch.dispatch-half-skill-ingestion — sets ``meta['system_prompt']`` (the skill's instruction body)
    and ``meta['tools']`` (its declared ``USES_TOOL`` targets), then binds the skill
    back into a runnable ``AGENT_SKILL`` CallableResource (idempotent) via
    :func:`persist_skill_as_runnable` so the next resolution is a pure prop read and
    the capability fabric carries the skill→tool edges. Best-effort: a binding
    failure leaves the skill resolvable (it still runs, just un-densified).

    ``source_is_resource`` is True when the node is already a ``CallableResource``
    (the AGENT_SKILL branch) — the binding updates props on that same node. When
    False (a bare ``:Skill`` node), writing a CallableResource onto the ``:Skill``
    id would NOT relabel it, so the runnable resource is written under a distinct
    ``resource:<skill_id>`` and a ``BINDS_RUNNABLE`` edge links the skill to it.
    """
    body = (system_prompt or "").strip()
    # Cold AGENT_SKILL nodes carry only a code path — read the SKILL.md body once.
    if not body and skill_code_path:
        try:
            import os

            if os.path.isfile(skill_code_path):
                raw = open(  # noqa: SIM115
                    skill_code_path, encoding="utf-8", errors="replace"
                ).read()
                # Strip YAML frontmatter so the prompt is the instruction body.
                if raw.startswith("---"):
                    _, _, rest = raw.partition("\n---")
                    raw = rest.lstrip("\n-") or raw
                body = raw.strip()
        except Exception as exc:  # noqa: BLE001 — body read is best-effort
            logger.debug("[ORCH-1.96] skill body read failed for %s: %s", name, exc)

    # Declared tools the skill needs (USES_TOOL edges), if any were materialized.
    tools: list[str] = []
    try:
        rows = engine.backend.execute(
            "MATCH (r) WHERE r.id = $sid "
            "MATCH (r)-[:USES_TOOL]->(t) RETURN t.name AS name, t.id AS id",
            {"sid": skill_id},
        )
        tools = [
            str(r.get("name") or r.get("id"))
            for r in (rows or [])
            if (r.get("name") or r.get("id"))
        ]
    except Exception as exc:  # noqa: BLE001
        logger.debug("[ORCH-1.96] skill tool lookup failed for %s: %s", name, exc)

    if body:
        meta["system_prompt"] = (
            f"You are the '{name}' skill. Follow these instructions to fulfil the "
            f"user's request, calling any available tools as needed:\n\n{body}"
        )
    meta["tools"] = [{"name": t, "description": t} for t in tools]
    meta["skill_id"] = skill_id

    # Densify: bind the skill into a runnable CallableResource so it persists. An
    # already-CallableResource node is updated in place; a bare :Skill node gets a
    # distinct resource:<id> sibling + a BINDS_RUNNABLE edge (a bare node can't be
    # relabelled to CallableResource on the backend).
    if body or tools:
        resource_id = skill_id if source_is_resource else f"resource:{skill_id}"
        try:
            from agent_utilities.knowledge_graph.enrichment.execute import (
                persist_skill_as_runnable,
            )

            persist_skill_as_runnable(
                getattr(engine, "backend", engine),
                skill_id=resource_id,
                name=name,
                system_prompt=body,
                description=description,
                tools=tools,
            )
            if not source_is_resource:
                with contextlib.suppress(Exception):
                    engine.link_nodes(skill_id, resource_id, "BINDS_RUNNABLE")
        except Exception as exc:  # noqa: BLE001 — binding is best-effort
            logger.debug("[ORCH-1.96] skill→runnable bind failed for %s: %s", name, exc)


def _bind_skill_to_owning_server(
    engine: IntelligenceGraphEngine,
    meta: dict[str, Any],
    skill_code_path: str,
    agent_name: str,
) -> None:
    """Upgrade a package-bundled skill to a single-server agent bound to its server.

    CONCEPT:AU-ORCH.execution.skill-bound-server-tools — a skill whose code path is under
    ``.../agents/<pkg>/.../skills/...`` is authored to drive the ``<pkg>`` MCP server's
    tools. Resolved as a bare skill it runs prompt-only (no tools) and can only describe
    a task. This finds the owning ``Server`` node (trying ``<pkg>`` and ``<pkg>-mcp`` —
    the package dir and the deployed server name can differ, e.g. tunnel-manager →
    tunnel-manager-mcp), then sets ``type="server"`` + ``url`` + the server's ``tools``
    so the run routes single-server (F1 task-aware selection applies) while KEEPING the
    skill's instructions as the system prompt. Best-effort: a miss leaves the skill
    prompt-only (unchanged behaviour).
    """
    import re

    m = re.search(r"/agents/([^/]+)/", str(skill_code_path or ""))
    if not m or not getattr(engine, "backend", None):
        return
    pkg = m.group(1)
    candidates = [pkg] if pkg.endswith("-mcp") else [f"{pkg}-mcp", pkg]
    for name in candidates:
        try:
            rows = engine.backend.execute(
                "MATCH (s:Server) WHERE s.name = $name OR s.id = $sid "
                "RETURN s.id AS sid, s.name AS name, s.url AS url, s.env AS env",
                {"name": name, "sid": f"srv:{name}"},
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("[ORCH-skill-bind] server lookup failed for '%s': %s", name, e)
            continue
        if not rows or not rows[0].get("url"):
            continue
        srv = rows[0]
        skill_prompt = meta.get("system_prompt", "")
        meta["type"] = "server"
        meta["server_id"] = srv.get("sid", "")
        meta["url"] = srv.get("url", "")
        meta["env"] = srv.get("env", "")
        meta["skill_of_server"] = srv.get("name", "")
        try:
            trows = engine.backend.execute(
                "MATCH (s:Server {id: $sid})-[:PROVIDES]->(r:CallableResource) "
                "RETURN r.name AS name, r.description AS description",
                {"sid": meta["server_id"]},
            )
            meta["tools"] = [
                {"name": r.get("name", ""), "description": r.get("description", "")}
                for r in (trows or [])
            ]
        except Exception as e:  # noqa: BLE001
            logger.debug("[ORCH-skill-bind] tool fetch failed: %s", e)
        # The skill's SOP drives the run; the server's tools execute it.
        if skill_prompt:
            meta["system_prompt"] = skill_prompt
        logger.info(
            "[ORCH-skill-bind] skill '%s' bound to server '%s' (%d tools)",
            agent_name,
            srv.get("name", ""),
            len(meta.get("tools", [])),
        )
        return


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
            "r.skill_code_path AS skill_path, r.system_prompt AS system_prompt",
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
                # CONCEPT:AU-ORCH.dispatch.dispatch-half-skill-ingestion — make the ingested skill actually RUNNABLE, not
                # just resolvable: load its instruction body as the system prompt and
                # its declared tools so the spawned LLM is primed with the skill's
                # behaviour. ``ingest_agent_skill`` writes only a ``skill_code_path``,
                # so read the SKILL.md body on a cold node and bind it back onto the
                # resource (idempotent) so the next resolution is a pure prop read.
                meta["type"] = "skill"
                _hydrate_skill_runnable(
                    engine,
                    meta,
                    skill_id=row.get("rid", "") or f"skill:{agent_name}",
                    name=agent_name,
                    description=row.get("description", "") or "",
                    system_prompt=row.get("system_prompt", "") or "",
                    skill_code_path=row.get("skill_path", "") or "",
                )
                # CONCEPT:AU-ORCH.execution.skill-bound-server-tools — a PACKAGE-BUNDLED skill exists to
                # DRIVE its MCP server's tools; resolved as a bare skill it runs prompt-only
                # and can only DESCRIBE a task, never execute it. If the skill's code path
                # identifies an owning server, upgrade it to a single-server agent — bind
                # that server's toolset (task-selected via F1) and run the skill's
                # instructions AS the system prompt against real tools.
                _bind_skill_to_owning_server(
                    engine, meta, row.get("skill_path", "") or "", agent_name
                )
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

    # --- Search 2b: bare ``:Skill`` nodes (CONCEPT:AU-ORCH.dispatch.dispatch-half-skill-ingestion) ---------------
    # ``skill_workflow_ingest`` (KG-2.97) writes plain ``:Skill`` nodes as the
    # ``USES_SKILL`` targets of workflow steps; they are search corpus, not
    # CallableResources, so ``run_agent`` could never dispatch them. Bind such a
    # skill into a runnable ``AGENT_SKILL`` CallableResource on first resolution
    # (reusing ``persist_skill_as_runnable``) so an ingested skill name becomes a
    # real dispatch target — closing the "ingested ≠ executable" gap.
    try:
        skill_rows = engine.backend.execute(
            "MATCH (s:Skill) WHERE s.name = $name OR s.id = $sid "
            "RETURN s.id AS sid, s.name AS name, s.description AS description, "
            "s.body AS body, s.instruction AS instruction, s.content AS content",
            {"name": agent_name, "sid": f"skill:{agent_name}"},
        )
        if skill_rows:
            row = skill_rows[0]
            skill_id = row.get("sid", "") or f"skill:{agent_name}"
            body = (
                row.get("body")
                or row.get("instruction")
                or row.get("content")
                or row.get("description")
                or ""
            )
            meta["type"] = "skill"
            _hydrate_skill_runnable(
                engine,
                meta,
                skill_id=skill_id,
                name=row.get("name", agent_name) or agent_name,
                description=row.get("description", "") or "",
                system_prompt=str(body),
                skill_code_path="",
                source_is_resource=False,
            )
            logger.info(
                "[ORCH-1.96] Resolved '%s' as a runnable ingested skill", agent_name
            )
            return meta
    except Exception as e:
        logger.debug("Skill lookup failed for '%s': %s", agent_name, e)
    # --- Search 2b: AgentTemplate nodes (KG-bound dispatchable personas) ---
    # CONCEPT:AU-ORCH.dispatch.seeded-agent-template — a built-in/seeded AgentTemplate (e.g. the
    # ``agent-utilities-expert``) binds a system-prompt node + toolsets + model
    # preference. Resolve it by name, then recover the linked Prompt's body via
    # the USES_PROMPT edge so the persona actually drives the run. Toolset binding
    # of ``toolset_ids`` into live MCP toolsets is the run_agent execution seam.
    try:
        tmpl_rows = engine.backend.execute(
            "MATCH (at:AgentTemplate) WHERE at.name = $name OR at.id = $tid "
            "RETURN at.id AS tid, at.system_prompt_id AS spid, "
            "at.toolset_ids AS toolsets, at.model_preference AS model",
            {"name": agent_name, "tid": f"at:{agent_name}"},
        )
        if tmpl_rows:
            row = tmpl_rows[0]
            meta["type"] = "agent_template"
            meta["model_preference"] = row.get("model") or ""
            toolsets = row.get("toolsets") or []
            if isinstance(toolsets, str):
                with contextlib.suppress(Exception):
                    import json as _json

                    toolsets = _json.loads(toolsets)
            meta["capabilities"] = list(toolsets) if isinstance(toolsets, list) else []
            spid = row.get("spid") or ""
            if spid:
                with contextlib.suppress(Exception):
                    prow = engine.backend.execute(
                        "MATCH (p:Prompt) WHERE p.id = $pid "
                        "RETURN p.system_prompt AS body",
                        {"pid": spid},
                    )
                    if prow and prow[0].get("body"):
                        meta["system_prompt"] = str(prow[0]["body"])
            logger.info(
                "[ORCH-1.100] Resolved '%s' as AgentTemplate (%d toolset(s))",
                agent_name,
                len(meta["capabilities"]),
            )
            return meta
    except Exception as e:
        logger.debug("AgentTemplate lookup failed for '%s': %s", agent_name, e)

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

    CONCEPT:AU-ORCH.routing.mcp-child-error-unwrap / OS-5.32 — a spawned agent that binds a jwt-protected
    fleet server (``*.arpa``) over SSE/streamable-HTTP must carry the same
    service-account bearer the multiplexer attaches to its children, or the
    call is rejected ``401`` (the toolset connected unauthenticated). Reuses the
    one minting path (``client_credentials.child_auth_header``): opt-in via
    ``MCP_CLIENT_AUTH=oidc-client-credentials`` (Bearer) or ``basic`` (Basic), an
    inert ``{}`` otherwise, and a failure degrades to no header (unchanged
    behaviour when disabled).
    """
    try:
        from agent_utilities.mcp.client_credentials import child_auth_header

        return child_auth_header(None)
    except Exception:  # noqa: BLE001 — auth is best-effort; never block a spawn
        return {}


def _toolset_for_id(engine: IntelligenceGraphEngine, toolset_id: str) -> Any:
    """Resolve ONE AgentTemplate ``toolset_id`` to a live MCP toolset.

    CONCEPT:AU-ORCH.adapter.transport-toolset-factory — the binding seam that turns a KG-bound persona's declared
    toolsets into tools the local LLM can actually call. Resolution reuses the
    existing Server/mcp_config + fleet-URL machinery (no new binder, no new
    transport code):

    1. Prefer an explicit served ``url`` on a ``:Server`` node in the KG (the
       mcp_config-derived registry), when one exists.
    2. Otherwise fall back to the homelab fleet served-URL convention
       (``http://<id>.<domain>/mcp``, :func:`_fleet_server_url`) — the same
       resolution the FOCUSED-TOOLS path (ORCH-1.74) uses; the fleet's ~58
       ``*-mcp`` servers plus ``graph-os`` and the ``mcp-multiplexer`` are all
       served there.

    The toolset carries the OIDC service-account bearer (:func:`_spawn_auth_headers`)
    so jwt-protected ``*.arpa`` servers don't reject the call ``401``. Returns the
    bound ``MCPToolset`` (id-tagged for filtering), or ``None`` for an empty id.
    """
    tid = (toolset_id or "").strip()
    if not tid:
        return None

    from agent_utilities.core.config import DEFAULT_SSL_VERIFY
    from agent_utilities.mcp.toolset_factory import build_http_toolset

    url = ""
    # 1. Prefer an explicit served URL recorded on a Server node (mcp_config registry).
    try:
        if engine and getattr(engine, "backend", None):
            rows = engine.backend.execute(
                "MATCH (s:Server) WHERE s.name = $name OR s.id = $sid "
                "RETURN s.url AS url",
                {"name": tid, "sid": f"srv:{tid}"},
            )
            if rows:
                cand = str(rows[0].get("url") or "").strip()
                if cand and not cand.startswith("stdio://"):
                    url = cand
    except Exception as exc:  # noqa: BLE001 — registry lookup is best-effort
        logger.debug("[ORCH-1.101] Server-node URL lookup failed for %s: %s", tid, exc)

    # 2. Fleet served-URL convention (the focused-tools resolution).
    url = url or _fleet_server_url(tid)

    return build_http_toolset(
        url,
        headers=_spawn_auth_headers() or None,
        verify=DEFAULT_SSL_VERIFY,
        timeout=60,
        toolset_id=tid,
    )


def _resolve_toolset_ids(
    engine: IntelligenceGraphEngine, toolset_ids: list[str]
) -> list[Any]:
    """Bind an AgentTemplate's ``toolset_ids`` into a list of live MCP toolsets.

    CONCEPT:AU-ORCH.adapter.transport-toolset-factory — each id is resolved by :func:`_toolset_for_id`; a single
    id that fails to bind is skipped (logged) rather than dropping the whole set,
    so the persona still gets every toolset that DID resolve (e.g. ``graph-os``
    for grounding) even if one server is unreachable.
    """
    bound: list[Any] = []
    for tid in toolset_ids or []:
        try:
            ts = _toolset_for_id(engine, tid)
        except Exception as exc:  # noqa: BLE001 — one bad id must not drop the rest
            logger.warning("[ORCH-1.101] failed to bind toolset_id %r: %s", tid, exc)
            ts = None
        if ts is not None:
            bound.append(ts)
    return bound


async def _prime_recent_mementos(
    engine: IntelligenceGraphEngine,
    source: str,
    limit: int = 3,
) -> list[str]:
    """Return the recent compressed mementos for ``source`` WITHOUT blocking the loop.

    CONCEPT:AU-KG.memory.refresh-per-session-memento — reads the per-session memento cache first (zero I/O). The cache
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


async def _prime_code_context(
    engine: IntelligenceGraphEngine,
    task: str,
    *,
    execution_profile: str | None = None,
) -> str | None:
    """Prime the KG's synthesized view of the task's code area (CONCEPT:AU-KG.retrieval.task-start-kg-priming).

    The task-start half of "query the code KG before you grep": when a run's task
    references a code symbol/area, inject the ``code_context`` answer (definition +
    call chain + concept + citations) into the run's context the way mementos prime
    a chat turn — so the agent inherits how the area works instead of opening with
    grep. Best-effort and run off the event loop; skipped on the latency-sensitive
    chat profile and when no real ``:Code`` anchor matches.
    """
    if (execution_profile or "").strip().lower() == "chat":
        return None
    if not task or len(task) < 8:
        return None
    try:
        from agent_utilities.knowledge_graph.retrieval.code_context import (
            build_code_context,
        )

        result = await asyncio.to_thread(
            build_code_context, engine, query=task, intent="how", top_k=6
        )
    except Exception as e:  # noqa: BLE001 — priming is best-effort
        logger.debug("Failed to prime code_context: %s", e)
        return None
    if not result or not result.get("anchors"):
        return None
    answer = str(result.get("answer", "")).strip()
    cites = result.get("citations", [])[:6]
    cite_lines = "\n".join(
        f"- {c.get('symbol')} @ {c.get('file')}:{c.get('line')}" for c in cites
    )
    cap = result.get("capability_id", "")
    return (
        f"{answer}\n\nCited (read only what you must edit):\n{cite_lines}\n"
        f"[code_context capability_id={cap} — after the task, report reads_avoided "
        f"via graph_feedback]"
    )


def _build_execution_config(
    engine: IntelligenceGraphEngine,
    agent_name: str,
    agent_meta: dict[str, Any],
    memento_source: str | None = None,
    execution_profile: str | ExecutionProfile | None = None,
    recent_mementos: list[str] | None = None,
    code_context_prime: str | None = None,
) -> dict[str, Any]:
    """Build a graph execution config dict from KG-resolved agent metadata.

    This produces the same config shape that ``create_graph_agent()`` and
    ``run_graph()`` expect, but tailored to the specific agent being executed.

    CONCEPT:AU-ECO.messaging.universal-graph-agent — ``memento_source`` selects WHICH stream of compressed
    mementos primes the run's context. It defaults to ``agent_name`` (an agent's
    own past runs), but a session-scoped caller (e.g. a chat channel) passes its
    session key so successive turns share continuity through the core memory: the
    prior turns of THAT conversation are recalled as mementos, not via a bespoke
    per-surface history query.

    CONCEPT:AU-KG.memory.refresh-per-session-memento — ``recent_mementos`` is the already-primed memento list (read
    off the event loop by :func:`_prime_recent_mementos`). When ``None`` (the legacy
    direct callers / tests), we fall back to a synchronous fetch here so the function
    stays self-contained, but the hot reply path always passes the primed list so no
    blocking backend round-trip runs on the loop.

    CONCEPT:AU-ORCH.execution.chat-profile-timeouts — ``execution_profile`` ("chat" vs default "task") selects the
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

    # Tag prompts: the agent itself + any capabilities. CONCEPT:AU-ORCH.dispatch.seeded-agent-template — when
    # resolution recovered the agent's real system prompt (e.g. a seeded
    # AgentTemplate persona like ``agent-utilities-expert``), drive the run with
    # that full persona instead of the bare "Specialized agent" stub.
    resolved_prompt = str(agent_meta.get("system_prompt") or "").strip()
    tag_prompts = {
        agent_name: resolved_prompt or f"Specialized agent: {agent_name}",
    }
    for cap in agent_meta.get("capabilities", []):
        if cap and cap != agent_name:
            tag_prompts[cap] = f"Capability: {cap}"

    # Prime recent Mementos into the sawtooth context. The hot reply path supplies them
    # already (read off the loop, CONCEPT:AU-KG.memory.refresh-per-session-memento); only a direct caller that passed
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

    # CONCEPT:AU-KG.retrieval.task-start-kg-priming — prime the KG's synthesized view of the task's code area so the
    # run learns how it works (with file:line citations) before reaching for grep.
    if code_context_prime:
        tag_prompts["code_context"] = (
            "How this code area works (from the KG — read only the cited "
            f"file:line you must edit):\n{code_context_prime}"
        )

    # Tool descriptions from KG
    for tool in agent_meta.get("tools", []):
        tool_name = tool.get("name", "")
        if tool_name:
            tag_prompts[tool_name] = tool.get("description", tool_name)

    # CONCEPT:AU-ORCH.execution.chat-profile-timeouts — chat profile bounds node timeouts to the chat budget; the task
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
        # CONCEPT:AU-ORCH.execution.per-job-shape-construction/1.68 — carry the constructed shape to the graph deps so each node
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
                from agent_utilities.mcp.toolset_factory import build_stdio_toolset

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

                    server = build_stdio_toolset(command, args, env=env_vars)
                    logger.debug(
                        "[agent_runner] Created stdio MCPToolset command=%s args=%s",
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
                        build_stdio_toolset(command, args, env=merged_env)
                    )
            else:
                from agent_utilities.mcp.toolset_factory import build_http_toolset

                config["mcp_toolsets"].append(
                    build_http_toolset(
                        url,
                        headers=_spawn_auth_headers() or None,
                        verify=DEFAULT_SSL_VERIFY,
                        timeout=60,
                    )
                )
        except Exception as e:
            logger.warning(
                "[ORCH-1.21] Failed to bind MCP toolset for '%s': %s", agent_name, e
            )

    # CONCEPT:AU-ORCH.adapter.transport-toolset-factory — a KG-bound AgentTemplate (e.g. ``agent-utilities-expert``)
    # declares its toolsets as ``toolset_ids`` (surfaced as ``capabilities`` by
    # ``_resolve_agent_from_kg``). Resolution recovered the persona prompt but, until
    # now, NOT live tools: the binding above only fires for ``type=="server"`` agents
    # with a URL, so the template ran prompt-only and HALLUCINATED. Bind each declared
    # toolset into a live MCP toolset so the persona can actually query graph-os / the
    # fleet and GROUND its answers (query-the-KG-then-answer). Reuses the same
    # Server/fleet-URL resolution + toolset_factory — no new binder.
    if agent_meta.get("type") == "agent_template":
        bound = _resolve_toolset_ids(engine, agent_meta.get("capabilities", []))
        if bound:
            config["mcp_toolsets"].extend(bound)
            logger.info(
                "[ORCH-1.101] Bound %d toolset(s) for AgentTemplate '%s': %s",
                len(bound),
                agent_name,
                agent_meta.get("capabilities"),
            )
        else:
            logger.warning(
                "[ORCH-1.101] AgentTemplate '%s' declared toolsets %s but none bound — "
                "it will run prompt-only",
                agent_name,
                agent_meta.get("capabilities"),
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


def _is_bound_template_agent(
    agent_meta: dict[str, Any], config: dict[str, Any]
) -> bool:
    """True when a resolved AgentTemplate has its toolset_ids bound to live toolsets.

    CONCEPT:AU-ORCH.adapter.transport-toolset-factory — such a persona (e.g. ``agent-utilities-expert``) runs a
    DIRECT grounding loop (its persona prompt + its bound toolsets), not the
    planning graph: the multi-agent router would over-decompose the ask and the
    persona/tools would never drive a single query-then-answer turn. The bound
    toolsets are exactly what lets it query graph-os and stop hallucinating.
    """
    return bool(
        agent_meta.get("type") == "agent_template" and config.get("mcp_toolsets")
    )


# Above this many tools on one server, bind only the task-relevant subset. A fleet
# server can expose hundreds (container-manager-mcp: 314); handing every schema to one
# agent makes the LLM call hang and the run silently fall through to a hallucinating
# toolless graph. Kept as a module constant per Configuration discipline (one correct
# value, auto-behaviour — not a knob).
_MAX_BOUND_TOOLS = 20

# Wall-clock budget for a single-server direct tool loop. Generous enough for a
# legitimate multi-step, multi-tool run, but far below the MCP client timeout so a
# blocking tool fails loud in minutes instead of hanging for the full client budget
# (CONCEPT:AU-ORCH.execution.delegation-wall-clock). One correct value, not a knob.
_EXECUTE_AGENT_WALL_CLOCK_S = 300.0


def _fleet_server_failed_result(agent_name: str, error: str) -> dict[str, Any]:
    """A degraded result for a resolved fleet-server delegation that failed.

    CONCEPT:AU-ORCH.execution.no-silent-hallucination — returned INSTEAD of falling through to the
    toolless graph, so a named-server delegation that could not run its real tools
    surfaces the failure (picked up as ``degraded`` by ``_delegation_degraded`` →
    truthful RunTrace + negative feedback) rather than a confident fabrication.
    """
    return {
        "status": "failed",
        "results": {
            "output": (
                f"Delegation to fleet server '{agent_name}' could not produce a "
                f"tool-grounded result ({error}). Refusing to fall back to a general "
                f"answer, which would fabricate tool output."
            )
        },
        "metadata": {"degraded": True, "outcome": "fleet_server_failed"},
    }


def _lexical_top_k_tools(task: str, tools: list[dict[str, Any]], k: int) -> list[str]:
    """Fast, dependency-free relevance ranking of tool names against the task.

    Scores each tool by task-word overlap on its name (weighted) + description.
    Returns up to ``k`` names with a non-zero score, most relevant first; ``[]``
    when nothing matches (caller then hard-caps). No LLM/embedding round-trip, so
    it can never re-introduce the latency this whole mechanism exists to avoid.
    """
    import re

    words = {w for w in re.findall(r"[a-z0-9]{3,}", task.lower())}
    if not words:
        return []
    scored: list[tuple[int, str]] = []
    for t in tools:
        name = str(t.get("name") or "")
        if not name:
            continue
        nlow = name.lower()
        dlow = str(t.get("description") or "").lower()
        score = sum(1 for w in words if w in dlow) + 3 * sum(
            1 for w in words if w in nlow
        )
        if score:
            scored.append((score, name))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [n for _s, n in scored][:k]


def _match_designated_to_names(ranked_ids: list[str], name_set: set[str]) -> list[str]:
    """Map KG-designation resource ids back to this server's tool names, in order."""
    out: list[str] = []
    seen: set[str] = set()
    for rid in ranked_ids:
        rid_s = str(rid)
        base = rid_s.split(":")[-1].split("/")[-1].split("__")[-1]
        cand = rid_s if rid_s in name_set else (base if base in name_set else "")
        if not cand:
            cand = next((n for n in name_set if n and rid_s.endswith(n)), "")
        if cand and cand not in seen:
            seen.add(cand)
            out.append(cand)
    return out


async def _select_relevant_tool_names(
    engine: IntelligenceGraphEngine | None,
    task: str,
    tools: list[dict[str, Any]],
    *,
    agent_name: str = "",
    max_tools: int = _MAX_BOUND_TOOLS,
) -> list[str] | None:
    """Pick the top-K task-relevant tools when a server exposes too many.

    CONCEPT:AU-ORCH.execution.task-aware-tool-selection — returns ``None`` when the server is small
    enough to bind wholesale. Otherwise: a fast lexical ranker over tool name +
    description (top-K by task-word overlap), then a hard cap. Always yields a focused,
    callable set so the single-server agent runs fast instead of stalling on hundreds of
    schemas.

    NOTE: this deliberately does NOT call the KG capability index on the per-delegation
    hot path. That index is embedding-backed and builds on first use — a cold-start
    round-trip that can take tens of seconds, i.e. it re-introduces the exact stall this
    mechanism exists to prevent, and a thread-bounded timeout still orphans the slow
    build. Lexical selection is deterministic and sub-millisecond; a pre-warmed
    capability-index ranker is the right future enhancement, not a live blocking call.
    """
    names = [str(t.get("name")) for t in (tools or []) if t.get("name")]
    if len(names) <= max_tools:
        return None

    selected = _lexical_top_k_tools(task, tools, max_tools)
    if selected:
        logger.info(
            "[ORCH-tool-select] lexical chose %d/%d tools for '%s'",
            len(selected),
            len(names),
            agent_name,
        )
        return selected

    # Hard cap — nothing matched lexically, but never hand the agent hundreds of schemas.
    logger.info(
        "[ORCH-tool-select] hard-capped %d/%d tools for '%s'",
        max_tools,
        len(names),
        agent_name,
    )
    return names[:max_tools]


async def _execute_single_server(
    config: dict[str, Any],
    task: str,
    max_steps: int,
    agent_meta: dict[str, Any],
    agent_name: str,
) -> dict[str, Any]:
    """Run a single-MCP-server agent directly against its bound toolset.

    CONCEPT:AU-ORCH.routing.mcp-child-error-unwrap — a named MCP-server agent must actually USE that server's
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
    # call — fail loudly instead (CONCEPT:AU-ORCH.session.carry-invoker).
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
    # than producing a zero-tool agent (CONCEPT:AU-ORCH.routing.mcp-child-error-unwrap).
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

    # CONCEPT:AU-ORCH.execution.delegation-wall-clock — bound the direct tool loop with a wall-clock.
    # ``usage_limits`` caps model round-trips but NOT time: a fleet tool that blocks (e.g.
    # a systems-manager telemetry call shelling to a stuck host command) hangs the whole
    # delegation for the full client timeout (observed: 1800s) and piles engine
    # connections. A hung delegation is worse than a failed one — time out and raise so
    # the caller records it as a degraded/failed run (fail-loud), never an indefinite hang.
    try:
        result = await asyncio.wait_for(
            agent.run(prompt, **run_kwargs),
            timeout=_EXECUTE_AGENT_WALL_CLOCK_S,
        )
    except TimeoutError as exc:
        raise RuntimeError(
            f"single-server agent '{agent_name}' exceeded the "
            f"{_EXECUTE_AGENT_WALL_CLOCK_S:.0f}s wall-clock budget — a bound tool likely "
            f"blocked; failing loud instead of hanging"
        ) from exc
    output = (
        getattr(result, "output", None)
        if getattr(result, "output", None) is not None
        else getattr(result, "data", None) or getattr(result, "content", None) or result
    )
    # CONCEPT:AU-KG.temporal.message-history-read — carry the per-tool-call provenance up to run_agent, which
    # persists it as :ToolCall nodes on the run's RunTrace. This is the deterministic
    # MCP tool-loop, so it is exactly where real tool calls happen and are visible.
    return {
        "results": {"output": str(output)},
        "tool_calls": _extract_tool_calls(result),
    }


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
    other deployments via ``FLEET_MCP_DOMAIN`` (CONCEPT:AU-ORCH.execution.focused-tools-altitude)."""
    domain = (setting("FLEET_MCP_DOMAIN", "arpa") or "arpa").strip().strip(".")
    return f"http://{server}.{domain}/mcp"


def _focused_tools_prompt(servers: list[str], config: dict[str, Any]) -> str:
    """System prompt for the focused-tools agent: keep any conversational persona the
    config carries, name the bound capabilities, and BIAS toward parallel tool calls
    (CONCEPT:AU-ORCH.execution.focused-tools-altitude)."""
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
    """FOCUSED-TOOLS altitude (CONCEPT:AU-ORCH.execution.focused-tools-altitude): the ontology lexical gate named concrete
    fleet server(s), so bind ONLY those servers' toolsets (least privilege) and run ONE direct
    agent loop — no planner, no usage_guard / memory_selection / expert fan-out / verifier. The
    agent is biased to call independent tools in parallel; ActionPolicy still governs each call.
    Reuses :func:`_execute_single_server` (which binds a LIST of toolsets) for the loop itself.
    """
    from agent_utilities.core.config import DEFAULT_SSL_VERIFY
    from agent_utilities.mcp.toolset_factory import build_http_toolset

    servers = [s for s in (getattr(shape, "tool_servers", ()) or ()) if s]
    if not servers:
        raise RuntimeError("focused-tools shape carried no servers")

    toolsets: list[Any] = []
    for srv in servers:
        url = _fleet_server_url(srv)
        toolsets.append(
            build_http_toolset(
                url,
                headers=_spawn_auth_headers() or None,
                verify=DEFAULT_SSL_VERIFY,
                timeout=60,
            )
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
    (CONCEPT:AU-ORCH.execution.direct-completion-shape). A ``direct_complete`` shape must NOT enter the graph: a functional
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

    # CONCEPT:AU-ORCH.execution.direct-completion-shape — a direct-completion shape answers with one lean local-model round and
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
    # CONCEPT:AU-ORCH.execution.orchestration-flow-mermaid — Orchestration execution-flow mermaid-diagram surfacing in graph_orchestrate responses
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
    skill_used: str = "",
    bound_server: str = "",
    skill_id: str = "",
) -> None:
    """Record an execution trace in the KG for auditability.

    CONCEPT:AU-ORCH.routing.mcp-child-error-unwrap — Execution provenance tracking.

    Creates a ``RunTrace`` node linked to the agent's Server/Resource node,
    enabling full audit trail of agent invocations. When a package skill drove the
    run, ``skill_used``/``bound_server`` record which skill ran and which server's
    tools it bound, and a ``USES_SKILL`` edge is written (CONCEPT:AU-ORCH.execution.skill-utilization-provenance)
    so "which runs used skill X, and what tools did it drive" is a single traversal.
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
    if skill_used:
        props["skill_used"] = skill_used
    if bound_server:
        props["bound_server"] = bound_server

    # Stamp the originating identity + correlation so the audit trail answers
    # "which tenant/actor ran this, and which agents share its run?" as a
    # tenant-scoped graph query (CONCEPT:AU-OS.observability.run-wide-correlation-id + OS-5.14 + KG-2.60).
    _stamp_run_identity(props)

    try:
        engine.add_node(trace_id, "RunTrace", properties=props)

        if engine.backend:
            # EXECUTED_ON links to the actual server whose tools ran — the bound server
            # for a skill-driven run (agent_name is the skill, not a Server), else the
            # agent's own server node.
            server_name = bound_server or agent_name
            engine.backend.execute(
                "MATCH (s:Server {id: $sid}), (t:RunTrace {id: $tid}) "
                "MERGE (t)-[:EXECUTED_ON]->(s)",
                {"sid": f"srv:{server_name}", "tid": trace_id},
            )
            # Skill-utilization provenance: which skill's SOP drove this run. Match the
            # skill node by ID — the engine cannot resolve a node by a non-id property
            # (name) in a write, which silently dropped this edge; EXECUTED_ON matches by
            # id and works, so mirror it. Prefer the resolved skill_id; fall back to the
            # standard ``skill:<name>`` id.
            if skill_used:
                rid = skill_id or f"skill:{skill_used}"
                engine.backend.execute(
                    "MATCH (r:CallableResource {id: $rid}), (t:RunTrace {id: $tid}) "
                    "MERGE (t)-[:USES_SKILL]->(r)",
                    {"rid": rid, "tid": trace_id},
                )
    except Exception as e:
        logger.debug("Failed to record execution trace: %s", e)


# ---------------------------------------------------------------------------
# Internal: degraded-outcome detection + self-healing feedback
# (CONCEPT:AU-ORCH.execution.degraded-no-data-outcome / AU-AHE.evaluation.action-outcome-feedback)
# ---------------------------------------------------------------------------

_DELEGATION_DEGRADED_SENTINELS = (
    "unable to find specific data",
    "could not be generated",
)


# Markers of a tool RESULT that is actually an error report (the MCP tool returned an
# error string as normal content rather than raising) — used to score :ToolCall status
# and the all-tools-errored degradation signal (CONCEPT:AU-ORCH.execution.all-tool-calls-errored).
_TOOL_ERROR_MARKERS = (
    "error executing",
    "traceback (most recent",
    "has no attribute",
    "exception:",
    "failed:",
    "is not defined",
)


def _result_looks_like_error(text: str) -> bool:
    """True when a tool result string is an error report, not real data."""
    low = (text or "").strip().lower()
    if not low:
        return False
    return low.startswith("error") or any(m in low for m in _TOOL_ERROR_MARKERS)


def _tool_call_errored(tc: Any) -> bool:
    """True when a captured tool call failed (explicit error, or error-shaped result)."""
    if not isinstance(tc, dict):
        return False
    if tc.get("error"):
        return True
    return _result_looks_like_error(str(tc.get("result") or ""))


def _delegation_degraded(result: Any) -> bool:
    """True when a delegation produced a non-answer (no data / empty / sentinel / all tools errored).

    CONCEPT:AU-ORCH.execution.degraded-no-data-outcome — the trust-critical signal: a run that
    routed through the graph and gathered zero results returns a plausible-but-empty
    "…unable to find specific data…" sentinel that was previously recorded as
    ``status="completed"``. Reads the structured ``degraded`` flag the graph
    synthesizer stamps into ``GraphResponse.metadata``; also flags a run that DID call
    tools but every call errored (CONCEPT:AU-ORCH.execution.all-tool-calls-errored — no tool-grounded
    result); falls back to an output-text sentinel / empty-output check so the
    single-server and focused-tools paths are covered too. Never raises.
    """
    try:
        output = ""
        if isinstance(result, dict):
            meta = result.get("metadata")
            if isinstance(meta, dict) and meta.get("degraded"):
                return True
            # A run that called tools but every call errored produced no grounded
            # result (e.g. 13 k8s calls all 'has no attribute') — degraded, not success.
            tcs = result.get("tool_calls")
            if (
                isinstance(tcs, list)
                and tcs
                and all(_tool_call_errored(tc) for tc in tcs)
            ):
                return True
            res = result.get("results")
            if isinstance(res, dict):
                output = str(res.get("output") or "")
            if not output:
                output = str(result.get("output") or "")
        else:
            output = str(result or "")
        low = output.strip().lower()
        if not low:
            return True
        return any(s in low for s in _DELEGATION_DEGRADED_SENTINELS)
    except Exception:  # noqa: BLE001 — a detector must never break the run path
        return False


def _record_degraded_feedback(
    engine: IntelligenceGraphEngine | None,
    agent_name: str,
    task: str,
    result: Any,
) -> None:
    """Feed a degraded delegation back as a negative action-outcome.

    CONCEPT:AU-AHE.evaluation.action-outcome-feedback — the self-healing half of fail-loud: a run
    that produced no usable data records ``success=False`` on the ``agent:<name>``
    reward-EMA, so routing/optimization learns to prefer delegations that actually
    achieve the goal instead of silently repeating a non-answer. Best-effort.
    """
    if not engine:
        return
    try:
        from agent_utilities.knowledge_graph.adaptation.feedback import FeedbackService

        feedback = FeedbackService.from_engine(engine)
    except Exception:  # noqa: BLE001 — feedback is optional
        return
    if feedback is None:
        return
    output = ""
    if isinstance(result, dict):
        res = result.get("results")
        if isinstance(res, dict):
            output = str(res.get("output") or "")
    with contextlib.suppress(Exception):
        feedback.record_action_outcome(
            f"agent:{agent_name}",
            success=False,
            observed=output[:200],
            query=task[:200],
            reason="delegation_degraded_no_data",
            agent_id=agent_name,
        )


# ---------------------------------------------------------------------------
# Internal: per-tool-call provenance (CONCEPT:AU-KG.temporal.message-history-read)
# ---------------------------------------------------------------------------

# Per-tool-call provenance extraction lives in the shared leaf module
# (orchestration/tool_provenance.py) so the multi-agent graph executor can surface the
# SAME :ToolCall provenance as this direct loop without a circular import
# (CONCEPT:AU-KG.temporal.message-history-read).
from agent_utilities.orchestration.tool_provenance import (  # noqa: E402
    extract_tool_calls as _extract_tool_calls,
)
from agent_utilities.orchestration.tool_provenance import (  # noqa: E402
    sanitize_tool_args as _sanitize_tool_args,  # noqa: F401  (re-exported for callers/tests)
)


def _persist_tool_calls(
    engine: IntelligenceGraphEngine | None,
    run_id: str,
    agent_name: str,
    server: str,
    tool_calls: list[dict[str, Any]],
) -> int:
    """Persist each tool call as a ``:ToolCall`` node linked to the run's RunTrace.

    CONCEPT:AU-KG.temporal.message-history-read — the run-level RunTrace (ORCH-1.21) said *that* a delegation
    ran; this makes the individual tool calls first-class, queryable provenance so
    Claude can ask "what tools did the local LLM call, with what args, what result"
    over graph-os. Each call also feeds ``action_outcome`` (AHE-3.62) so the
    reward-EMA densifies on the tools that actually worked — visibility and learning
    from the same seam. Best-effort: a provenance write must never fail the run.
    """
    if not engine or not tool_calls:
        return 0
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    trace_id = f"trace:{run_id}"
    written = 0
    try:
        from agent_utilities.knowledge_graph.adaptation.feedback import FeedbackService

        feedback = FeedbackService.from_engine(engine)
    except Exception:  # noqa: BLE001 — reward is optional
        feedback = None
    for i, tc in enumerate(tool_calls):
        tc_id = f"toolcall:{run_id.split(':', 1)[-1]}:{i}"
        # A tool that returned an error STRING as normal content (the MCP tool caught its
        # own exception) has no explicit ``error`` but is still a failure — score it as
        # such so provenance queries can filter real failures (AU-ORCH.execution.all-tool-calls-errored).
        ok = not _tool_call_errored(tc)
        props: dict[str, Any] = {
            "run_id": run_id,
            "agent_name": agent_name,
            "server": server,
            "tool_name": tc.get("tool_name", ""),
            "args": tc.get("args", ""),
            "result_preview": tc.get("result", "")[:2000],
            "error": tc.get("error", "")[:500],
            "status": "ok" if ok else "error",
            "sequence": i,
            "timestamp": ts,
        }
        _stamp_run_identity(props)
        try:
            engine.add_node(tc_id, "ToolCall", properties=props)
            # link_nodes writes backend-FIRST (durable), unlike add_edge's
            # best-effort compute-cache path — so the provenance edge survives in
            # the epistemic-graph for graph-os traversal queries.
            engine.link_nodes(trace_id, tc_id, "MADE_TOOL_CALL")
            written += 1
        except Exception as exc:  # noqa: BLE001
            logger.debug("[KG-2.296] ToolCall persist failed (%s): %s", tc_id, exc)
            continue
        if feedback is not None and tc.get("tool_name"):
            try:
                feedback.record_action_outcome(
                    f"tool:{tc['tool_name']}",
                    success=ok,
                    observed=tc.get("result", "")[:200],
                    reason="tool_call_outcome",
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("[KG-2.296] tool action_outcome failed: %s", exc)
    if written:
        logger.info(
            "[KG-2.296] run %s: persisted %d ToolCall node(s) under %s",
            run_id,
            written,
            trace_id,
        )
    return written


# ---------------------------------------------------------------------------
# Internal: ARPO step-credit read-back (CONCEPT:AU-AHE.reward.this-is-read-back)
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

    CONCEPT:AU-AHE.reward.this-is-read-back — this is the read-back half of agent-step policy
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
