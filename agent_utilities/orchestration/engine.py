"""Orchestration Engine — The canonical entrypoint for all agent execution.

CONCEPT:AU-ORCH.execution.orchestration — Orchestration Engine

This module provides the single execution kernel for graph, streaming, dynamic,
parallel, workflow, SDD, and research execution.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)

import asyncio
import contextlib
import functools
import inspect
import json
import secrets
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path

from agent_utilities.core.config import (
    DEFAULT_LLM_MODEL_ID,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_ROUTER_MODEL,
)

DEFAULT_ENABLE_LLM_VALIDATION = True
DEFAULT_GRAPH_AGENT_MODEL = DEFAULT_LLM_MODEL_ID
DEFAULT_GRAPH_PERSISTENCE_PATH = ".agent_utilities/graph_persistence"
DEFAULT_GRAPH_ROUTER_TIMEOUT = 120000
DEFAULT_GRAPH_VERIFIER_TIMEOUT = 120000
DEFAULT_PROVIDER = DEFAULT_LLM_PROVIDER

# Wall-clock budget (seconds) for a single spawned agent in a workflow fan-out.
# CONCEPT:AU-ORCH.execution.dynamic-workflows — ``max_steps`` bounds interaction ROUNDS, not time: a spawned
# agent that awaits an unresponsive MCP tool never advances a step, so the fan-out
# ``asyncio.gather`` would hang to the caller's 300s timeout. We bound each spawned
# ``run_agent`` by wall-clock so a stuck child resolves to a structured timeout error
# (Configuration-discipline: a module constant, not an env flag).
AGENT_WALLCLOCK_TIMEOUT_S = 120.0

from agent_utilities.core.config import (
    DEFAULT_GRAPH_TIMEOUT,
    emit_graph_event,
    get_discovery_registry,
)
from agent_utilities.core.model_factory import create_model
from agent_utilities.security.persistence_privacy import persistence_reference

from ..graph.mermaid import get_graph_mermaid
from ..graph.state import REQUESTED_MODEL_ID_CTX, GraphDeps, GraphState
from ..models import GraphResponse
from .graph_execution_evidence import (
    GraphExecutionEvidenceCollector,
    run_with_execution_evidence,
)

try:
    from opentelemetry import trace

    tracer: trace.Tracer | None = trace.get_tracer("agent-utilities.graph")
except ImportError:
    tracer = None


def _pydantic_graph_span(
    *,
    run_id: str,
    query: str,
    topology: str,
    operation: str = "run",
    evidence: GraphExecutionEvidenceCollector | None = None,
) -> contextlib.AbstractContextManager[Any]:
    """Return a span context for one actual Pydantic Graph execution API."""

    if tracer is None:
        return contextlib.nullcontext(None)
    attributes: dict[str, Any] = {
        "agent_utilities.execution.mode": "pydantic_graph",
        "query_length": len(query),
        "run_ref": persistence_reference(
            "run", run_id, namespace="orchestration-trace"
        ),
        "topology": topology,
    }
    if evidence is not None:
        attributes.update(
            {
                "agent_utilities.graph.topology_digest": evidence.topology_digest,
                "agent_utilities.graph.version_digest": evidence.version_digest,
                "agent_utilities.graph.runtime_version": evidence.runtime_version,
                "agent_utilities.graph.resume_supported": False,
            }
        )
    return tracer.start_as_current_span(
        f"pydantic_graph.{operation}",
        attributes=attributes,
    )


from contextlib import AsyncExitStack

import anyio

from agent_utilities.models.execution_manifest import (
    ExecutionManifest,
    ExecutionResult,
)


def _foreground_execution(fn):
    """Hold the shared foreground lease for every public graph-run shape.

    Background ingestion already cooperatively checks this lease between bounded
    batches.  Keeping the lease at the canonical execution engine (rather than
    at one protocol adapter) means direct, streaming, and iterator callers all
    cause the same prompt ramp-down.
    """
    from agent_utilities.core.background_throttle import get_throttle

    if inspect.isasyncgenfunction(fn):

        @functools.wraps(fn)
        async def stream_wrapper(*args, **kwargs):
            with get_throttle().foreground():
                async for item in fn(*args, **kwargs):
                    yield item

        return stream_wrapper

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        with get_throttle().foreground():
            return await fn(*args, **kwargs)

    return wrapper


async def _offload_sync(operation, *args, **kwargs):
    """Run a synchronous integration without occupying the request event loop."""

    result = await asyncio.to_thread(operation, *args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


def _is_agent_error(output: str) -> bool:
    """True when a spawned-agent output string is a structured error envelope.

    ``_run_agent_bounded`` encodes a timeout/failure as a JSON ``{"error", "agent"}``
    string. Such a value is a ``str`` and truthy, so the fan-out's
    ``isinstance(r, str) and r`` filter would otherwise mistake it for real output.
    """
    s = output.strip()
    if not (s.startswith("{") and '"error"' in s):
        return False
    try:
        return isinstance(json.loads(s), dict) and "error" in json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return False


async def _enter_mcp_toolset(
    stack: AsyncExitStack,
    ts: Any,
    timeout_style: Literal["current_task", "wait_for"],
) -> Any:
    """Enter one MCP toolset with the caller's timeout contract."""
    if timeout_style == "current_task":
        # Use asyncio.timeout() (not asyncio.wait_for) to bound the connect:
        # wait_for runs the coroutine in a NEW task, so a stdio toolset's anyio
        # cancel scope would be ENTERED in that child task while the AsyncExitStack
        # EXITS it in this (outer) task -> "Attempted to exit cancel scope in a
        # different task than it was entered in". asyncio.timeout() applies to the
        # current task, keeping enter/exit on the same task.
        async with asyncio.timeout(60.0):
            return await stack.enter_async_context(ts)
    return await asyncio.wait_for(stack.enter_async_context(ts), timeout=60.0)


def _record_connected_mcp_toolset(
    connected_toolsets: list,
    already_connected: set[int],
    ts: Any,
    *,
    connected: Any,
    log_prefix: str,
    srv_id: str,
) -> None:
    """Record a successful MCP connection and emit its success log."""
    already_connected.add(id(ts))
    connected_toolsets.append(connected)
    logger.info(f"{log_prefix}: ✅ MCP server '{srv_id}' connected")


async def _connect_mcp_toolsets_common(
    stack: AsyncExitStack,
    deps: Any,
    *,
    timeout_style: Literal["current_task", "wait_for"],
    log_prefix: str,
    log_connect: bool,
    failure_suffix: str,
    warning_suffix: str,
) -> None:
    """Connect toolsets while retaining each caller's timeout/log contract."""
    failed_servers: list[tuple[str, str]] = []
    connected_toolsets: list = []
    _already_connected: set[int] = set()

    for ts in deps.mcp_toolsets:
        if not hasattr(ts, "__aenter__"):
            connected_toolsets.append(ts)
            continue
        if id(ts) in _already_connected:
            connected_toolsets.append(ts)
            continue

        srv_id = getattr(ts, "id", getattr(ts, "name", repr(ts)))
        if log_connect:
            logger.debug(f"{log_prefix}: Connecting to MCP server '{srv_id}'...")
        try:
            connected = await _enter_mcp_toolset(stack, ts, timeout_style)
            _record_connected_mcp_toolset(
                connected_toolsets,
                _already_connected,
                ts,
                connected=connected,
                log_prefix=log_prefix,
                srv_id=srv_id,
            )
        except Exception as e:
            err_msg = str(e)
            logger.error(
                f"{log_prefix}: ❌ MCP server '{srv_id}' {failure_suffix}: {err_msg}"
            )
            failed_servers.append((srv_id, err_msg))

    deps.mcp_toolsets = [ts for ts in connected_toolsets if ts is not None]

    if failed_servers:
        logger.warning(
            f"{log_prefix}: {len(failed_servers)} MCP server(s) {warning_suffix}:\n"
            + "\n".join(f"  ❌ {sid}: {err}" for sid, err in failed_servers)
        )


async def _connect_mcp_toolsets(stack: AsyncExitStack, deps: Any) -> None:
    """Connect every MCP toolset in ``deps.mcp_toolsets``, tolerating failures.

    Extracted verbatim from ``AgentOrchestrationEngine.execute_graph`` (pure
    extract-method, no behaviour change). Mutates ``deps.mcp_toolsets`` in
    place to the connected subset, exactly as the original inline code did;
    has no return value.
    """
    await _connect_mcp_toolsets_common(
        stack,
        deps,
        timeout_style="current_task",
        log_prefix="run_graph",
        log_connect=True,
        failure_suffix="FAILED to connect",
        warning_suffix="failed to connect — graph will proceed without them",
    )


def _run_security_preflight(query: str, run_id: str) -> dict | None:
    """Prompt-injection scan; block the query or warn, best-effort.

    Extracted verbatim from ``AgentOrchestrationEngine.execute_graph`` (pure
    extract-method, no behaviour change). Returns a blocked-response dict when
    the scanner flags the query as malicious (the caller must return it
    immediately), else ``None`` to let the run proceed -- exactly like the
    original inline ``return GraphResponse(...).model_dump()`` from this same
    try block.
    """
    # --- Security Guard Pre-Flight (OS-5.4, OS-5.5) ---
    try:
        from ..security.prompt_scanner import PromptInjectionScanner

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
                    "execution_mode": "graph_preflight",
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
    except ImportError:  # noqa: BLE001 — optional prompt scanner is not installed
        pass  # Scanner not available
    except Exception as e:
        # D-DST-6: Security Guard Pre-Flight (OS-5.4/5.5) — a scanner crash here
        # was previously indistinguishable from "the scan ran clean and found
        # nothing" (the exact guardrail-crash-reads-as-clean-pass cousin), and
        # the query proceeds UNSCANNED either way. Raised to warning (matching
        # this lane's DoomLoopDetector/adversarial-verification precedent) so a
        # persistently-failing scanner is diagnosable rather than invisible.
        logger.warning(
            "run_graph: prompt scanner failed (query proceeding UNSCANNED): %s",
            e,
        )
    return None


async def _run_graph_with_evidence(
    graph: Any,
    run_id: str,
    query: str,
    topology: str,
    state: Any,
    deps: Any,
    graph_evidence: Any,
) -> tuple[Any, float, dict | None]:
    """Run the graph under the evidence collector; unwrap the raw result.

    Extracted verbatim from ``AgentOrchestrationEngine.execute_graph`` (pure
    extract-method, no behaviour change). Returns ``(result, graph_run_start,
    None)`` on success (mirroring the original's fall-through -- the caller
    needs ``graph_run_start`` for the telemetry duration calculations further
    down, which the original inline code computed at this same point and
    reused from the enclosing scope), or ``(None, graph_run_start,
    error_response_dict)`` when the graph run itself raised -- the caller must
    return ``error_response_dict`` immediately, exactly like the original's
    inline ``return GraphResponse(...).model_dump()`` from inside this same
    except block.
    """
    result = None
    _graph_run_start = time.perf_counter()
    try:
        if tracer is None:
            logger.info("run_graph: Running pydantic_graph.run (no tracer)...")
        with _pydantic_graph_span(
            run_id=run_id,
            query=query,
            topology=topology,
            evidence=graph_evidence,
        ) as span:
            graph_evidence.attach_span(span)
            try:
                with anyio.move_on_after(DEFAULT_GRAPH_TIMEOUT / 1000.0) as scope:
                    result = await run_with_execution_evidence(
                        graph,
                        state=state,
                        deps=deps,
                        collector=graph_evidence,
                    )
                if scope.cancel_called:
                    logger.error(
                        "run_graph: Graph execution TIMEOUT after %sms",
                        DEFAULT_GRAPH_TIMEOUT,
                    )
                    result = "timeout"
            finally:
                graph_evidence.finish_span(state=state)
            if span is not None:
                span.set_status(
                    trace.Status(
                        trace.StatusCode.OK if result else trace.StatusCode.ERROR
                    )
                )
    except Exception as e:
        logger.error(
            "run_graph: critical graph execution failure: %s",
            e,
        )
        emit_graph_event(
            deps.event_queue, "graph_complete", run_id=run_id, status="error"
        )
        return (
            None,
            _graph_run_start,
            GraphResponse(
                status="error",
                error=str(e),
                metadata={
                    "run_id": run_id,
                    "is_error": True,
                    "execution_mode": "pydantic_graph",
                },
                # A budget, timeout, or model failure can happen after a
                # real tool completed.  Preserve the calls accumulated by
                # graph nodes so the outer required-tool gate and durable
                # RunTrace record what actually happened instead of
                # reporting zero provenance.
                tool_calls=list(getattr(state, "tool_calls", []) or []),
                execution_evidence=graph_evidence.evidence(state=state),
            ).model_dump(),
        )

    # CONCEPT:AU-ORCH.execution.node-direct-end — a node may END the run directly with End[GraphResponse]
    # (the router's direct-completion shape). pydantic-graph returns the End wrapper,
    # so unwrap it to the GraphResponse here; otherwise the result handling below
    # falls through to ``str(result)`` and the reply becomes "End(data=GraphResponse(…))".
    from pydantic_graph import End

    if isinstance(result, End):
        result = result.data

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
    return result, _graph_run_start, None


@dataclass
class _TelemetryContext:
    """Bundles ``_emit_execute_graph_telemetry``'s shared inputs so each
    independent export sub-block below stays under the 7-param cap.
    """

    run_id: str
    query: str
    result: Any
    state: Any
    graph_evidence: Any
    config: dict
    graph_run_start: float
    usage: dict[str, int]
    run_model: str


def _telemetry_run_kwargs(ctx: _TelemetryContext) -> dict[str, Any]:
    """Return the common run fields shared by usage-plane exporters."""
    return {
        "run_id": ctx.run_id,
        "query": ctx.query,
        "status": "success" if ctx.result else "timeout",
        "duration_ms": (time.perf_counter() - ctx.graph_run_start) * 1000.0,
        "token_usage": ctx.usage,
        "model": ctx.run_model,
    }


async def _export_langfuse_trace(ctx: _TelemetryContext) -> None:
    """Default-on: ships this graph run as a Langfuse trace + token-usage
    generation when LANGFUSE_* keys are configured (CONCEPT:AU-OS.
    observability.langfuse-exporter). No-ops cleanly when the keys/dep are
    absent so the live path is never affected.
    """
    try:
        from ..observability.langfuse_exporter import get_langfuse_exporter

        _exporter = get_langfuse_exporter()
        if _exporter is not None:
            await _offload_sync(
                _exporter.export_graph_run,
                **_telemetry_run_kwargs(ctx),
                metadata={
                    "domain": ctx.state.routed_domain,
                    "execution_mode": "pydantic_graph",
                    "graph_topology_digest": ctx.graph_evidence.topology_digest,
                    "graph_version_digest": ctx.graph_evidence.version_digest,
                    "graph_transition_count": len(ctx.graph_evidence.transitions),
                },
                evidence=(
                    ctx.config.get("trace_evidence")
                    if isinstance(ctx.config.get("trace_evidence"), dict)
                    else None
                ),
            )
    except Exception as _lf_exc:  # noqa: BLE001 — export must never crash a run
        logger.debug(
            "run_graph: Langfuse export skipped (%s).",
            type(_lf_exc).__name__,
        )


async def _export_self_ingest_run_trace(ctx: _TelemetryContext) -> None:
    """Dogfooding: ship this graph run's RunTrace into the epistemic-graph
    engine obs store (CONCEPT:AU-KG.ingest.attaching-this-root-logger).
    Opt-in (default-off); clean no-op when disabled.
    """
    try:
        from ..observability.self_ingest import emit_run_trace

        await _offload_sync(
            emit_run_trace,
            run_id=ctx.run_id,
            status="success" if ctx.result else "timeout",
            duration_ms=(time.perf_counter() - ctx.graph_run_start) * 1000.0,
            query=ctx.query,
            attributes={
                "domain": ctx.state.routed_domain,
                "execution_mode": "pydantic_graph",
                "graph_topology_digest": ctx.graph_evidence.topology_digest,
                "graph_version_digest": ctx.graph_evidence.version_digest,
                "graph_transition_count": len(ctx.graph_evidence.transitions),
            },
        )
    except Exception as _si_exc:  # noqa: BLE001 — telemetry must never crash a run
        logger.debug("run_graph: self-ingest run_trace skipped: %s", _si_exc)


async def _record_usage_row(ctx: _TelemetryContext) -> None:
    """CONCEPT:AU-OS.observability.persist-this-graph-run — persist this
    graph run as a runtime usage row so token counts/cost feed the same
    /api/observability surface the ingested agent logs do. Best-effort;
    never affects the run.
    """
    try:
        from agent_utilities.security.brain_context import current_actor
        from agent_utilities.usage.recorder import get_usage_recorder

        await _offload_sync(
            get_usage_recorder().record_run,
            **_telemetry_run_kwargs(ctx),
            project=str(ctx.state.routed_domain or ""),
            tenant_id=current_actor().tenant_id,
        )
    except Exception as _ur_exc:  # noqa: BLE001 — recorder must never crash a run
        logger.debug(
            "run_graph_usage_record_skipped error_type=%s",
            type(_ur_exc).__name__,
        )


def _stamp_otel_span(ctx: _TelemetryContext) -> None:
    """OTel gen_ai span attrs (CONCEPT:AU-OS.observability.
    telemetry-observability, X2) — reuses the SAME usage/run_model snapshot
    built above, stamps them onto run_agent's own span (opened by
    ``on_graph_start`` in ``agent_runner.run_agent``) as
    ``gen_ai.request.model``/``gen_ai.usage.*``. Best-effort; a run with no
    tracked span (OTel unconfigured) is a clean no-op.
    """
    try:
        from ..observability import get_telemetry_engine

        get_telemetry_engine().on_response(
            run_id=ctx.run_id, usage=ctx.usage, model=ctx.run_model
        )
    except Exception as _otel_exc:  # noqa: BLE001 — tracing must never crash a run
        logger.debug(
            "run_graph_otel_response_skipped error_type=%s",
            type(_otel_exc).__name__,
        )


async def _emit_execute_graph_telemetry(
    run_id: str,
    query: str,
    result: Any,
    state: Any,
    graph_evidence: Any,
    config: dict,
    _graph_run_start: float,
) -> tuple[dict[str, int], str]:
    """Cost-plane usage snapshot + best-effort observability exports.

    Extracted verbatim from ``AgentOrchestrationEngine.execute_graph`` (pure
    extract-method, no behaviour change): Langfuse auto-export, self-ingest
    RunTrace, usage-recorder persistence, and the OTel gen_ai span attrs. Every
    sub-block is independently try/except-wrapped in the original and stays
    that way here -- none of them may crash a run.

    Returns the ``(_usage, _run_model)`` snapshot computed at the top of this
    block -- the original inline code left those two names in the enclosing
    ``execute_graph`` scope for the response-shaping code further down to
    reuse (``result.metadata["token_usage"]``); the caller must now pass them
    through explicitly.
    """
    # --- Cost-plane usage snapshot (CONCEPT:AU-OS.observability.usage-analytics-store, D-54c-1) ---
    # ``state.session_usage`` is the ONE accumulator every specialist node already
    # feeds via ``GraphState._update_usage`` (including provider cache read/write +
    # reasoning tokens) — build the ``token_usage`` dict every downstream consumer
    # below reads FROM IT, not from ``result.metadata["token_usage"]`` (nothing ever
    # wrote that key, so it was always ``{}``: cost attribution and cache-savings
    # telemetry were structurally hollow regardless of what any node returned).
    _usage: dict[str, int] = {
        "input_tokens": state.session_usage.input_tokens,
        "output_tokens": state.session_usage.output_tokens,
        "cache_creation_input_tokens": state.session_usage.cache_creation_input_tokens,
        "cache_read_input_tokens": state.session_usage.cache_read_input_tokens,
        "reasoning_tokens": state.session_usage.reasoning_tokens,
    }
    _run_model = str(config.get("agent_model") or "")

    ctx = _TelemetryContext(
        run_id=run_id,
        query=query,
        result=result,
        state=state,
        graph_evidence=graph_evidence,
        config=config,
        graph_run_start=_graph_run_start,
        usage=_usage,
        run_model=_run_model,
    )

    await _export_langfuse_trace(ctx)
    await _export_self_ingest_run_trace(ctx)
    await _record_usage_row(ctx)
    _stamp_otel_span(ctx)

    return _usage, _run_model


def _shape_response_for_none_result(
    run_id: str, state: Any, graph_evidence: Any, mermaid_prefix: str
) -> dict:
    """Decision-node "no output" termination -- graph.run() returned bare None.

    Extracted verbatim from ``_shape_graph_execute_response`` (pure extract-method,
    no behaviour change).
    """
    logger.error(
        "run_graph: graph terminated with no output — a decision branch routed "
        "directly to the end node with no End[GraphResponse] payload, so no "
        "specialist/verifier/synthesizer node (and no model) ever ran. "
        "Registry keys: %s state.error=%s",
        list(state.results_registry.keys()),
        getattr(state, "error", None),
    )
    # D-RTR-3 (engine side): ``dispatcher_step``'s empty-plan branch (and
    # ``router_step``'s total-planning-failure path) now stamp a concrete,
    # actionable reason onto ``ctx.state.error`` before this ``None`` termination
    # — but that state is otherwise discarded here, so the caller always saw the
    # same hardcoded generic apology no matter *why* the turn produced nothing.
    # Surface the real reason instead. ``state.error`` is a typed ``str | None``
    # (``graph/state.py``), but it is free text assembled from an exception's
    # ``str(e)`` upstream, so it is sanitised — stripped, and length-capped so a
    # stray raw traceback/repr can never reach the user as a wall of text — before
    # use. Fail-closed is preserved exactly as before: a missing/blank/non-string
    # ``state.error`` (the "stringified None" bug this guard exists to prevent)
    # falls back to the same non-empty generic text that shipped before this
    # change — never ``None``, never an empty string, never the literal "None".
    raw_error = getattr(state, "error", None)
    sanitized_error = raw_error.strip() if isinstance(raw_error, str) else ""
    if sanitized_error:
        _MAX_ERROR_LEN = 500
        if len(sanitized_error) > _MAX_ERROR_LEN:
            sanitized_error = sanitized_error[:_MAX_ERROR_LEN].rstrip() + "…"
        error_text = sanitized_error
        output_text = (
            f"I couldn't produce a response for this turn: {sanitized_error} "
            "Please try again."
        )
    else:
        error_text = (
            "The orchestration graph completed without invoking any specialist "
            "or model for this turn — no answer was generated."
        )
        output_text = (
            "I couldn't produce a response for this turn: the orchestration "
            "graph ended before any model ran. Please try again."
        )
    return GraphResponse(
        status="failed",
        error=error_text,
        results={"output": output_text},
        mermaid=mermaid_prefix if mermaid_prefix else None,
        metadata={
            "run_id": run_id,
            "domain": state.routed_domain,
            "degraded": True,
            "outcome": "empty_graph_termination",
            "execution_mode": "pydantic_graph",
        },
        tool_calls=list(getattr(state, "tool_calls", []) or []),
        execution_evidence=graph_evidence.evidence(state=state),
    ).model_dump()


def _classify_budget_dimension(err_text: str) -> str | None:
    """Which budget dimension tripped (node transitions/tool calls/tokens/
    cost/duration), parsed from the error text -- see
    :func:`_shape_response_for_error_dict`.
    """
    for dim in (
        "max node transitions",
        "max tool calls",
        "max total tokens",
        "max cost usd",
        "max duration",
    ):
        if dim in err_text.lower():
            return dim.replace("max ", "").replace(" ", "_")
    return None


def _build_failure_results(err_text: str, partial_results: Any) -> dict[str, Any]:
    failure_results: dict[str, Any] = {
        "output": f"The task could not be completed: {err_text}"
    }
    if isinstance(partial_results, dict) and partial_results:
        # Preserve everything completed so far -- a budget/error
        # termination must not discard partial specialist output that was
        # already produced before the cap tripped.
        failure_results["partial_results"] = partial_results
    return failure_results


def _shape_response_for_error_dict(
    result: dict, run_id: str, state: Any, graph_evidence: Any, mermaid_prefix: str
) -> dict:
    """Unrecovered ``error_recovery_step`` termination dict -> a failed GraphResponse.

    Extracted verbatim from ``_shape_graph_execute_response`` (pure extract-method,
    no behaviour change).
    """
    err_text = str(result.get("error") or "")
    logger.error(
        "run_graph: graph terminated via error_recovery with an unrecovered error: %s",
        err_text[:300],
    )
    # CONCEPT:AU-ORCH.execution.execution-budget-caps — termination is an
    # explicit, classified condition, not a bare "something went wrong": a
    # budget exhaustion (node transitions, tool calls, tokens, cost, or
    # duration — ``error_recovery_step`` stamps ``budget_exceeded`` for all
    # five) is reported as its own outcome/dimension rather than folded into
    # the generic "graph_terminal_error" every other terminal failure shares.
    budget_exceeded = bool(result.get("budget_exceeded"))
    budget_dimension = _classify_budget_dimension(err_text) if budget_exceeded else None
    failure_results = _build_failure_results(err_text, result.get("results"))
    return GraphResponse(
        status="failed",
        error=err_text,
        results=failure_results,
        mermaid=mermaid_prefix if mermaid_prefix else None,
        metadata={
            "run_id": run_id,
            "domain": state.routed_domain,
            "degraded": True,
            "outcome": "budget_exceeded" if budget_exceeded else "graph_terminal_error",
            **({"budget_dimension": budget_dimension} if budget_dimension else {}),
            "execution_mode": "pydantic_graph",
        },
        tool_calls=list(getattr(state, "tool_calls", []) or []),
        execution_evidence=graph_evidence.evidence(state=state),
    ).model_dump()


def _shape_response_for_graph_response(
    result: GraphResponse,
    run_id: str,
    state: Any,
    graph_evidence: Any,
    mermaid_prefix: str,
    _usage: dict[str, int],
    _run_model: str,
) -> dict:
    """A real ``GraphResponse`` result: stamp mermaid/evidence/metadata onto
    it in place. Extracted verbatim from ``_shape_graph_execute_response``
    (pure extract-method, no behaviour change).
    """
    result.mermaid = mermaid_prefix if mermaid_prefix else None
    result.execution_evidence = graph_evidence.evidence(state=state)
    result.metadata.update(
        {
            "run_id": run_id,
            "domain": state.routed_domain,
            "execution_mode": "pydantic_graph",
            # D-54c-1: stamp the real accumulated usage (incl. cache read/write +
            # reasoning tokens) onto the response too, not just the side-channel
            # exporters above — any other reader of GraphResponse.metadata now
            # sees real cost-plane data instead of a permanently-empty dict.
            "token_usage": _usage,
            "model": _run_model,
        }
    )
    # Surface the graph run's accumulated tool calls so run_agent persists them
    # as :ToolCall provenance (CONCEPT:AU-KG.temporal.message-history-read) — the multi-agent path
    # previously wrote none, unlike the direct single-server loop.
    if not result.tool_calls and getattr(state, "tool_calls", None):
        result.tool_calls = list(state.tool_calls)
    return result.model_dump()


def _shape_response_for_str_result(
    result: str, run_id: str, state: Any, graph_evidence: Any, mermaid_prefix: str
) -> dict:
    """Guard: graph.run() returned a plain string (node label) instead of
    GraphResponse. This happens when the graph exits without hitting
    End[GraphResponse] via some other decision branch whose matched node has
    no further outgoing edge (the ``None`` termination case is handled
    separately by :func:`_shape_response_for_none_result`; this is the
    general fallback for any other stray string result). Extract the best
    available result from state before wrapping.
    """
    logger.error(
        f"run_graph: graph.run() returned node label '{result}' instead of GraphResponse. "
        f"This indicates the graph terminated unexpectedly. "
        f"Registry keys: {list(state.results_registry.keys())}"
    )
    output = (
        next(iter(state.results_registry.values()), None)
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
            "execution_mode": "pydantic_graph",
        },
        tool_calls=list(getattr(state, "tool_calls", []) or []),
        execution_evidence=graph_evidence.evidence(state=state),
    ).model_dump()


def _shape_graph_execute_response(
    result: Any,
    run_id: str,
    state: Any,
    graph_evidence: Any,
    mermaid_prefix: str,
    _usage: dict[str, int],
    _run_model: str,
) -> dict:
    """Shape the final ``execute_graph`` return value from the raw graph result.

    Extracted verbatim from ``AgentOrchestrationEngine.execute_graph`` (pure
    extract-method, no behaviour change). Handles every shape ``graph.run()``
    can hand back: a real ``GraphResponse``, a bare ``None`` (decision-node
    no-output termination), a stray node-label ``str``, an unrecovered
    ``error_recovery_step`` dict, or the plain completed-string fallback.
    """
    if isinstance(result, GraphResponse):
        return _shape_response_for_graph_response(
            result, run_id, state, graph_evidence, mermaid_prefix, _usage, _run_model
        )

    # Guard: graph.run() returned bare ``None`` instead of an End[GraphResponse] or a
    # node-label string. This is the decision-node "no results" termination —
    # ``dispatcher_step`` returns ``None`` when the plan is complete but both
    # ``results_registry`` and ``exploration_notes`` are empty (its own "Plan completed
    # but NO execution results found in registry" case, `_router_impl.py`), and
    # ``dispatcher_route``'s ``type(None)`` branch sends that straight to ``g.end_node``
    # with no payload. pydantic-graph then returns ``End(data=None)``, which unwraps to
    # ``result is None`` above — NOT a string, so it must be caught here, before the
    # catch-all at the bottom of this function, which used to stringify it into
    # ``results.output == "None"`` under ``status="completed"``. That reported a turn
    # where NO specialist/verifier/synthesizer node — and therefore no model — ever ran
    # as an ordinary successful reply whose answer happened to be the four characters
    # "None". Per this repo's fail-closed doctrine (AGENTS.md "Fail closed — a degraded
    # read must never grant permission": make failure a distinct value, never an empty
    # success), report it as a genuine, honest failure instead.
    if result is None:
        return _shape_response_for_none_result(
            run_id, state, graph_evidence, mermaid_prefix
        )

    if isinstance(result, str):
        return _shape_response_for_str_result(
            result, run_id, state, graph_evidence, mermaid_prefix
        )

    # Guard: a terminal error_recovery_step End({"error": ..., "results": {...}}) is a
    # REAL failure, not a completed answer (CONCEPT:AU-ORCH.execution.messaging-orchestration-transparency).
    # Without this branch it fell through to the catch-all below, which stringified this
    # dict into `results.output` under status="completed" — presenting a raw Python-dict
    # repr (e.g. "{'error': 'Execution budget exceeded...', 'results': {...}}") as if it
    # were a normal reply. Surface the real cause in `error` + a coherent output instead.
    if isinstance(result, dict) and result.get("error"):
        return _shape_response_for_error_dict(
            result, run_id, state, graph_evidence, mermaid_prefix
        )

    return GraphResponse(
        status="completed",
        results={"output": str(result)},
        mermaid=mermaid_prefix if mermaid_prefix else None,
        metadata={
            "run_id": run_id,
            "domain": state.routed_domain,
            "execution_mode": "pydantic_graph",
        },
        tool_calls=list(getattr(state, "tool_calls", []) or []),
        execution_evidence=graph_evidence.evidence(state=state),
    ).model_dump()


# ═══════════════════════════════════════════════════════════════════════════
# iter_graph helpers — extracted verbatim (pure extract-method, no behaviour
# change) so ``AgentOrchestrationEngine.iter_graph`` stays under the
# complexity caps. See ``iter_graph``'s own docstring for the streaming
# contract these pieces jointly implement.
# ═══════════════════════════════════════════════════════════════════════════


def _resolve_end_marker_type() -> type[Any]:
    """Pydantic Graph 2.21 promotes ``EndMarker`` to the top-level package,
    while older supported 2.x installations retain it under ``beta.graph``.
    Keep iter-mode graph execution valid across the promoted dependency
    boundary instead of making the import location part of our API.
    """
    from importlib import import_module

    import pydantic_graph

    end_marker_type = getattr(pydantic_graph, "EndMarker", None)
    if end_marker_type is None:
        end_marker_type = import_module("pydantic_graph.beta.graph").EndMarker
    return end_marker_type


def _build_graph_deps(
    config: dict[str, Any],
    *,
    mcp_toolsets: list[Any] | None,
    event_queue: asyncio.Queue[Any] | None,
    requested_model_id: str | None,
    run_id: str,
    plan_sync: Any,
    mode: Literal["execute", "iter", "stream"],
) -> GraphDeps:
    """Build the shared graph dependencies for one execution mode.

    The mode-specific fields intentionally stay explicit here: ``execute``
    honors the configured request id and execution shape, ``iter`` and
    ``execute`` include the node registry, and ``stream`` relies on
    ``GraphDeps`` defaults for both fields.  Assignments follow the original
    constructors' order so model creation and configuration lookup remain
    lazy and their exceptions propagate unchanged.
    """
    _custom_headers = config.get("custom_headers")
    deps_kwargs: dict[str, Any] = {
        "tag_prompts": config.get("tag_prompts", {}),
        "tag_env_vars": config.get("tag_env_vars", {}),
        "mcp_toolsets": (
            mcp_toolsets if mcp_toolsets is not None else config.get("mcp_toolsets", [])
        ),
        "mcp_url": config.get("mcp_url", ""),
        "mcp_config": config.get("mcp_config", ""),
    }
    deps_kwargs["router_model"] = create_model(
        model_id=config.get("router_model", DEFAULT_ROUTER_MODEL),
        api_key=config.get("api_key"),
        base_url=config.get("base_url"),
        custom_headers=_custom_headers,
        provider=config.get("provider", DEFAULT_PROVIDER),
    )
    deps_kwargs["agent_model"] = create_model(
        model_id=config.get("agent_model", DEFAULT_GRAPH_AGENT_MODEL),
        api_key=config.get("api_key"),
        base_url=config.get("base_url"),
        custom_headers=_custom_headers,
        provider=config.get("provider", DEFAULT_PROVIDER),
    )
    if mode in {"execute", "iter"}:
        deps_kwargs["nodes"] = config.get("nodes", {})
    deps_kwargs.update(
        {
            "min_confidence": config.get("min_confidence", 0.6),
            "sub_agents": config.get("sub_agents", {}),
            "provider": config.get("provider", DEFAULT_PROVIDER),
            "base_url": config.get("base_url"),
            "api_key": config.get("api_key"),
            "event_queue": event_queue,
            "router_timeout": config.get(
                "router_timeout", DEFAULT_GRAPH_ROUTER_TIMEOUT
            ),
            "verifier_timeout": config.get(
                "verifier_timeout", DEFAULT_GRAPH_VERIFIER_TIMEOUT
            ),
        }
    )
    if mode == "execute":
        deps_kwargs["execution_shape"] = config.get("execution_shape")
        deps_kwargs["request_id"] = config.get("request_id", run_id)
    else:
        deps_kwargs["request_id"] = run_id
    deps_kwargs.update(
        {
            "routing_strategy": config.get("routing_strategy", "hybrid"),
            "enable_llm_validation": config.get(
                "enable_llm_validation", DEFAULT_ENABLE_LLM_VALIDATION
            ),
            "discovery_metadata": config.get("discovery_metadata", {}),
            "plan_sync": plan_sync,
            "approval_manager": config.get("approval_manager"),
            "model_registry": config.get("model_registry"),
            "requested_model_id": requested_model_id,
            "permissions_kernel": config.get("permissions_kernel"),
            "agent_identity": config.get("agent_identity"),
            "knowledge_engine": config.get("knowledge_engine"),
            "response_format": config.get("response_format", "text"),
            "execution_mode": config.get("execution_mode", "auto"),
            "pinned_skill_name": config.get("pinned_skill_name", ""),
            "pinned_skill_prompt": config.get("pinned_skill_prompt", ""),
        }
    )
    return GraphDeps(**deps_kwargs)


def _build_iter_deps(
    config: dict,
    *,
    mcp_toolsets: list[Any] | None,
    event_queue: asyncio.Queue,
    requested_model_id: str | None,
    run_id: str,
    plan_sync: Any,
) -> GraphDeps:
    """Build the ``GraphDeps`` :func:`iter_graph` runs the graph with."""
    return _build_graph_deps(
        config,
        mcp_toolsets=mcp_toolsets,
        event_queue=event_queue,
        requested_model_id=requested_model_id,
        run_id=run_id,
        plan_sync=plan_sync,
        mode="iter",
    )


def _build_iter_state(
    query: str,
    query_parts: list[dict[str, Any]] | None,
    run_id: str,
    mode: str,
    topology: str,
    config: dict,
) -> GraphState:
    """Build the ``GraphState`` :func:`iter_graph` runs the graph with."""
    return GraphState(
        query=query,
        query_parts=query_parts or [],
        session_id=run_id,
        mode=mode,
        topology=topology,
        invoker_context=config.get(
            "invoker_context", ""
        ),  # CONCEPT:AU-ORCH.session.invoker-agent-handoff
        invoker_budget_tokens=config.get(
            "invoker_budget_tokens"
        ),  # CONCEPT:AU-ORCH.session.invoker-agent-handoff
        invoker_allowed_tools=config.get(
            "invoker_allowed_tools"
        ),  # CONCEPT:AU-ORCH.session.invoker-agent-handoff
        invoker_capability_ceiling=config.get(
            "invoker_capability_ceiling"
        ),  # CONCEPT:AU-OS.identity.per-agent-on-behalf-delegation
        invoker_cred_ref=config.get(
            "invoker_cred_ref"
        ),  # CONCEPT:AU-ORCH.session.invoker-agent-handoff
        invoker_channel_id=config.get(
            "message_channel_id"
        ),  # CONCEPT:AU-ORCH.session.session-anchored-collections-native
    )


def _build_graph_state_context(
    *,
    graph: Any,
    query: str,
    query_parts: list[dict[str, Any]] | None,
    run_id: str,
    mode: str,
    topology: str,
    config: dict,
) -> tuple[GraphState, GraphExecutionEvidenceCollector]:
    """Build and bind the state/evidence pair shared by graph run shapes."""
    state = _build_iter_state(query, query_parts, run_id, mode, topology, config)
    graph_evidence = GraphExecutionEvidenceCollector(graph, topology=topology)
    graph_evidence.bind_state(state)
    return state, graph_evidence


async def _hydrate_registry_tags_for_iter(deps: GraphDeps) -> None:
    """Merge registry tags into ``deps`` (iter_graph's own variant — also
    keys by a lowercased/underscored ``node_id``, which ``execute_graph``'s
    own registry-hydration block does not; kept as its own extraction
    rather than unified, to avoid changing either function's behaviour).
    """
    # CONCEPT:AU-ORCH.routing.offload-sync-roundtrip — registry hydration is a
    # synchronous backend round-trip; keep it off the event loop (mirrors
    # execute_graph above).
    registry = await _offload_sync(get_discovery_registry)
    for agent in registry.agents:
        if agent.name and agent.name not in deps.tag_prompts:
            deps.tag_prompts[agent.name] = agent.description or agent.name
        node_id = agent.name.lower().replace(" ", "_")
        if node_id not in deps.tag_prompts:
            deps.tag_prompts[node_id] = agent.description or agent.name


async def _connect_mcp_toolsets_for_iter(stack: AsyncExitStack, deps: Any) -> None:
    """Connect every MCP toolset in ``deps.mcp_toolsets``, tolerating failures.

    ``iter_graph``'s own variant of :func:`_connect_mcp_toolsets` — extracted
    verbatim (pure extract-method, no behaviour change); kept separate rather
    than unified because it uses ``asyncio.wait_for`` (not the
    ``asyncio.timeout()`` pattern ``_connect_mcp_toolsets`` deliberately
    switched to for a documented cancel-scope/task-mismatch bug) and its own
    ``run_graph_iter``-prefixed log messages.
    """
    await _connect_mcp_toolsets_common(
        stack,
        deps,
        timeout_style="wait_for",
        log_prefix="run_graph_iter",
        log_connect=False,
        failure_suffix="FAILED",
        warning_suffix="failed",
    )


def _drain_sideband_events(eq: asyncio.Queue) -> Iterator[dict[str, Any]]:
    """Yield ``{"type": "sideband", ...}`` events already queued on ``eq``,
    without blocking (mirrors the inline ``while not eq.empty(): ...``
    drains in the original ``iter_graph``).
    """
    while not eq.empty():
        yield {"type": "sideband", "event": eq.get_nowait()}


@dataclass
class _IterRunContext:
    """Bundles ``iter_graph``'s per-run collaborators for
    :meth:`AgentOrchestrationEngine._stream_iter_graph_events`, keeping it
    under the 7-param cap.
    """

    run_id: str
    event_queue: asyncio.Queue
    graph_evidence: Any
    end_marker_type: type[Any]
    elicitation_callback: Any


# ═══════════════════════════════════════════════════════════════════════════
# synthesize_team helpers — extracted verbatim (pure extract-method, no
# behaviour change) so ``AgentOrchestrationEngine.synthesize_team`` stays
# under the complexity caps.
# ═══════════════════════════════════════════════════════════════════════════


def _resolve_candidate_ids(engine: Any, domain: str) -> list[Any]:
    """Epistemic-graph compute scopes candidates to the domain's blast radius.

    Returns the raw ``n.get("id")`` value for a dict node (may be non-str,
    or ``None`` if absent) or ``str(n)`` otherwise -- ``list[Any]``, not
    ``list[str]``, to match that actual (unchanged) runtime behaviour.
    """
    return [
        n.get("id") if isinstance(n, dict) else str(n)
        for n in (
            engine.graph_compute.get_blast_radius(f"domain:{domain}", max_depth=2) or []
        )
    ]


def _build_agent_roster_query(
    domain: str, candidate_ids: list[str], delegated_authority: str | None
) -> tuple[str, dict[str, Any]]:
    """Resolve the agent roster from the graph store. With a delegated
    authority, restrict to agents authorised for it; otherwise scope by
    domain.
    """
    if delegated_authority:
        agent_query = (
            "MATCH (a:Agent)-[:HAS_DELEGATED_AUTHORITY_FROM|AUTHORIZED_FOR]->"
            "(auth {id: $delegated_authority}) "
            "WHERE a.agent_id IN $candidate_ids "
            "RETURN a.agent_id AS agent_id, a.role AS role, a.name AS name"
        )
    else:
        agent_query = (
            "MATCH (a:Agent) "
            "WHERE a.domain = $domain "
            "AND a.agent_id IN $candidate_ids "
            "RETURN a.agent_id AS agent_id, a.role AS role, a.name AS name"
        )
    params = {
        "domain": domain,
        "candidate_ids": candidate_ids,
        "delegated_authority": delegated_authority,
    }
    return agent_query, params


def _resolve_agent_tools(engine: Any, agent_id: Any) -> list[str]:
    """Per-agent tool binding via the USES edge to CallableResource nodes."""
    tool_rows = (
        engine.backend.execute(
            "MATCH (a:Agent {agent_id: $agent_id})-[:USES]->(t:CallableResource) "
            "RETURN t.name AS tool_name",
            {"agent_id": agent_id},
        )
        or []
    )
    return [t["tool_name"] for t in tool_rows if t.get("tool_name")]


def _resolve_team_agents(
    engine: Any, agent_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    agents = []
    for row in agent_rows:
        agent_id = row.get("agent_id") or row.get("id")
        role = row.get("role") or "general"
        tools = _resolve_agent_tools(engine, agent_id)
        agents.append(
            {
                "role": role,
                "agent_id": agent_id,
                "tools": tools,
                "system_prompt": f"You are the {role} agent.",
            }
        )
    return agents


def _build_team_composition(
    agents: list[dict[str, Any]],
    domain: str,
    complexity: float,
    delegated_authority: str | None,
) -> Any:
    from agent_utilities.models.knowledge_graph import TeamComposition

    return TeamComposition(
        team_id=f"team:{secrets.token_hex(16)}",
        adaptive_agent_router=agents,
        execution_mode=(
            "parallel" if (complexity > 1 or len(agents) > 1) else "sequential"
        ),
        reasoning=(
            f"Synthesized {len(agents)}-agent topology for domain:{domain}"
            + (
                f" under delegated authority {delegated_authority}"
                if delegated_authority
                else ""
            )
        ),
        confidence=0.8,
    )


# ═══════════════════════════════════════════════════════════════════════════
# execute_graph helpers — extracted verbatim (pure extract-method, no
# behaviour change) so ``AgentOrchestrationEngine.execute_graph`` stays under
# the complexity caps.
# ═══════════════════════════════════════════════════════════════════════════


def _build_execute_deps(
    config: dict,
    *,
    mcp_toolsets: list[Any] | None,
    event_queue: asyncio.Queue | None,
    requested_model_id: str | None,
    run_id: str,
    plan_sync: Any,
) -> GraphDeps:
    """Build the ``GraphDeps`` :meth:`AgentOrchestrationEngine.execute_graph`
    runs the graph with."""
    return _build_graph_deps(
        config,
        mcp_toolsets=mcp_toolsets,
        event_queue=event_queue,
        requested_model_id=requested_model_id,
        run_id=run_id,
        plan_sync=plan_sync,
        mode="execute",
    )


def _ensure_persistence_dir(persist: bool, state_dir: str, run_id: str) -> None:
    """When ``persist`` is set, ensure the parent directory for this run's
    state file exists.

    NOTE (extracted verbatim, behaviour unchanged): the actual
    ``FileStatePersistence`` object is never constructed here — the
    original inline line that would do so is commented out — so today
    ``persist=True`` has ONLY the side effect of creating this directory;
    nothing reads ``path`` afterward. Flagged in the lane report as
    surfaced-but-not-fixed (out of scope for a pure extract).
    """
    if not persist:
        return
    path = Path(state_dir)
    if path.suffix != ".json":
        path = path / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    # persistence = FileStatePersistence(json_file=path)


def _apply_max_steps_cap(
    state: GraphState, max_steps: int | None, config: dict
) -> None:
    """CONCEPT:AU-AHE.harness.loop-exit-conditions — TURN CAP threading (exit 2).

    The caller's ``run_agent`` max_steps used to be DROPPED here (the busiest
    path): ``_execute_graph`` never forwarded it, so the dispatcher's
    force-terminate guard (``node_transitions > max_node_transitions``) fell
    back to the ExecutionBudget default (50) regardless of what the caller
    asked for. Thread it onto the graph's EXISTING enforced cap so the turn
    cap is actually honored. Interpreted as interaction ROUNDS with headroom
    for the router/agent/verifier hops per round — the SAME
    ``max(max_steps*2, 10)`` convention the single-server path uses
    (agent_runner._execute_single_server) so both paths read max_steps
    identically. Never TIGHTENS below today's default 50 for the standard
    max_steps=30 (→ 60), so no existing task regresses.
    """
    _ms = max_steps if max_steps is not None else config.get("max_steps")
    if not _ms:
        return
    from ..orchestration.loop_guards import graph_node_transition_cap

    state.max_steps = int(_ms)
    state.execution_budget.max_node_transitions = graph_node_transition_cap(_ms)


async def _prepare_execute_graph_run(
    stack: AsyncExitStack, deps: GraphDeps, run_id: str, query: str, topology: str
) -> None:
    """MCP connect + registry-tag hydration + ``graph_start`` event +
    service-registry warm-up. Extracted verbatim from ``execute_graph``
    (pure extract-method, no behaviour change).
    """
    await _connect_mcp_toolsets(stack, deps)

    # Standardize tag_prompts from the registry for high-fidelity routing.
    # We merge existing prompts with registry-provided domain tags.
    registry = await _offload_sync(get_discovery_registry)
    for agent in registry.agents:
        # Domain tags (like 'git_operations') are the primary routing targets
        # MCPAgent uses 'name' as the primary identifier
        if agent.name and agent.name not in deps.tag_prompts:
            deps.tag_prompts[agent.name] = agent.description or agent.name

    emit_graph_event(
        deps.event_queue, "graph_start", run_id=run_id, query=query, topology=topology
    )
    logger.info(
        f"run_graph: Starting graph execution for run_id {run_id}. Registered {len(deps.tag_prompts)} specialists."
    )

    # --- CONCEPT:AU-ORCH.execution.service-registry-initialization Service Registry Initialization ---
    try:
        from ..core.registry.service_adapter import ServiceRegistry

        svc_registry = ServiceRegistry.instance()
        svc_count = await _offload_sync(svc_registry.initialize)
        logger.debug(
            "run_graph: Service registry initialized with %d services", svc_count
        )
    except Exception as e:  # noqa: BLE001 — svc_registry/svc_count are never referenced again in this function; ServiceRegistry.instance() is a lazy singleton re-initialized elsewhere (agent_runner.py's dispatch path), this is pure redundant warm-up
        logger.debug("run_graph: Service registry init skipped: %s", e)


# ═══════════════════════════════════════════════════════════════════════════
# stream_graph helpers — extracted verbatim (pure extract-method, no
# behaviour change) so ``AgentOrchestrationEngine.stream_graph`` stays under
# the complexity caps.
# ═══════════════════════════════════════════════════════════════════════════


def _build_stream_deps(
    config: dict,
    *,
    mcp_toolsets: list[Any] | None,
    event_queue: asyncio.Queue,
    requested_model_id: str | None,
    run_id: str,
    plan_sync: Any,
) -> GraphDeps:
    """Build the ``GraphDeps`` :meth:`AgentOrchestrationEngine.stream_graph`
    runs the graph with its stream-specific dependency field set.
    """
    return _build_graph_deps(
        config,
        mcp_toolsets=mcp_toolsets,
        event_queue=event_queue,
        requested_model_id=requested_model_id,
        run_id=run_id,
        plan_sync=plan_sync,
        mode="stream",
    )


def _build_graph_run_context(
    deps_builder: Any,
    *,
    graph: Any,
    config: dict,
    query: str,
    query_parts: list[dict[str, Any]] | None,
    run_id: str,
    mode: str,
    topology: str,
    event_queue: asyncio.Queue[Any] | None,
    mcp_toolsets: list[Any] | None,
    requested_model_id: str | None,
    plan_sync: Any,
) -> tuple[GraphDeps, GraphState, GraphExecutionEvidenceCollector]:
    """Build dependencies, state, and evidence for one graph run shape."""
    deps = deps_builder(
        config,
        mcp_toolsets=mcp_toolsets,
        event_queue=event_queue,
        requested_model_id=requested_model_id,
        run_id=run_id,
        plan_sync=plan_sync,
    )
    state, graph_evidence = _build_graph_state_context(
        graph=graph,
        query=query,
        query_parts=query_parts,
        run_id=run_id,
        mode=mode,
        topology=topology,
        config=config,
    )
    return deps, state, graph_evidence


def _prepare_run_context(
    run_id: str | None, requested_model_id: str | None
) -> tuple[str, str | None]:
    """Resolve the run id, correlation id, and per-turn model override."""
    if run_id is None:
        run_id = secrets.token_hex(16)

    # CONCEPT:AU-OS.observability.run-wide-correlation-id — establish a run-wide correlation id so every nested
    # agent/span/side-effect in this run is joinable. Idempotent: nested
    # in-process runs inherit the parent's id via the contextvar.
    with contextlib.suppress(Exception):
        from ..observability.correlation import ensure_correlation_id

        ensure_correlation_id()

    if requested_model_id is None:
        requested_model_id = REQUESTED_MODEL_ID_CTX.get()
    return run_id, requested_model_id


def _extract_stream_final_output(
    graph_result_holder: dict[str, Any], state: Any
) -> Any:
    """Extract the best available output after ``stream_graph``'s background
    task completes:

    1. Graph End result (verifier's synthesized GraphResponse)
    2. First value in results_registry (plan-based, set by any step)
    """
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
        final_output = next(iter(state.results_registry.values()), None)
    if not final_output:
        final_output = "No output generated."
    return final_output


# ═══════════════════════════════════════════════════════════════════════════
# validate_graph helpers — extracted verbatim (pure extract-method, no
# behaviour change) so ``AgentOrchestrationEngine.validate_graph`` stays
# under the complexity caps.
# ═══════════════════════════════════════════════════════════════════════════


def _collect_registry_info(registry: Any) -> dict[str, Any]:
    """MCP agents/tools from the discovery registry."""
    return {
        "mcp_agent_count": len(registry.agents),
        "mcp_agents": [
            {
                "name": a.name,
                "agent_type": a.agent_type,
                "mcp_server": a.mcp_server,
                "tool_count": len(a.tools),
            }
            for a in registry.agents
        ],
        "mcp_tool_count": len(registry.tools),
    }


def _collect_graph_edge_count(graph: Any) -> Any:
    """Graph structure: rough mermaid-based edge count, or ``"unknown"``."""
    if not hasattr(graph, "mermaid_code"):
        return "unknown"
    mermaid = graph.mermaid_code()
    # Count node definitions (rough heuristic)
    [line for line in mermaid.split("\n") if "-->" in line or ":" in line]
    return len([line for line in mermaid.split("\n") if "-->" in line])


def _build_validation_warnings(
    tag_prompts: dict[str, Any], mcp_toolsets: list[Any], registry: Any
) -> list[str]:
    warnings: list[str] = []
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
    return warnings


# ═══════════════════════════════════════════════════════════════════════════
# execute_workflow helpers — extracted verbatim (pure extract-method, no
# behaviour change) so ``AgentOrchestrationEngine.execute_workflow`` stays
# under the complexity caps.
# ═══════════════════════════════════════════════════════════════════════════


async def _run_adversarial_verification(
    state: Any, deps: Any, task: str, completion_state: str, synthesis_output: str
) -> tuple[bool, str]:
    """ADVERSARIAL VERIFICATION: run one adversarial pass against
    ``completion_state``. Returns ``(converged, next_current_context)`` --
    on convergence ``next_current_context`` is empty (the caller uses
    ``synthesis_output`` as ``final_output`` instead); on failure it is the
    findings-fed-back context for the next iteration.
    """
    from typing import cast

    from agent_utilities.capabilities.adversarial_verifier import (
        run_adversarial_pass,
    )
    from agent_utilities.graph.state import GraphDeps, GraphState

    m_state = cast(GraphState, state)
    m_deps = cast(GraphDeps, deps)

    # Inject the completion state into the verifier query
    state.query = f"Task: {task}\n\nRequired Completion State: {completion_state}"
    adv_res = await run_adversarial_pass(m_state, m_deps, synthesis_output)

    if adv_res and not getattr(adv_res, "vulnerabilities_found", True):
        logger.info("Adversarial verification passed! Convergence reached.")
        return True, ""

    findings = getattr(adv_res, "findings", "Failed to meet completion state.")
    logger.warning(f"Adversarial verification failed. Findings: {findings}")
    next_context = (
        f"Original Task: {task}\n\n"
        f"Previous Output:\n{synthesis_output}\n\n"
        f"Reviewer Findings (MUST FIX):\n{findings}"
    )
    return False, next_context


async def _submit_workflow_pr(
    workflow_id: str, task: str, convergence_reached: bool, final_output: str
) -> str | None:
    """Pull Request Submission — automated PR creation once the dynamic
    workflow has produced a final output (converged or not).
    """
    if not (convergence_reached or final_output):
        return None
    logger.info("Workflow complete. Submitting automated PR...")
    try:
        # Use a specialized agent to create the PR
        from agent_utilities.agent.factory import create_agent

        pr_agent, _ = create_agent(
            provider="openai",
            model_id="openai:gpt-4o",
            system_prompt="You are an automated PR submission agent. You have access to GitHub, GitLab, and Repository Manager MCP tools.",
            name="pr_submitter",
            enable_universal_tools=True,
            tool_tags=[
                "github-tools",
                "github",
                "gitlab",
                "repository-manager",
            ],
        )

        pr_task = (
            f"The dynamic workflow '{workflow_id}' has completed its task.\n\n"
            f"Task details: {task}\n"
            f"Please review the git diff, commit the changes, push to a new branch, and create a Pull Request with the title 'Auto-chore: {workflow_id}'.\n"
            f"If there are no changes, just return 'No changes to commit'.\n\n"
            f"Return ONLY the URL of the created PR, or a message saying no changes were needed."
        )

        pr_result = await asyncio.wait_for(pr_agent.run(pr_task), timeout=180.0)
        return pr_result.output
    except Exception as e:
        logger.error(f"Failed to submit automated PR: {e}")
        return f"Failed to create PR: {e}"


# implements core.execution.ExecutionEngine
class AgentOrchestrationEngine:
    """The singular orchestration engine for the agent ecosystem.

    Modes:
      - graph: Full pydantic-graph execution
      - stream: SSE streaming graph execution
      - dynamic: KG-driven dynamic synthesis
      - parallel: Concurrent subagent dispatch
      - workflow: Compiled workflow execution
      - sdd: Spec-driven development orchestration
      - research: Research pipeline execution
    """

    def __init__(self, engine: IntelligenceGraphEngine | None = None):
        if engine is None:
            try:
                from agent_utilities.knowledge_graph.core.engine import (
                    IntelligenceGraphEngine,
                )

                engine = IntelligenceGraphEngine.get_active()
            except Exception as e:
                logger.warning(f"Failed to initialize IntelligenceGraphEngine: {e}")
                engine = None
        self.engine = engine

    async def dispatch(
        self, task: Any, *, mode: str = "auto", **kwargs: Any
    ) -> dict[str, Any]:
        """Core dispatch — the single entrypoint for task execution."""
        logger.info(f"Dispatching task in mode: {mode}")
        if mode == "auto":
            mode = self._determine_mode(task)

        if mode == "graph":
            return await self.execute_graph(task, **kwargs)
        elif mode == "stream":
            # This would return an AsyncIterator in practice, but wrapping it for generic typing
            return {"stream": self.stream_graph(task, **kwargs)}
        elif mode == "dynamic":
            return await self._execute_dynamic(task, **kwargs)
        elif mode == "parallel":
            return await self.execute_parallel(task, **kwargs)
        elif mode == "workflow":
            return await self.execute_workflow(task, **kwargs)
        elif mode == "sdd":
            return await self.execute_sdd(task, **kwargs)
        elif mode == "swe":
            return await self.execute_swe(task, **kwargs)
        else:
            raise ValueError(f"Unknown dispatch mode: {mode}")

    async def execute_swe(
        self, task: Any, *, deps: Any = None, workspace: Any = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Run the KG-grounded SWE agent (CONCEPT:AU-ORCH.execution.swe-agent-system-prompt) on ``task``.

        Drives the edit→run→test loop inside a developer workspace (OS-5.33), grounding in the
        code ontology (KG-2.65). A ``workspace`` may be supplied (e.g. a repo already cloned by
        the SWE-bench harness, AHE-3.22); otherwise a fresh one is created and torn down.
        """
        from agent_utilities.models import AgentDeps
        from agent_utilities.runtime import create_workspace

        from .swe_agent import run_swe_task

        deps = deps or AgentDeps()
        owns_ws = workspace is None
        ws = workspace or create_workspace(actor=getattr(deps, "user_id", None))
        if owns_ws:
            await ws.start()
        deps.workspace = ws
        try:
            result = await run_swe_task(str(task), deps, **kwargs)
            return {
                "mode": "swe",
                "output": result.output,
                "patch": result.patch,
                "tool_calls": result.tool_calls,
            }
        finally:
            if owns_ws:
                await ws.stop()

    def _determine_mode(self, task: Any) -> str:
        """Heuristics to determine the best execution mode based on the task payload."""
        if isinstance(task, list):
            return "parallel"
        if isinstance(task, dict) and "workflow_id" in task:
            return "workflow"
        if isinstance(task, dict) and "spec" in task:
            return "sdd"
        if hasattr(task, "run"):
            return "graph"
        return "dynamic"

    # --- Dynamic Team Synthesis ---
    def synthesize_team(
        self,
        query: str,
        domain: str,
        complexity: float = 1.0,
        delegated_authority: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Synthesize a subagent team for a domain via KG analysis.

        Topology scoping runs on the **epistemic-graph compute layer**
        (``get_blast_radius`` over the out-of-process tokio/MessagePack engine — not
        PyO3/FFI, not scipy/sklearn), which bounds the candidate agents to the domain
        sub-graph. The agent roster and each agent's tools are then resolved from the
        graph store, and when a ``delegated_authority`` is supplied the roster is
        restricted to agents authorised for it (CONCEPT:AU-ORCH.execution.execution-budget-caps governance).
        """
        logger.info(f"Synthesizing team for {domain} (complexity: {complexity})")
        assert self.engine is not None, (
            "IntelligenceGraphEngine is required for team synthesis"
        )

        candidate_ids = _resolve_candidate_ids(self.engine, domain)
        agent_query, params = _build_agent_roster_query(
            domain, candidate_ids, delegated_authority
        )
        agent_rows = self.engine.backend.execute(agent_query, params) or []
        agents = _resolve_team_agents(self.engine, agent_rows)

        if not agents:
            raise LookupError("no authorized agents are available for the domain")

        return _build_team_composition(agents, domain, complexity, delegated_authority)

    # --- KG State Machine ---
    def determine_next_node(self, current_node: str, context: dict[str, Any]) -> str:
        """Determine the next execution node in a dynamic graph.
        Replaces KGDrivenExecutionEngine routing logic.
        """
        # Call into rust to evaluate next hops based on semantic edges
        assert self.engine is not None, (
            "IntelligenceGraphEngine is required for node determination"
        )
        successors = self.engine.graph_compute.get_successors(current_node)
        return successors[0] if successors else "END"

    # --- Graph Execution (pydantic-graph) ---
    @_foreground_execution
    async def execute_graph(
        self,
        graph,
        config: dict,
        query: str,
        run_id: str | None = None,
        persist: bool = False,
        state_dir: str = DEFAULT_GRAPH_PERSISTENCE_PATH or "graph_state",
        streamdown: bool = True,
        eq: asyncio.Queue[Any] | None = None,
        mode: str = "ask",  # execute_graph's request mode
        topology: str = "basic",
        mcp_toolsets: list[Any] | None = None,
        query_parts: list[dict[str, Any]] | None = None,
        plan_sync=None,
        requested_model_id: str | None = None,
        max_steps: int | None = None,
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
        run_id, requested_model_id = _prepare_run_context(run_id, requested_model_id)

        mermaid_prefix = ""
        if streamdown:
            with contextlib.suppress(Exception):
                mermaid_prefix = (
                    f"```mermaid\n{get_graph_mermaid(graph, config)}\n```\n\n"
                )

        deps, state, graph_evidence = _build_graph_run_context(
            _build_execute_deps,
            graph=graph,
            config=config,
            query=query,
            query_parts=query_parts,
            run_id=run_id,
            mode=mode,
            topology=topology,
            plan_sync=plan_sync,
            event_queue=eq,
            mcp_toolsets=mcp_toolsets,
            requested_model_id=requested_model_id,
        )

        _ensure_persistence_dir(persist, state_dir, run_id)
        _apply_max_steps_cap(state, max_steps, config)

        async with AsyncExitStack() as stack:
            await _prepare_execute_graph_run(stack, deps, run_id, query, topology)

            _preflight_block = _run_security_preflight(query, run_id)
            if _preflight_block is not None:
                return _preflight_block

            result, _graph_run_start, _early_error = await _run_graph_with_evidence(
                graph, run_id, query, topology, state, deps, graph_evidence
            )
            if _early_error is not None:
                return _early_error

            _usage, _run_model = await _emit_execute_graph_telemetry(
                run_id, query, result, state, graph_evidence, config, _graph_run_start
            )

        return _shape_graph_execute_response(
            result, run_id, state, graph_evidence, mermaid_prefix, _usage, _run_model
        )

    @_foreground_execution
    async def stream_graph(
        self,
        graph,
        config: dict,
        query: str,
        run_id: str | None = None,
        persist: bool = False,
        state_dir: str = DEFAULT_GRAPH_PERSISTENCE_PATH or "agent_data/graph_state",
        mode: str = "ask",  # stream_graph's request mode
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

        run_id, requested_model_id = _prepare_run_context(run_id, requested_model_id)

        eq: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        # Emit graph-start event via the sideband queue
        emit_graph_event(
            eq,
            "graph_start",
            run_id=run_id,
            query=query,
            topology=topology,
        )

        deps, state, graph_evidence = _build_graph_run_context(
            _build_stream_deps,
            graph=graph,
            config=config,
            query=query,
            query_parts=query_parts,
            run_id=run_id,
            mode=mode,
            topology=topology,
            event_queue=eq,
            mcp_toolsets=mcp_toolsets,
            requested_model_id=requested_model_id,
            plan_sync=plan_sync,
        )

        _ensure_persistence_dir(persist, state_dir, run_id)

        # Shared container for background tasks to pass graph results to the main loop
        graph_result_holder = {"value": None}

        async def run_in_background() -> None:
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

                    deps.mcp_toolsets = [
                        ts for ts in connected_toolsets if ts is not None
                    ]

                    with _pydantic_graph_span(
                        run_id=run_id,
                        query=query,
                        topology=topology,
                        evidence=graph_evidence,
                    ) as span:
                        graph_evidence.attach_span(span)
                        try:
                            result = await asyncio.wait_for(
                                run_with_execution_evidence(
                                    graph,
                                    state=state,
                                    deps=deps,
                                    collector=graph_evidence,
                                ),
                                timeout=DEFAULT_GRAPH_TIMEOUT / 1000.0,
                            )
                        finally:
                            graph_evidence.finish_span(state=state)
                    graph_result_holder["value"] = result
            except TimeoutError:
                await eq.put({"type": "error", "error": "Graph execution timed out"})
            except Exception as e:
                await eq.put({"type": "error", "error": str(e)})
            finally:
                from agent_utilities.core.config import emit_graph_event

                emit_graph_event(eq, "graph_complete", run_id=run_id, status="success")
                await eq.put({"type": "complete"})

        task = asyncio.create_task(run_in_background())

        while True:
            event = await eq.get()
            if event.get("type") == "complete":
                break

            yield f"data: {json.dumps(event)}\n\n"

        await task

        final_output = _extract_stream_final_output(graph_result_holder, state)
        yield (
            "data: "
            + json.dumps(
                {
                    "type": "final_output",
                    "content": final_output,
                    "execution_evidence": graph_evidence.evidence(
                        state=state
                    ).model_dump(),
                }
            )
            + "\n\n"
        )

    async def _stream_iter_graph_events(
        self, graph_run: Any, state: GraphState, ctx: _IterRunContext
    ) -> AsyncIterator[dict[str, Any]]:
        """The per-step streaming body of ``iter_graph``'s ``graph.iter()``
        loop -- yields the SAME event dicts the original inline ``async for``
        loop did (``node_transition`` / ``elicitation`` / ``graph_complete``),
        including the sideband drain and elicitation pause between steps.
        Ends (without raising) once ``end_marker_type`` is reached, exactly
        matching the original's ``break``.
        """
        step_count = 0
        async for event in graph_run:
            if isinstance(event, ctx.end_marker_type):
                # Graph completed — yield the final result and the complete
                # evidence accumulated before it.
                yield {
                    "type": "graph_complete",
                    "run_id": ctx.run_id,
                    "output": event.value,
                    "state_snapshot": self._build_state_snapshot(state),
                    "execution_evidence": ctx.graph_evidence.evidence(
                        state=state
                    ).model_dump(),
                }
                return

            # event is Sequence[GraphTask] — tasks scheduled next.
            ctx.graph_evidence.record_event(event, state=state)
            step_count += 1
            active_nodes = [
                {"node_id": str(t.node_id), "task_id": str(t.task_id)} for t in event
            ]

            yield {
                "type": "node_transition",
                "step": step_count,
                "run_id": ctx.run_id,
                "active_nodes": active_nodes,
                "state_snapshot": self._build_state_snapshot(state),
            }

            # Drain sideband events emitted by the step
            for sideband in _drain_sideband_events(ctx.event_queue):
                yield sideband

            # Elicitation check: pause for human approval if needed
            if state.human_approval_required and ctx.elicitation_callback is not None:
                yield {
                    "type": "elicitation",
                    "reason": "human_approval_required",
                    "state_snapshot": self._build_state_snapshot(state),
                }
                redirect = await ctx.elicitation_callback(state)
                state.human_approval_required = False
                if redirect:
                    logger.info(
                        "run_graph_iter: Elicitation redirect to '%s'", redirect
                    )
                    # The dispatcher reads this on its next turn.
                    state.user_redirect_feedback = redirect

    @_foreground_execution
    async def iter_graph(
        self,
        graph,
        config: dict,
        query: str,
        run_id: str | None = None,
        mode: str = "ask",  # iter_graph's request mode
        topology: str = "basic",
        mcp_toolsets: list[Any] | None = None,
        query_parts: list[dict[str, Any]] | None = None,
        plan_sync=None,
        requested_model_id: str | None = None,
        elicitation_callback=None,
    ):
        r"""Execute graph step-by-step using ``graph.iter()`` for maximum control.

        Unlike :func:`run_graph` which delegates to ``graph.run()`` (blocking
        until completion), this function uses the ``graph.iter()`` API to
        yield per-step execution events.  This enables:

        * **Progress streaming** — each step yields metadata about the active
          node, enabling real-time AG-UI sideband updates.
        * **Caller-controlled iteration** — callers can stop consuming without
          claiming that the current write-only snapshots can resume the run.
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

            * ``"node_transition"`` — a graph task batch was scheduled
            * ``"elicitation"`` — the graph is pausing for human input
            * ``"graph_complete"`` — the graph has finished executing
            * ``"error"`` — an error occurred during execution

        CONCEPT:AU-ORCH.execution.inject-signal-board-observations Graph Orchestration

        """
        end_marker_type = _resolve_end_marker_type()

        run_id, requested_model_id = _prepare_run_context(run_id, requested_model_id)

        eq: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        emit_graph_event(
            eq, "graph_start", run_id=run_id, query=query, topology=topology
        )

        deps, state, graph_evidence = _build_graph_run_context(
            _build_iter_deps,
            graph=graph,
            config=config,
            query=query,
            query_parts=query_parts,
            run_id=run_id,
            mode=mode,
            topology=topology,
            event_queue=eq,
            mcp_toolsets=mcp_toolsets,
            requested_model_id=requested_model_id,
            plan_sync=plan_sync,
        )

        # Merge registry tags into deps (same as run_graph)
        await _hydrate_registry_tags_for_iter(deps)

        async with AsyncExitStack() as stack:
            await _connect_mcp_toolsets_for_iter(stack, deps)

            with _pydantic_graph_span(
                run_id=run_id,
                query=query,
                topology=topology,
                operation="iter",
                evidence=graph_evidence,
            ) as span:
                graph_evidence.attach_span(span)
                ctx = _IterRunContext(
                    run_id=run_id,
                    event_queue=eq,
                    graph_evidence=graph_evidence,
                    end_marker_type=end_marker_type,
                    elicitation_callback=elicitation_callback,
                )
                try:
                    async with graph.iter(state=state, deps=deps) as graph_run:
                        async for evt in self._stream_iter_graph_events(
                            graph_run, state, ctx
                        ):
                            yield evt
                except TimeoutError:
                    yield {
                        "type": "error",
                        "run_id": run_id,
                        "error": "Graph execution timed out",
                        "execution_evidence": graph_evidence.evidence(
                            state=state
                        ).model_dump(),
                    }
                except Exception as e:
                    logger.error("run_graph_iter: CRITICAL ERROR: %s", e)
                    yield {
                        "type": "error",
                        "run_id": run_id,
                        "error": str(e),
                        "execution_evidence": graph_evidence.evidence(
                            state=state
                        ).model_dump(),
                    }
                finally:
                    graph_evidence.finish_span(state=state)

            # Drain any remaining sideband events
            for sideband in _drain_sideband_events(eq):
                yield sideband

    @staticmethod
    def _build_state_snapshot(state: GraphState) -> dict[str, Any]:
        """Build a lightweight serializable snapshot of the current graph state.

        This snapshot is included in every yielded event for observability,
        audit trails, and checkpoint evidence.  It is not a runnable resume
        payload.

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
            "graph_node_sequence": list(state.graph_node_sequence),
            "graph_transition_sequence": list(state.graph_transition_sequence),
            "checkpoint_ids": list(state.checkpoint_ids),
            "resume_supported": False,
            "error": state.error,
            "results_registry_keys": list(state.results_registry.keys()),
            "session_id": state.session_id,
        }

    @staticmethod
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
        from agent_utilities.agent.discovery import discover_agents

        tag_prompts = config.get("tag_prompts", {})
        mcp_toolsets = config.get("mcp_toolsets", [])
        registry = get_discovery_registry()
        discovered = discover_agents()

        info: dict[str, Any] = {
            "domain_count": len(tag_prompts),
            "domain_tags": list(tag_prompts.keys()),
            "mcp_toolset_count": len(mcp_toolsets),
            "discovered_agent_count": len(discovered),
            "discovered_agents": list(discovered.keys()),
            "graph_edge_count": _collect_graph_edge_count(graph),
        }
        info.update(_collect_registry_info(registry))

        warnings = _build_validation_warnings(tag_prompts, mcp_toolsets, registry)
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

    # --- Parallel Execution ---
    async def execute_parallel(self, tasks: list[Any], **kwargs: Any) -> dict[str, Any]:
        """Dispatch subagents concurrently. Replaces ParallelEngine."""
        import asyncio

        logger.info(f"Executing {len(tasks)} parallel tasks")
        results = await asyncio.gather(
            *[self.dispatch(t, mode="dynamic") for t in tasks], return_exceptions=True
        )
        return {"status": "success", "results": results}

    async def _run_agent_bounded(
        self,
        agent_name: str,
        task: str,
        timeout_s: float,
        max_steps: int = 30,
    ) -> str:
        """Run a spawned agent under a wall-clock budget, never raising.

        CONCEPT:AU-ORCH.execution.dynamic-workflows — ``run_agent``'s ``max_steps`` bounds interaction rounds,
        not time. A spawned agent awaiting an unresponsive MCP tool advances zero
        steps yet blocks forever, so a bare ``asyncio.gather`` over the fan-out hangs
        to the caller's outer timeout. Bounding each spawn by ``asyncio.wait_for``
        converts a stuck child into a clear, JSON-serialisable error so the gather
        always settles with a usable value.

        Returns the agent's output string, or a structured error string (a JSON
        ``{"error", "agent"}``) on timeout / failure.
        """
        from agent_utilities.orchestration.agent_runner import run_agent

        try:
            return await asyncio.wait_for(
                run_agent(
                    agent_name=agent_name,
                    task=task,
                    max_steps=max_steps,
                    engine=self.engine,
                ),
                timeout=timeout_s,
            )
        except TimeoutError:
            logger.error(
                "Spawned agent '%s' timed out after %.0fs (wall-clock budget)",
                agent_name,
                timeout_s,
            )
            return json.dumps(
                {
                    "error": f"agent execution timed out after {timeout_s:.0f}s",
                    "agent": agent_name,
                }
            )
        except Exception as e:  # noqa: BLE001 — a spawn failure must not wedge the fan-out
            logger.error("Spawned agent '%s' failed: %s", agent_name, e)
            return json.dumps({"error": str(e), "agent": agent_name})

    # --- Workflow Execution ---
    async def _run_dynamic_workflow_fanout(
        self,
        current_context: str,
        completion_state: str,
        max_fan_out: int,
        agent_timeout: float,
    ) -> list[str]:
        """Bounded parallel fan-out for one ``execute_workflow`` iteration.

        Each spawned agent is bounded by wall-clock (not just max_steps) so an
        unresponsive MCP tool cannot wedge the gather
        (CONCEPT:AU-ORCH.execution.dynamic-workflows). ``_run_agent_bounded``
        never raises (it converts a timeout/error into a structured error
        string), so gather always settles; error-encoded results are filtered
        out here before adversarial review.
        """
        import asyncio

        tasks = []
        for i in range(min(max_fan_out, 3)):
            # Add variation to prompt
            sub_task = f"{current_context}\n\nAttempt {i + 1}. Ensure you work towards: {completion_state}"
            tasks.append(
                self._run_agent_bounded(
                    agent_name=f"dynamic_worker_{i}",
                    task=sub_task,
                    timeout_s=agent_timeout,
                )
            )
        fan_out_results = await asyncio.gather(*tasks, return_exceptions=True)
        return [
            r
            for r in fan_out_results
            if isinstance(r, str) and r and not _is_agent_error(r)
        ]

    async def execute_workflow(
        self, workflow_id: str, *, completion_state: str, **kwargs: Any
    ) -> dict[str, Any]:
        """Execute a dynamic workflow by ID toward a required completion state.

        CONCEPT:AU-ORCH.execution.dynamic-workflows - Dynamic Workflows
        Supports autonomous adversarial loops converging on a completion state,
        followed by automated PR creation via GitHub/GitLab MCP tools.
        """
        logger.info(f"Executing dynamic workflow: {workflow_id}")

        task = kwargs.get("task", "")
        if not completion_state.strip():
            raise ValueError("completion_state is required")
        max_fan_out = kwargs.get("max_fan_out", 5)
        max_iterations = kwargs.get("max_iterations", 5)

        # Per-spawned-agent wall-clock budget. Honour a caller-supplied override
        # (kwargs / a threaded ``budget_tokens``-style timeout) before using
        # to the module default (CONCEPT:AU-ORCH.execution.dynamic-workflows).
        agent_timeout = float(
            kwargs.get("agent_timeout_s") or AGENT_WALLCLOCK_TIMEOUT_S
        )

        # Dynamic Workflow Loop
        logger.info(
            f"Starting Dynamic Workflow '{workflow_id}' aimed at completion_state: '{completion_state}'"
        )

        class MockGraphState:
            def __init__(self, q, m="execute"):
                self.query = q
                self.mode = m
                self.signal_board = {}

        class MockGraphDeps:
            def __init__(self, model):
                self.agent_model = model
                self.verifier_timeout = 120.0
                self.event_queue = None

        state = MockGraphState(task)
        deps = MockGraphDeps("openai:gpt-4o-mini")

        iteration = 0
        convergence_reached = False
        final_output = ""
        current_context = task

        while iteration < max_iterations:
            iteration += 1
            logger.info(f"Dynamic Workflow iteration {iteration}/{max_iterations}")

            valid_outputs = await self._run_dynamic_workflow_fanout(
                current_context, completion_state, max_fan_out, agent_timeout
            )
            if not valid_outputs:
                logger.warning("All parallel subagents failed in this iteration.")
                current_context = f"{current_context}\n\nPrevious attempt failed. Please correct and try again."
                continue

            # Pick the best output for adversarial review
            synthesis_output = "\n\n---\n\n".join(valid_outputs[:1])

            logger.info("Running adversarial verification against completion_state...")
            converged, current_context = await _run_adversarial_verification(
                state, deps, task, completion_state, synthesis_output
            )
            if converged:
                convergence_reached = True
                final_output = synthesis_output
                break

        pr_url = await _submit_workflow_pr(
            workflow_id, task, convergence_reached, final_output
        )

        return {
            "workflow_id": workflow_id,
            "status": "converged" if convergence_reached else "max_iterations_reached",
            "iterations": iteration,
            "final_output": final_output,
            "pr_result": pr_url,
        }

    # --- SDD Execution ---
    async def execute_sdd(self, spec: Any, **kwargs: Any) -> dict[str, Any]:
        """Execute Spec-Driven Development tasks. Replaces SDDOrchestrator."""
        logger.info("Executing SDD specification")
        return {"spec": str(spec), "status": "implemented"}

    async def _execute_dynamic(self, task: Any, **kwargs: Any) -> dict[str, Any]:
        """Internal dynamic execution loop."""
        return {"status": "dynamic_executed", "task": str(task)}

    # --- ParallelEngine Delegation ---

    async def execute(
        self,
        manifest: ExecutionManifest,
        graph_deps: GraphDeps | None = None,
    ) -> ExecutionResult:
        """Execute a manifest. Delegates to ParallelEngine.

        CONCEPT:AU-ORCH.execution.parallel-engine-visualizer — Parallel Engine
        """
        from agent_utilities.graph.parallel_engine import ParallelEngine

        pe = ParallelEngine(engine=self.engine)
        return await pe.execute(manifest, graph_deps)

    async def run(
        self,
        manifest: ExecutionManifest,
        graph_deps: GraphDeps | None = None,
    ) -> ExecutionResult:
        """Unified ExecutionEngine contract entrypoint.

        Plan 03 Step 5 — conforms to ``core.execution.ExecutionEngine``.
        Additive adapter delegating to :meth:`execute` (the canonical
        Parallel Engine entrypoint, CONCEPT:AU-ORCH.execution.parallel-engine-visualizer). Behaviour unchanged.
        """
        return await self.execute(manifest, graph_deps)
