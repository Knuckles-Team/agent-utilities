"""Agent Runner — CONCEPT:AU-ORCH.routing.mcp-child-error-unwrap KG-to-LLM Execution Bridge.

Bridges the ``graph_orchestrate`` MCP tool to the
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
import inspect
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agent_utilities.core.config import setting
from agent_utilities.orchestration.execution_contract import (
    ExecutionMode,
    missing_required_tools,
    validate_execution_mode,
    validate_pydantic_graph_contract,
    validate_tool_contract,
)
from agent_utilities.orchestration.response_format import (
    ResponseFormat,
    validate_response_format,
)
from agent_utilities.orchestration.run_identity import new_run_id

if TYPE_CHECKING:
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
    from agent_utilities.orchestration.execution_profile import ExecutionProfile

logger = logging.getLogger(__name__)


async def _call_without_blocking(
    operation: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    """Run a capability without executing synchronous work on the event loop."""

    if inspect.iscoroutinefunction(operation):
        return await operation(*args, **kwargs)
    result = await asyncio.to_thread(operation, *args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


def _resolve_context_ref(
    engine: IntelligenceGraphEngine,
    context_ref: str,
    run_id: str,
) -> str | None:
    """Resolve and provenance-link a persisted context in one blocking worker."""

    rows = engine.query_cypher(
        "MATCH (c:ContextBlob) WHERE c.id = $id "
        "RETURN c.id AS id, c.content AS content",
        {"id": context_ref},
    )
    if not rows or not rows[0].get("content"):
        return None
    add_edge = getattr(engine, "add_edge", None)
    if callable(add_edge):
        with contextlib.suppress(Exception):
            from agent_utilities.observability.trace_ontology import trace_id

            add_edge(trace_id(run_id), context_ref, "HAS_CONTEXT")
    return str(rows[0]["content"])


def _anchor_run_to_session(
    engine: IntelligenceGraphEngine,
    *,
    session_id: str,
    run_id: str,
) -> None:
    """Persist the session/run relationship through the synchronous graph API."""

    from agent_utilities.observability.trace_ontology import trace_id

    snode = f"session:{session_id}"
    engine.add_node(
        snode, "Session", properties={"id": snode, "session_id": session_id}
    )
    engine.add_edge(snode, trace_id(run_id), "HAS_RUN")


def _render_agent_result(
    output: Any,
    *,
    run_id: str,
    return_mermaid: bool,
    mermaid: Any = None,
    channel_id: str | None = None,
    run_summary: dict[str, Any] | None = None,
    execution_evidence: dict[str, Any] | None = None,
) -> str:
    """Render one agent result through the sole public delegation envelope.

    CONCEPT:AU-ORCH.execution.messaging-orchestration-transparency — ``run_summary`` (when a
    caller opts in) rides the SAME envelope as ``mermaid``/``channel_id``: additive, opt-in,
    and never changes the bare-string contract for a caller that asks for none of the three.
    """

    output_text = str(output)
    if not return_mermaid and not channel_id and run_summary is None:
        return output_text
    payload: dict[str, Any] = {"output": output_text, "run_id": run_id}
    if return_mermaid:
        payload["mermaid"] = mermaid
    if channel_id:
        payload["channel_id"] = channel_id
    if run_summary is not None:
        payload["run_summary"] = run_summary
    if execution_evidence is not None:
        payload["execution_evidence"] = execution_evidence
    return json.dumps(payload, default=str)


def _trace_evidence_for_run(
    *,
    model_ref: str,
    model_class: str,
    skill_used: str = "",
    skill_instruction_digest: str = "",
) -> dict[str, str]:
    """Build the closed opaque attribution contract exported with a run trace."""

    from agent_utilities.security.persistence_privacy import persistence_reference

    evidence = {"model_ref": model_ref, "model_class": model_class}
    if skill_used and skill_instruction_digest:
        evidence.update(
            {
                "skill_ref": persistence_reference(
                    "skill", skill_used, namespace="execution-trace"
                ),
                "skill_body_ref": persistence_reference(
                    "skill_body",
                    skill_instruction_digest,
                    namespace="skill-validation",
                ),
            }
        )
    return evidence


# CONCEPT:AU-ORCH.execution.passthrough-identity — prompt-only universal entrypoints that must flow through the full
# multi-agent graph AS THEMSELVES, never resolved to a KG specialist. Resolving one is pure
# waste (a multi-second semantic search) and actively wrong (it mis-binds the universal
# messaging assistant to an unrelated tag). Keep this to genuine pass-through identities.
_PASSTHROUGH_AGENTS = frozenset({"messaging-assistant"})

# CONCEPT:AU-AHE.harness.loop-exit-conditions — ERROR THRESHOLD (exit 7) on
# agent_runner's execution path. A process-wide per-agent ConsecutiveFailureGuard
# lifts the engine breaker's threshold+reset semantics to delegation OUTCOMES: a
# degraded/no-data run increments the agent's consecutive-failure run; any
# successful run resets it. This does NOT abort a single ``run_agent`` (that would
# change its one-shot contract); it TRACKS the signal + warns when tripped so a
# LOOP driving repeated ``run_agent`` calls (e.g. ``LoopController.run_loop``) can
# halt to the terminal ``error_threshold_exceeded`` instead of burning its turn
# cap on an agent that keeps failing. Consult it via
# :func:`consecutive_agent_failures`.
_AGENT_FAILURE_GUARDS: dict[str, Any] = {}


def _agent_failure_guard(agent_name: str) -> Any:
    """Return (creating if needed) the shared failure guard for ``agent_name``."""
    from agent_utilities.orchestration.loop_guards import ConsecutiveFailureGuard

    guard = _AGENT_FAILURE_GUARDS.get(agent_name)
    if guard is None:
        from agent_utilities.core.config import config as _cfg

        threshold = int(getattr(_cfg, "kg_loop_max_consecutive_failures", 3))
        guard = _AGENT_FAILURE_GUARDS[agent_name] = ConsecutiveFailureGuard(
            threshold=threshold
        )
    return guard


def consecutive_agent_failures(agent_name: str) -> int:
    """Current consecutive-degraded-run count for ``agent_name`` (0 if none).

    The read side of the exit-7 guard on the execution path: a loop driving
    repeated delegations can consult this and terminate ``error_threshold_exceeded``
    once it reaches the configured threshold.
    """
    guard = _AGENT_FAILURE_GUARDS.get(agent_name)
    return int(getattr(guard, "count", 0)) if guard is not None else 0


def _record_agent_outcome(agent_name: str, *, degraded: bool) -> None:
    """Feed one delegation outcome into the per-agent failure guard (exit 7)."""
    guard = _agent_failure_guard(agent_name)
    if degraded:
        if guard.record_failure():
            logger.warning(
                "[CONCEPT:AU-AHE.harness.loop-exit-conditions] agent %r has %d "
                "consecutive degraded runs (threshold %d) — a driving loop should "
                "halt to error_threshold_exceeded.",
                agent_name,
                guard.count,
                guard.threshold,
            )
    else:
        guard.record_success()


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


# ---------------------------------------------------------------------------
# Internal: run_summary — messaging-orchestration transparency
# (CONCEPT:AU-ORCH.execution.messaging-orchestration-transparency)
# ---------------------------------------------------------------------------


def _extract_failure_text(result: Any) -> str:
    """Best-effort REAL cause text from a degraded/failed structured result.

    Prefers an explicit ``error`` field (``GraphResponse``-shaped critical failures), then a
    ``metadata`` error/reason, then the rendered output text — which is where
    ``_fleet_server_failed_result`` puts its already-composed, truthful failure string. Only
    when NONE of those carry anything does this fall back to a synthesized "no output"
    description — so ``raw_failure`` is NEVER empty for a degraded/failed outcome, which is
    what lets :func:`_build_run_summary` always populate ``failure`` (never a silent drop back
    to the old hardcoded sentinel).
    """
    if not isinstance(result, dict):
        text = str(result or "").strip()
        return text or "the run produced no output"
    err = result.get("error")
    if err:
        return str(err)
    meta = result.get("metadata")
    if isinstance(meta, dict):
        m_err = meta.get("error") or meta.get("failure_reason")
        if m_err:
            return str(m_err)
    res = result.get("results")
    if isinstance(res, dict) and res.get("output"):
        return str(res["output"])
    return "the run produced no usable output"


def _build_run_summary(
    *,
    route: dict[str, Any],
    outcome: str,
    stage_reached: str,
    run_id: str,
    raw_failure: str | None,
    execution_mode: str = "other",
) -> dict[str, Any]:
    """Assemble the structured, chat-renderable summary of one delegation's routing + outcome.

    CONCEPT:AU-ORCH.execution.messaging-orchestration-transparency — every terminal outcome of
    ``run_agent`` carries this (via ``_render_agent_result``'s opt-in ``run_summary``) so a
    failure is a troubleshooting ENTRY POINT: a translated cause plus a ``trace_ref`` into the
    durable ``RunTrace`` this SAME run already writes (``_record_execution_trace`` /
    ``observability.trace_ontology``) — never an opaque "something failed". ``outcome`` is one
    of ``ok`` | ``degraded`` | ``failed`` | ``timeout``; ``failure`` is present iff
    ``raw_failure`` is given.
    """
    from agent_utilities.observability.trace_ontology import trace_id as _trace_id

    summary: dict[str, Any] = {
        "route": route,
        "outcome": outcome,
        "stage_reached": stage_reached,
        "trace_ref": _trace_id(run_id),
        "execution_mode": execution_mode,
    }
    if raw_failure:
        from agent_utilities.orchestration.failure_translation import (
            build_failure_detail,
        )

        summary["failure"] = build_failure_detail(raw_failure)
    return summary


# ---------------------------------------------------------------------------
# Internal: ProgressEvent stream — checkpoint-by-checkpoint execution transparency
# (CONCEPT:AU-ORCH.execution.messaging-orchestration-transparency)
# ---------------------------------------------------------------------------
#
# A LONG delegation is otherwise a black box: the caller sees nothing between "started"
# and the final answer. ``run_agent`` therefore emits a small ``ProgressEvent`` at each of
# its EXISTING checkpoints (routing decision, focused-tool binding, each fleet tool result,
# the evidence/retrieval-quality gate, the durable RunTrace write, final synthesis, and the
# terminal done/failure) to an OPTIONAL, caller-supplied ``progress_sink``. An entrypoint
# (the messaging router — Telegram/Mattermost/Teams) renders that stream into a live,
# edit-in-place status message so a long run is transparent step-by-step. This is the
# core-side half of the Universal-capability split: the STREAM is built once here; every
# entrypoint only RENDERS it.
#
# Two invariants make this strictly additive and safe on the just-stabilized delegation
# path: (1) ``progress_sink`` defaults to ``None`` and :func:`_emit` returns IMMEDIATELY on
# ``None`` — constructing no event and touching nothing — so a run with no sink behaves
# byte-for-byte as before; (2) every sink invocation is fire-and-forget: bounded by a short
# wall-clock and fully exception-isolated, so a slow or failing sink can NEVER stall or fail
# the actual run (only a genuine cancellation of the RUN itself propagates).

ProgressStage = str  # one of: start route evidence_gate tool_call tool_result
#                       synthesis checkpoint done failure
ProgressStatus = str  # one of: started ok degraded failed

# Defensive wall-clock ceiling for ONE sink invocation. A well-behaved sink (the messaging
# renderer coalesces edits and returns fast) never approaches this; it exists only so a
# misbehaving/hung sink cannot pause the run indefinitely.
_PROGRESS_SINK_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class ProgressEvent:
    """One checkpoint in a ``run_agent`` execution, streamed to an optional sink.

    CONCEPT:AU-ORCH.execution.messaging-orchestration-transparency — the additive, opt-in
    progress channel. Fields:

    * ``run_id``   — the run this event belongs to (the same handle used for ``trace_ref``).
    * ``stage``    — ``start`` | ``route`` | ``evidence_gate`` | ``tool_call`` |
      ``tool_result`` | ``synthesis`` | ``checkpoint`` | ``done`` | ``failure``.
    * ``status``   — ``started`` | ``ok`` | ``degraded`` | ``failed``.
    * ``detail``   — short human string (a server name, the route ``why``, a translated
      failure), safe to render straight into a chat surface.
    * ``evidence`` — small structured extras (servers, trace_ref, failure category, …);
      the paper's "evidence gating / checkpoint traces" surfaced as data.
    * ``ts``       — wall-clock emit time (``time.time()``).
    """

    run_id: str
    stage: ProgressStage
    status: ProgressStatus
    detail: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    ts: float = 0.0


# A sink is any async callable taking one ProgressEvent. ``None`` means "no streaming".
ProgressSink = Callable[[ProgressEvent], Awaitable[None]]


async def _emit(
    sink: ProgressSink | None,
    *,
    run_id: str,
    stage: ProgressStage,
    status: ProgressStatus,
    detail: str = "",
    evidence: dict[str, Any] | None = None,
) -> None:
    """Fire-and-forget one :class:`ProgressEvent` to ``sink``. No-op when ``sink`` is None.

    CONCEPT:AU-ORCH.execution.messaging-orchestration-transparency — the ONE choke point for
    progress emission, so the two safety invariants live in a single place:

    * **No-op default.** ``sink is None`` (the default for every existing caller) returns
      before constructing a ``ProgressEvent`` or importing anything — the run is byte-for-byte
      unchanged.
    * **Cannot break the run.** The sink call is bounded by ``_PROGRESS_SINK_TIMEOUT_S`` and
      every exception it raises (including a sink-side timeout) is swallowed. The SOLE thing
      that propagates is a genuine :class:`asyncio.CancelledError` of the surrounding run — a
      progress sink must never mask the run's own cancellation semantics.
    """
    if sink is None:
        return
    try:
        event = ProgressEvent(
            run_id=run_id,
            stage=stage,
            status=status,
            detail=detail,
            evidence=evidence or {},
            ts=time.time(),
        )
        await asyncio.wait_for(sink(event), timeout=_PROGRESS_SINK_TIMEOUT_S)
    except asyncio.CancelledError:
        # The RUN is being cancelled — never swallow that (it drives the timeout trace path).
        raise
    except BaseException as exc:  # noqa: BLE001 — a sink must NEVER affect the run outcome
        logger.debug(
            "run_agent: progress_sink emit swallowed (stage=%s, exc=%s)",
            stage,
            type(exc).__name__,
        )


def _record_delegation_over_budget(
    agent_name: str, elapsed_s: float, outcome: str
) -> None:
    """CONCEPT:AU-AHE.harness.runtime-reliability-loop — feed a delegation that ran over (a
    fraction of) its wall-clock budget into the runtime-reliability detect→gap loop.

    A run that is slow-but-not-wrong — an O(N) retrieval regression, engine contention — is
    invisible to the reward flywheel (success × speed still scores it a success), and a hard
    timeout is a caller-side cancellation that never becomes a graded run. Recording the
    over-budget elapsed makes a repeated pattern on one agent a SOURCE_RUNTIME gap. This is a
    buffer append only (all exceptions swallowed); no engine write on the run's hot path.
    """
    try:
        from agent_utilities.observability.runtime_signals import (
            KIND_DELEGATION_OVER_BUDGET,
            record_runtime_signal,
        )

        record_runtime_signal(
            KIND_DELEGATION_OVER_BUDGET,
            agent_name or "unknown",
            {
                "elapsed_s": round(float(elapsed_s), 1),
                "budget_s": _EXECUTE_AGENT_WALL_CLOCK_S,
                "outcome": outcome,
            },
        )
    except Exception:  # noqa: BLE001 — emission must never affect the run
        pass


def _prepare_spawn_delegation(
    agent_name: str,
    run_id: str,
    config: dict[str, Any],
) -> Any:
    """Build the per-agent on-behalf-of delegation for this spawn, or ``None`` when off.

    CONCEPT:AU-OS.identity.per-agent-on-behalf-delegation — connects the three primitives for
    THIS spawn: resolve the ultimate caller principal + ceiling, append ``agent:<name>:<run_id>``
    to the (possibly nested) delegation chain, RFC 8693-exchange the caller's token, and mint the
    run-scoped token. Populates ``config['invoker_capability_ceiling']`` so ``apply_tool_scope``
    can intersect the spawn's tools with the principal ceiling. Every step is best-effort and
    logged; a missing caller token or IdP simply leaves that leg empty (delegation still records
    the chain for provenance). Returns ``None`` in ``off`` mode (legacy identity).
    """
    from agent_utilities.security import delegation as _deleg

    mode = _deleg.delegation_mode()
    if mode is _deleg.DelegationMode.OFF:
        return None

    principal = _deleg.resolve_principal_identity()
    parent = _deleg.current_delegation()
    parent_chain = parent.chain if parent is not None else ()
    ultimate = (
        principal.principal or (parent.principal if parent else "") or "service:local"
    )

    # Ceiling flows to the spawned agent's GraphState so apply_tool_scope can narrow tools.
    if principal.ceiling:
        config["invoker_capability_ceiling"] = list(principal.ceiling)

    # RFC 8693 on-behalf-of exchange (decision 1) — best-effort; the authoritative chain is the
    # delegation array below, the exchanged token rides the envelope's oidc_token when supported.
    label = _deleg.agent_instance_label(agent_name, run_id)
    oidc_token: str | None = None
    try:
        from agent_utilities.mcp.delegated_auth import (
            exchange_token_for_agent,
            get_user_token,
        )

        if get_user_token():
            oidc_token = exchange_token_for_agent(label)
    except Exception as exc:  # noqa: BLE001 — exchange is best-effort (no IdP / grant not enabled)
        logger.info(
            "[delegation] on-behalf-of token exchange unavailable (%s); "
            "chain recorded without a delegated OIDC token",
            type(exc).__name__,
        )

    # Run-scoped token (decision 3) — endpoint scope from the resolved tool allow-list; fails
    # closed when delegation is on and no signing secret is configured.
    run_token = ""
    expires_at: float | None = None
    try:
        run_token, expires_at = _deleg.mint_spawn_run_token(
            run_id,
            principal=ultimate,
            tenant=principal.tenant,
            allowed_tools=config.get("invoker_allowed_tools"),
        )
    except Exception:
        # In `on` mode a missing secret MUST surface (config-contract); re-raise. In warn/off the
        # mint uses the ephemeral secret and does not reach here.
        if mode is _deleg.DelegationMode.ON:
            raise
        logger.warning("[delegation] run-token mint skipped", exc_info=False)

    delegation = _deleg.build_spawn_delegation(
        agent_name=agent_name,
        run_id=run_id,
        principal=ultimate,
        ceiling=principal.ceiling,
        run_token=run_token,
        parent_chain=parent_chain,
        oidc_token=oidc_token,
        expires_at=expires_at,
        mode=mode,
    )
    logger.info(
        "[delegation] mode=%s spawn=%s principal=%s chain_len=%d ceiling=%d "
        "run_token=%s oidc=%s",
        mode.value,
        label,
        "set" if principal.principal else "ambient",
        len(delegation.chain),
        len(delegation.ceiling),
        "minted" if run_token else "none",
        "exchanged" if oidc_token else "none",
    )
    return delegation


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
    reasoning_effort: str | None = None,
    model_class: str = "standard",
    response_format: ResponseFormat = "text",
    run_id: str | None = None,
    include_run_summary: bool = False,
    progress_sink: ProgressSink | None = None,
    required_tools: list[str] | None = None,
    skill_name: str | None = None,
    tool_server: str | None = None,
    execution_mode: ExecutionMode = "auto",
) -> str:
    """Execute a named agent using the KG-backed pydantic-graph pipeline.

    CONCEPT:AU-ORCH.routing.mcp-child-error-unwrap — KG-to-LLM Execution Bridge

    This is the primary entry point for ``graph_orchestrate``.
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
        return_mermaid: CONCEPT:AU-ORCH.execution.orchestration-flow-mermaid — when True,
            return a JSON object containing ``output`` and ``mermaid`` (null when no
            diagram was produced). Default False preserves the bare-string contract relied on
            by internal callers (e.g. the dynamic-workflow fan-out in
            ``engine.execute_workflow``, which filters on ``isinstance(r, str)``).
        run_id: CONCEPT:AU-ORCH.execution.messaging-orchestration-transparency — an optional
            caller-minted run handle (``run_identity.new_run_id()``). A caller that wants a
            ``trace_ref`` to survive even a HARD cancellation of this call (e.g. the messaging
            router's reply-budget ``asyncio.wait_for`` wall, which tears this coroutine down
            before it can return anything) pre-generates one and passes it here, then reuses
            the SAME id to build its own trace reference regardless of how this call ends.
            ``None`` (the default) mints a fresh one, unchanged from prior behavior.
        include_run_summary: CONCEPT:AU-ORCH.execution.messaging-orchestration-transparency —
            when True, the envelope (forced on, like ``return_mermaid``/``channel_id``) carries
            a ``run_summary`` — ``{route, outcome, stage_reached, trace_ref, failure?}`` — for
            EVERY terminal outcome (success, degraded, failed, or a best-effort one recorded
            just before a cancellation re-raises). Opt-in and additive: default False keeps the
            bare-string contract bit-for-bit for every existing caller.
        progress_sink: CONCEPT:AU-ORCH.execution.messaging-orchestration-transparency — an
            OPTIONAL async callable ``(ProgressEvent) -> Awaitable[None]`` that receives a
            checkpoint-by-checkpoint stream of this run (routing, focused-tool binding, each
            fleet tool result, the evidence/retrieval-quality gate, the durable trace write,
            synthesis, and the terminal done/failure) so a long delegation is transparent
            step-by-step in a chat surface. Strictly additive and default ``None``: with no
            sink the run behaves byte-for-byte as before (see :func:`_emit`). Every emission
            is fire-and-forget and fully exception-isolated — a slow or failing sink can never
            stall or fail the run.

    Returns:
        The synthesized result string from the graph execution, or, when ``return_mermaid``,
        ``open_channel``/``session_id``, or ``include_run_summary`` is set, a JSON string with
        ``output`` plus whichever of ``mermaid``/``channel_id``/``run_summary`` were requested.

    """
    response_format = validate_response_format(response_format)
    requested_execution_mode = validate_execution_mode(execution_mode)
    allowed_tools, required_tools = validate_tool_contract(
        allowed_tools, required_tools
    )
    if skill_name and skill_name != agent_name:
        raise ValueError("skill_name must match the dispatched agent_name")
    if tool_server and not skill_name:
        raise ValueError("tool_server requires skill_name")
    validate_pydantic_graph_contract(
        requested_execution_mode,
        skill_name=skill_name,
        tool_server=tool_server,
        allowed_tools=allowed_tools,
    )
    run_id = run_id or new_run_id()
    start_time = time.monotonic()
    actual_execution_mode = "other"
    logger.info(
        "[ORCH-1.21] Starting agent execution: agent=%s, run_id=%s, task=%.100s...",
        agent_name,
        run_id,
        task,
    )
    # CONCEPT:AU-ORCH.execution.messaging-orchestration-transparency — first checkpoint of the
    # progress stream (no-op when progress_sink is None). Lets a chat surface post its "working
    # on it…" status the instant the delegation begins, not only when the answer lands.
    await _emit(
        progress_sink,
        run_id=run_id,
        stage="start",
        status="started",
        detail=agent_name,
        evidence={"agent": agent_name},
    )
    # CONCEPT:AU-OS.observability.telemetry-observability (X2) — one OTel span per
    # run_agent execution, closed by _record_execution_trace on EVERY exit path
    # (success/degraded/failed/enterprise). Best-effort: OTel unconfigured (the
    # default) makes this a clean no-op, never affects the run.
    try:
        from agent_utilities.observability import get_telemetry_engine

        get_telemetry_engine().on_graph_start(
            run_id=run_id, agent_id=agent_name, query=task
        )
    except Exception as exc:  # noqa: BLE001 — tracing must never break a run
        logger.debug(
            "run_agent: OTel span start skipped (exception_type=%s)",
            type(exc).__name__,
        )

    # Step 1: Resolve engine
    engine = engine or _get_or_create_engine()

    if agent_name.lower() == "enterprise":
        from agent_utilities.graph.manifest_generators import manifest_for_enterprise
        from agent_utilities.graph.parallel_engine import ParallelEngine

        logger.info(
            "[ORCH-1.9] Executing full Enterprise Autonomous Company orchestration"
        )
        manifest = await _call_without_blocking(manifest_for_enterprise, task, engine)
        pe = ParallelEngine(engine=engine)

        try:
            pe_result = await pe.execute(manifest)
            duration_ms = (time.monotonic() - start_time) * 1000
            await _call_without_blocking(
                _record_execution_trace,
                engine,
                run_id,
                "enterprise",
                task,
                status="completed",
                duration_ms=duration_ms,
                result_preview=str(pe_result)[:500],
                execution_mode="parallel_engine",
            )
            return _render_agent_result(
                pe_result, run_id=run_id, return_mermaid=return_mermaid
            )
        except Exception as e:
            logger.error("[ORCH-1.9] Enterprise execution failed: %s", e)
            await _call_without_blocking(
                _record_execution_trace,
                engine,
                run_id,
                "enterprise",
                task,
                status="failed",
                error=str(e),
                execution_mode="parallel_engine",
            )
            return _render_agent_result(
                f"Enterprise execution failed: {e}",
                run_id=run_id,
                return_mermaid=return_mermaid,
            )

    # Step 1b: Check if agent_name maps to a native ServiceRegistry capability (e.g. trading_swarm)
    try:
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
                result = None
                handled = False
                if hasattr(instance, "analyze"):
                    handled = True
                    # Specifically for TradingSwarm
                    try:
                        task_data = json.loads(task)
                    except Exception:
                        task_data = {"raw_task": task}

                    result = await _call_without_blocking(instance.analyze, task_data)
                elif hasattr(instance, "select_pattern"):
                    handled = True
                    # Specifically for SubagentPatternRouter
                    result = await _call_without_blocking(
                        instance.select_pattern, needs_collaboration=True
                    )
                elif hasattr(instance, "run"):
                    handled = True
                    result = await _call_without_blocking(instance.run, task)
                elif hasattr(instance, "execute"):
                    handled = True
                    result = await _call_without_blocking(instance.execute, task)
                if handled:
                    await _call_without_blocking(
                        _record_execution_trace,
                        engine,
                        run_id,
                        agent_name,
                        task,
                        status="completed",
                        duration_ms=(time.monotonic() - start_time) * 1000,
                        result_preview=str(result)[:500],
                        execution_mode="service_registry",
                    )
                    return _render_agent_result(
                        result, run_id=run_id, return_mermaid=return_mermaid
                    )
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

    shape = await _call_without_blocking(
        plan_execution_shape,
        task,
        profile_hint=execution_profile,
        engine=engine,
    )

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
    # An explicit agent is a routing constraint, not a hint for the task lexical
    # planner.  In particular, the focused-tools shape is planned from ``task``
    # independently of ``agent_name``; skipping resolution here let that shape bind
    # a different server even when the caller had pinned one.
    if (
        shape.resolve_agent or agent_name.strip()
    ) and agent_name.strip().lower() not in _PASSTHROUGH_AGENTS:
        agent_meta = await _call_without_blocking(
            _resolve_agent_from_kg, engine, agent_name
        )
    else:
        agent_meta = _unresolved_agent_meta()

    if skill_name:
        if not agent_meta.get("skill_id"):
            reason, degraded = await _call_without_blocking(
                _skill_unrunnable_reason, engine, skill_name
            )
            if degraded:
                # D-SNV-5: the precondition READ failed (e.g. a transient
                # engine/session error) — that is not an honest negative and
                # must never be phrased as "is not runnable", which a caller
                # or an operator reading the log would take as a confirmed,
                # actionable finding about the skill itself.
                raise RuntimeError(
                    f"could not determine whether skill '{skill_name}' is "
                    f"runnable: {reason}"
                )
            raise LookupError(
                f"ingested skill '{skill_name}' is not runnable: {reason}"
            )
        if tool_server:
            agent_meta = await _call_without_blocking(
                _bind_explicit_tool_server,
                engine,
                agent_meta,
                tool_server,
                skill_name,
                allowed_tools,
            )

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
        model_class=model_class,
        allowed_tools=allowed_tools,
    )
    config["response_format"] = response_format
    config["execution_mode"] = requested_execution_mode
    if skill_name:
        config["pinned_skill_name"] = skill_name
        config["pinned_skill_prompt"] = str(agent_meta.get("system_prompt") or "")
    # CONCEPT:AU-ORCH.session.carry-invoker — carry the invoker's curated context + token budget into the spawn.
    # context_ref resolves a persisted ContextBlob (cross-process handoff): fetch its content
    # from the epistemic-graph and link it to this run's RunTrace for provenance.
    if context_ref and not context:
        try:
            context = await _call_without_blocking(
                _resolve_context_ref, engine, context_ref, run_id
            )
        except Exception as _ctx_exc:  # noqa: BLE001
            logger.warning(
                "context_ref %s resolution failed: %s", context_ref, _ctx_exc
            )
    if context:
        config["invoker_context"] = context
    # CONCEPT:AU-AHE.harness.loop-exit-conditions — BUDGET CAP (exit 3), native by
    # default. A top-level ``run_agent`` gets a token budget even when the caller
    # passed none, so the ``UsageLimits.total_tokens_limit`` hard cap is threaded
    # onto EVERY spawned agent (the single-server loop and the graph spawn sites,
    # which enforce it via pydantic-ai UsageLimits) — not only explicit invoker
    # handoffs. The ResourceOptimizer session token budget is the default; a caller
    # may still pass an explicit ``budget_tokens`` (honored verbatim) and a
    # deployment can raise/lower ``SESSION_TOKEN_BUDGET``.
    effective_budget_tokens = budget_tokens
    if effective_budget_tokens is None:
        from agent_utilities.core.resource_optimizer import DEFAULT_TOKEN_BUDGET

        effective_budget_tokens = DEFAULT_TOKEN_BUDGET
    if effective_budget_tokens:
        config["invoker_budget_tokens"] = int(effective_budget_tokens)
    _bind_native_skill_toolset(
        config=config,
        agent_meta=agent_meta,
        agent_name=agent_name,
    )
    if cred_ref:
        config["invoker_cred_ref"] = cred_ref
    # CONCEPT:AU-ORCH.execution.delegation-reasoning-off — reasoning is an opt-in capability
    # (like RLM): a run that needs deliberation turns it ON per-execution by passing an
    # effort ("low"/"medium"/"high"); otherwise the deterministic tool loop leaves it OFF
    # (the fleet default). Threaded onto config so _execute_single_server can honor it.
    if reasoning_effort:
        config["reasoning_effort"] = str(reasoning_effort)
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

        channel_id = await _call_without_blocking(
            agent_channel.open_channel, engine, session_id or run_id, run_id
        )
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
    _skill_instruction_digest = str(
        agent_meta.get("skill_instruction_digest", "") or ""
    )
    from agent_utilities.security.persistence_privacy import persistence_reference

    _model_ref = persistence_reference(
        "model", config.get("agent_model"), namespace="orchestration-run"
    )
    _model_class = str(config.get("selected_model_class") or "")
    config["trace_evidence"] = _trace_evidence_for_run(
        model_ref=_model_ref,
        model_class=_model_class,
        skill_used=_skill_used,
        skill_instruction_digest=_skill_instruction_digest,
    )

    # CONCEPT:AU-OS.identity.per-agent-on-behalf-delegation — resolve THIS spawn's on-behalf-of
    # identity (exchange + chain + run-token + ceiling) once, up front. It is bound as ambient
    # for the execution block below (so the spawn's engine calls carry the delegation envelope in
    # `on` mode) and passed explicitly to the RunTrace so provenance records the chain regardless
    # of context scope. `off` mode returns None (legacy identity, zero overhead).
    # CONCEPT:AU-ORCH.routing.offload-sync-roundtrip — in `on`/`warn` mode this performs a
    # SYNCHRONOUS RFC 8693 token-exchange HTTP POST to the IdP token endpoint (a real network
    # round-trip, not just a KG hit); run it off the event loop like every other blocking
    # capability in this function.
    _spawn_delegation = await _call_without_blocking(
        _prepare_spawn_delegation, agent_name, run_id, config
    )

    # Step 4: Execute. A resolved single MCP-server agent runs a DETERMINISTIC
    # direct tool loop (bind only that server's toolset, no router); anything else
    # goes through the full multi-agent orchestration graph. Routing a one-server
    # task through the graph let the LLM router/dispatcher mis-route it (e.g. to a
    # verifier that ran on empty results), so the server's tools were never called.
    from agent_utilities.security.delegation import (
        enter_delegation as _enter_delegation,
    )
    from agent_utilities.security.delegation import (
        reset_delegation as _reset_delegation,
    )

    _delegation_token = _enter_delegation(_spawn_delegation)
    # CONCEPT:AU-ORCH.execution.messaging-orchestration-transparency — tracked alongside the
    # dispatch below (NOT re-derived from ``agent_meta``/``shape`` after the fact) so
    # ``route``/``stage_reached`` always reflect the branch that ACTUALLY ran, including a
    # fallback sub-branch (e.g. bound-template -> full graph). Read by the failure-exit branch
    # below AND by ``_build_run_summary`` on the success/degraded exit (Step 5).
    route: dict[str, Any] = {}
    stage_reached = "dispatch"
    try:
        if requested_execution_mode == "pydantic_graph":
            actual_execution_mode = "pydantic_graph"
            route = {
                "agents": ["pydantic-graph", agent_name],
                "servers": [tool_server] if tool_server else [],
                "why": "caller forced the governed pydantic-graph graph.run route",
            }
            stage_reached = "pydantic-graph"
            await _emit(
                progress_sink,
                run_id=run_id,
                stage="route",
                status="ok",
                detail=str(route["why"]),
                evidence={
                    "agents": route["agents"],
                    "servers": route["servers"],
                    "skill": skill_name or "",
                },
            )
            result = await _execute_graph(
                config=config,
                query=task,
                run_id=run_id,
                max_steps=max_steps,
                agent_meta=agent_meta,
                agent_name=agent_name,
            )
        elif _is_bound_template_agent(agent_meta, config):
            # CONCEPT:AU-ORCH.adapter.transport-toolset-factory — a KG-bound persona (e.g. agent-utilities-expert)
            # runs a DIRECT grounding loop: its recovered persona prompt drives the
            # run and its now-bound toolsets (graph-os + the fleet) let it query the
            # KG and ground the answer, instead of the prompt-only run that
            # hallucinated. Takes precedence over the generic focused-tools lexical
            # gate because the template DECLARES its own toolsets. A failure falls
            # through to the full graph (never drops the turn).
            route = {
                "agents": [agent_name],
                "servers": [],
                "why": "KG-bound persona template with pre-bound toolsets",
            }
            stage_reached = f"bound-template: {agent_name}"
            await _emit(
                progress_sink,
                run_id=run_id,
                stage="route",
                status="ok",
                detail=str(route["why"]),
                evidence={"agents": route["agents"], "servers": route["servers"]},
            )
            try:
                actual_execution_mode = "single_server_agent"
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
                stage_reached = (
                    f"bound-template: {agent_name} (fallback: multi-agent-graph)"
                )
                actual_execution_mode = "pydantic_graph"
                result = await _execute_graph(
                    config=config,
                    query=task,
                    run_id=run_id,
                    max_steps=max_steps,
                    agent_meta=agent_meta,
                    agent_name=agent_name,
                )
        elif getattr(shape, "tool_servers", ()) and agent_meta.get("type") != "server":
            # CONCEPT:AU-ORCH.execution.focused-tools-altitude — FOCUSED-TOOLS altitude: the lexical gate named concrete fleet
            # server(s), so bind exactly those toolsets and run ONE direct agent loop (parallel
            # tool calls) instead of the planning graph, which over-decomposes a named-tool ask
            # into a multi-step plan + expert fan-out.
            _focused_servers = list(getattr(shape, "tool_servers", ()) or ())
            route = {
                "agents": [],
                "servers": _focused_servers,
                "why": "lexical gate matched named fleet server(s) for this task",
            }
            stage_reached = f"tool-call: {','.join(_focused_servers)}"
            await _emit(
                progress_sink,
                run_id=run_id,
                stage="route",
                status="ok",
                detail=str(route["why"]),
                evidence={"servers": _focused_servers},
            )
            # CONCEPT:AU-ORCH.execution.messaging-orchestration-transparency — surface the
            # ORCH-1.74 focused-tools binding ("binding N server(s)") as a tool_call checkpoint,
            # so the chat surface shows WHICH fleet tools this run is about to reach before the
            # (possibly slow) parallel tool loop runs.
            await _emit(
                progress_sink,
                run_id=run_id,
                stage="tool_call",
                status="started",
                detail=", ".join(_focused_servers),
                evidence={"servers": _focused_servers},
            )
            try:
                actual_execution_mode = "single_server_agent"
                result = await _execute_focused_tools(
                    task=task,
                    shape=shape,
                    config=config,
                    agent_name=agent_name,
                    max_steps=max_steps,
                )
            except Exception as e:  # noqa: BLE001
                # CONCEPT:AU-ORCH.execution.focused-tools-fail-closed — this branch is entered ONLY
                # because ``shape.tool_servers`` (the live-KG-ontology lexical match against the
                # TASK, resolved in ``plan_execution_shape`` independently of ``agent_name``) named
                # concrete fleet server(s) — that is the branch guard itself, so it is ALWAYS a
                # server-name delegation, regardless of whether the top-level ``agent_name``
                # happens to also resolve as a KG ``:Server`` (it usually does NOT: ``agent_name``
                # is frequently a generic/passthrough identity like the messaging assistant, while
                # the REAL delegation target is ``shape.tool_servers``). The previous fail-closed
                # gate tested ``agent_meta.get("type") == "server"`` — the WRONG variable — so a
                # genuine named-server delegation whose real tools could not be reached (server
                # never registered / 0 :Server nodes, unreachable, auth failure, ...) silently fell
                # through to the toolless multi-agent graph and could fabricate a plausible-looking
                # answer stamped "completed" — exactly the confident-hallucination failure
                # AU-ORCH.execution.no-silent-hallucination exists to catch. There is no legitimate
                # fallthrough once a concrete server target is named, so always fail closed here.
                err = _flatten_exception_group(e)
                servers = list(getattr(shape, "tool_servers", ()) or ())
                logger.warning(
                    "[ORCH-1.74] focused-tools path failed for fleet server(s) %s (%s); "
                    "surfacing degraded instead of hallucinating via the graph.",
                    servers,
                    err,
                )
                result = _fleet_server_failed_result(
                    agent_name or ",".join(servers), err
                )
        elif _is_single_server_agent(agent_meta, config):
            actual_execution_mode = "single_server_agent"
            route = {
                "agents": [],
                "servers": [agent_name],
                "why": "resolved as a single configured MCP server",
            }
            stage_reached = f"tool-call: {agent_name}"
            await _emit(
                progress_sink,
                run_id=run_id,
                stage="route",
                status="ok",
                detail=str(route["why"]),
                evidence={"servers": [agent_name]},
            )
            await _emit(
                progress_sink,
                run_id=run_id,
                stage="tool_call",
                status="started",
                detail=agent_name,
                evidence={"servers": [agent_name]},
            )
            result = await _execute_single_server(
                config=config,
                task=task,
                max_steps=max_steps,
                agent_meta=agent_meta,
                agent_name=agent_name,
                bound_tool_grounding=True,
            )
        else:
            actual_execution_mode = "pydantic_graph"
            route = {
                "agents": ["multi-agent-graph"],
                "servers": [],
                "why": "no named server/template matched; routed to the multi-agent planning graph",
            }
            stage_reached = "multi-agent-graph"
            await _emit(
                progress_sink,
                run_id=run_id,
                stage="route",
                status="ok",
                detail=str(route["why"]),
                evidence={"agents": ["multi-agent-graph"]},
            )
            result = await _execute_graph(
                config=config,
                query=task,
                run_id=run_id,
                max_steps=max_steps,
                agent_meta=agent_meta,
                agent_name=agent_name,
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
            # CONCEPT:AU-ORCH.execution.messaging-orchestration-transparency — this branch
            # re-raises immediately, so the ordinary failure trace below NEVER runs for a
            # cancelled run (Step 5 never runs either). Without this, a caller-side wall-clock/
            # reply-budget timeout (e.g. the messaging router's ``asyncio.wait_for``) leaves
            # ZERO durable trace for this run_id — a ``trace_ref`` handed to the caller (a
            # caller that pre-generated one via the ``run_id=`` param specifically so it
            # survives a cancellation) would resolve to nothing. Best-effort record a
            # "timeout" RunTrace with whatever route/stage this run reached before it was cut
            # off, so that trace_ref is a REAL troubleshooting entry point.
            #
            # DELIBERATELY SYNCHRONOUS — do NOT route this one through
            # ``_call_without_blocking``/``asyncio.to_thread`` like the rest of this
            # function's graph I/O. Cancellation has ALREADY been delivered to this
            # coroutine, so any ``await`` here is a suspension point that a second
            # ``cancel()`` (a supervisor retrying, or shutdown) re-raises through —
            # measured: under a double-cancel the awaited variant loses the write
            # entirely and this trace_ref resolves to nothing again, which is the exact
            # bug this block exists to prevent. A bounded, once-per-cancelled-run
            # blocking write is the correct trade against silently losing the only
            # durable record of a timed-out run.
            try:
                _record_execution_trace(
                    engine,
                    run_id,
                    agent_name,
                    task,
                    status="timeout",
                    duration_ms=(time.monotonic() - start_time) * 1000,
                    error=(
                        f"execution cancelled at stage={stage_reached!r} "
                        "(caller-side wall-clock/reply-budget timeout)"
                    ),
                    skill_used=_skill_used,
                    bound_server=_bound_server,
                    skill_id=_skill_id,
                    skill_instruction_digest=_skill_instruction_digest,
                    model_ref=_model_ref,
                    model_class=_model_class,
                    model_name=str(config.get("agent_model") or ""),
                    execution_mode=actual_execution_mode,
                    delegation=_spawn_delegation,
                )
            except Exception as trace_exc:  # noqa: BLE001 — best-effort; never block cancellation
                logger.debug(
                    "run_agent: best-effort timeout-trace write failed: %s", trace_exc
                )
            # CONCEPT:AU-AHE.harness.runtime-reliability-loop — a caller-side wall-clock
            # timeout is a delegation definitively over budget that never becomes a graded
            # run; record it (fire-and-forget) so a repeated pattern surfaces as a gap.
            _record_delegation_over_budget(
                agent_name, time.monotonic() - start_time, "timeout"
            )
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
        # CONCEPT:AU-KG.retrieval.fail-closed-grounding-contract — a `required`-policy
        # GroundingUnavailableError lands here (it is a PermissionError, caught by this
        # broad handler like any other failure): the run is truthfully recorded as
        # `status="failed"`, never silently answered ungrounded.
        from agent_utilities.core.contextual_model import grounding_snapshot as _gs

        _grounding_degraded, _grounding_reason = _gs()
        # Record failure provenance
        await _call_without_blocking(
            _record_execution_trace,
            engine,
            run_id,
            agent_name,
            task,
            status="failed",
            error=err_msg,
            skill_used=_skill_used,
            bound_server=_bound_server,
            skill_id=_skill_id,
            skill_instruction_digest=_skill_instruction_digest,
            model_ref=_model_ref,
            model_class=_model_class,
            model_name=str(config.get("agent_model") or ""),
            execution_mode=actual_execution_mode,
            delegation=_spawn_delegation,
            grounding_status="degraded" if _grounding_degraded else "grounded",
            grounding_reason=_grounding_reason,
        )
        # ARPO read-back (CONCEPT:AU-AHE.reward.this-is-read-back): failed runs carry step credit too
        # (a correct step in a failed trajectory must not be penalized).
        await _call_without_blocking(
            _write_step_credit,
            engine,
            run_id,
            agent_name,
            None,
            success=False,
        )
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
        # CONCEPT:AU-ORCH.execution.messaging-orchestration-transparency — stream the terminal
        # failure using the SAME translated text the run_summary carries, so the transparency
        # the user gets in-flight matches the final footer (never a bare "something failed").
        from agent_utilities.observability.trace_ontology import (
            trace_id as _trace_id_fail,
        )
        from agent_utilities.orchestration.failure_translation import (
            translate_failure as _translate_failure,
        )

        _fail_xlate = _translate_failure(err_msg)
        await _emit(
            progress_sink,
            run_id=run_id,
            stage="failure",
            status="failed",
            detail=_fail_xlate.translated,
            evidence={
                "category": _fail_xlate.category,
                "hint": _fail_xlate.hint,
                "stage_reached": stage_reached,
                "trace_ref": _trace_id_fail(run_id),
            },
        )
        return _render_agent_result(
            f"Agent execution failed: {err_msg}",
            run_id=run_id,
            return_mermaid=return_mermaid,
            channel_id=channel_id,
            run_summary=(
                _build_run_summary(
                    route=route,
                    outcome="failed",
                    stage_reached=stage_reached,
                    run_id=run_id,
                    raw_failure=err_msg,
                    execution_mode=actual_execution_mode,
                )
                if include_run_summary
                else None
            ),
        )
    finally:
        # CONCEPT:AU-OS.identity.per-agent-on-behalf-delegation — release the spawn's ambient
        # delegation on EVERY exit of the execution block (success, failure-return, or a
        # re-raised CancelledError), so it never leaks into the caller's context or a sibling
        # run. The success-path provenance below runs under legacy identity and receives the
        # delegation explicitly.
        _reset_delegation(_delegation_token)

    # Preserve graph evidence across the tool-grounding gate below. A missing
    # required ToolCall can replace the user-facing result with a truthful
    # failure envelope, but it must not erase the topology that reached that
    # failure.
    graph_execution_evidence = (
        result.get("execution_evidence")
        if isinstance(result, dict)
        and isinstance(result.get("execution_evidence"), dict)
        else None
    )

    # A caller that requested tools, or explicitly selected a server, must be grounded
    # by a real captured ToolCall.  Text that merely *looks* like a tool invocation is
    # model output, not provenance, and must never be reported as a successful run.
    tool_required = (
        bool(allowed_tools)
        or bool(required_tools)
        or agent_meta.get("type") == "server"
    )
    if tool_required and not _has_grounded_tool_call(result):
        result = _fleet_server_failed_result(
            agent_name,
            "tool-required execution finished without recorded ToolCall provenance",
        )
    elif required_tools and isinstance(result, dict):
        calls = result.get("tool_calls")
        observed = [
            str(call.get("tool_name") or "")
            for call in (calls if isinstance(calls, list) else [])
            if isinstance(call, dict)
        ]
        observed_aliases: dict[str, str] = {}
        server_name = tool_server or _bound_server
        public_prefix = (
            _configured_fleet_server_prefix(server_name) if server_name else ""
        )
        if public_prefix and server_name:
            from agent_utilities.mcp.multiplexer import clean_tool_name

            observed_aliases = {
                clean_tool_name(public_prefix, server_name, name): name
                for name in observed
                if name
            }
        missing = missing_required_tools(
            required_tools,
            observed,
            observed_aliases=observed_aliases,
        )
        if missing:
            result = _fleet_server_failed_result(
                tool_server or agent_name,
                "required tools produced no ToolCall provenance: " + ", ".join(missing),
                tool_calls=calls if isinstance(calls, list) else [],
            )
    if graph_execution_evidence is not None and isinstance(result, dict):
        result["execution_evidence"] = graph_execution_evidence

    # Step 5: Record provenance. A delegation that fell through to the graph's "no data"
    # sentinel (or returned an empty answer) is a DEGRADED outcome, not a success —
    # returning a confident-empty "completed" is the failure this fixes. Detect it so the
    # RunTrace status is truthful, the reward/shape learning is not poisoned by a
    # non-answer, and the failure is fed back so routing self-corrects next time
    # (CONCEPT:AU-ORCH.execution.degraded-no-data-outcome; F2/F5).
    degraded = _delegation_degraded(result)
    # CONCEPT:AU-KG.retrieval.fail-closed-grounding-contract — a run that proceeded
    # under an explicit best_effort/none grounding opt-in with at least one
    # ungrounded model call is likewise NOT a plain success: fold it into the SAME
    # `degraded` flag that already gates the RunTrace status, the consecutive-
    # failure guard, the degraded-feedback write, ARPO step credit, and shape-policy
    # reward below, so a degraded-grounding run is never learned from as a success.
    from agent_utilities.core.contextual_model import grounding_snapshot as _gs

    _grounding_degraded, _grounding_reason = _gs()
    if _grounding_degraded:
        degraded = True
    if isinstance(result, dict):
        _result_metadata = result.setdefault("metadata", {})
        if isinstance(_result_metadata, dict):
            actual_execution_mode = str(
                _result_metadata.get("execution_mode") or actual_execution_mode
            )
            _result_metadata["execution_mode"] = actual_execution_mode
    # CONCEPT:AU-ORCH.execution.messaging-orchestration-transparency — the REAL cause of a
    # degraded outcome (e.g. _fleet_server_failed_result's already-composed truthful message,
    # or a GraphResponse.error from a critical graph failure) so BOTH the durable RunTrace
    # below and the run_summary (attached to `result.metadata` + returned to any caller that
    # asked for it) carry it — never the old hardcoded "delegation produced no usable data"
    # sentinel, which discarded a real, already-known cause.
    _raw_failure = _extract_failure_text(result) if degraded else None
    run_summary = _build_run_summary(
        route=route,
        outcome="degraded" if degraded else "ok",
        stage_reached=stage_reached,
        run_id=run_id,
        raw_failure=_raw_failure,
        execution_mode=actual_execution_mode,
    )
    if isinstance(result, dict):
        _meta = result.setdefault("metadata", {})
        if isinstance(_meta, dict):
            _meta["run_summary"] = run_summary
    # CONCEPT:AU-ORCH.execution.messaging-orchestration-transparency — stream each fleet tool
    # invocation this run actually made (the SAME per-:ToolCall provenance persisted just
    # below), so a chat surface shows tools resolving one by one during a long parallel loop.
    # Guarded on ``progress_sink is not None`` so the None default skips the loop entirely.
    if progress_sink is not None and isinstance(result, dict):
        for _tc in result.get("tool_calls") or []:
            if not isinstance(_tc, dict):
                continue
            _tc_name = str(_tc.get("tool_name") or "tool")
            _tc_err = str(_tc.get("error") or "")
            await _emit(
                progress_sink,
                run_id=run_id,
                stage="tool_result",
                status="failed" if _tc_err else "ok",
                detail=_tc_name,
                evidence={"error": _tc_err[:200]} if _tc_err else {},
            )
    # CONCEPT:AU-ORCH.execution.messaging-orchestration-transparency — the evidence gate (the
    # paper's evidence-gating) SURFACED: when a degraded outcome translates to the
    # retrieval-quality signature, the run was blocked because nothing cleared the relevance
    # bar. Reuse the EXISTING failure_translation so the streamed text matches the footer.
    if _raw_failure:
        from agent_utilities.orchestration.failure_translation import (
            translate_failure as _translate_gate,
        )

        _gate = _translate_gate(_raw_failure)
        if _gate.category == "retrieval_quality":
            await _emit(
                progress_sink,
                run_id=run_id,
                stage="evidence_gate",
                status="failed",
                detail=_gate.translated,
                evidence={"category": _gate.category, "hint": _gate.hint},
            )
    # CONCEPT:AU-AHE.harness.loop-exit-conditions — ERROR THRESHOLD (exit 7):
    # feed this delegation's outcome into the per-agent consecutive-failure guard
    # (threshold + reset-on-success), so a loop driving repeated run_agent calls
    # can halt to error_threshold_exceeded. Tracking only — never aborts this run.
    _record_agent_outcome(agent_name, degraded=degraded)
    duration_ms = (time.monotonic() - start_time) * 1000
    # CONCEPT:AU-AHE.harness.runtime-reliability-loop — a run that ate most of its wall-clock
    # budget is a slow-not-wrong signal the reward flywheel can't see. Fire-and-forget.
    if (
        duration_ms
        >= _DELEGATION_BUDGET_WARN_FRACTION * _EXECUTE_AGENT_WALL_CLOCK_S * 1000
    ):
        _record_delegation_over_budget(
            agent_name, duration_ms / 1000.0, "degraded" if degraded else "ok"
        )
    _trace_recorded = await _call_without_blocking(
        _record_execution_trace,
        engine,
        run_id,
        agent_name,
        task,
        status="degraded" if degraded else "completed",
        duration_ms=duration_ms,
        result_preview=str(result)[:500],
        error=_raw_failure,
        skill_used=_skill_used,
        bound_server=_bound_server,
        skill_id=_skill_id,
        skill_instruction_digest=_skill_instruction_digest,
        model_ref=_model_ref,
        model_class=_model_class,
        model_name=str(config.get("agent_model") or ""),
        execution_mode=actual_execution_mode,
        graph_execution_evidence=graph_execution_evidence,
        tool_call_count=(
            len(result["tool_calls"])
            if isinstance(result, dict) and isinstance(result.get("tool_calls"), list)
            else None
        ),
        delegation=_spawn_delegation,
        grounding_status="degraded" if _grounding_degraded else "grounded",
        grounding_reason=_grounding_reason,
    )
    # CONCEPT:AU-ORCH.execution.messaging-orchestration-transparency — the paper's
    # "checkpointing / traces" surfaced: the durable RunTrace this run just wrote IS its
    # checkpoint. Stream it with the trace_ref so the caller can deep-link into the run's
    # provenance while it is still fresh.
    # D-DST-6: gate the "run trace recorded" report on _trace_recorded — _record_execution_trace
    # now returns False when the KG write actually failed, so this checkpoint no longer
    # tells the caller a trace exists when it may not (write-then-mark-seen).
    if progress_sink is not None:
        from agent_utilities.observability.trace_ontology import (
            trace_id as _trace_id_ck,
        )

        await _emit(
            progress_sink,
            run_id=run_id,
            stage="checkpoint",
            status="degraded" if (degraded or not _trace_recorded) else "ok",
            detail="run trace recorded"
            if _trace_recorded
            else "run trace write failed",
            evidence={"trace_ref": _trace_id_ck(run_id)} if _trace_recorded else {},
        )
    # CONCEPT:AU-KG.temporal.message-history-read — persist each tool call the local LLM made as a :ToolCall
    # node on this run's RunTrace, so the delegated action is fully visible over
    # graph-os ("what tools, what args, what result"). Best-effort, never breaks.
    if isinstance(result, dict) and result.get("tool_calls"):
        await _call_without_blocking(
            _persist_tool_calls,
            engine,
            run_id,
            agent_name,
            agent_name,
            result["tool_calls"],
        )
    # Self-healing (CONCEPT:AU-AHE.evaluation.action-outcome-feedback): a degraded run teaches the
    # reward-EMA that this agent/task-class produced a non-answer, so routing prefers
    # actions that actually achieve the goal. Best-effort; never breaks the run.
    if degraded:
        await _call_without_blocking(
            _record_degraded_feedback, engine, agent_name, task, result
        )
    # ARPO read-back (CONCEPT:AU-AHE.reward.this-is-read-back): credit the intermediate agent-steps of
    # this run into the capability reward-EMA so routing learns from the steps,
    # not only the final answer. Guarded — never breaks the run path.
    await _call_without_blocking(
        _write_step_credit,
        engine,
        run_id,
        agent_name,
        result,
        success=not degraded,
    )
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
        with contextlib.suppress(Exception):
            await _call_without_blocking(
                _anchor_run_to_session,
                engine,
                session_id=session_id,
                run_id=run_id,
            )

    logger.info(
        "[ORCH-1.21] Agent execution complete: agent=%s, run_id=%s, duration=%.0fms",
        agent_name,
        run_id,
        duration_ms,
    )

    # CONCEPT:AU-ORCH.execution.messaging-orchestration-transparency — the run has produced its
    # result and is about to render the final answer. Stream a synthesis milestone, then the
    # terminal ``done`` (carrying the same outcome + trace_ref the run_summary holds). Both
    # precede — and so cover — BOTH return shapes below (the dict envelope and the bare string).
    if progress_sink is not None:
        from agent_utilities.observability.trace_ontology import (
            trace_id as _trace_id_done,
        )

        await _emit(
            progress_sink,
            run_id=run_id,
            stage="synthesis",
            status="degraded" if degraded else "ok",
            detail="composing the final answer",
        )
        await _emit(
            progress_sink,
            run_id=run_id,
            stage="done",
            status="degraded" if degraded else "ok",
            detail=((_raw_failure or "")[:200] if degraded else "completed"),
            evidence={
                "outcome": "degraded" if degraded else "ok",
                "trace_ref": _trace_id_done(run_id),
            },
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
        elif result.get("error"):
            # CONCEPT:AU-ORCH.execution.messaging-orchestration-transparency — a
            # GraphResponse-shaped critical failure (engine.execute_graph's catch-all, or a
            # terminal error_recovery_step) carries its cause in `error`, not
            # `results.output`. Surface that cause directly rather than falling through to a
            # raw Python-dict-repr dump of the whole result (`str(result)`) — exactly the
            # illegible "some sort of failure" a chat user would otherwise see.
            output_str = f"The request could not be completed: {result['error']}"
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
        return _render_agent_result(
            output_str,
            run_id=run_id,
            return_mermaid=return_mermaid,
            mermaid=mermaid,
            channel_id=channel_id,
            run_summary=run_summary if include_run_summary else None,
            execution_evidence=graph_execution_evidence,
        )

    return _render_agent_result(
        result,
        run_id=run_id,
        return_mermaid=return_mermaid,
        channel_id=channel_id,
        run_summary=run_summary if include_run_summary else None,
    )


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

    return IntelligenceGraphEngine.get_or_create(backend=backend, db_path=db_path)


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


def _skill_unrunnable_reason(
    engine: IntelligenceGraphEngine, skill_name: str
) -> tuple[str, bool]:
    """Explain WHY ``skill_name`` appears unrunnable, naming the unmet precondition.

    CONCEPT:AU-ORCH.dispatch.named-runnable-precondition — "not found or
    runnable" collapsed four very different states into one unactionable
    message: never ingested, ingested-but-body-less, harvested-but-the-child-
    was-unreachable, and present-but-not-dispatchable. The cross-process
    harvest records the first unmet precondition on the ``Skill`` node
    (``runnable_blocked_by``), so the failure can name it. Diagnostic only —
    it never makes a skill runnable, and any failure to *read* the reason is
    reported as such rather than masquerading as "not found".

    Returns ``(reason, degraded)``. ``degraded=True`` means the precondition
    itself could not be READ (e.g. a transient engine/session failure) — that
    is NOT an honest negative and the caller must not phrase it as "is not
    runnable" (D-SNV-5): a failed read proves nothing about runnability, only
    that this diagnostic could not complete. ``degraded=False`` means the
    graph was actually consulted and the returned reason is a real finding.
    """
    try:
        # ``backend.execute`` is the parameterized read path the rest of this
        # module resolves agents with; ``engine.query_cypher`` does not bind a
        # ``WHERE`` predicate the same way on every backend build.
        rows = engine.backend.execute(
            "MATCH (s:Skill) WHERE s.name = $name "
            "RETURN s.runnable_blocked_by AS blocked, s.mcp_server AS server",
            {"name": skill_name},
        )
    except Exception as exc:  # noqa: BLE001 — the reason lookup is diagnostic;
        # a failure here must be REPORTED (with its cause), never silently
        # downgraded to "not found", which would be a different claim entirely.
        logger.warning(
            "[ORCH-1.96] could not read the unrunnable reason for %s (%s)",
            skill_name,
            type(exc).__name__,
            exc_info=True,
        )
        return (
            f"its blocking precondition could not be read ({type(exc).__name__}: {exc})",
            True,
        )
    if not rows:
        return (
            "no Skill node with that name is ingested (unmet precondition "
            "'skill_ingested')",
            False,
        )
    row = rows[0] or {}
    blocked = str(row.get("blocked") or "").strip()
    server = str(row.get("server") or "").strip()
    where = f" served by '{server}'" if server else ""
    if blocked:
        return f"unmet precondition '{blocked}'{where}", False
    # No recorded block reason. Do NOT assume the runnable resource is missing —
    # asserting an absence without checking is how a diagnostic starts lying.
    # Check, then report whichever is actually true.
    from agent_utilities.knowledge_graph.ingestion.skill_workflow_ingest import (
        skill_reference,
    )

    slug = skill_reference(skill_name).removeprefix("skill://")
    try:
        bound = engine.backend.execute(
            "MATCH (r:CallableResource) WHERE r.id = $rid RETURN r.id AS id",
            {"rid": f"resource:skill:{slug}"},
        )
    except Exception as exc:  # noqa: BLE001 — diagnostic only; say so plainly.
        logger.warning(
            "[ORCH-1.96] could not confirm the runnable resource for %s (%s)",
            skill_name,
            type(exc).__name__,
            exc_info=True,
        )
        return (
            f"its runnable resource could not be confirmed ({type(exc).__name__}: {exc})",
            True,
        )
    if bound:
        # The resource EXISTS, so resolution failed later — in
        # ``_hydrate_skill_runnable``, which fails closed on an incomplete body,
        # a missing digest, or a digest that no longer matches the body.
        return (
            f"its runnable resource exists{where} but failed hydration — its "
            "instruction body, digest, or source_ref is incomplete or no longer "
            "matches (unmet precondition 'runnable_metadata_intact')",
            False,
        )
    return (
        f"it is ingested{where} but has no runnable CallableResource "
        "(unmet precondition 'skill_body_served')",
        False,
    )


def _hydrate_skill_runnable(
    engine: IntelligenceGraphEngine,
    meta: dict[str, Any],
    *,
    skill_id: str,
    name: str,
    system_prompt: str,
    instruction_digest: str,
    source_ref: str,
) -> None:
    """Populate ``meta`` so an ingested skill runs with its own instructions + tools.

    CONCEPT:AU-ORCH.dispatch.dispatch-half-skill-ingestion — sets ``meta['system_prompt']`` (the skill's instruction body)
    and ``meta['tools']`` (its declared ``USES_TOOL`` targets), then binds the skill
    from its already-persisted ``CallableResource``. Local discovery paths are
    never consulted at execution time: an incomplete resource fails closed.
    """
    body = (system_prompt or "").strip()
    if not body or not instruction_digest or not source_ref.startswith("skill://"):
        raise RuntimeError("runnable skill metadata is incomplete")
    from agent_utilities.knowledge_graph.ingestion.skill_workflow_ingest import (
        runnable_skill_digest,
    )

    if runnable_skill_digest(body) != instruction_digest:
        raise RuntimeError("runnable skill instruction digest mismatch")

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
    except Exception as exc:
        # D-DST-6: a KG failure here silently produces meta["tools"]=[] --
        # indistinguishable from "this skill genuinely declares zero tools".
        # Mark the distinction explicitly instead of conflating a lookup
        # failure with a confirmed-empty toolset (both this function's own
        # docstring and the "skills prompt-only" behavior this feeds into
        # depend on knowing which one actually happened).
        meta["tools_lookup_failed"] = True
        logger.warning("[ORCH-1.96] skill tool lookup failed for %s: %s", name, exc)

    meta["system_prompt"] = (
        f"You are the '{name}' skill. Follow these instructions to fulfil the "
        f"user's request, calling any available tools as needed:\n\n{body}"
    )
    meta["tools"] = [{"name": t, "description": t} for t in tools]
    meta["skill_id"] = skill_id
    meta["skill_instruction_digest"] = instruction_digest
    meta["skill_source_ref"] = source_ref


def _bind_skill_to_owning_server(
    engine: IntelligenceGraphEngine,
    meta: dict[str, Any],
    provider_ref: str,
    agent_name: str,
) -> None:
    """Upgrade a package-bundled skill to a single-server agent bound to its server.

    CONCEPT:AU-ORCH.execution.skill-bound-server-tools — a skill contributed by a fleet
    provider is authored to drive that provider's MCP server tools. Resolved as
    a bare skill it runs prompt-only (no tools) and can only describe a task. This uses
    the privacy-safe ``provider://`` identity persisted at ingestion (filesystem paths
    are intentionally not retained), trying ``<provider>`` and ``<provider>-mcp``. It
    then sets ``type="server"`` + the server's tools so the run routes single-server
    while KEEPING the skill's instructions as the system prompt. Best-effort: a miss
    leaves the skill prompt-only.
    """
    provider = str(provider_ref or "").removeprefix("provider://").strip()
    if (
        not provider
        or provider in {"agent-utilities", "configured-overlay", "xdg-local"}
        or not getattr(engine, "backend", None)
    ):
        return
    candidates = (
        [provider] if provider.endswith("-mcp") else [f"{provider}-mcp", provider]
    )
    for name in candidates:
        try:
            rows = engine.backend.execute(
                "MATCH (s:Server) WHERE s.name = $name OR s.id = $sid "
                "RETURN s.id AS sid, s.name AS name, s.url AS url, s.env AS env",
                {"name": name, "sid": f"srv:{name}"},
            )
        except Exception as e:  # noqa: BLE001 — no meta mutation has happened yet in this loop iteration; on failure it just tries the next candidate name, matching this function's documented "a miss leaves the skill prompt-only" fallback
            logger.debug("[ORCH-skill-bind] server lookup failed for '%s': %s", name, e)
            continue
        if not rows or not rows[0].get("url"):
            continue
        srv = rows[0]
        # D-DST-6: fetch the server's verified tool catalog BEFORE committing
        # meta["type"]="server"/server_id/url/env. Committing those first (the
        # prior order) meant a tool-fetch failure left meta believing it was
        # bound to a real server while meta["tools"] still held the SKILL's
        # own USES_TOOL names from _hydrate_skill_runnable -- a load-bearing
        # state mismatch, not the benign prompt-only fallback this function's
        # docstring promises. On failure, try the next candidate exactly like
        # the server-lookup failure above already does.
        try:
            trows = engine.backend.execute(
                "MATCH (s:Server {id: $sid})-[:PROVIDES]->(r:CallableResource) "
                "RETURN r.name AS name, r.description AS description",
                {"sid": srv.get("sid", "")},
            )
        except Exception as e:  # noqa: BLE001 — meta is not yet mutated at this point; failure just tries the next candidate name (or falls through to prompt-only), matching the documented fallback contract
            logger.warning(
                "[ORCH-skill-bind] tool fetch failed for server '%s', trying next candidate: %s",
                name,
                e,
            )
            continue
        tools = [
            {"name": r.get("name", ""), "description": r.get("description", "")}
            for r in (trows or [])
        ]
        skill_prompt = meta.get("system_prompt", "")
        meta["type"] = "server"
        meta["server_id"] = srv.get("sid", "")
        meta["url"] = srv.get("url", "")
        meta["env"] = srv.get("env", "")
        meta["skill_of_server"] = srv.get("name", "")
        meta["tools"] = tools
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
        ``tools`` (list of tool names), ``capabilities``, ``toolset_id``, and
        ``system_prompt``. Live transports are resolved from AgentConfig, never KG.

    """
    meta: dict[str, Any] = {
        "type": "unknown",
        "server_id": "",
        "tools": [],
        "capabilities": [],
        "toolset_id": "",
        "endpoint_ref": "",
        "system_prompt": "",
        "skill_instruction_digest": "",
        "skill_source_ref": "",
    }

    if not engine or not engine.backend:
        logger.warning("[ORCH-1.21] No KG backend — using empty agent metadata")
        return meta

    # --- Search 1: Server nodes (MCP servers) ---
    try:
        server_rows = engine.backend.execute(
            "MATCH (s:Server) WHERE s.name = $name OR s.id = $sid "
            "RETURN s.id AS sid, s.name AS name, s.server_ref AS server_ref, "
            "s.tool_count AS tc",
            {"name": agent_name, "sid": f"srv:{agent_name}"},
        )
        if server_rows:
            row = server_rows[0]
            meta["type"] = "server"
            meta["server_id"] = row.get("sid", "")
            meta["toolset_id"] = row.get("name", "") or agent_name

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
    except Exception as e:  # noqa: BLE001 — Search 1 of 4 cascading resolution stages (Server → CallableResource → AgentTemplate → semantic search); a failure here falls through to the next stage, meta is still unmutated defaults at this point
        logger.debug("Server lookup failed for '%s': %s", agent_name, e)

    # --- Search 2: CallableResource nodes (skills, A2A agents) ---
    try:
        resource_rows = engine.backend.execute(
            "MATCH (r:CallableResource) WHERE r.name = $name "
            "RETURN r.id AS rid, r.resource_type AS rtype, r.description AS description, "
            "r.system_prompt AS system_prompt, r.instruction_digest AS instruction_digest, "
            "r.source_ref AS source_ref, r.provider_ref AS provider_ref, "
            "r.endpoint AS endpoint_ref",
            {"name": agent_name},
        )
        if resource_rows:
            row = resource_rows[0]
            rtype = row.get("rtype", "")
            if rtype == "A2A_AGENT":
                meta["type"] = "a2a"
                meta["endpoint_ref"] = row.get("endpoint_ref", "")
            elif rtype == "AGENT_SKILL":
                # A runnable skill is complete graph state: its instruction body,
                # digest, and neutral source reference were persisted at ingestion.
                meta["type"] = "skill"
                _hydrate_skill_runnable(
                    engine,
                    meta,
                    skill_id=row.get("rid", "") or f"skill:{agent_name}",
                    name=agent_name,
                    system_prompt=row.get("system_prompt", "") or "",
                    instruction_digest=row.get("instruction_digest", "") or "",
                    source_ref=row.get("source_ref", "") or "",
                )
                # CONCEPT:AU-ORCH.execution.skill-bound-server-tools — a PACKAGE-BUNDLED skill exists to
                # DRIVE its MCP server's tools; resolved as a bare skill it runs prompt-only
                # and can only DESCRIBE a task, never execute it. If the skill's provider
                # identifies an owning server, upgrade it to a single-server agent — bind
                # that server's toolset (task-selected via F1) and run the skill's
                # instructions AS the system prompt against real tools.
                _bind_skill_to_owning_server(
                    engine, meta, row.get("provider_ref", "") or "", agent_name
                )
            else:
                meta["type"] = "resource"
            logger.info(
                "[ORCH-1.21] Resolved '%s' as %s",
                agent_name,
                meta["type"],
            )
            return meta
    except Exception as e:  # noqa: BLE001 — Search 2 of 4 cascading resolution stages; a failure (including one re-raised out of _hydrate_skill_runnable/_bind_skill_to_owning_server) falls through to the AgentTemplate search stage
        logger.debug("Resource lookup failed for '%s': %s", agent_name, e)

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
    except Exception as e:  # noqa: BLE001 — Search 2b of 4 cascading resolution stages; a failure falls through to the explicit-fleet-pin check and semantic search
        logger.debug("AgentTemplate lookup failed for '%s': %s", agent_name, e)

    # An explicit server pin must remain authoritative even while the durable
    # capability graph is being refreshed. The live fleet catalog is already the
    # transport authority used by the multiplexer, so an exact configured name is
    # a safe fallback identity; semantic search must never rebound it to a
    # different server named in the task text.
    if agent_name in _configured_fleet_server_names():
        meta["type"] = "server"
        meta["server_id"] = f"srv:{agent_name}"
        meta["toolset_id"] = agent_name
        logger.info(
            "[ORCH-1.21] Resolved explicit '%s' from the live MCP fleet catalog",
            agent_name,
        )
        return meta

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
    except Exception as e:  # noqa: BLE001 — final cascade stage; on failure meta keeps its type="unknown" default, which is the correct signal that no resolution strategy succeeded
        logger.debug("Semantic search failed for '%s': %s", agent_name, e)

    return meta


def _bind_explicit_tool_server(
    engine: IntelligenceGraphEngine,
    skill_meta: dict[str, Any],
    tool_server: str,
    skill_name: str,
    allowed_tools: list[str] | None,
) -> dict[str, Any]:
    """Bind one exact fleet catalog to an already-validated ingested skill."""

    server_name = str(tool_server or "").strip()
    if not server_name:
        raise ValueError("tool_server is required")
    server_meta = _resolve_agent_from_kg(engine, server_name)
    if server_meta.get("type") != "server":
        raise LookupError(f"configured MCP server '{server_name}' was not found")

    declared = {
        str(tool.get("name") or "").strip()
        for tool in server_meta.get("tools") or []
        if isinstance(tool, dict) and str(tool.get("name") or "").strip()
    }
    requested = set(allowed_tools or [])
    if declared and not requested.issubset(declared):
        unknown = sorted(requested - declared)
        raise PermissionError(
            "allowed_tools contains tools outside the configured server catalog: "
            + ", ".join(unknown)
        )

    combined = dict(server_meta)
    for key in (
        "skill_id",
        "skill_instruction_digest",
        "skill_source_ref",
        "system_prompt",
    ):
        combined[key] = skill_meta.get(key, "")
    combined["skill_of_server"] = server_name
    combined["toolset_id"] = server_name
    logger.info(
        "[ORCH-1.21] Bound ingested skill '%s' to explicit MCP server '%s' (%d tools)",
        skill_name,
        server_name,
        len(server_meta.get("tools") or []),
    )
    return combined


# ---------------------------------------------------------------------------
# Internal: Config Construction
# ---------------------------------------------------------------------------


def _configured_fleet_server_names() -> frozenset[str]:
    """Return exact server identities from the active multiplexer catalog."""
    try:
        from agent_utilities.mcp.multiplexer import (
            MCPMultiplexer,
            _resolve_config_path,
        )

        config_path = _resolve_config_path(str(setting("MCP_CONFIG", "") or "") or None)
        catalog = MCPMultiplexer(config_path).load_catalog()
        if isinstance(catalog, dict):
            return frozenset(
                str(name).strip()
                for name in catalog
                if isinstance(name, str) and name.strip()
            )
    except Exception as exc:  # noqa: BLE001 — KG resolution remains available
        logger.debug("Live MCP fleet catalog lookup failed: %s", exc)
    return frozenset()


def _configured_fleet_server_prefix(server_name: str) -> str:
    """Return the collision-safe public prefix for one configured server."""
    try:
        from agent_utilities.mcp.multiplexer import (
            MCPMultiplexer,
            _resolve_config_path,
        )

        config_path = _resolve_config_path(str(setting("MCP_CONFIG", "") or "") or None)
        mux = MCPMultiplexer(config_path)
        if server_name in mux.load_catalog():
            return mux.server_prefix(server_name)
    except Exception as exc:  # noqa: BLE001 — raw tool names still work
        logger.debug("Live MCP fleet prefix lookup failed for %s: %s", server_name, exc)
    return ""


def _configured_model_for_class(model_class: str) -> Any:
    """Resolve an explicit runtime class to one exact AgentConfig model tier."""
    from agent_utilities.core.config import config as agent_config

    levels = {"economy": "light", "standard": "normal"}
    requested = str(model_class or "").strip().casefold()
    if requested not in levels:
        raise ValueError("model_class must be economy or standard")
    matches = [
        model
        for model in agent_config.chat_models
        if str(model.intelligence_level).strip().casefold() == levels[requested]
    ]
    if not matches:
        raise RuntimeError(f"configured {requested} model class is unavailable")
    return matches[0]


def _spawn_auth() -> Any:
    """Refresh-capable service-account auth for spawned REMOTE MCP toolsets.

    CONCEPT:AU-ORCH.routing.mcp-child-error-unwrap / OS-5.32 — a spawned agent that binds a jwt-protected
    fleet server over SSE/streamable-HTTP must carry the same
    service-account identity the multiplexer attaches to its children, or the
    call is rejected ``401``. Reuse ``client_credentials.child_auth`` so a
    long-lived streamable-HTTP session pulls a current token per request and
    force-refreshes once on ``401``. A static Authorization header is invalid
    here: it freezes the token minted when the toolset was built and makes
    delegated calls fail after that token expires. Authentication disabled
    explicitly returns ``None``; configuration and mint failures remain
    fail-closed.
    """
    from agent_utilities.mcp.client_credentials import child_auth

    return child_auth(None)


def _toolset_for_id(
    _engine: IntelligenceGraphEngine,
    toolset_id: str,
    *,
    allowed_tools: list[str] | None = None,
) -> Any:
    """Resolve ONE AgentTemplate ``toolset_id`` to a live MCP toolset.

    CONCEPT:AU-ORCH.adapter.transport-toolset-factory — the binding seam that turns a KG-bound persona's declared
    toolsets into tools the local LLM can actually call. Resolution reuses the
    existing Server/mcp_config + fleet-URL machinery (no new binder, no new
    transport code). ``graph-os`` is the one deliberate exception: it is this
    process, so it binds only the caller-granted native tools and never opens a
    self-HTTP connection.

    Resolution uses only the active AgentConfig URL template or the live fleet
    configuration. KG ``Server.url`` values are opaque provenance references,
    never transport endpoints.

    The toolset carries refresh-capable OIDC service-account auth
    (:func:`_spawn_auth`) so JWT-protected servers do not reject calls after the
    initial token expires. Returns the
    bound ``MCPToolset`` (id-tagged for filtering), or ``None`` for an empty id.
    """
    tid = (toolset_id or "").strip()
    if not tid:
        return None

    if tid == "graph-os":
        requested = [
            str(name).strip() for name in allowed_tools or [] if str(name).strip()
        ]
        if not requested:
            raise PermissionError(
                "graph-os AgentTemplate binding requires an explicit bounded tool allow-list"
            )
        if len(requested) != len(set(requested)):
            raise ValueError("GraphOS tool allow-list contains duplicates")

        from agent_utilities.mcp.kg_server import (
            REGISTERED_TOOLS,
            build_native_graphos_toolset,
            ensure_tools_registered,
        )

        ensure_tools_registered()
        if "graph_orchestrate" in requested:
            raise PermissionError("recursive native GraphOS delegation is forbidden")
        if any(name not in REGISTERED_TOOLS for name in requested):
            raise PermissionError(
                "GraphOS tool allow-list contains a tool not declared by graph-os"
            )
        return build_native_graphos_toolset(requested, toolset_id=tid)

    from agent_utilities.mcp.toolset_factory import build_http_toolset

    url = _fleet_server_url(tid)
    if not url:
        raise RuntimeError(
            "toolset endpoint is unresolved; configure FLEET_MCP_URL_TEMPLATE "
            "or a remote URL in the active MCP fleet configuration"
        )

    return build_http_toolset(
        url,
        auth=_spawn_auth(),
        timeout=60,
        toolset_id=tid,
    )


def _resolve_toolset_ids(
    engine: IntelligenceGraphEngine,
    toolset_ids: list[str],
    *,
    allowed_tools: list[str] | None = None,
) -> list[Any]:
    """Bind an AgentTemplate's ``toolset_ids`` into a list of live MCP toolsets.

    CONCEPT:AU-ORCH.adapter.transport-toolset-factory — each id is resolved by :func:`_toolset_for_id`; a single
    Every declared toolset must resolve. A missing endpoint or credential fails
    the binding instead of silently running the persona with reduced authority.
    """
    bound: list[Any] = []
    for tid in toolset_ids or []:
        ts = _toolset_for_id(engine, tid, allowed_tools=allowed_tools)
        if ts is None:
            raise RuntimeError("declared toolset could not be bound")
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
    except Exception:  # noqa: BLE001
        logger.debug("Memento priming unavailable")
        return []

    cache = SessionMementoCache.instance()
    cached = cache.get(source)
    if cached is not None:
        return cached

    try:
        mementos = await asyncio.to_thread(get_recent_mementos, engine, source, limit)
    except Exception:  # noqa: BLE001
        logger.debug("Failed to prime privacy-safe Mementos")
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


def _bind_native_skill_toolset(
    *,
    config: dict[str, Any],
    agent_meta: dict[str, Any],
    agent_name: str,
) -> None:
    """Bind exactly the caller-allowed native tools declared by a KG skill.

    Runnable skill instructions and any ``USES_TOOL`` edges are authoritative
    graph state. Domain skills may receive only a narrower subset of declared
    tools. Platform skills intentionally own no GraphOS verbs, so their explicit
    caller allow-list is authorized by the caller session and the spawned signed
    identity instead. The resulting in-process toolset dispatches through
    GraphOS's verified tool boundary; no self-HTTP or raw function call exists.
    """

    if agent_meta.get("type") != "skill":
        return
    requested = [
        str(name).strip()
        for name in config.get("invoker_allowed_tools") or []
        if str(name).strip()
    ]
    if not requested:
        return
    if len(requested) != len(set(requested)):
        raise ValueError("delegated skill tool allow-list contains duplicates")

    declared = {
        str(item.get("name") or "").strip()
        for item in agent_meta.get("tools") or []
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    }
    if declared and not set(requested).issubset(declared):
        raise PermissionError("delegated skill requested an undeclared tool")

    from agent_utilities.mcp.kg_server import (
        REGISTERED_TOOLS,
        build_native_graphos_toolset,
        ensure_tools_registered,
    )

    ensure_tools_registered()
    if "graph_orchestrate" in requested:
        raise PermissionError("recursive native GraphOS delegation is forbidden")
    if any(name not in REGISTERED_TOOLS for name in requested):
        raise RuntimeError("delegated skill native tool is unavailable")
    config.setdefault("mcp_toolsets", []).append(
        build_native_graphos_toolset(requested, toolset_id=agent_name)
    )


def _build_execution_config(
    engine: IntelligenceGraphEngine,
    agent_name: str,
    agent_meta: dict[str, Any],
    memento_source: str | None = None,
    execution_profile: str | ExecutionProfile | None = None,
    recent_mementos: list[str] | None = None,
    code_context_prime: str | None = None,
    model_class: str = "standard",
    allowed_tools: list[str] | None = None,
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
    off the event loop by :func:`_prime_recent_mementos`). When ``None`` (direct
    library callers), we perform a synchronous fetch here so the function
    stays self-contained, but the hot reply path always passes the primed list so no
    blocking backend round-trip runs on the loop.

    CONCEPT:AU-ORCH.execution.chat-profile-timeouts — ``execution_profile`` ("chat" vs default "task") selects the
    per-node timeout budget: the chat profile bounds router/verifier to tens of seconds
    so a degraded backend fails fast inside the chat budget rather than at 300 s.
    """
    from agent_utilities.core.config import (
        DEFAULT_GRAPH_ROUTER_TIMEOUT,
        DEFAULT_GRAPH_VERIFIER_TIMEOUT,
        DEFAULT_MIN_CONFIDENCE,
    )
    from agent_utilities.orchestration.execution_profile import (
        resolve_execution_profile,
    )

    profile = resolve_execution_profile(execution_profile)
    selected_model = _configured_model_for_class(model_class)
    selected_class = str(model_class).strip().casefold()

    # Tag prompts: the agent itself + any capabilities. CONCEPT:AU-ORCH.dispatch.seeded-agent-template — when
    # resolution recovered the agent's real system prompt (e.g. a seeded
    # AgentTemplate persona like ``agent-utilities-expert``), drive the run with
    # that full persona instead of the bare generic "Specialized agent" placeholder.
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
        except Exception:  # noqa: BLE001
            logger.debug("Failed to fetch privacy-safe Mementos for context")
            recent_mementos = []
    if recent_mementos:
        memento_text = "\n\n---\n\n".join(recent_mementos)
        tag_prompts["mementos"] = (
            f"Past Context Mementos (Compressed State):\n{memento_text}"
        )

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
        "router_model": selected_model.id,
        "agent_model": selected_model.id,
        "selected_model_class": selected_class,
        "router_timeout": router_timeout,
        "verifier_timeout": verifier_timeout,
        "execution_profile": profile.name,
        # CONCEPT:AU-ORCH.execution.per-job-shape-construction/1.68 — carry the constructed shape to the graph deps so each node
        # can decide whether to run its work or pass through for this job.
        "execution_shape": profile,
        "min_confidence": DEFAULT_MIN_CONFIDENCE,
        "valid_domains": tuple(tag_prompts.keys()),
        "provider": selected_model.provider,
        "base_url": selected_model.base_url,
        "api_key_ref": selected_model.api_key_ref,
        "nodes": {},
        "sub_agents": {},
        "routing_strategy": "hybrid",
        "enable_llm_validation": False,
        "discovery_metadata": {},
    }
    if allowed_tools:
        config["invoker_allowed_tools"] = list(allowed_tools)

    # Bind a server only through the live fleet configuration. KG server nodes
    # carry capability identity and opaque provenance, never executable transport.
    if agent_meta.get("type") == "server":
        toolset_id = str(agent_meta.get("toolset_id") or agent_name)
        toolset = _toolset_for_id(engine, toolset_id)
        if toolset is None:
            raise RuntimeError("server toolset could not be bound")
        config["mcp_toolsets"].append(toolset)

    # CONCEPT:AU-ORCH.adapter.transport-toolset-factory — a KG-bound AgentTemplate (e.g. ``agent-utilities-expert``)
    # declares its toolsets as ``toolset_ids`` (surfaced as ``capabilities`` by
    # ``_resolve_agent_from_kg``). Resolution recovered the persona prompt but, until
    # now, NOT live tools: the binding above only fires for ``type=="server"`` agents
    # with a URL, so the template ran prompt-only and HALLUCINATED. Bind each declared
    # toolset into a live MCP toolset so the persona can actually query graph-os / the
    # fleet and GROUND its answers (query-the-KG-then-answer). Reuses the same
    # Server/fleet-URL resolution + toolset_factory — no new binder.
    if agent_meta.get("type") == "agent_template":
        declared_tools = {
            str(tool.get("name") or "").strip()
            for tool in agent_meta.get("tools") or []
            if isinstance(tool, dict) and str(tool.get("name") or "").strip()
        }
        requested_tools = set(config.get("invoker_allowed_tools") or [])
        if declared_tools and not requested_tools.issubset(declared_tools):
            raise PermissionError("AgentTemplate requested an undeclared tool")
        bound = _resolve_toolset_ids(
            engine,
            agent_meta.get("capabilities", []),
            allowed_tools=config.get("invoker_allowed_tools"),
        )
        if bound:
            config["mcp_toolsets"].extend(bound)
            logger.info(
                "[ORCH-1.101] Bound %d toolset(s) for AgentTemplate '%s': %s",
                len(bound),
                agent_name,
                agent_meta.get("capabilities"),
            )
        elif agent_meta.get("capabilities"):
            logger.warning(
                "[ORCH-1.101] AgentTemplate '%s' declared toolsets %s but none bound — "
                "execution is rejected",
                agent_name,
                agent_meta.get("capabilities"),
            )
            raise RuntimeError("declared agent-template toolsets could not be bound")

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

# Fraction of the wall-clock budget above which a COMPLETED run is still recorded as a
# runtime-reliability "over budget" signal (CONCEPT:AU-AHE.harness.runtime-reliability-loop).
# A run that succeeds but eats ~most of its budget is a slow-not-wrong regression the reward
# flywheel never penalizes; this makes a repeated pattern visible. Named constant, not a knob.
_DELEGATION_BUDGET_WARN_FRACTION = 0.8


def _fleet_server_failed_result(
    agent_name: str,
    error: str,
    *,
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
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
        # Keep the zero explicit: the RunTrace writer distinguishes a known
        # ungrounded tool-required execution from a legacy result with no
        # provenance field at all.
        "tool_calls": list(tool_calls or []),
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
    *,
    bound_tool_grounding: bool = False,
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
    from contextlib import nullcontext

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
        public_prefix = _configured_fleet_server_prefix(agent_name)
        filtered: list[Any] = []
        for ts in toolsets:
            _filter = getattr(ts, "filtered", None)
            if not callable(_filter):
                raise RuntimeError(
                    f"toolset {type(ts).__name__!r} does not support tool filtering; "
                    f"cannot enforce allowed_tools={sorted(allow_set)} for agent "
                    f"'{agent_name}'"
                )
            if public_prefix:
                from agent_utilities.mcp.multiplexer import clean_tool_name

                filtered.append(
                    _filter(
                        lambda _ctx, td, _a=allow_set, _p=public_prefix, _s=agent_name: (
                            td.name in _a or clean_tool_name(_p, _s, td.name) in _a
                        )
                    )
                )
            else:
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
        api_key=config.get("api_key_ref"),
        mcp_toolsets=toolsets,
        # CONCEPT:AU-ORCH.execution.focused-tools-altitude — this direct-execution loop
        # (focused-tools / single-server / bound-template) binds EXACTLY the toolsets
        # resolved above (server-granular least privilege), each already carrying the
        # spawn service-account bearer. Pin ``mcp_url``/``mcp_config`` empty so
        # ``create_agent`` does NOT fall back to the process defaults
        # (``DEFAULT_MCP_URL`` / ``DEFAULT_MCP_CONFIG``) and silently reload the WHOLE
        # deployment fleet unauthenticated — that fallback pulled in graph-os's own
        # ``graph-os`` self-entry and every other child, hairpinning an HTTP client to
        # our own gateway (``401`` on ``/mcp``) and drowning the least-privilege intent.
        mcp_url="",
        mcp_config="",
        enable_skills=False,
        enable_universal_tools=False,
        name=agent_name,
        system_prompt=system_prompt,
        # CONCEPT:AU-ORCH.execution.delegation-reasoning-off — reasoning is a CAPABILITY,
        # not a default: this deterministic "call the bound tool(s), report the result"
        # loop leaves it OFF (the fleet default via create_model) so chain-of-thought
        # doesn't stack ~18x per-turn latency across the model→tool→model turns until the
        # run blows the wall-clock and is mis-attributed to a blocked tool. An execution
        # that genuinely needs deliberation opts IN by setting ``reasoning_effort`` on the
        # run config (e.g. "low"/"high"), mirroring how RLM is enabled when needed — None
        # here inherits the model/per-agent setting rather than forcing it off.
        reasoning_effort=config.get("reasoning_effort"),
    )

    prompt = task
    ctx_blob = config.get("invoker_context")
    if ctx_blob:
        prompt = f"Context:\n{ctx_blob}\n\nTask:\n{task}"

    run_kwargs: dict[str, Any] = {"message_history": []}
    # CONCEPT:AU-ORCH.execution.execution-budget-caps — always bound a single
    # request's input tokens, regardless of whether an invoker token/step budget
    # was set: a fleet tool that silently ignores an unknown ``limit`` argument
    # and returns a huge payload (the real 212 KB production incident) must not
    # blow this run in one request even when the caller never requested a budget.
    from agent_utilities.orchestration.loop_guards import (
        DEFAULT_PER_REQUEST_INPUT_TOKENS_LIMIT,
    )

    limit_kwargs: dict[str, Any] = {
        "per_request_input_tokens_limit": DEFAULT_PER_REQUEST_INPUT_TOKENS_LIMIT
    }
    budget = config.get("invoker_budget_tokens")
    if budget:
        limit_kwargs["total_tokens_limit"] = int(budget)
    # max_steps bounds model round-trips; keep headroom for tool call/response turns.
    if max_steps:
        limit_kwargs["request_limit"] = max(int(max_steps) * 2, 10)
    from pydantic_ai.usage import UsageLimits

    run_kwargs["usage_limits"] = UsageLimits(**limit_kwargs)

    # CONCEPT:AU-ORCH.execution.delegation-wall-clock — bound the direct tool loop with a wall-clock.
    # ``usage_limits`` caps model round-trips but NOT time: a fleet tool that blocks (e.g.
    # a systems-manager telemetry call shelling to a stuck host command) hangs the whole
    # delegation for the full client timeout (observed: 1800s) and piles engine
    # connections. A hung delegation is worse than a failed one — time out and raise so
    # the caller records it as a degraded/failed run (fail-loud), never an indefinite hang.
    from agent_utilities.core.contextual_model import use_bound_tool_grounding

    try:
        # The server and callable tool surface are already explicit and
        # least-privilege-bound above.  In this focused path, authenticated MCP
        # results are the evidence; a broad KG retrieval before every model turn
        # adds contention without adding grounding.  The trusted context scope
        # installs a transport-owned tool-grounding contract while ToolCall and
        # RunTrace persistence below retain the full provenance.
        grounding_scope = (
            use_bound_tool_grounding() if bound_tool_grounding else nullcontext()
        )
        with grounding_scope:
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
        "metadata": {"execution_mode": "single_server_agent"},
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
    """Resolve a fleet MCP URL exclusively from deployment configuration.

    ``FLEET_MCP_URL_TEMPLATE`` must contain ``{server}``, for example
    ``https://{server}.example.test/mcp``. No site-specific domain is assumed.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", server or ""):
        raise RuntimeError("fleet MCP server identifier is invalid")
    template = str(setting("FLEET_MCP_URL_TEMPLATE", "") or "").strip()
    if template:
        if "{server}" not in template:
            raise RuntimeError("FLEET_MCP_URL_TEMPLATE must contain '{server}'")
        candidate = template.replace("{server}", server).rstrip("/")
    else:
        from agent_utilities.mcp.multiplexer import (
            MCPMultiplexer,
            _resolve_config_path,
            _resolve_runtime_value,
        )

        config_path = _resolve_config_path(str(setting("MCP_CONFIG", "") or "") or None)
        config = MCPMultiplexer(config_path).load_catalog().get(server)
        if not isinstance(config, dict):
            return ""
        candidate = _resolve_runtime_value(config.get("url", ""), sensitive=False)
        if not candidate:
            return ""

    from urllib.parse import urlsplit

    parsed = urlsplit(candidate)
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or len(candidate) > 8_192
    ):
        raise RuntimeError("fleet MCP endpoint is invalid")
    if parsed.scheme.lower() == "http":
        # CONCEPT:AU-ORCH.execution.focused-tools-fleet-egress — this twin of the
        # multiplexer's egress gate (mcp/multiplexer.py, "Remote MCP child requires
        # HTTPS outside loopback") previously exempted ONLY loopback, so a fleet reached
        # over plain HTTP behind a TLS-terminating ingress — legitimately declared via
        # MCP_HTTP_ALLOWED_PRIVATE_HOSTS, which the multiplexer already honors — still
        # hard-failed HERE, surfacing as the ORCH-1.74 focused-tools degrade (the agent
        # graph "couldn't reach github-mcp"). Honor the SAME allowlist so the two gates
        # agree; a host absent from it still requires HTTPS.
        from agent_utilities.core.config import config as _agent_config

        _allowed_http_hosts = {
            "localhost",
            "127.0.0.1",
            "::1",
            *(host.lower() for host in _agent_config.mcp_http_allowed_private_hosts),
        }
        if parsed.hostname.lower() not in _allowed_http_hosts:
            raise RuntimeError("fleet MCP endpoint requires HTTPS outside loopback")
    return candidate


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
    from agent_utilities.mcp.toolset_factory import build_http_toolset

    servers = [s for s in (getattr(shape, "tool_servers", ()) or ()) if s]
    if not servers:
        raise RuntimeError("focused-tools shape carried no servers")

    toolsets: list[Any] = []
    for srv in servers:
        url = _fleet_server_url(srv)
        if not url:
            raise RuntimeError(
                "focused-tools delegation requires FLEET_MCP_URL_TEMPLATE"
            )
        toolsets.append(
            build_http_toolset(
                url,
                auth=_spawn_auth(),
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
        bound_tool_grounding=True,
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
    from pydantic_ai import ModelSettings

    from agent_utilities.core.config import DEFAULT_EXTRA_BODY
    from agent_utilities.core.contextual_model import create_context_agent
    from agent_utilities.core.model_factory import (
        create_model,
        merge_extra_body,
        reasoning_wire_directives,
    )

    model_id = getattr(shape, "model_id", None) if shape is not None else None
    reason_on = (
        bool(getattr(shape, "enable_reasoning", False)) if shape is not None else False
    )
    budget = getattr(shape, "router_timeout", None) if shape is not None else None
    # CONCEPT:AU-ORCH.execution.delegation-reasoning-off — reasoning rides on core
    # ModelSettings.thinking (True -> reasoning on, False -> off) AND the raw vLLM
    # extra_body directive (``reasoning_wire_directives``). ``thinking`` ALONE
    # regressed silently: pydantic-ai only forwards it into the request when the
    # model's profile is recognized as reasoning-capable (OpenAI's o-series/gpt-5
    # naming only), which qwen/qwen3.6-27b served through the generic ``openai``
    # provider is NOT — so a bare ``thinking=False`` here was dropped on the floor
    # and the model's own default (thinking ON) won on every direct-complete turn,
    # costing ~22s instead of sub-second (this is the shape most routine short
    # replies — e.g. the Telegram messaging path — take). The raw directive is
    # MERGED into (never replaces) DEFAULT_EXTRA_BODY's own deployment knobs.
    _reason_effort = "medium" if reason_on else "none"
    _extra_body = merge_extra_body(
        dict(DEFAULT_EXTRA_BODY or {}), reasoning_wire_directives(_reason_effort)
    )
    _direct_system_prompt = (
        "You are a helpful assistant. Respond naturally and concisely."
    )
    _direct_model_settings: Any = ModelSettings(
        thinking=reason_on,
        extra_body=_extra_body or None,
        max_tokens=1024,
        timeout=budget or 30.0,
    )
    try:
        # D-54c-4 — the direct-completion fast path builds its own agent + explicit
        # model_settings and never runs through attach_profile_resolver, so fold the
        # provider-native prompt-cache directive here too
        # (CONCEPT:AU-ORCH.optimization.provider-prompt-cache). The system prompt is a
        # fixed constant, so this is the highest-hit-rate site in the whole fleet.
        from agent_utilities.caching.prompt_cache import fold_prompt_cache_hint

        _direct_model_settings = fold_prompt_cache_hint(
            _direct_model_settings,
            system_prompt=_direct_system_prompt,
            model_identity=model_id,
        )
    except Exception:  # noqa: BLE001 - prompt-cache hint is best-effort
        pass
    agent = create_context_agent(
        model=create_model(model_id=model_id, reasoning_effort=_reason_effort),
        system_prompt=_direct_system_prompt,
        model_settings=_direct_model_settings,
    )
    res = await agent.run(query)
    return {
        "status": "completed",
        "results": {"output": str(res.output)},
        "metadata": {
            "direct_complete": True,
            "domain": "conversational",
            "execution_mode": "direct_completion",
        },
    }


async def _execute_graph(
    config: dict[str, Any],
    query: str,
    run_id: str,
    max_steps: int,
    agent_meta: dict[str, Any],
    agent_name: str,
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
    # Structured responses have one authority: the graph synthesizer backed by a
    # Pydantic output schema. The lean chat fast path is intentionally text-only,
    # so a JSON request must reach GraphDeps and the final synthesizer.
    if config.get("response_format", "text") == "json":
        _direct = False
    if config.get("execution_mode") == "pydantic_graph":
        _direct = False
    if _direct:
        try:
            return await _run_direct_completion(query, _shape)
        except Exception as e:  # noqa: BLE001 — a failed lean answer falls through to the graph
            logger.warning(
                "[ORCH-1.68] direct completion failed (%s); falling through to the graph.",
                e,
            )

    # Build graph from config
    graph, full_config = await _call_without_blocking(
        create_graph_agent,
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
        name=agent_name,
        capabilities=tuple(config.get("invoker_allowed_tools") or ()),
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
        # CONCEPT:AU-AHE.harness.loop-exit-conditions — TURN CAP (exit 2). Forward
        # the caller's max_steps onto the multi-agent graph's enforced budget
        # instead of dropping it (the audit-flagged unthreaded gap on this path).
        max_steps=max_steps,
    )

    return result


# ---------------------------------------------------------------------------
# Internal: Provenance Tracking
# ---------------------------------------------------------------------------


def _stamp_run_identity(props: dict[str, Any], delegation: Any = None) -> None:
    """Add opaque tenant/actor/correlation references + the delegation chain to an audit record.

    Best-effort: identity and correlation are ambient context, so any failure
    (no actor in scope, observability not wired) leaves the record unstamped
    rather than failing the write.

    CONCEPT:AU-OS.identity.per-agent-on-behalf-delegation (decision 6) — when a spawn runs under
    a delegation (``warn`` or ``on``), the full principal→…→agent chain is stamped onto the
    ``:RunTrace`` so provenance answers "which real caller ran this spawn, through which agents?"
    as a single query. The ultimate principal is referenced opaquely (privacy); the agent-
    instance ids are not sensitive and are kept verbatim.
    """
    from agent_utilities.security.persistence_privacy import persistence_reference

    try:
        from agent_utilities.security.brain_context import current_actor

        actor = current_actor()
        if actor.actor_id:
            props.setdefault(
                "actor_ref",
                persistence_reference("actor", actor.actor_id, namespace="run-trace"),
            )
        if actor.tenant_id:
            props.setdefault(
                "tenant_ref",
                persistence_reference("tenant", actor.tenant_id, namespace="run-trace"),
            )
    except Exception as exc:  # pragma: no cover - identity best-effort  # noqa: BLE001 — actor/tenant identity is ambient enrichment on the RunTrace; per this function's documented contract, an unstamped record (not a failed write) is the correct degraded outcome
        logger.debug("run identity stamp skipped: %s", exc)

    try:
        from agent_utilities.security.delegation import current_delegation

        deleg = delegation if delegation is not None else current_delegation()
        if deleg is not None and getattr(deleg, "chain", None):
            props.setdefault(
                "delegation_chain",
                [
                    persistence_reference(
                        "actor", deleg.principal, namespace="run-trace"
                    )
                    if entry == deleg.principal
                    else entry
                    for entry in deleg.chain
                ],
            )
            props.setdefault("delegation_agent_instance", deleg.agent_instance_id)
            props.setdefault("delegation_mode", deleg.mode.value)
    except Exception as exc:  # pragma: no cover - delegation stamp best-effort  # noqa: BLE001 — delegation-chain stamping is ambient enrichment on the RunTrace; an unstamped record is the documented best-effort fallback, not a failed write
        logger.debug("delegation chain stamp skipped: %s", exc)
    try:
        from agent_utilities.observability.correlation import get_correlation_id

        cid = get_correlation_id()
        if cid:
            props.setdefault(
                "correlation_ref",
                persistence_reference("correlation", cid, namespace="run-trace"),
            )
    except Exception as exc:  # pragma: no cover - correlation best-effort  # noqa: BLE001 — correlation-id stamping is ambient enrichment on the RunTrace; an unstamped record is the documented best-effort fallback, not a failed write
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
    skill_instruction_digest: str = "",
    model_ref: str = "",
    model_class: str = "",
    model_name: str = "",
    tool_call_count: int | None = None,
    execution_mode: str = "other",
    graph_execution_evidence: dict[str, Any] | None = None,
    delegation: Any = None,
    grounding_status: str = "",
    grounding_reason: str = "",
) -> bool:
    """Record an execution trace in the KG for auditability.

    CONCEPT:AU-ORCH.routing.mcp-child-error-unwrap — Execution provenance tracking.

    Creates a ``RunTrace`` node linked to the agent's Server/Resource node,
    enabling a privacy-safe audit trail of agent invocations. When a package skill
    drove the run, opaque ``skill_ref``/``server_ref`` properties identify the
    attribution class and a ``USES_SKILL`` edge is written (CONCEPT:AU-ORCH.execution.skill-utilization-provenance)
    so "which runs used skill X, and what tools did it drive" is a single traversal.

    ``model_name``/``tool_call_count`` are OTel-only (X2): unlike ``model_ref``
    (an opaque reference persisted onto the KG ``RunTrace`` node below),
    ``model_name`` is the plain model identifier stamped onto the run's OTel
    span as ``gen_ai.request.model`` — never written to the graph.

    Returns:
        True iff the RunTrace/Outcome graph state was actually written (D-DST-6:
        the caller must not report "run trace recorded" to progress_sink when
        this returns False — see the caller in run_agent).
    """
    # This is every exit path of run_agent's dispatch (success/degraded/failed/
    # enterprise) — closing the run's OTel span HERE (before the ``engine``
    # guard below) guarantees the span :meth:`on_graph_start` opened is always
    # closed, independent of whether the KG write that follows runs at all.
    try:
        from agent_utilities.observability import get_telemetry_engine

        get_telemetry_engine().on_graph_end(
            run_id=run_id,
            status=status,
            duration_ms=float(duration_ms or 0.0),
            model=model_name,
            tool_call_count=tool_call_count,
            execution_mode=execution_mode,
            graph_execution_evidence=graph_execution_evidence,
        )
    except Exception as exc:  # noqa: BLE001 — tracing must never break a run
        logger.debug(
            "run_agent: OTel span end skipped (exception_type=%s)",
            type(exc).__name__,
        )

    if not engine:
        return False

    from agent_utilities.observability.trace_ontology import (
        OUTCOME_NODE_LABEL,
        TRACE_NODE_LABEL,
        TRACE_PRODUCED_OUTCOME_EDGE,
        outcome_id,
        outcome_properties,
        trace_properties,
    )
    from agent_utilities.observability.trace_ontology import (
        trace_id as canonical_trace_id,
    )

    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    trace_id = canonical_trace_id(run_id)
    props = trace_properties(
        run_id=run_id,
        agent_name=agent_name,
        task=task,
        status=status,
        timestamp=ts,
        error=error,
        duration_ms=duration_ms,
        result_preview=result_preview,
        skill_used=skill_used,
        bound_server=bound_server,
        execution_mode=execution_mode,
        graph_execution_evidence=graph_execution_evidence,
        grounding_status=grounding_status,
        grounding_reason=grounding_reason,
    )
    if model_ref:
        props["model_ref"] = model_ref
    if model_class:
        props["model_class"] = model_class
    if skill_instruction_digest:
        props["skill_instruction_digest"] = skill_instruction_digest

    # Stamp the originating identity + correlation so the audit trail answers
    # "which tenant/actor ran this, and which agents share its run?" as a
    # tenant-scoped graph query (CONCEPT:AU-OS.observability.run-wide-correlation-id + OS-5.14 + KG-2.60).
    _stamp_run_identity(props, delegation=delegation)

    try:
        engine.add_node(trace_id, TRACE_NODE_LABEL, properties=props)
        oid = outcome_id(run_id)
        engine.add_node(
            oid,
            OUTCOME_NODE_LABEL,
            properties=outcome_properties(
                run_id=run_id,
                status=status,
                timestamp=ts,
                event_sequence=int(props["event_sequence"]),
                feedback=error or status,
            ),
        )
        engine.link_nodes(trace_id, oid, TRACE_PRODUCED_OUTCOME_EDGE)

        if engine.backend:
            # EXECUTED_ON links to the actual server whose tools ran — the bound server
            # for a skill-driven run (agent_name is the skill, not a Server), else the
            # agent's own server node.
            #
            # A comma-pattern MATCH plus an edge MERGE both exceed the
            # engine's native Cypher write subset (one leading MATCH, MERGE on
            # a single bare node only;
            # epistemic-graph/crates/eg-query/src/cypher/parser.rs:1184);
            # ``link_nodes`` dispatches through the typed engine API for a
            # native authority (which -- unlike the portable Cypher fallback
            # used for a non-native store -- requires the Server/skill
            # resource to already exist) and falls back to the portable
            # multi-clause Cypher for a non-native store, mirroring
            # ``record_outcome``'s TRACE_PRODUCED_OUTCOME_EDGE link above.
            # Each link is caught locally: the RunTrace/OutcomeEvaluation
            # nodes above are ALREADY durably written by this point, so a
            # missing auxiliary Server/skill node (same silent-no-op the
            # original MATCH gave a non-native store) must not flip this
            # function's return to False and make a successfully recorded
            # trace look unrecorded.
            server_name = bound_server or agent_name
            try:
                engine.link_nodes(trace_id, f"srv:{server_name}", "EXECUTED_ON")
            except Exception as exc:  # noqa: BLE001 — auxiliary EXECUTED_ON edge only; the RunTrace/Outcome nodes are already persisted, logged and skipped rather than reported as a trace-recording failure
                logger.debug(
                    "EXECUTED_ON link skipped for trace %r (server=%r): %s",
                    trace_id,
                    server_name,
                    exc,
                )
            # Skill-utilization provenance: which skill's SOP drove this run. Match the
            # skill node by ID — the engine cannot resolve a node by a non-id property
            # (name) in a write, which silently dropped this edge; EXECUTED_ON matches by
            # id and works, so mirror it. Prefer the resolved skill_id; fall back to the
            # canonical ``resource:skill:<name>`` id.
            if skill_used:
                rid = skill_id or f"resource:skill:{skill_used}"
                try:
                    engine.link_nodes(trace_id, rid, "USES_SKILL")
                except Exception as exc:  # noqa: BLE001 — auxiliary USES_SKILL edge only; same rationale as EXECUTED_ON above
                    logger.debug(
                        "USES_SKILL link skipped for trace %r (skill=%r): %s",
                        trace_id,
                        rid,
                        exc,
                    )
    except Exception as e:
        # D-DST-6 + D-DG-7 (reconciliation-gate-2 resolution of two lanes that
        # edited this handler concurrently).
        #
        # D-DG-7 argued this failure "can never be surfaced to the caller", so
        # only the log level mattered. That was true when it was written and is
        # NOT true now: D-DST-6 gave this function a bool return that run_agent
        # actually consumes (`_trace_recorded`, ~line 1561) to gate its
        # progress_sink "run trace recorded" checkpoint — previously reported
        # unconditionally, i.e. write-then-mark-seen on the harness's own
        # provenance layer. So the bool contract is kept.
        #
        # D-DG-7's other two points stand on their own and are kept as well:
        # this write is the ONLY persistence of the run's RunTrace/Outcome
        # nodes, and a run reporting status="ok" with a trace_ref pointing at a
        # node that was never written is a production failure — invisible to
        # the reward/evolution flywheel and to anyone reading the trace back.
        # Hence `error` (not `warning`) and the run/trace ids in the message.
        # %-style, not an f-string, so the args stay lazy and pass through
        # core/log_privacy.py's sanitizer.
        logger.error(
            "Failed to record execution trace (run_id=%r, trace_id=%r): %s",
            run_id,
            trace_id,
            e,
        )
        return False

    return True


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


def _has_grounded_tool_call(result: Any) -> bool:
    """Return whether execution captured at least one real ToolCall record.

    A fenced JSON snippet such as ``{"tool": "repos"}`` is only model text.  The
    executor's tool loop is the authority: it records actual invocations in the
    structured ``tool_calls`` list that is later persisted as ``:ToolCall`` nodes.
    """
    if not isinstance(result, dict):
        return False
    calls = result.get("tool_calls")
    return bool(
        isinstance(calls, list)
        and any(
            isinstance(call, dict) and str(call.get("tool_name") or "").strip()
            for call in calls
        )
    )


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

# Common id-shaped keys a tool call's sanitized args carry when the call acted on
# an existing KG entity (an incident, ticket, spec proposal, governance gate, …).
# Checked in order; the first present, existing node wins (CONCEPT:AU-KG.audit.tool-call-acted-on-reverse-index).
_TOOL_ARG_TARGET_KEYS = (
    "node_id",
    "id",
    "entity_id",
    "target_id",
    "ticket_id",
    "incident_id",
    "spec_id",
)


def _extract_tool_call_target(args: Any) -> str:
    """Best-effort candidate target-entity id from a tool call's sanitized args.

    ``args`` is the compact, secret-redacted JSON string ``sanitize_tool_args``
    produces. Returns the first recognizable id-shaped key's value, or ``""`` when
    none is present / args isn't decodable JSON.
    """
    if not args or not isinstance(args, str):
        return ""
    try:
        import json as _json

        decoded = _json.loads(args)
    except Exception:  # noqa: BLE001
        return ""
    if not isinstance(decoded, dict):
        return ""
    for key in _TOOL_ARG_TARGET_KEYS:
        val = decoded.get(key)
        if isinstance(val, str) and val:
            return val
    return ""


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
    from agent_utilities.observability.trace_ontology import (
        TOOL_CALL_NODE_LABEL,
        TRACE_USED_TOOL_EDGE,
        tool_call_properties,
    )
    from agent_utilities.observability.trace_ontology import (
        trace_id as canonical_trace_id,
    )

    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    trace_id = canonical_trace_id(run_id)
    written = 0
    try:
        from agent_utilities.knowledge_graph.adaptation.feedback import FeedbackService

        feedback = FeedbackService.from_engine(engine)
    except Exception:  # noqa: BLE001 — reward is optional
        feedback = None
    for i, tc in enumerate(tool_calls):
        tc_id = f"toolcall:{trace_id.removeprefix('trace:')}:{i}"
        # A tool that returned an error STRING as normal content (the MCP tool caught its
        # own exception) has no explicit ``error`` but is still a failure — score it as
        # such so provenance queries can filter real failures (AU-ORCH.execution.all-tool-calls-errored).
        ok = not _tool_call_errored(tc)
        props = tool_call_properties(
            run_id=run_id,
            tool_name=str(tc.get("tool_name", "")),
            args=tc.get("args", ""),
            result=tc.get("result", ""),
            error=tc.get("error", ""),
            status="ok" if ok else "error",
            sequence=i,
            timestamp=ts,
        )
        _stamp_run_identity(props)
        try:
            engine.add_node(tc_id, TOOL_CALL_NODE_LABEL, properties=props)
            # link_nodes writes backend-FIRST (durable), unlike add_edge's
            # best-effort compute-cache path — so the provenance edge survives in
            # the epistemic-graph for graph-os traversal queries.
            engine.link_nodes(trace_id, tc_id, TRACE_USED_TOOL_EDGE)
            written += 1
            # CONCEPT:AU-KG.audit.tool-call-acted-on-reverse-index (G23) — when the call's args
            # name an existing entity, link the ToolCall to it so
            # Orchestrator.get_tool_calls_for_target can reconstruct "what happened to
            # X" without a per-call round trip at read time. Best-effort: never fails
            # the run, and never vivifies a phantom target node.
            target_id = _extract_tool_call_target(tc.get("args", ""))
            if target_id and target_id != tc_id:
                try:
                    if engine.graph.has_node(target_id):
                        engine.link_nodes(tc_id, target_id, "ACTED_ON")
                except Exception as exc:  # noqa: BLE001 — ACTED_ON reverse-index edge is a documented best-effort enrichment (see comment above); the ToolCall node itself already persisted successfully before this sub-step runs
                    logger.debug(
                        "[KG-2.296] ACTED_ON link skipped (%s -> %s): %s",
                        tc_id,
                        target_id,
                        exc,
                    )
        except Exception as exc:  # noqa: BLE001 — 'written' is only incremented on the prior success path (line ~3793); a failure here correctly excludes this ToolCall from the persisted count and moves on to the next one
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
            except Exception as exc:  # noqa: BLE001 — reward-EMA feedback write is a side effect of persisting the ToolCall (already durable); its failure doesn't affect the ToolCall count or any caller-visible status
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
