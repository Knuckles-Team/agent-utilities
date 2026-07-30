#!/usr/bin/env python3
from __future__ import annotations

"""Mandatory evidence compilation at the model transport boundary.

Every Pydantic AI agent is built by :func:`create_context_agent`, which passes its
model through :func:`wrap_model_with_context` before construction.  The wrapper
resolves the verified ``GraphSession`` before touching a retriever, compiles an
evidence bundle, and prepends that bundle to both streaming and non-streaming
requests — this is the delegated-run (``execute_agent``/``execute_workflow``,
CONCEPT:AU-ORCH.execution.orchestration-flow-mermaid) default context-assembly path: every agent the
orchestration graph constructs (router/dispatcher/planner/specialist/single-server/
direct-completion — ``agent/factory.py``, ``graph/executor.py``,
``graph/hierarchical_planner.py``, ``graph/_router_impl.py``,
``orchestration/agent_runner.py``) is built through :func:`create_context_agent`, so
its model calls are compiled/cited/provenance-ranked by construction, ON BY
DEFAULT, with no separate opt-in required (CONCEPT:AU-KG.retrieval.context-compiler, W3.7).

There is no ``skip_context`` ARGUMENT: no per-call/per-agent parameter lets a task
or prompt disable its own grounding (a malicious or careless caller cannot opt out).
The one escape hatch is the deployment-level ``MODEL_CONTEXT_COMPILER_ENABLED``
setting (config-contract style — :func:`agent_utilities.core.config.setting`,
default ``True``, see :func:`_context_compiler_enabled`): an operator can disable
compilation for the whole process for a genuine disable case (e.g. a
cost-sensitive bulk/offline run with no retrieval-worthy KG content), never as a
per-request bypass. With the escape hatch at its default (on), code that genuinely
has no evidence source (compilation enabled, no engine registered) still fails
closed in authenticated production profiles.

The module owns no provider client and stores no prompt, identity, endpoint, or
filesystem value.  Persisted bundle caching remains the ContextCompiler's job and
uses only privacy-safe opaque references.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal

from agent_utilities.core.config import setting
from agent_utilities.knowledge_graph.core.session import (
    GraphSession,
    graph_session_required,
    resolve_session,
)

if TYPE_CHECKING:
    from agent_utilities.knowledge_graph.retrieval.context_compiler import ContextBundle

logger = logging.getLogger(__name__)

__all__ = [
    "ContextCompilationError",
    "GroundingPolicy",
    "GroundingUnavailableError",
    "compile_model_context",
    "create_context_agent",
    "current_grounding_policy",
    "disable_context_agent_instrumentation",
    "grounding_snapshot",
    "instrument_context_agents",
    "set_context_compiler_engine",
    "set_context_compiler_cache",
    "get_context_compiler_cache",
    "use_bound_tool_grounding",
    "use_context_compiler_engine",
    "use_grounding_policy",
    "wrap_model_with_context",
]

_CONTEXT_MARKER = "[agent-utilities:compiled-evidence:v1]"
_BOUND_TOOL_GROUNDING_MARKER = "[agent-utilities:bound-tool-grounding:v1]"
_DEGRADED_GROUNDING_MARKER = "[agent-utilities:degraded-evidence:v1]"
_bound_tool_grounding: ContextVar[bool] = ContextVar(
    "agent_utilities_bound_tool_grounding",
    default=False,
)

# CONCEPT:AU-KG.retrieval.fail-closed-grounding-contract — the explicit grounding
# policy a caller opts into. ``"required"`` (the default) is FAIL CLOSED: a compile
# timeout, a compile error, or a retrieval-quality-gate failure refuses the model
# call outright (:class:`GroundingUnavailableError`) rather than silently sending an
# ungrounded request. ``"best_effort"``/``"none"`` are explicit per-run opt-ins (see
# :func:`use_grounding_policy`) that let the request proceed degraded — but ALWAYS
# with an explicit marker in the messages themselves (:data:`_DEGRADED_GROUNDING_MARKER`)
# and on the current OTel span, and the run-level outcome is tracked via
# :func:`grounding_snapshot` so a caller (``agent_runner.run_agent``) can refuse to
# record a degraded run as a plain success for reward/learning purposes.
GroundingPolicy = Literal["required", "best_effort", "none"]

_grounding_policy: ContextVar[str] = ContextVar(
    "agent_utilities_grounding_policy", default="required"
)
# Aggregate, run-scoped outcome: was ANY model call in the current grounding_scope
# degraded, and why (the FIRST reason is kept — the earliest/root cause).
#
# This is a ContextVar holding a MUTABLE dict, deliberately, and NOT a pair of
# plain bool/str ContextVars. A ContextVar *write* made inside a child asyncio
# Task is invisible to the parent (``asyncio.create_task``/``TaskGroup`` — and
# ``asyncio.to_thread`` — each run in a COPY of the context), and the model call
# that discovers the degradation frequently runs one task-boundary below the
# ``use_grounding_policy`` scope that ``run_agent`` reads back. Measured: with
# plain-value ContextVars the degraded outcome was silently lost across a single
# ``create_task`` hop, so a degraded run was recorded as a plain success — the
# exact defect this contract exists to prevent. The dict REFERENCE is what the
# child inherits, so an in-place mutation is visible to the parent scope that
# installed it.
_grounding_outcome: ContextVar[dict[str, Any] | None] = ContextVar(
    "agent_utilities_grounding_outcome", default=None
)

# Wall-clock budget for the mandatory evidence compilation at the model-transport
# boundary. Compilation is a blocking, engine-contended retrieval (hybrid_retriever
# → graph_compute → the eg client's sync_wrapper → future.result()); running it
# inline on the asyncio event loop is what SIGKILLed graph-os (a contended
# retrieval blocked the loop so long that even status-only /health timed out).
# It is therefore run OFF the loop (asyncio.to_thread) and BOUNDED by this budget:
# a quick retrieval still returns the full compiled bundle, while a slow/contended
# one degrades to ungrounded passthrough FAST instead of stalling the delegation
# (and the whole process) past its wall clock. This is a liveness guard, not an
# operator tunable, so it is a named module constant per configuration discipline
# rather than a new env knob. ~10s sits below typical delegation/model timeouts
# while still letting a genuinely-slow-but-progressing retrieval complete.
_CONTEXT_COMPILE_TIMEOUT_S = 10.0

# CONCEPT:AU-KG.retrieval.context-compile-circuit-breaker — per-process breaker over
# the bounded compile call (:func:`_compiled_evidence_and_bundle_bounded`). Paying
# _CONTEXT_COMPILE_TIMEOUT_S once is a liveness guard; paying it N times in a row
# when retrieval is CONSISTENTLY too slow/broken is itself what blows a caller's
# reply budget (e.g. messaging's reply window) — the engine calls were fast, the
# retrieval LEGS were repeatedly slow. After _CTX_COMPILE_BREAKER_THRESHOLD
# consecutive degradations (timeout or error) the breaker OPENs: for
# _CTX_COMPILE_BREAKER_COOLDOWN_S it returns the degraded ``(messages, None)``
# immediately, skipping the to_thread+wait_for round trip entirely. Once the
# cooldown elapses the breaker is implicitly HALF-OPEN — the very next call is let
# through as a single probe: success CLOSES the breaker (streak resets to 0),
# failure re-OPENs it for another cooldown. Any successful compile at any time
# resets the streak. Just two module-level primitives + time.monotonic(), no lock:
# this sits on the hot path and a benign race at the reopen boundary costs at most
# one extra probe attempt, never incorrect output.
_CTX_COMPILE_BREAKER_THRESHOLD = 3
_CTX_COMPILE_BREAKER_COOLDOWN_S = 30.0

_ctx_compile_degradation_streak = 0
_ctx_compile_breaker_reopen_at = 0.0  # monotonic timestamp; 0.0 == breaker CLOSED

_compiler_engine: Any | None = None
_compiler_cache: Any | None = None
_WRAPPER_CLASS: Any | None = None
_MISSING_MODEL = object()


class ContextCompilationError(PermissionError):
    """A model invocation could not establish its governed evidence context."""


class GroundingUnavailableError(ContextCompilationError):
    """Raised when governed evidence is unavailable and the caller requires it.

    CONCEPT:AU-KG.retrieval.fail-closed-grounding-contract. Raised only when the
    ambient :data:`GroundingPolicy` is ``"required"`` (the default) AND evidence
    compilation timed out, errored, the per-process circuit breaker is open, or the
    retrieval-quality gate rejected every candidate. This is the fail-closed half of
    the contract: the request never reaches the model ungrounded. A caller that
    genuinely tolerates degraded operation opts in via :func:`use_grounding_policy`.
    """


@contextmanager
def use_grounding_policy(policy: GroundingPolicy = "required") -> Iterator[None]:
    """Scope the grounding policy (and its outcome tracking) for one delegated run.

    CONCEPT:AU-KG.retrieval.fail-closed-grounding-contract. ``"required"`` (the
    default — this is ALSO what every caller gets who never opens this scope) fails
    the request closed on a compile timeout/error/breaker-open or a retrieval-quality
    gate failure. ``"best_effort"``/``"none"`` opt into degraded operation: the
    request still proceeds, but every degraded model call carries an explicit marker
    (:data:`_DEGRADED_GROUNDING_MARKER`) in its messages and on the current OTel
    span, and :func:`grounding_snapshot` reports the aggregate outcome for the whole
    scope so the caller can refuse to record a degraded run as a plain success.

    Nesting resets outcome tracking to a fresh ``(False, "")`` for this scope, so a
    caller that opens one scope per delegated run (``Orchestrator.execute_agent``)
    gets a clean per-run signal rather than one contaminated by an unrelated prior
    run sharing the same task/context.
    """

    policy_token = _grounding_policy.set(policy)
    # A FRESH dict per scope (see :data:`_grounding_outcome`) — nesting therefore
    # still gets clean per-run tracking, while every child task/thread spawned
    # inside the scope mutates the SAME object this frame will read back.
    outcome_token = _grounding_outcome.set({"degraded": False, "reason": ""})
    try:
        yield
    finally:
        _grounding_policy.reset(policy_token)
        _grounding_outcome.reset(outcome_token)


def current_grounding_policy() -> str:
    """The ambient :data:`GroundingPolicy` for the current scope (default ``"required"``)."""

    return _grounding_policy.get()


def grounding_snapshot() -> tuple[bool, str]:
    """``(was_degraded, reason)`` for the current :func:`use_grounding_policy` scope.

    ``was_degraded`` is True iff ANY model call so far in this scope ran without
    genuinely compiled evidence (compile timeout/error/breaker-open, or a
    retrieval-quality gate failure) under an opted-in ``"best_effort"``/``"none"``
    policy. ``reason`` is the FIRST such reason recorded (the root cause). Read this
    after the delegated run completes (e.g. ``agent_runner._record_execution_trace``)
    to stamp the RunTrace/OTel span, and to gate reward/learning writes so a
    degraded run is never recorded as a plain success.

    Outside any :func:`use_grounding_policy` scope this is ``(False, "")``: with no
    scope the policy is the default ``"required"``, which RAISES rather than
    degrading, so there is no degraded outcome to aggregate.
    """

    outcome = _grounding_outcome.get()
    if outcome is None:
        return False, ""
    return bool(outcome.get("degraded")), str(outcome.get("reason") or "")


def _annotate_grounding_span(status: str, reason: str = "") -> None:
    """Best-effort: widen the CURRENT OTel span with the grounding outcome.

    Mirrors ``TelemetryEngine.annotate_epistemic``/``annotate_context_compiler``'s
    posture (``observability/__init__.py``): never opens a new span, never raises,
    a clean no-op wherever no tracing pipeline is active.
    """
    try:
        from agent_utilities.observability import get_telemetry_engine

        get_telemetry_engine().annotate_grounding(status=status, reason=reason)
    except Exception as exc:  # noqa: BLE001 — tracing must never break a model call
        logger.debug(
            "grounding span annotation failed (exception_type=%s)", type(exc).__name__
        )


def _mark_grounding_degraded(reason: str) -> None:
    """Record one degraded model call against the run-scoped outcome (first reason wins)."""

    outcome = _grounding_outcome.get()
    if outcome is None:
        # No :func:`use_grounding_policy` scope is open (a direct model call, a
        # harness). Install one lazily so the outcome is still readable by a
        # same-context caller — the pre-existing behaviour. It cannot propagate
        # UP out of a child task (nothing above installed a container to mutate),
        # which is exactly why the real delegated path opens the scope in
        # ``Orchestrator.execute_agent`` before ``run_agent`` reads it back.
        outcome = {"degraded": False, "reason": ""}
        _grounding_outcome.set(outcome)
    # Mutated IN PLACE, never rebound — that is what makes the write visible
    # to the parent frame across a child-task/thread context copy.
    if not outcome.get("degraded"):
        outcome["reason"] = reason
    outcome["degraded"] = True
    _annotate_grounding_span("degraded", reason)


class _EmptyEvidenceSource:
    """Development-only explicit empty source; never used in an auth-required profile."""

    def search_hybrid(
        self, query: str, *, top_k: int = 8, as_of: str | None = None
    ) -> list[dict[str, Any]]:
        del query, top_k, as_of
        return []

    def retrieve_epistemic_view(self, query: str, *, top_k: int = 8) -> dict[str, Any]:
        del query, top_k
        return {}


def set_context_compiler_engine(engine: Any | None) -> None:
    """Install the process evidence source during graph/runtime bootstrap."""

    global _compiler_engine
    _compiler_engine = engine


@contextmanager
def use_context_compiler_engine(engine: Any) -> Iterator[None]:
    """Temporarily install an evidence source and restore the exact prior source.

    The compiler source is process-wide bootstrap state. Callers that keep this
    scope open across an ``await`` must serialize those scopes; the bundled-skill
    validator does so with its direct-case execution lock.
    """

    previous = _compiler_engine
    set_context_compiler_engine(engine)
    try:
        yield
    finally:
        set_context_compiler_engine(previous)


def set_context_compiler_cache(cache: Any | None) -> None:
    """Install an optional shared KV backend for compiled bundles."""

    global _compiler_cache
    _compiler_cache = cache


def get_context_compiler_cache() -> Any | None:
    """Return the process's configured KV-cache backend, or ``None`` (CONCEPT:
    AU-KG.retrieval.context-compiler-kv-seam).

    Read-side counterpart to :func:`set_context_compiler_cache` — lets the
    ``tms_revalidation`` maintenance task (W3.2 TMS live-wiring) reach the SAME
    backend :func:`compile_model_context` stores compiled bundles into, so a
    bundle the engine's TMS marks stale can be dropped from the shared cache.
    """

    return _compiler_cache


@contextmanager
def use_bound_tool_grounding() -> Iterator[None]:
    """Use authenticated bound-tool results as the request's factual authority.

    This trusted orchestration-only mode is for a direct agent whose callable
    surface has already been reduced to one explicitly selected MCP server (or a
    focused subset of it).  There is no retrieval question to answer before the
    tool call: the tool response and its persisted ``ToolCall``/``RunTrace`` are
    the evidence.  The transport still receives a compiler-owned grounding
    contract, while the expensive KG evidence search is skipped.

    The scope is process-internal and cannot be selected by request content.
    """

    token = _bound_tool_grounding.set(True)
    try:
        yield
    finally:
        _bound_tool_grounding.reset(token)


def _active_engine() -> Any | None:
    if _compiler_engine is not None:
        return _compiler_engine
    try:
        from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

        return IntelligenceGraphEngine.get_active()
    except Exception:
        return None


def compile_model_context(
    query: str,
    *,
    session: GraphSession | None = None,
    engine: Any | None = None,
    model_version: str = "",
    snapshot: str = "",
) -> ContextBundle:
    """Compile the sole evidence bundle allowed to reach a model invocation."""

    from agent_utilities.knowledge_graph.retrieval.context_compiler import (
        ContextCompiler,
    )

    authority = resolve_session(session, required_scope="kg:read")
    source = engine or _active_engine()
    if source is None:
        if graph_session_required():
            raise ContextCompilationError(
                "authenticated model invocation requires a configured ContextCompiler engine"
            )
        source = _EmptyEvidenceSource()

    try:
        budget = max(64, int(setting("MODEL_CONTEXT_TOKEN_BUDGET", "2000") or 2000))
    except (TypeError, ValueError):
        budget = 2000
    return ContextCompiler(source).compile(
        str(query or ""),
        authority,
        token_budget=budget,
        kv_backend=_compiler_cache,
        model_version=str(model_version or ""),
        redaction_version=str(
            setting("MODEL_CONTEXT_REDACTION_VERSION", "permissioning-v1")
            or "permissioning-v1"
        ),
        evidence_ordering_version=str(
            setting("MODEL_CONTEXT_ORDERING_VERSION", "context-mmr-v1")
            or "context-mmr-v1"
        ),
        snapshot=str(snapshot or authority.catalog_epoch or ""),
    )


def _part_text(part: Any) -> str:
    content = getattr(part, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list | tuple):
        return " ".join(str(item) for item in content if isinstance(item, str))
    return ""


def _query_from_messages(messages: list[Any]) -> str:
    for message in reversed(messages):
        for part in reversed(list(getattr(message, "parts", ()) or ())):
            kind = str(getattr(part, "part_kind", ""))
            if kind in {"user-prompt", "system-prompt"}:
                text = _part_text(part).strip()
                # Marker text is not a capability: a caller is allowed to ask
                # about it and that complete turn must still drive retrieval.
                # A genuinely compiled request is returned early by
                # ``_already_compiled`` before this query extractor runs.
                if text:
                    return text
    return ""


def _already_compiled(messages: list[Any]) -> bool:
    if not messages:
        return False
    parts = list(getattr(messages[0], "parts", ()) or ())
    if not parts:
        return False
    # The wrapper always prepends exactly this leading system part. Trusting a
    # marker anywhere else would let caller-controlled history suppress the
    # mandatory compiler. The marker is an idempotency sentinel, not an auth
    # token, and its position is therefore part of the boundary contract.
    first = parts[0]
    return str(getattr(first, "part_kind", "")) == "system-prompt" and _part_text(
        first
    ).lstrip().startswith(_CONTEXT_MARKER)


def _context_compiler_enabled() -> bool:
    """The deployment-level escape hatch for the mandatory evidence boundary.

    CONCEPT:AU-KG.retrieval.context-compiler (W3.7 default-on wiring). Config-contract
    style (:func:`agent_utilities.core.config.setting`) — default ``True``
    reproduces the pre-existing mandatory-compilation behavior byte-for-byte, so a
    fresh/default config takes the compiler path with no action required. There is
    deliberately no per-call parameter: only ``MODEL_CONTEXT_COMPILER_ENABLED`` (an
    operator-level, whole-process setting sourced from ``config.json``/env, never
    from request/task content) can flip it off, for a genuine disable case (e.g. a
    cost-sensitive bulk/offline run against a corpus with no retrieval-worthy KG
    content yet) — see the module docstring.
    """

    return bool(setting("MODEL_CONTEXT_COMPILER_ENABLED", True))


def _compiled_evidence_and_bundle(
    messages: list[Any], model_name: str
) -> tuple[list[Any], ContextBundle | None]:
    """Compile governed evidence for ``messages`` and return ``(messages, bundle)``.

    The bundle-returning half of :func:`_compile_messages` — a caller that also
    wants to attribute a metric to THIS call's compiled evidence (Seam-6 TTFT
    labeling by ``kv_cache_hit``, CONCEPT:AU-KG.retrieval.context-compiler-kv-seam) needs the
    :class:`~agent_utilities.knowledge_graph.retrieval.context_compiler.ContextBundle`
    object, not just the rendered messages. ``bundle`` is ``None`` when compilation
    did not run this call — the messages were already compiled (idempotency
    marker present), or the :func:`_context_compiler_enabled` escape hatch is off —
    so "genuinely compiled this call" is distinguishable from "passthrough".
    """
    if _already_compiled(messages):
        return messages, None
    if _bound_tool_grounding.get():
        from pydantic_ai.messages import ModelRequest, SystemPromptPart

        evidence = ModelRequest(
            parts=[
                SystemPromptPart(
                    content=(
                        f"{_CONTEXT_MARKER}\n"
                        f"{_BOUND_TOOL_GROUNDING_MARKER}\n"
                        "The authenticated, pre-bound MCP tool results are the only "
                        "factual authority for this request. Call the bound tool before "
                        "making factual claims. Ground the answer in its returned data "
                        "and provenance, or state that the tool produced no evidence."
                    )
                )
            ]
        )
        return [evidence, *messages], None
    if not _context_compiler_enabled():
        logger.warning(
            "[CONCEPT:AU-KG.retrieval.context-compiler] MODEL_CONTEXT_COMPILER_ENABLED=false — "
            "sending this model request WITHOUT compiled evidence (ungrounded, "
            "no citations/proof-graph). This is a deployment-level escape hatch, "
            "not the default; re-enable it unless this run intentionally opted out."
        )
        return messages, None
    query = _query_from_messages(messages)
    bundle = compile_model_context(query, model_version=model_name)
    from pydantic_ai.messages import ModelRequest, SystemPromptPart

    evidence = ModelRequest(
        parts=[
            SystemPromptPart(
                content=(
                    f"{_CONTEXT_MARKER}\n"
                    "This governed evidence bundle is the only factual context "
                    "for the request. Cite it or state that evidence is absent.\n\n"
                    f"{bundle.as_text()}"
                )
            )
        ]
    )
    return [evidence, *messages], bundle


def _compile_messages(messages: list[Any], model_name: str) -> list[Any]:
    return _compiled_evidence_and_bundle(messages, model_name)[0]


def _record_retrieval_degraded(
    reason: str, model_name: str, *, error_type: str = ""
) -> None:
    """CONCEPT:AU-AHE.harness.runtime-reliability-loop — feed a retrieval degradation into
    the runtime-reliability detect→gap loop.

    A compilation that keeps timing out / erroring at the model-transport boundary is the
    event-loop-blocking retrieval that SIGKILLed graph-os — slow, not wrong, so it never
    penalized a reward and stayed invisible to the run-quality flywheel. Recording it makes
    a repeated degradation a SOURCE_RUNTIME (recommendation) gap. Fire-and-forget: a buffer
    append only, all exceptions swallowed, never any engine write on this hot path.
    """
    try:
        from agent_utilities.observability.runtime_signals import (
            KIND_RETRIEVAL_DEGRADED,
            record_runtime_signal,
        )

        detail: dict[str, Any] = {
            "timeout_s": _CONTEXT_COMPILE_TIMEOUT_S,
            "reason": reason,
        }
        if error_type:
            detail["error_type"] = error_type
        record_runtime_signal(KIND_RETRIEVAL_DEGRADED, "model_context_compile", detail)
    except Exception:  # noqa: BLE001 — emission must never affect the model call
        pass


def _ctx_compile_breaker_is_open() -> bool:
    """CONCEPT:AU-KG.retrieval.context-compile-circuit-breaker — OPEN check.

    True only while the streak has reached the threshold AND the cooldown set
    by the last trip/re-trip has not yet elapsed. Once the cooldown elapses this
    returns False again (implicitly HALF-OPEN) without any state mutation here —
    the next call through :func:`_compiled_evidence_and_bundle_bounded` is the
    probe attempt itself.
    """
    return (
        _ctx_compile_degradation_streak >= _CTX_COMPILE_BREAKER_THRESHOLD
        and time.monotonic() < _ctx_compile_breaker_reopen_at
    )


def _ctx_compile_note_degradation() -> None:
    """Record one degradation (timeout or error) against the breaker.

    CONCEPT:AU-KG.retrieval.context-compile-circuit-breaker. Increments the
    consecutive-degradation streak and, once it reaches
    ``_CTX_COMPILE_BREAKER_THRESHOLD``, (re)opens the breaker for another
    ``_CTX_COMPILE_BREAKER_COOLDOWN_S`` — this covers both the initial trip and a
    failed HALF-OPEN probe re-opening it. No lock: a benign race with a
    concurrent caller costs at most one extra probe attempt.
    """
    global _ctx_compile_degradation_streak, _ctx_compile_breaker_reopen_at
    _ctx_compile_degradation_streak += 1
    if _ctx_compile_degradation_streak >= _CTX_COMPILE_BREAKER_THRESHOLD:
        _ctx_compile_breaker_reopen_at = (
            time.monotonic() + _CTX_COMPILE_BREAKER_COOLDOWN_S
        )


def _ctx_compile_note_success() -> None:
    """Reset the breaker after a successful compile (CONCEPT:AU-KG.retrieval.context-compile-circuit-breaker)."""
    global _ctx_compile_degradation_streak, _ctx_compile_breaker_reopen_at
    _ctx_compile_degradation_streak = 0
    _ctx_compile_breaker_reopen_at = 0.0


def _degrade_for_policy(
    messages: list[Any],
    model_name: str,
    *,
    reason: str,
    compiled_bundle: ContextBundle | None = None,
) -> tuple[list[Any], ContextBundle | None]:
    """Apply the ambient :data:`GroundingPolicy` to a degraded/no-evidence outcome.

    CONCEPT:AU-KG.retrieval.fail-closed-grounding-contract — the single choke point
    both :func:`_compiled_evidence_and_bundle_bounded`'s timeout/error/breaker-open
    branches and its retrieval-quality-gate-failure branch route through.

    ``"required"`` (the default): marks the run degraded (for the RunTrace/OTel
    span this run's caller reads back via :func:`grounding_snapshot`) and RAISES
    :class:`GroundingUnavailableError` — the request never reaches the model.

    ``"best_effort"``/``"none"``: marks the run degraded and returns messages with
    an explicit ``_DEGRADED_GROUNDING_MARKER`` system-prompt prefix instructing the
    model to never present an ungrounded claim as sourced from the knowledge graph.
    ``compiled_bundle`` is returned as-is (not forced to ``None``) when the caller
    already has a real (possibly empty) :class:`ContextBundle` — e.g. the retrieval
    genuinely ran but the quality gate rejected every candidate — so TTFT/KV-cache
    metrics attribution is unaffected; it is a marker of "no usable evidence", not
    "no compute happened".
    """
    policy = _grounding_policy.get()
    _mark_grounding_degraded(reason)
    if policy == "required":
        raise GroundingUnavailableError(
            f"governed model invocation for {model_name!r} could not establish "
            f"compiled evidence ({reason}); grounding policy is 'required' (the "
            "default) — refusing to send an ungrounded request. Pass "
            "grounding='best_effort' or grounding='none' to explicitly opt into "
            "degraded operation."
        )
    from pydantic_ai.messages import ModelRequest, SystemPromptPart

    status = "none" if policy == "none" else "degraded"
    evidence = ModelRequest(
        parts=[
            SystemPromptPart(
                content=(
                    f"{_CONTEXT_MARKER}\n{_DEGRADED_GROUNDING_MARKER}\n"
                    f"grounding: {status}\nreason: {reason}\n"
                    "No governed evidence bundle could be established for this "
                    "request (the caller explicitly opted into degraded grounding "
                    f"via grounding={policy!r}). Do not present any claim as "
                    "sourced from the knowledge graph; state plainly that no "
                    "verified evidence was available."
                )
            )
        ]
    )
    return [evidence, *messages], compiled_bundle


async def _compiled_evidence_and_bundle_bounded(
    messages: list[Any], model_name: str
) -> tuple[list[Any], ContextBundle | None]:
    """Non-blocking, bounded wrapper over :func:`_compiled_evidence_and_bundle`.

    The compilation it wraps is a synchronous, engine-contended retrieval whose
    innermost hop (``graph_compute`` → the eg client's ``sync_wrapper`` →
    ``future.result()``) BLOCKS the calling thread. On the model-transport
    boundary that thread is the MAIN asyncio event loop, so a slow/contended
    retrieval stalls every other coroutine on it — including status-only
    ``/health`` — which is what SIGKILLed graph-os during a delegation. Running it
    via :func:`asyncio.to_thread` moves the block off the loop (``to_thread``
    propagates the ambient contextvars, so the verified ``GraphSession`` and the
    resource-priority class still reach the retriever), and :func:`asyncio.wait_for`
    bounds it by :data:`_CONTEXT_COMPILE_TIMEOUT_S`.

    CONCEPT:AU-KG.retrieval.fail-closed-grounding-contract: on timeout, any
    compilation error, an OPEN circuit breaker, or a retrieval-quality-gate
    failure (the compiled bundle carries no usable evidence), the outcome is
    decided by the ambient :data:`GroundingPolicy` (:func:`_degrade_for_policy`) —
    ``"required"`` (the default) RAISES rather than silently degrading; an
    explicit ``"best_effort"``/``"none"`` opt-in degrades with a visible marker.
    The fast path is unchanged: a quick, quality-gate-passing retrieval returns the
    full compiled bundle.
    """
    # Bound-tool grounding is a deterministic compiler-owned prefix: it performs
    # no engine retrieval, so it neither pays a thread hop nor inherits the KG
    # retrieval circuit breaker's degraded/ungrounded state.
    if _bound_tool_grounding.get():
        return _compiled_evidence_and_bundle(messages, model_name)

    if _ctx_compile_breaker_is_open():
        logger.debug(
            "[CONCEPT:AU-KG.retrieval.context-compile-circuit-breaker] breaker OPEN "
            "(%d consecutive degradations) — skipping compile, ~%.1fs left in cooldown.",
            _ctx_compile_degradation_streak,
            _ctx_compile_breaker_reopen_at - time.monotonic(),
        )
        return _degrade_for_policy(messages, model_name, reason="circuit_breaker_open")

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(_compiled_evidence_and_bundle, messages, model_name),
            timeout=_CONTEXT_COMPILE_TIMEOUT_S,
        )
    except (
        TimeoutError
    ):  # asyncio.wait_for's timeout (asyncio.TimeoutError is this alias)
        logger.warning(
            "[CONCEPT:AU-KG.retrieval.context-compiler] evidence compilation exceeded "
            "%.1fs and was abandoned; applying the grounding policy instead of "
            "silently sending this model request without compiled evidence.",
            _CONTEXT_COMPILE_TIMEOUT_S,
        )
        _record_retrieval_degraded("timeout", model_name)
        _ctx_compile_note_degradation()
        return _degrade_for_policy(messages, model_name, reason="timeout")
    except Exception as exc:  # noqa: BLE001 — best-effort boundary, never block
        logger.warning(
            "[CONCEPT:AU-KG.retrieval.context-compiler] evidence compilation failed "
            "(%s); applying the grounding policy instead of silently sending this "
            "model request without compiled evidence.",
            type(exc).__name__,
        )
        _record_retrieval_degraded("error", model_name, error_type=type(exc).__name__)
        _ctx_compile_note_degradation()
        return _degrade_for_policy(
            messages, model_name, reason=f"error:{type(exc).__name__}"
        )

    _ctx_compile_note_success()
    governed, bundle = result
    if bundle is not None and getattr(bundle, "retrieval_quality_gate_failed", False):
        reason = "quality_gate:" + (
            getattr(bundle, "retrieval_quality_reason", "") or "low_relevance_topk"
        )
        logger.warning(
            "[CONCEPT:AU-KG.retrieval.context-compiler] retrieval quality gate "
            "failed (%s); the compiled evidence bundle carries no usable "
            "candidates — applying the grounding policy.",
            reason,
        )
        # Not a compile-latency/error signal — never trips the compile-time
        # circuit breaker, which exists for a slow/broken retrieval PATH, not a
        # genuinely-sparse-or-noisy result for one query.
        return _degrade_for_policy(
            messages, model_name, reason=reason, compiled_bundle=bundle
        )
    _annotate_grounding_span("compiled")
    return governed, bundle


async def _compile_messages_bounded(messages: list[Any], model_name: str) -> list[Any]:
    return (await _compiled_evidence_and_bundle_bounded(messages, model_name))[0]


def _record_delegated_run_ttft(duration_s: float, bundle: ContextBundle | None) -> None:
    """Seam-6 TTFT surrogate for the delegated-run model-transport boundary (X6).

    CONCEPT:AU-KG.retrieval.context-compiler-kv-seam. Mirrors ``knowledge_graph.retrieval.
    context_compiler_serving._record_ttft`` — same histogram
    (``agent_utilities_context_compiler_ttft_seconds``), same best-effort/never-raises
    posture — but labeled ``path="delegated_run"`` so this interactive
    execute_agent/execute_workflow call population is never averaged together
    with the batch/enrichment ``bundle_chat_completion`` population under the same
    bucket (see that metric's ``path`` label). ``bundle is None`` (passthrough —
    already-compiled or the escape hatch is off) records nothing: there is no
    compiled evidence to attribute the call's latency to.
    """
    if bundle is None:
        return
    try:
        from agent_utilities.observability.gateway_metrics import (
            CONTEXT_COMPILER_TTFT,
        )

        CONTEXT_COMPILER_TTFT.labels(
            kv_cache_hit=str(bool(bundle.kv_cache_hit)).lower(),
            path="delegated_run",
        ).observe(duration_s)
    except Exception as exc:  # noqa: BLE001 — metrics must never break a model call
        logger.debug("delegated-run ttft metric recording failed: %s", exc)


def _wrapper_class() -> Any | None:
    global _WRAPPER_CLASS
    if _WRAPPER_CLASS is not None:
        return _WRAPPER_CLASS
    try:
        from pydantic_ai.models.wrapper import WrapperModel
    except Exception:
        return None

    class _ContextCompiledModel(WrapperModel):  # type: ignore[misc, valid-type]
        _agent_utilities_context_wrapper = True

        async def request(
            self, messages: list[Any], model_settings: Any, mrp: Any
        ) -> Any:
            governed, bundle = await _compiled_evidence_and_bundle_bounded(
                messages, self.model_name
            )
            start = time.perf_counter()
            try:
                return await super().request(governed, model_settings, mrp)
            finally:
                _record_delegated_run_ttft(time.perf_counter() - start, bundle)

        async def count_tokens(
            self, messages: list[Any], model_settings: Any, mrp: Any
        ) -> Any:
            governed = await _compile_messages_bounded(messages, self.model_name)
            return await super().count_tokens(governed, model_settings, mrp)

        async def compact_messages(
            self, request_context: Any, *, instructions: str | None = None
        ) -> Any:
            governed = await _compile_messages_bounded(
                request_context.messages, self.model_name
            )
            return await super().compact_messages(
                replace(request_context, messages=governed),
                instructions=instructions,
            )

        @asynccontextmanager
        async def request_stream(
            self,
            messages: list[Any],
            model_settings: Any,
            mrp: Any,
            run_context: Any | None = None,
        ) -> AsyncIterator[Any]:
            governed, bundle = await _compiled_evidence_and_bundle_bounded(
                messages, self.model_name
            )
            start = time.perf_counter()
            try:
                async with super().request_stream(
                    governed, model_settings, mrp, run_context
                ) as response:
                    yield response
            finally:
                _record_delegated_run_ttft(time.perf_counter() - start, bundle)

    _WRAPPER_CLASS = _ContextCompiledModel
    return _WRAPPER_CLASS


def wrap_model_with_context(model: Any) -> Any:
    """Return the mandatory ContextCompiler wrapper, idempotently."""

    if isinstance(model, str):
        # Pydantic AI accepts convenient model-name strings, but they otherwise
        # bypass the model-factory transport choke point entirely. Resolve them
        # through the factory before wrapping; the second call is a real Model
        # and therefore cannot recurse through this branch.
        from agent_utilities.core.model_factory import create_model

        provider, model_id = (None, model)
        if ":" in model:
            prefix, candidate = model.split(":", 1)
            if prefix.lower() in {
                "openai",
                "anthropic",
                "google",
                "gemini",
                "groq",
                "mistral",
                "huggingface",
                "ollama",
                "deepseek",
            }:
                provider, model_id = prefix.lower(), candidate
                if provider == "gemini":
                    provider = "google"
        return create_model(provider=provider, model_id=model_id)
    if getattr(model, "_agent_utilities_context_wrapper", False):
        return model
    wrapper = _wrapper_class()
    if wrapper is None:
        raise ContextCompilationError(
            "pydantic-ai WrapperModel is required for governed model invocation"
        )
    return wrapper(model)


def create_context_agent(
    model: Any = _MISSING_MODEL,
    *,
    permissions_kernel: Any | None = None,
    agent_identity: Any | None = None,
    permission_engine: Any | None = None,
    default_capabilities: bool = True,
    **kwargs: Any,
) -> Any:
    """Construct a Pydantic AI agent behind the mandatory context boundary.

    This is the sole application constructor for Pydantic AI agents.  It requires
    an explicit model, idempotently installs the transport wrapper, and only then
    instantiates the agent.  Runtime model injection therefore cannot bypass the
    compiler by handing an arbitrary model object directly to ``Agent``.

    This is ALSO the single capability-composition seam.
    With ``default_capabilities=True`` (the default),
    every agent it builds — the graph executor, the ~20 KG-internal sites, delegated
    agents, and ``create_agent`` alike — receives the default-ON reliability capability
    set (stuck-loop detection, context warnings, tool-output eviction, live Memento
    compaction, and a hooks capability), merged by concrete type over any capabilities
    the caller already supplied so nothing is double-added. ``create_agent`` assembles
    its own richly-configured list from the same
    :func:`~agent_utilities.capabilities.composition.default_runtime_capabilities` and
    passes ``default_capabilities=False`` so its explicit opt-out flags win.
    """

    if model is _MISSING_MODEL or model is None:
        raise ContextCompilationError(
            "governed agent construction requires an explicit model"
        )
    if default_capabilities:
        from agent_utilities.capabilities.composition import (
            default_runtime_capabilities,
            merge_capabilities,
        )
        from agent_utilities.capabilities.hooks import HooksCapability

        supplied = list(kwargs.get("capabilities", ()) or ())
        defaults = default_runtime_capabilities()
        defaults.append(HooksCapability(hooks=[]))
        kwargs["capabilities"] = merge_capabilities(supplied, defaults)
    toolsets = list(kwargs.get("toolsets", ()) or ())
    raw_mcp_bound = any(
        hasattr(toolset, "list_tools") or hasattr(toolset, "direct_call_tool")
        for toolset in toolsets
    )
    if raw_mcp_bound:
        if permissions_kernel is None or agent_identity is None:
            raise PermissionError(
                "raw MCP toolsets require an explicitly injected permission context"
            )
        from agent_utilities.security.permissions_kernel import (
            verify_permission_context,
        )
        from agent_utilities.security.tool_guard import flag_mcp_tool_definitions

        permission_context = verify_permission_context(
            permissions_kernel,
            agent_identity,
        )
        if permission_context is None:  # defensive: an explicit pair was supplied
            raise PermissionError("MCP permission context is required")
        kwargs["toolsets"] = flag_mcp_tool_definitions(
            toolsets,
            permissions_kernel=permission_context.kernel,
            agent_identity=permission_context.identity,
            engine=permission_engine,
        )
    elif (permissions_kernel is None) != (agent_identity is None):
        raise PermissionError(
            "permission kernel and agent identity must be injected together"
        )
    elif permissions_kernel is not None and agent_identity is not None:
        from agent_utilities.security.permissions_kernel import (
            verify_permission_context,
        )

        verify_permission_context(permissions_kernel, agent_identity)
    from pydantic_ai import Agent as _PydanticAgent

    return _PydanticAgent(model=wrap_model_with_context(model), **kwargs)


def instrument_context_agents(settings: Any) -> Any:
    """Install process-wide instrumentation on the governed agent runtime."""

    from pydantic_ai import Agent as _PydanticAgent

    _PydanticAgent.instrument_all(settings)
    return getattr(_PydanticAgent, "_instrument_default", None)


def disable_context_agent_instrumentation() -> None:
    """Disable process-wide instrumentation on the governed agent runtime."""

    from pydantic_ai import Agent as _PydanticAgent

    _PydanticAgent.instrument_all(False)
