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
from dataclasses import replace
from typing import TYPE_CHECKING, Any

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
    "compile_model_context",
    "create_context_agent",
    "disable_context_agent_instrumentation",
    "instrument_context_agents",
    "set_context_compiler_engine",
    "set_context_compiler_cache",
    "get_context_compiler_cache",
    "use_context_compiler_engine",
    "wrap_model_with_context",
]

_CONTEXT_MARKER = "[agent-utilities:compiled-evidence:v1]"

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

    On timeout OR any compilation error the request DEGRADES to the ungrounded
    passthrough ``(messages, None)`` — byte-for-byte the shape used when the
    evidence is already compiled or the escape hatch is off — because the evidence
    boundary is best-effort at the transport edge and liveness must win over a
    contended retrieval. Degradation is logged loudly so an ungrounded call is
    always observable. The fast path is unchanged: a quick retrieval returns the
    full compiled bundle.

    CONCEPT:AU-KG.retrieval.context-compile-circuit-breaker: when the breaker is
    already OPEN (``_CTX_COMPILE_BREAKER_THRESHOLD`` consecutive degradations,
    cooldown not yet elapsed) this returns the degraded ``(messages, None)``
    immediately — skipping the to_thread+wait_for round trip — so a persistently
    broken/contended retrieval is not re-paid ``_CONTEXT_COMPILE_TIMEOUT_S`` on
    every single call.
    """
    if _ctx_compile_breaker_is_open():
        logger.debug(
            "[CONCEPT:AU-KG.retrieval.context-compile-circuit-breaker] breaker OPEN "
            "(%d consecutive degradations) — skipping compile, ~%.1fs left in cooldown.",
            _ctx_compile_degradation_streak,
            _ctx_compile_breaker_reopen_at - time.monotonic(),
        )
        return messages, None

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(_compiled_evidence_and_bundle, messages, model_name),
            timeout=_CONTEXT_COMPILE_TIMEOUT_S,
        )
        _ctx_compile_note_success()
        return result
    except TimeoutError:  # asyncio.wait_for's timeout (asyncio.TimeoutError is this alias)
        logger.warning(
            "[CONCEPT:AU-KG.retrieval.context-compiler] evidence compilation exceeded "
            "%.1fs and was abandoned; sending this model request WITHOUT compiled "
            "evidence (ungrounded) to keep the event loop responsive.",
            _CONTEXT_COMPILE_TIMEOUT_S,
        )
        _record_retrieval_degraded("timeout", model_name)
        _ctx_compile_note_degradation()
        return messages, None
    except Exception as exc:  # noqa: BLE001 — best-effort boundary, never block/raise
        logger.warning(
            "[CONCEPT:AU-KG.retrieval.context-compiler] evidence compilation failed "
            "(%s); sending this model request WITHOUT compiled evidence (ungrounded).",
            type(exc).__name__,
        )
        _record_retrieval_degraded("error", model_name, error_type=type(exc).__name__)
        _ctx_compile_note_degradation()
        return messages, None


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
