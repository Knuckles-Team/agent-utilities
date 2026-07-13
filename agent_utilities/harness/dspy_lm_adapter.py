from __future__ import annotations

"""Endpoint-safe DSPy LM adapter (CONCEPT:AU-AHE.optimization.endpoint-safe-dspy-optimization — a concurrency-bounded, priority-yielding, usage-tracked DSPy LM).

DSPy defaults to its own litellm-backed ``LM`` when no ``dspy.configure(lm=...)``/
``dspy.context(lm=...)`` is installed. Unconfigured, that default LM would call the
fleet's vLLM endpoint DIRECTLY — bypassing:

* :mod:`agent_utilities.core.model_concurrency` — the per-model concurrency
  controller every other LLM caller in this codebase routes through
  (``resolve_capacity`` / the shared priority gate it backs);
* :mod:`agent_utilities.core.resource_priority` — the resource-priority edict
  (interactive/orchestration ALWAYS admitted ahead of background work on the
  SAME shared LLM endpoint);
* :mod:`agent_utilities.core.background_throttle` — the global background-work
  throttle (yields to interactive/bulk-ingest work).

On a deployment with a HARD cap on parallel LLM endpoints, an unthrottled DSPy
optimizer — which can itself fan out many concurrent generations across
candidate programs / bootstrapped demos — could oversubscribe the endpoint and
starve interactive/orchestration traffic. This module is the ONE place DSPy
optimization resolves its LM from: :func:`dspy_optimization_guard` is the context
manager every DSPy-optimization entrypoint wraps its work in.
"""

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

logger = logging.getLogger(__name__)

try:  # OpenTelemetry is optional — degrade to logging-only when absent.
    from opentelemetry import trace as _otel_trace

    _tracer: Any = _otel_trace.get_tracer("agent-utilities.dspy_optimization")
except Exception:  # noqa: BLE001 - tracing is best-effort, never a hard dependency
    _otel_trace = None
    _tracer = None

__all__ = [
    "resolve_dspy_chat_model",
    "build_dspy_lm",
    "dspy_optimization_guard",
    "optimization_span",
]

# Lazily-built dspy.LM subclass + lazily-built shared token-usage tracker — both
# resolved once and cached, mirroring the ``_tracing_model_cls`` idiom in
# ``harness.tracing`` so this module stays importable (no hard ``dspy`` import at
# module scope) in a lean install without the ``[agent]`` extra.
_LM_CLASS: Any = None
_LM_CLASS_RESOLVED = False
_tracker_instance: Any = None


def resolve_dspy_chat_model() -> tuple[str, str | None, str | None]:
    """Resolve ``(model_id, base_url, api_key)`` for DSPy from the SAME configured
    fleet chat model every other caller uses — never a hardcoded id/endpoint, and
    never a direct env read (CONCEPT:AU-OS.config.model-factory-passthrough).

    Mirrors :func:`agent_utilities.core.model_factory._create_model_impl`'s
    resolution order: the operator's ``config.default_chat_model``
    (``config.chat_models`` with ``intelligence_level='normal'``, else the first
    configured model) wins; when NO chat model is registered at all, falls back to
    ``config.openai_base_url``/``config.openai_api_key`` with the same last-resort
    literal id ``create_model`` uses.
    """
    from agent_utilities.core.config import config

    model = config.default_chat_model
    if model is not None:
        base_url = model.base_url or config.openai_base_url
        api_key = model.api_key if model.api_key is not None else config.openai_api_key
        return model.id, base_url, api_key
    return "qwen/qwen3.6-27b", config.openai_base_url, config.openai_api_key


def _tracker() -> Any:
    """The shared :class:`TokenUsageTracker` DSPy usage is recorded through."""
    global _tracker_instance
    if _tracker_instance is None:
        from agent_utilities.observability.token_tracker import TokenUsageTracker

        _tracker_instance = TokenUsageTracker()
    return _tracker_instance


def _record_dspy_usage(model_key: str, response: Any, latency_s: float) -> None:
    """Best-effort usage telemetry for one DSPy generation, tagged
    ``source=dspy_optimization`` (CONCEPT:AU-OS.observability.granular-token-analytics) — the SAME
    :class:`TokenUsageTracker` every other caller records through, so DSPy's
    consumption counts against the same visibility/cap as everything else. Never
    raises — telemetry must never break an optimization pass.
    """
    try:
        usage = getattr(response, "usage", None)
        if not usage:
            return
        get = (
            usage.get
            if isinstance(usage, dict)
            else lambda k, d=0: getattr(usage, k, d)
        )
        prompt_tokens = int(get("prompt_tokens", 0) or 0)
        completion_tokens = int(get("completion_tokens", 0) or 0)
        if not (prompt_tokens or completion_tokens):
            return
        from agent_utilities.observability.token_tracker import TokenUsageRecord

        _tracker().record(
            TokenUsageRecord(
                agent_name="dspy_optimization",
                model_name=model_key,
                prompt_tokens=prompt_tokens,
                response_tokens=completion_tokens,
                metadata={
                    "source": "dspy_optimization",
                    "latency_s": round(latency_s, 3),
                },
            )
        )
    except Exception as e:  # noqa: BLE001 - telemetry is best-effort
        logger.debug("dspy usage telemetry skipped: %s", e)


def _throttled_lm_cls() -> Any:
    """Lazily build (once) the ``dspy.LM`` subclass that routes every generation
    through the shared endpoint guards. Returns ``None`` when DSPy is not
    installed — the caller then skips DSPy configuration entirely (no ``dspy``
    import in this process means no DSPy call can happen at all, so there is
    nothing to protect).
    """
    global _LM_CLASS, _LM_CLASS_RESOLVED
    if _LM_CLASS_RESOLVED:
        return _LM_CLASS
    _LM_CLASS_RESOLVED = True
    try:
        import dspy
    except Exception:  # noqa: BLE001 - DSPy is an optional [agent] extra
        return None

    class _ConcurrencyBoundDSPyLM(dspy.LM):  # type: ignore[misc, valid-type]
        """A ``dspy.LM`` gated by the per-model concurrency controller +
        resource-priority edict every other LLM caller in this codebase routes
        through (CONCEPT:AU-AHE.optimization.endpoint-safe-dspy-optimization). Composition,
        not reimplementation: still litellm under the hood via ``dspy.LM``, so every
        provider/streaming/caching behavior is inherited unchanged — only the
        admission + telemetry around each call is added.
        """

        def __init__(self, *, au_model_key: str, **kwargs: Any) -> None:
            self._au_model_key = au_model_key
            super().__init__(**kwargs)

        def forward(
            self, prompt: str | None = None, messages: Any = None, **kwargs: Any
        ) -> Any:
            from agent_utilities.core.resource_priority import (
                PriorityClass,
                current_priority,
                priority_slot_sync,
            )

            # Untagged ⇒ BACKGROUND_INGESTION (CONCEPT:AU-ORCH.scheduling.resource-priority-edict):
            # DSPy optimization is background work by construction — it must never
            # look "interactive" to the shared gate even if a caller forgot to tag it.
            prio = current_priority() or PriorityClass.BACKGROUND_INGESTION
            t0 = time.monotonic()
            # Bounded to the model's model_concurrency.resolve_capacity (the SAME
            # capacity every other fan-out over this model respects) AND yields to
            # interactive/orchestration contention on the shared endpoint.
            with priority_slot_sync(self._au_model_key, priority=prio):
                response = super().forward(prompt=prompt, messages=messages, **kwargs)
            _record_dspy_usage(self._au_model_key, response, time.monotonic() - t0)
            return response

        async def aforward(
            self, prompt: str | None = None, messages: Any = None, **kwargs: Any
        ) -> Any:
            from agent_utilities.core.resource_priority import (
                PriorityClass,
                current_priority,
                priority_slot,
            )

            prio = current_priority() or PriorityClass.BACKGROUND_INGESTION
            t0 = time.monotonic()
            async with priority_slot(self._au_model_key, priority=prio):
                response = await super().aforward(
                    prompt=prompt, messages=messages, **kwargs
                )
            _record_dspy_usage(self._au_model_key, response, time.monotonic() - t0)
            return response

    _ConcurrencyBoundDSPyLM.__name__ = "ConcurrencyBoundDSPyLM"
    _ConcurrencyBoundDSPyLM.__qualname__ = "ConcurrencyBoundDSPyLM"
    _LM_CLASS = _ConcurrencyBoundDSPyLM
    return _LM_CLASS


def build_dspy_lm(**overrides: Any) -> Any:
    """Build the throttled DSPy LM for the operator's configured fleet chat model.

    Resolves model/base_url/key from ``config.chat_models``/``config.default_chat_model``
    (never hardcoded, never a direct env read — see :func:`resolve_dspy_chat_model`).
    Returns ``None`` when DSPy is not installed.
    """
    cls = _throttled_lm_cls()
    if cls is None:
        return None
    model_id, base_url, api_key = resolve_dspy_chat_model()
    kwargs: dict[str, Any] = {}
    if base_url:
        kwargs["api_base"] = base_url
    kwargs["api_key"] = api_key or "EMPTY"
    kwargs.update(overrides)
    return cls(au_model_key=model_id, model=f"openai/{model_id}", **kwargs)


@contextmanager
def dspy_optimization_guard(op_label: str = "dspy_optimization") -> Iterator[bool]:
    """The ONE guard every DSPy-optimization entrypoint wraps its work in
    (CONCEPT:AU-AHE.optimization.endpoint-safe-dspy-optimization) — the scheduled
    ``dspy_optimization`` tick AND the on-demand ``graph_orchestrate
    action=optimize_component`` surface both funnel through
    :func:`agent_utilities.harness.dspy_optimization.run_component_optimization`,
    which installs this guard around its dispatch.

    1. Tags the ambient :class:`PriorityClass` ``BACKGROUND_INGESTION`` for the
       duration of the block (CONCEPT:AU-ORCH.scheduling.resource-priority-edict).
       DSPy optimization is background work by definition, so every nested DSPy LM
       call resolves that priority via ``current_priority()`` and yields to
       interactive/orchestration load on the shared model gate, never starving it.
    2. Takes a slot on the global background-work throttle
       (:mod:`agent_utilities.core.background_throttle`) — best-effort: if the
       (fixed, small) throttle is saturated by other background daemons, this
       proceeds anyway under the per-model priority gate alone, since THAT gate
       (not this coarser job-level one) is the authoritative endpoint-safety net.
    3. Installs the throttled/telemetry-recording LM for the block via
       ``dspy.context(lm=...)`` — deliberately NOT ``dspy.configure(...)``. DSPy
       restricts ``configure`` to a single owning thread/async task and RAISES on
       a second call from a different one — exactly what happens here, since the
       scheduler tick and an MCP request run on different threads/async tasks.
       ``dspy.context(...)`` is DSPy's own documented, safe-from-any-thread-or-
       async-task equivalent for repeated/concurrent overrides.

    Yields ``True`` when a throttled LM was actually installed (DSPy reachable),
    ``False`` when DSPy is unavailable (the block still runs — every DSPy call site
    itself no-ops without DSPy; this just skips the no-op ``dspy.context``).
    """
    from agent_utilities.core.background_throttle import get_throttle
    from agent_utilities.core.resource_priority import PriorityClass, priority_scope

    lm = build_dspy_lm()
    with priority_scope(PriorityClass.BACKGROUND_INGESTION):
        with get_throttle().background_slot() as acquired:
            if not acquired:
                logger.info(
                    "%s: background-work throttle saturated; proceeding on the "
                    "per-model priority gate alone",
                    op_label,
                )
            if lm is None:
                logger.debug(
                    "%s: DSPy not installed; skipping LM configuration", op_label
                )
                yield False
                return
            import dspy

            with dspy.context(lm=lm):
                yield True


def optimization_span(name: str):
    """An OTEL span for one DSPy-optimization unit of work, or a no-op context
    manager when OpenTelemetry is not configured. ``with optimization_span(...) as
    span:`` — ``span`` is ``None`` when tracing is inactive; guard attribute-setting
    accordingly."""
    if _tracer is None:
        return nullcontext(None)
    return _tracer.start_as_current_span(name)
