"""Automatic embedder endpoint failover (CONCEPT:AU-KG.enrichment.each-call-resolves-active).

The embedding plane runs against a **primary** embedder endpoint (e.g. a dedicated
``gr1080-embed.arpa`` box, ``gpu_group="gr1080"``). When that endpoint is down or
unreachable, embeds must not simply fail — they transparently **fail over to a
configured fallback** endpoint (e.g. the shared ``vllm-embed.arpa`` on the GB10,
``gpu_group="gb10"``), and route **back automatically** once the primary recovers.

This module is the single resolver of *which* embedder endpoint is active right
now. It is consulted by:

* :func:`agent_utilities.core.embedding_utilities.create_embedding_model` — so the
  cached embedding client is built for the ACTIVE endpoint (the cache keys on the
  resolved base_url, so it swaps to the fallback's client on failover and back).
* :func:`agent_utilities.knowledge_graph.enrichment.semantic.make_embed_fn` — so the
  batched fan-out gates on the ACTIVE endpoint's **model key**, which makes the
  whole capacity guard (server_ceiling / adaptive capacity / GPU-group budget)
  resolve the active endpoint's config. In fallback mode the key is
  ``embedding:fallback`` → ``Config.gpu_group`` returns the fallback's ``gpu_group``
  (``gb10``), so the fan-out shares the GB10 **joint budget** with the generator and
  can never OOM the shared box — exactly the operator's requirement.

**The failover trigger reuses the existing per-endpoint circuit breaker**
(CONCEPT:AU-ORCH.routing.load-shedding-backoff): the PRIMARY embedder's breaker (keyed ``embedding``) is fed by
the primary embed fan-out itself. When a connection-failure / timeout / overload
trips it OPEN, :meth:`ModelCircuitBreaker.is_tripped` returns ``True`` and this
router selects the fallback. Recovery is automatic and needs no extra machinery:
once the breaker's backoff cooldown elapses ``is_tripped`` flips back to ``False``,
so the next batch returns to the PRIMARY and the breaker's own HALF_OPEN probe
decides whether it truly recovered (close → stay primary) or re-opens (→ fallback
again next round). No fallback configured ⇒ always primary (zero behaviour change).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)

__all__ = [
    "PRIMARY_KEY",
    "FALLBACK_KEY",
    "EmbeddingEndpoint",
    "active_embedding_endpoint",
    "embedding_endpoint_status",
    "reset_embedding_failover",
]

#: Capacity-guard model key for the PRIMARY embedder. ``make_embed_fn`` feeds this
#: key's circuit breaker, so a primary outage trips THIS breaker → failover.
PRIMARY_KEY = "embedding"
#: Capacity-guard model key for the FALLBACK embedder. ``Config._resolve_model_config``
#: resolves it to the primary's ``.fallback`` config, so server_ceiling / adaptive
#: capacity / gpu_group budget all key off the fallback endpoint while failed-over.
FALLBACK_KEY = "embedding:fallback"


@dataclass(frozen=True)
class EmbeddingEndpoint:
    """The embedder endpoint that is active right now (CONCEPT:AU-KG.enrichment.each-call-resolves-active)."""

    #: Capacity-guard model key — :data:`PRIMARY_KEY` or :data:`FALLBACK_KEY`. Drives
    #: the per-endpoint gate, breaker, and GPU-group budget bucket.
    model_key: str
    model_id: str | None
    provider: str | None
    base_url: str | None
    api_key: str | None
    oauth2: dict[str, object] | None
    gpu_group: str | None
    is_fallback: bool


# --- observability: track + log endpoint transitions ------------------------
_obs_lock = threading.Lock()
_last_active_key: str | None = None
_failover_count = 0
_recovery_count = 0


def _observe(endpoint: EmbeddingEndpoint) -> None:
    """Log + count primary↔fallback transitions (CONCEPT:AU-KG.enrichment.each-call-resolves-active). Never raises."""
    global _last_active_key, _failover_count, _recovery_count
    with _obs_lock:
        prev = _last_active_key
        if prev == endpoint.model_key:
            return
        _last_active_key = endpoint.model_key
        # First resolution (prev is None) is not a transition — just record it.
        if prev is None:
            return
        if endpoint.is_fallback:
            _failover_count += 1
            logger.warning(
                "embedding failover: PRIMARY embedder unreachable — routing embeds "
                "to FALLBACK endpoint %s (gpu_group=%s, key=%s); the GPU-group "
                "budget for that group now governs embed concurrency.",
                endpoint.base_url,
                endpoint.gpu_group,
                endpoint.model_key,
            )
        else:
            _recovery_count += 1
            logger.info(
                "embedding recovery: PRIMARY embedder %s reachable again — routing "
                "embeds back to PRIMARY (gpu_group=%s, key=%s).",
                endpoint.base_url,
                endpoint.gpu_group,
                endpoint.model_key,
            )


def _endpoint_from_cfg(
    model_key: str, cfg: object, *, is_fallback: bool
) -> EmbeddingEndpoint:
    """Build an :class:`EmbeddingEndpoint` from an ``EmbeddingModelConfig``."""
    gpu_group: str | None = None
    try:
        from agent_utilities.core.config import config

        gpu_group = config.gpu_group(model_key)
    except Exception:  # noqa: BLE001 — grouping is best-effort observability
        tag = getattr(cfg, "gpu_group", None)
        gpu_group = str(tag).strip().lower() if tag else None
    return EmbeddingEndpoint(
        model_key=model_key,
        model_id=getattr(cfg, "id", None),
        provider=getattr(cfg, "provider", None),
        base_url=getattr(cfg, "base_url", None),
        api_key=getattr(cfg, "api_key", None),
        oauth2=getattr(cfg, "oauth2", None),
        gpu_group=gpu_group,
        is_fallback=is_fallback,
    )


def active_embedding_endpoint() -> EmbeddingEndpoint:
    """Resolve the embedder endpoint to use right now (CONCEPT:AU-KG.enrichment.each-call-resolves-active).

    Returns the PRIMARY endpoint unless a fallback is configured AND the primary
    embedder's circuit breaker (CONCEPT:AU-ORCH.routing.load-shedding-backoff) is actively tripped, in which
    case it returns the FALLBACK endpoint (key :data:`FALLBACK_KEY`). Transparent to
    callers, automatic in both directions, and fail-safe: any resolution error or a
    missing config collapses to a bare PRIMARY endpoint (no failover), so embedding
    never breaks because of this router.
    """
    cfg = None
    try:
        from agent_utilities.core.config import config

        cfg = config.default_embedding_model
    except Exception:  # noqa: BLE001 — no config → bare primary, no failover
        cfg = None
    if cfg is None:
        return EmbeddingEndpoint(
            model_key=PRIMARY_KEY,
            model_id=None,
            provider=None,
            base_url=None,
            api_key=None,
            oauth2=None,
            gpu_group=None,
            is_fallback=False,
        )

    primary = _endpoint_from_cfg(PRIMARY_KEY, cfg, is_fallback=False)
    fallback_cfg = getattr(cfg, "fallback", None)
    if fallback_cfg is None:
        _observe(primary)
        return primary

    # Failover decision: route to the fallback only while the PRIMARY embedder is
    # actively shedding/unreachable (its breaker is OPEN within cooldown). Recovery
    # is automatic — once the cooldown elapses is_tripped() flips False and we
    # return to the primary (its HALF_OPEN probe then confirms or re-opens).
    tripped = False
    try:
        from agent_utilities.core.model_circuit_breaker import get_circuit_breaker

        tripped = get_circuit_breaker(PRIMARY_KEY).is_tripped()
    except Exception:  # noqa: BLE001 — breaker is best-effort; default to primary
        tripped = False

    endpoint = (
        _endpoint_from_cfg(FALLBACK_KEY, fallback_cfg, is_fallback=True)
        if tripped
        else primary
    )
    _observe(endpoint)
    return endpoint


def embedding_endpoint_status() -> dict[str, object]:
    """Queryable snapshot of the embedder failover state (CONCEPT:AU-KG.enrichment.each-call-resolves-active).

    Surfaces the active endpoint, whether it is the fallback, the primary breaker's
    state, and the cumulative failover/recovery counts — a small, stable signal for
    observability/doctor surfaces. Never raises.
    """
    endpoint = active_embedding_endpoint()
    breaker_snap: dict[str, object] = {}
    try:
        from agent_utilities.core.model_circuit_breaker import get_circuit_breaker

        breaker_snap = get_circuit_breaker(PRIMARY_KEY).snapshot()
    except Exception:  # noqa: BLE001 — observability must never raise
        breaker_snap = {}
    with _obs_lock:
        failovers, recoveries = _failover_count, _recovery_count
    return {
        "active_model_key": endpoint.model_key,
        "active_base_url": endpoint.base_url,
        "active_gpu_group": endpoint.gpu_group,
        "is_fallback": endpoint.is_fallback,
        "fallback_configured": endpoint.is_fallback or _fallback_is_configured(),
        "primary_breaker": breaker_snap,
        "failover_count": failovers,
        "recovery_count": recoveries,
    }


def _fallback_is_configured() -> bool:
    try:
        from agent_utilities.core.config import config

        cfg = config.default_embedding_model
        return bool(cfg is not None and getattr(cfg, "fallback", None) is not None)
    except Exception:  # noqa: BLE001
        return False


def reset_embedding_failover() -> None:
    """Reset the observability counters/state (test isolation). CONCEPT:AU-KG.enrichment.each-call-resolves-active."""
    global _last_active_key, _failover_count, _recovery_count
    with _obs_lock:
        _last_active_key = None
        _failover_count = 0
        _recovery_count = 0
