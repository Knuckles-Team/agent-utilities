"""Capability designation for the live router — RECEIVED from epistemic-graph.

Capability designation is an EG choice (``EG-DECISION-ENGINE-R172``): EG holds the
native capability index, applies capability/tenant/policy filters with ontology
subsumption, ranks by cosine + learned reward + ontology prior, and learns from
outcomes. agent-utilities keeps no index and does no scoring: this module only
forwards the router's question to the :class:`~agent_utilities.knowledge_graph.
retrieval.capability_designation_port.CapabilityDesignationPort` bound on the
engine (``engine.capability_designator``) and forwards outcomes back.

With no port bound, :func:`designate_specialists` returns ``None`` so routing
falls back to its keyword scan.
"""

from __future__ import annotations

import logging
from typing import Any

from agent_utilities.knowledge_graph.retrieval.capability_designation_port import (
    CapabilityDesignationRequest,
    designation_port_of,
)

logger = logging.getLogger(__name__)


def embed_query(query: str, embed_fn: Any = None) -> Any | None:
    """Resolve ``query`` to an embedding vector via ``embed_fn``, or the default model.

    The router embeds the query text; EG designates over the vector. Returns ``None`` (never raises) when no model is configured.
    """
    if embed_fn is None:
        from agent_utilities.core.embedding_utilities import create_embedding_model

        model = create_embedding_model()
        if model is None:
            return None
        embed_fn = model.get_text_embedding
    return embed_fn(query)


def designate_specialists(
    engine: Any,
    query: str,
    *,
    k: int = 5,
    required_caps: list[str] | None = None,
    tenant: str | None = None,
    policy_tags: list[str] | None = None,
    embed_fn: Any = None,
) -> list[str] | None:
    """Return the EG-designated top-``k`` callable resource ids for ``query``.

    Returns ``None`` when designation is unavailable (no EG port bound, no
    embedding model, or the port failed) so the caller falls back to its
    keyword scan. The ids are EG's order; nothing is re-ranked here.
    """
    port = designation_port_of(engine)
    if port is None:
        return None
    try:
        embedding = embed_query(query, embed_fn)
        if embedding is None:
            return None
        designations = port.designate(
            CapabilityDesignationRequest(
                query_vector=[float(v) for v in embedding],
                required_capabilities=tuple(required_caps or ()),
                required_policy_tags=tuple(policy_tags or ()),
                tenant=tenant,
                k=int(k),
            )
        )
        return [d.id for d in designations]
    except Exception as e:  # never break routing  # noqa: BLE001 — keyword-scan fallback is the documented degraded path
        logger.debug("EG capability designation unavailable, falling back: %s", e)
        return None


def record_capability_outcome(
    engine: Any,
    entity_id: str,
    *,
    success: bool | None = None,
    reward: float | None = None,
    source_ids: list[str] | None = None,
) -> float:
    """Forward a routing outcome to EG, which learns it (``EG-DECISION-ENGINE-R172.5``).

    Reward learning is an EG decision: agent-utilities computes no average and
    persists no reward. The outcome goes through the designation port bound on
    the engine; EG returns its updated reward. With no port bound (or a port
    failure) nothing is learned and the neutral ``0.5`` prior is returned. Never
    raises.

    ``source_ids`` (X-6 / Seam 3, CONCEPT:EG-KG.epistemic.truth-maintenance):
    when a caller passes the real id(s) of the observation the outcome came from
    (e.g. :class:`~agent_utilities.knowledge_graph.research.claim_flywheel.
    ClaimFlywheel.record_outcome`'s freshly-written ``ClaimOutcome`` node id —
    never fabricated) and EG accepted the outcome, this stamps a
    ``:DerivedFrom`` edge from ``entity_id`` to each id and registers
    ``entity_id`` as a live engine-side TruthMaintenance materialization off that
    provenance, so a later change to the observation marks the reward stale.
    Best-effort, never gates the forward above.
    """
    port = designation_port_of(engine)
    if port is None:
        return 0.5
    try:
        updated = port.record_outcome(entity_id, success=success, reward=reward)
    except Exception as e:  # noqa: BLE001 — outcome forwarding never breaks the caller
        logger.debug(
            "record_capability_outcome: EG outcome failed for %r: %s",
            entity_id,
            e,
        )
        return 0.5
    if updated is None:
        return 0.5
    if source_ids:
        _register_capability_reward_materialization(engine, entity_id, source_ids)
    return float(updated)


def _register_capability_reward_materialization(
    engine: Any, entity_id: str, source_ids: list[str]
) -> None:
    """X-6 / Seam 3 (CONCEPT:EG-KG.epistemic.truth-maintenance) — see
    :func:`record_capability_outcome`'s ``source_ids`` docstring. Best-effort;
    never raises into the caller's durable-write path."""
    for source_id in source_ids:
        try:
            engine.add_edge(entity_id, source_id, "DERIVED_FROM")
        except Exception as e:  # noqa: BLE001 — provenance edges are best-effort
            logger.debug(
                "record_capability_outcome: derived_from edge %r->%r failed: %s",
                entity_id,
                source_id,
                e,
            )
    try:
        engine.register_materialization(entity_id)
    except Exception as e:  # noqa: BLE001 — TMS registration is best-effort
        logger.debug(
            "record_capability_outcome: register_materialization failed for %r: %s",
            entity_id,
            e,
        )


__all__ = [
    "designate_specialists",
    "embed_query",
    "record_capability_outcome",
]
