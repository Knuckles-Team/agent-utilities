#!/usr/bin/python
from __future__ import annotations

"""Ontology-driven tool/agent routing — X-4 (CONCEPT:AU-P1-3).

The single X-4 entry point: combine the engine's filtered ANN (AU-P1-3), ontology
SUBSUMPTION (:mod:`agent_utilities.knowledge_graph.ontology.capability_hierarchy`),
and tenant/policy filters into ONE candidate-selection call, re-ranked by the
durable contextual bandit (:mod:`~.durable_outcome_store` /
:class:`~agent_utilities.knowledge_graph.retrieval.capability_index.CapabilityIndex`),
with a WHY-eligible explanation attached to every candidate. The engine is the
only candidate-selection and capability-property authority.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from agent_utilities.graph.routing.enrichers.capability_designation import (
    embed_query,
)
from agent_utilities.knowledge_graph.retrieval.capability_index import (
    compute_eligibility,
)

logger = logging.getLogger(__name__)

__all__ = [
    "RoutingCandidate",
    "route_capability_request",
    "explain_routing_eligibility",
]

_DEFAULT_REWARD_WEIGHT = 0.15


@dataclass
class RoutingCandidate:
    """One routed candidate: id, blended score, and the full WHY-eligible explanation."""

    id: str
    score: float
    eligibility: dict[str, Any] = field(default_factory=dict)


def _resolve_hierarchy(hierarchy: Any | None) -> Any:
    if hierarchy is not None:
        return hierarchy
    from agent_utilities.knowledge_graph.ontology.capability_hierarchy import (
        get_default_hierarchy,
    )

    return get_default_hierarchy()


def _read_reward(engine: Any, entity_id: str) -> float:
    """Read the durable reward, using the neutral statistical prior if absent."""
    try:
        from agent_utilities.knowledge_graph.retrieval.durable_outcome_store import (
            read_capability_reward,
        )

        durable = read_capability_reward(engine, entity_id)
        if durable is not None:
            return durable
    except Exception as e:  # noqa: BLE001 — durable read is best-effort
        logger.debug("route_capability_request: durable reward read failed: %s", e)

    return 0.5


def _fetch_node_properties(engine: Any, entity_id: str) -> dict[str, Any]:
    graph = getattr(engine, "graph", None)
    getter = getattr(graph, "_get_node_properties", None) if graph is not None else None
    if not callable(getter):
        return {}
    try:
        props = getter(entity_id)
        return props if isinstance(props, dict) else {}
    except Exception as e:  # noqa: BLE001 — best-effort
        logger.debug(
            "route_capability_request: property read failed for %r: %s", entity_id, e
        )
        return {}


def explain_routing_eligibility(
    engine: Any,
    entity_id: str,
    *,
    required_capability_type: str,
    tenant: str | None = None,
    policy_tags: list[str] | None = None,
    capability_hierarchy: Any | None = None,
) -> dict[str, Any]:
    """WHY ``entity_id`` was (or would be) eligible — ontology subsumption + policy
    + tenant + the calibrated bandit reward, computed from engine state.
    """
    hierarchy = _resolve_hierarchy(capability_hierarchy)
    props = _fetch_node_properties(engine, entity_id)

    caps = props.get("capabilities") or props.get("providesCapability") or []
    if isinstance(caps, str):
        caps = [caps]
    entity_tenant = props.get("tenant")
    entity_policy_tags = props.get("policy_tags") or props.get("policyTags") or []
    ontology_type = props.get("type") or props.get("node_type")

    reward = _read_reward(engine, entity_id)
    return compute_eligibility(
        id=entity_id,
        capabilities=caps,
        required_caps=[required_capability_type],
        tenant=entity_tenant,
        required_tenant=tenant,
        policy_tags=entity_policy_tags,
        required_policy_tags=policy_tags,
        reward=reward,
        ontology_type=ontology_type,
        hierarchy=hierarchy,
    )


def route_capability_request(
    engine: Any,
    query: str,
    *,
    required_capability_type: str,
    k: int = 5,
    tenant: str | None = None,
    policy_tags: list[str] | None = None,
    embed_fn: Any = None,
    capability_hierarchy: Any | None = None,
    reward_weight: float = _DEFAULT_REWARD_WEIGHT,
) -> list[RoutingCandidate]:
    """Route ``query`` to the best-eligible tools/agents for ``required_capability_type``.

    Combines, in order:

    1. **Candidate selection** — the engine's native filtered ANN (AU-P1-3), made
       ontology-subsumption-aware (X-4): a tool declaring a narrower ontology
       subtype of ``required_capability_type`` is a candidate, not just an exact
       string match (see ``knowledge_graph/ontology/capability_hierarchy.py``).
    2. **Policy/tenant filters** — pushed down with the same call (AU-P1-3).
    3. **Durable-bandit re-ranking** — every surviving candidate's cosine score is
       re-blended with its calibrated success-rate reward EMA
       (:mod:`~.durable_outcome_store`), so a historically-better tool outranks a
       merely-closer-in-embedding-space one.
    4. **Explainability** — each returned candidate carries the FULL eligibility
       dict (subsumption path, policy/tenant match, reward) via
       :func:`explain_routing_eligibility`.

    ``capability_hierarchy`` defaults to the bundled ontology's singleton
    (:func:`~agent_utilities.knowledge_graph.ontology.capability_hierarchy.
    get_default_hierarchy`) — subsumption is ON by default here (the top-level
    X-4 entry point).
    """
    hierarchy = _resolve_hierarchy(capability_hierarchy)
    embedding = embed_query(query, embed_fn)
    if embedding is None:
        return []

    required = [required_capability_type]
    from agent_utilities.core.release_channel import active_channel
    from agent_utilities.knowledge_graph.retrieval.engine_capability_search import (
        engine_filtered_search,
    )

    # Oversample so the durable reward can reorder the engine-selected pool.
    oversample = max(int(k) * 3, k)
    engine_hits = engine_filtered_search(
        engine,
        embedding,
        k=oversample,
        required_caps=required,
        tenant=tenant,
        policy_tags=policy_tags,
        capability_hierarchy=hierarchy,
        active_release_channel=active_channel(),
    )
    raw_candidates = list(engine_hits or [])

    if not raw_candidates:
        return []

    # Durable-bandit re-rank: blend cosine with the calibrated reward EMA, exactly
    # the engine routing policy uses.
    blended: list[tuple[str, float, float]] = []
    for nid, cosine in raw_candidates:
        reward = _read_reward(engine, nid)
        score = cosine + reward_weight * (reward - 0.5)
        blended.append((nid, score, reward))
    blended.sort(key=lambda t: t[1], reverse=True)

    out: list[RoutingCandidate] = []
    for nid, score, _reward in blended[: int(k)]:
        eligibility = explain_routing_eligibility(
            engine,
            nid,
            required_capability_type=required_capability_type,
            tenant=tenant,
            policy_tags=policy_tags,
            capability_hierarchy=hierarchy,
        )
        out.append(
            RoutingCandidate(id=nid, score=float(score), eligibility=eligibility)
        )
    return out
