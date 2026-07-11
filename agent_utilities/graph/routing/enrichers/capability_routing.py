#!/usr/bin/python
from __future__ import annotations

"""Ontology-driven tool/agent routing — X-4 (CONCEPT:AU-P1-3).

The single X-4 entry point: combine the engine's filtered ANN (AU-P1-3), ontology
SUBSUMPTION (:mod:`agent_utilities.knowledge_graph.ontology.capability_hierarchy`),
and tenant/policy filters into ONE candidate-selection call, re-ranked by the
durable contextual bandit (:mod:`~.durable_outcome_store` /
:class:`~agent_utilities.knowledge_graph.retrieval.capability_index.CapabilityIndex`),
with a WHY-eligible explanation attached to every candidate.

:func:`route_capability_request` is deliberately path-agnostic about WHERE the
candidate pool came from (the engine's own ANN, or the bounded in-process
fallback cache) — either way the final ranking blends in the SAME learned reward
EMA, so "the bandit prefers a historically-better tool" holds regardless of which
tier answered. :func:`explain_routing_eligibility` mirrors that: it reads the
candidate's properties straight from the engine when the in-process cache never
saw it (the common case once the engine's native filtered ANN is authoritative),
falling back to the cache only when the engine is unreachable.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from agent_utilities.graph.routing.enrichers.capability_designation import (
    embed_query,
    get_designation_index,
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
    """Best-effort reward lookup: durable engine property, else the in-process cache."""
    try:
        from agent_utilities.knowledge_graph.retrieval.durable_outcome_store import (
            read_capability_reward,
        )

        durable = read_capability_reward(engine, entity_id)
        if durable is not None:
            return durable
    except Exception as e:  # noqa: BLE001 — durable read is best-effort
        logger.debug("route_capability_request: durable reward read failed: %s", e)

    watcher = getattr(engine, "_capability_index_watcher", None)
    if watcher is not None:
        try:
            return watcher.index.reward_of(entity_id)
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "route_capability_request: in-process reward read failed: %s", e
            )
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
    + tenant + the calibrated bandit reward, computed ENGINE-NATIVE-FIRST.

    Unlike :func:`~.capability_designation.explain_capability_eligibility` (which
    only answers for an id resident in the bounded in-process cache), this reads
    the candidate's declared capabilities/tenant/policy tags directly from the
    engine's own node properties — so it answers for ANY candidate the engine's
    filtered ANN surfaced, not just ones the (possibly cold or lagging) in-process
    cache happens to have cached. Falls back to the in-process cache's properties
    only when the engine has no queryable node-properties surface at all (e.g. a
    pure in-process fallback engine in tests). Never raises; an entity unknown to
    both surfaces still gets a (fully ineligible) eligibility dict rather than
    ``None``, since a routing caller always needs a features dict to log/act on.
    """
    hierarchy = _resolve_hierarchy(capability_hierarchy)
    props = _fetch_node_properties(engine, entity_id)

    if props:
        caps = props.get("capabilities") or props.get("providesCapability") or []
        if isinstance(caps, str):
            caps = [caps]
        entity_tenant = props.get("tenant")
        entity_policy_tags = props.get("policy_tags") or props.get("policyTags") or []
        ontology_type = props.get("type") or props.get("node_type")
    else:
        # No engine node-properties surface reachable — fall back to whatever the
        # bounded in-process cache knows (may itself be empty/absent for this id).
        index = get_designation_index(engine, capability_hierarchy=hierarchy)
        caps = list(index.capabilities_of(entity_id)) if index is not None else []
        entity_tenant = None
        entity_policy_tags = []
        ontology_type = None

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

    1. **Candidate selection** — the engine's native filtered ANN (AU-P1-3) when
       reachable, else the bounded in-process fallback cache — both made
       ontology-subsumption-aware (X-4): a tool declaring a narrower ontology
       subtype of ``required_capability_type`` is a candidate, not just an exact
       string match (see ``knowledge_graph/ontology/capability_hierarchy.py``).
    2. **Policy/tenant filters** — pushed down with the same call (AU-P1-3).
    3. **Durable-bandit re-ranking** — every surviving candidate's cosine score is
       re-blended with its calibrated success-rate reward EMA
       (:mod:`~.durable_outcome_store`), so a historically-better tool outranks a
       merely-closer-in-embedding-space one, REGARDLESS of which tier (engine or
       fallback) supplied the candidate pool.
    4. **Explainability** — each returned candidate carries the FULL eligibility
       dict (subsumption path, policy/tenant match, reward) via
       :func:`explain_routing_eligibility`.

    ``capability_hierarchy`` defaults to the bundled ontology's singleton
    (:func:`~agent_utilities.knowledge_graph.ontology.capability_hierarchy.
    get_default_hierarchy`) — subsumption is ON by default here (the top-level
    X-4 entry point), unlike the lower-level primitives it composes, which stay
    opt-in for backward compatibility.
    """
    hierarchy = _resolve_hierarchy(capability_hierarchy)
    embedding = embed_query(query, embed_fn)
    if embedding is None:
        return []

    required = [required_capability_type]
    raw_candidates: list[tuple[str, float]] = []
    try:
        from agent_utilities.knowledge_graph.retrieval.engine_capability_search import (
            engine_filtered_search,
        )

        # Oversample so the bandit re-rank below has real headroom to reorder
        # within, not just re-score an already-Limit-truncated top-k.
        oversample = max(int(k) * 3, k)
        engine_hits = engine_filtered_search(
            engine,
            embedding,
            k=oversample,
            required_caps=required,
            tenant=tenant,
            policy_tags=policy_tags,
            capability_hierarchy=hierarchy,
        )
        if engine_hits is not None:
            raw_candidates = list(engine_hits)
    except Exception as e:  # noqa: BLE001 — engine path is best-effort, fallback below
        logger.debug("route_capability_request: engine path failed: %s", e)

    if not raw_candidates:
        index = get_designation_index(engine, capability_hierarchy=hierarchy)
        if index is not None and len(index):
            designations = index.designate(
                embedding,
                required_caps=required,
                k=max(int(k) * 3, k),
                tenant=tenant,
                required_policy_tags=policy_tags,
                reward_weight=0.0,  # re-blended uniformly below instead
            )
            raw_candidates = [(d.id, d.score) for d in designations]

    if not raw_candidates:
        return []

    # Durable-bandit re-rank: blend cosine with the calibrated reward EMA, exactly
    # the formula CapabilityIndex.designate() uses, applied uniformly regardless
    # of which tier supplied the candidate.
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
