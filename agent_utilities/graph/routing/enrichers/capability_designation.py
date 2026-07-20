"""Engine-native KG specialist designation and durable outcome learning.

Candidate selection is authoritative in epistemic-graph: capability, tenant,
policy, ontology, and vector filters execute in one native search. Python does
not maintain a duplicate capability index.
"""

from __future__ import annotations

import logging
from typing import Any

from agent_utilities.knowledge_graph.retrieval.capability_index import (
    compute_eligibility,
)

logger = logging.getLogger(__name__)


def embed_query(query: str, embed_fn: Any = None) -> Any | None:
    """Resolve ``query`` to an embedding vector, or ``None`` when unavailable."""
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
    capability_hierarchy: Any | None = None,
) -> list[str] | None:
    """Return ids selected by the engine's filtered ANN authority."""
    embedding = embed_query(query, embed_fn)
    if embedding is None:
        return None

    from agent_utilities.core.release_channel import active_channel
    from agent_utilities.knowledge_graph.retrieval.engine_capability_search import (
        engine_filtered_search,
    )

    hits = engine_filtered_search(
        engine,
        embedding,
        k=k,
        required_caps=required_caps,
        tenant=tenant,
        policy_tags=policy_tags,
        capability_hierarchy=capability_hierarchy,
        active_release_channel=active_channel(),
    )
    if hits is None:
        return None
    return [node_id for node_id, _score in hits]


def record_capability_outcome(
    engine: Any,
    entity_id: str,
    *,
    success: bool | None = None,
    reward: float | None = None,
    alpha: float = 0.3,
    source_ids: list[str] | None = None,
) -> float:
    """Persist a contextual-bandit outcome through the graph authority."""
    from agent_utilities.knowledge_graph.retrieval.durable_outcome_store import (
        persist_capability_reward,
    )

    updated = persist_capability_reward(
        engine,
        entity_id,
        success=success,
        reward=reward,
        alpha=alpha,
    )
    if updated is None:
        raise RuntimeError("capability outcome persistence is unavailable")
    if source_ids:
        _register_capability_reward_materialization(engine, entity_id, source_ids)
    return float(updated)


def _register_capability_reward_materialization(
    engine: Any, entity_id: str, source_ids: list[str]
) -> None:
    """Register the durable reward's source provenance and materialization."""
    for source_id in source_ids:
        engine.add_edge(entity_id, source_id, relationship_type="DERIVED_FROM")
    engine.register_materialization(entity_id)


def _node_properties(engine: Any, entity_id: str) -> dict[str, Any] | None:
    graph = getattr(engine, "graph", None)
    getter = getattr(graph, "_get_node_properties", None)
    if not callable(getter):
        return None
    props = getter(entity_id)
    return props if isinstance(props, dict) and props else None


def explain_capability_eligibility(
    engine: Any,
    entity_id: str,
    *,
    required_caps: list[str] | None = None,
    tenant: str | None = None,
    policy_tags: list[str] | None = None,
    capability_hierarchy: Any | None = None,
) -> dict[str, Any] | None:
    """Explain eligibility directly from authoritative node properties."""
    props = _node_properties(engine, entity_id)
    if props is None:
        return None
    capabilities = props.get("capabilities") or props.get("providesCapability") or []
    if isinstance(capabilities, str):
        capabilities = [capabilities]

    from agent_utilities.knowledge_graph.retrieval.durable_outcome_store import (
        read_capability_reward,
    )

    reward = read_capability_reward(engine, entity_id)
    return compute_eligibility(
        id=entity_id,
        capabilities=capabilities,
        required_caps=required_caps,
        tenant=props.get("tenant"),
        required_tenant=tenant,
        policy_tags=props.get("policy_tags") or props.get("policyTags") or [],
        required_policy_tags=policy_tags,
        reward=0.5 if reward is None else reward,
        ontology_type=props.get("type") or props.get("node_type"),
        hierarchy=capability_hierarchy,
    )


__all__ = [
    "designate_specialists",
    "embed_query",
    "explain_capability_eligibility",
    "record_capability_outcome",
]
