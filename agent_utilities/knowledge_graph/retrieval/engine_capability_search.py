#!/usr/bin/python
from __future__ import annotations

"""Engine-native filtered ANN for capability retrieval (CONCEPT:AU-P1-3).

The engine-native capability index. Capability/tenant/policy-restricted candidate
selection is now performed by the engine's own vector index — the SAME
``query.unified`` cross-modal plan (``Scan``/``Filter``/``Rank``/``Limit``,
CONCEPT:AU-KG.compute.kg-2) and native ``semantic_search`` ANN primitive that
:mod:`agent_utilities.knowledge_graph.retrieval.hybrid_retriever` already uses for
general retrieval — instead of the in-process hnswlib/numpy scan in
:mod:`.capability_index`. That module is now a *bounded, non-authoritative cache*;
this module is the authority.

Two engine-native paths, in preference order (mirrors
``docs/architecture/vector_index_lifecycle.md``'s retrieval tiers):

1. **Unified filtered plan** — ``Scan(label) |> Filter(caps/tenant/policy) |>
   Rank(query) |> Limit`` in ONE costed round-trip. The engine composes the
   capability/tenant/policy restriction with the vector ``Rank`` leg itself — this
   is the "filtered ANN" the engine provides natively; there is no Python-side
   pre-filter-then-scan.
2. **Native ANN + bounded post-filter** — when the connected engine has no
   ``query`` feature (a lean build), fall to the unseeded ``semantic_search`` kNN,
   over-fetch, and restrict to the filter-matching ids over the BOUNDED returned
   candidate pool (never a full-graph scan) — the same degrade pattern
   ``hybrid_retriever._engine_vector_search`` uses for corpus/path restriction.

Both paths return ``None`` when no engine vector surface is reachable at all,
signalling the caller to fall back to the bounded in-process
:class:`~.capability_index.CapabilityIndex` cache (dev / no engine configured).
Never raises — a plan the engine build doesn't understand (e.g. an older engine
without the ``Filter`` op) degrades to the next tier, exactly like every other
engine-surface consumer in this codebase.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["engine_filtered_search", "build_capability_filters"]


def build_capability_filters(
    required_caps: list[str] | None = None,
    tenant: str | None = None,
    policy_tags: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Render capability/tenant/policy restrictions into ``Filter`` op dicts.

    One ``Filter`` per condition so the engine can cost/reorder them independently
    (``reorder_filter_selectivity`` on :meth:`GraphComputeEngine.query_unified`).
    ``capabilities`` is filtered with an ``array_contains``-shaped op per required
    capability (a candidate must satisfy every one — AND semantics, matching
    :class:`.capability_index.CapabilityIndex`'s set-intersection filter); tenant is
    an equality-or-absent op; policy tags are ``array_contains`` per tag.
    """
    filters: list[dict[str, Any]] = []
    for cap in required_caps or ():
        filters.append(
            {"property": "capabilities", "op": "array_contains", "value": str(cap)}
        )
    if tenant is not None:
        filters.append(
            {"property": "tenant", "op": "eq_or_null", "value": str(tenant)}
        )
    for tag in policy_tags or ():
        filters.append(
            {"property": "policy_tags", "op": "array_contains", "value": str(tag)}
        )
    return filters


def _node_satisfies(
    props: dict[str, Any],
    required_caps: list[str] | None,
    tenant: str | None,
    policy_tags: list[str] | None,
) -> bool:
    """Bounded, in-Python re-check of the SAME predicate the engine ``Filter`` encodes.

    Used only to post-filter the small candidate pool the engine's unfiltered
    ``semantic_search`` already returned (tier 2) — never a full-graph scan.
    """
    if required_caps:
        caps = props.get("capabilities") or props.get("providesCapability") or []
        if isinstance(caps, str):
            caps = [caps]
        caps_set = {str(c) for c in caps}
        if not all(str(c) in caps_set for c in required_caps):
            return False
    if tenant is not None:
        node_tenant = props.get("tenant")
        if node_tenant not in (None, "", tenant):
            return False
    if policy_tags:
        tags = props.get("policy_tags") or props.get("policyTags") or []
        if isinstance(tags, str):
            tags = [tags]
        tags_set = {str(t) for t in tags}
        if not all(str(t) in tags_set for t in policy_tags):
            return False
    return True


def _batch_properties(graph: Any, ids: list[str]) -> dict[str, dict[str, Any]]:
    """Fetch node properties for ``ids`` in ONE round-trip when possible.

    Mirrors ``hybrid_retriever._batch_node_properties`` — batched via
    ``nodes.properties_batch`` when the client exposes it, else per-id via the
    resident projection. Best-effort: an unreadable id is simply absent.
    """
    out: dict[str, dict[str, Any]] = {}
    client = getattr(graph, "_client", None)
    nodes_ns = getattr(client, "nodes", None) if client is not None else None
    batch = getattr(nodes_ns, "properties_batch", None)
    if callable(batch):
        try:
            for nid, blob in (batch(ids) or {}).items():
                if isinstance(blob, dict):
                    out[str(nid)] = blob
            return out
        except Exception as e:  # noqa: BLE001 — degrade to per-id projection
            logger.debug("engine_capability_search: properties_batch failed: %s", e)
    getter = getattr(graph, "_get_node_properties", None)
    if callable(getter):
        for nid in ids:
            try:
                p = getter(nid)
                if isinstance(p, dict):
                    out[nid] = p
            except Exception:  # noqa: BLE001,S112
                continue
    return out


def engine_filtered_search(
    engine: Any,
    query_embedding: list[float],
    *,
    k: int = 5,
    required_caps: list[str] | None = None,
    tenant: str | None = None,
    policy_tags: list[str] | None = None,
    label: str = "",
) -> list[tuple[str, float]] | None:
    """Return ``(id, score)`` candidates from the engine's native filtered ANN.

    ``None`` means "no engine vector surface reachable at all" — the caller should
    fall back to the bounded in-process :class:`~.capability_index.CapabilityIndex`
    cache. An empty list is a real, authoritative answer: the engine ran the
    filtered plan and no entity qualified.
    """
    graph = getattr(engine, "graph", None)
    if graph is None:
        return None
    qvec = [float(x) for x in query_embedding]
    filters = build_capability_filters(required_caps, tenant, policy_tags)

    # Tier 1 — ONE unified cross-modal plan: Scan (optional) |> Filter* |> Rank |> Limit.
    query_unified = getattr(graph, "query_unified", None)
    if callable(query_unified):
        plan: list[dict[str, Any]] = []
        if label:
            plan.append({"Scan": {"label": label}})
        for f in filters:
            plan.append({"Filter": f})
        plan.append({"Rank": {"query": qvec}})
        plan.append({"Limit": {"k": int(k)}})
        try:
            rows = query_unified(plan) or []
            return [
                (str(r["id"]), float(r.get("score") or 0.0))
                for r in rows
                if isinstance(r, dict) and r.get("id") is not None
            ]
        except Exception as e:  # noqa: BLE001 — engine build without `query`/`Filter` -> tier 2
            logger.debug(
                "engine_capability_search: unified filtered plan unavailable, "
                "falling to native ANN + bounded post-filter: %s",
                e,
            )

    # Tier 2 — native unfiltered ANN, then a BOUNDED post-filter over the returned pool.
    semantic_search = getattr(graph, "semantic_search", None)
    if not callable(semantic_search):
        return None
    fetch_k = int(k) if not filters else max(int(k) * 4, 20)
    try:
        raw = semantic_search(qvec, fetch_k) or []
    except Exception as e:  # noqa: BLE001 — no engine ANN reachable at all
        logger.debug("engine_capability_search: native ANN unavailable: %s", e)
        return None

    candidates = [(str(nid), float(score)) for nid, score in raw if nid]
    if not filters:
        return candidates[: int(k)]

    ids = [nid for nid, _ in candidates]
    props_by_id = _batch_properties(graph, ids)
    out: list[tuple[str, float]] = []
    for nid, score in candidates:
        props = props_by_id.get(nid, {})
        if _node_satisfies(props, required_caps, tenant, policy_tags):
            out.append((nid, score))
            if len(out) >= int(k):
                break
    return out
