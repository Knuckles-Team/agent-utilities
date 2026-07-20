#!/usr/bin/python
from __future__ import annotations

"""Engine-native filtered ANN for capability retrieval (CONCEPT:AU-P1-3).

The engine-native capability index. Capability/tenant/policy-restricted candidate
selection is now performed by the engine's own vector index — the SAME
``query.unified`` cross-modal plan (``Scan``/``Filter``/``Rank``/``Limit``,
CONCEPT:AU-KG.compute.kg-2) and native ``semantic_search`` ANN primitive that
:mod:`agent_utilities.knowledge_graph.retrieval.hybrid_retriever` already uses for
general retrieval — instead of the in-process hnswlib/numpy scan in
:mod:`.capability_index`. This module is the designation authority.

Two engine-native paths, in preference order (mirrors
``docs/architecture/vector_index_lifecycle.md``'s retrieval stages):

1. **Unified filtered plan** — ``Scan(label) |> Filter(caps/tenant/policy) |>
   Rank(query) |> Limit`` in ONE costed round-trip. The engine composes the
   capability/tenant/policy restriction with the vector ``Rank`` leg itself — this
   is the "filtered ANN" the engine provides natively; there is no Python-side
   pre-filter-then-scan.
2. **Native ANN + bounded post-filter** — when unified planning is unavailable,
   fall to the unseeded ``semantic_search`` kNN, over-fetch, and restrict to the
   filter-matching ids over the BOUNDED returned candidate pool (never a
   full-graph scan) — the same degrade pattern
   ``hybrid_retriever._engine_vector_search`` uses for corpus/path restriction.

Both paths return ``None`` when no engine vector surface is reachable at all.
There is no designation fallback to an in-process capability index. If a unified
plan is rejected, the mandatory full artifact can still use its native ANN
surface; an engine without either vector surface is unavailable.

**X-4 — ontology-subsumption-aware push-down.** The ``capabilities`` restriction
is always hierarchy-aware: a required capability type is satisfied by a node
declaring it OR any ontology-narrower subtype. The engine's ``array_contains``
``Filter`` op is
exact-string only, so when (and only when) a required capability actually HAS
known subtypes, the capability ``Filter`` legs are left out of the pushed-down
plan (tenant/policy filters still push down) and the returned/oversampled rows are
post-filtered locally with :func:`_node_satisfies` against the subsumption-widened
set — the same bounded degrade pattern stage 2 already uses when unified planning
is unavailable, applied here to a semantic (not just capability-surface) gap. A required
capability with no known subtypes retains the exact filtered plan.
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
        filters.append({"property": "tenant", "op": "eq_or_null", "value": str(tenant)})
    for tag in policy_tags or ():
        filters.append(
            {"property": "policy_tags", "op": "array_contains", "value": str(tag)}
        )
    return filters


def _resolve_capability_hierarchy(hierarchy: Any | None) -> Any:
    """Return an injected hierarchy or the bundled current ontology hierarchy."""
    if hierarchy is not None:
        return hierarchy
    from agent_utilities.knowledge_graph.ontology.capability_hierarchy import (
        get_default_hierarchy,
    )

    return get_default_hierarchy()


def _capability_satisfied(caps_set: set[str], required: str, hierarchy: Any) -> bool:
    """True if ``required`` is declared literally, or (X-4) subsumed by a declared subtype."""
    if required in caps_set:
        return True
    return any(hierarchy.is_subtype_of(c, required) for c in caps_set)


def _node_satisfies(
    props: dict[str, Any],
    required_caps: list[str] | None,
    tenant: str | None,
    policy_tags: list[str] | None,
    *,
    capability_hierarchy: Any,
    active_release_channel: Any | None = None,
) -> bool:
    """Bounded, in-Python re-check of the SAME predicate the engine ``Filter`` encodes.

    Used to post-filter the small candidate pool the engine's unfiltered
    ``semantic_search`` already returned (tier 2), and — when ``capability_hierarchy``
    widens a required capability to a non-trivial subtype set (X-4) — also to
    post-filter the tier-1 unified-plan rows for the capability legs that plan
    could not push down. Never a full-graph scan either way.
    """
    if required_caps:
        caps = props.get("capabilities") or props.get("providesCapability") or []
        if isinstance(caps, str):
            caps = [caps]
        caps_set = {str(c) for c in caps}
        if not all(
            _capability_satisfied(caps_set, str(c), capability_hierarchy)
            for c in required_caps
        ):
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
    if active_release_channel is not None:
        from agent_utilities.core.release_channel import component_visible

        if not component_visible(props, active_release_channel):
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


def _local_capability_filter(
    graph: Any,
    candidates: list[tuple[str, float]],
    required_caps: list[str] | None,
    tenant: str | None,
    policy_tags: list[str] | None,
    capability_hierarchy: Any,
    active_release_channel: Any | None,
    k: int,
) -> list[tuple[str, float]]:
    """Batch-fetch properties for ``candidates`` and keep the first ``k`` that satisfy
    every filter (subsumption-aware — X-4). Shared by tier 1 (when a required
    capability has ontology subtypes the pushed-down plan couldn't express) and
    tier 2 (which never pushes any filter down at all)."""
    ids = [nid for nid, _ in candidates]
    props_by_id = _batch_properties(graph, ids)
    out: list[tuple[str, float]] = []
    for nid, score in candidates:
        props = props_by_id.get(nid, {})
        if _node_satisfies(
            props,
            required_caps,
            tenant,
            policy_tags,
            capability_hierarchy=capability_hierarchy,
            active_release_channel=active_release_channel,
        ):
            out.append((nid, score))
            if len(out) >= int(k):
                break
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
    capability_hierarchy: Any | None = None,
    active_release_channel: Any | None = None,
) -> list[tuple[str, float]] | None:
    """Return ``(id, score)`` candidates from the engine's native filtered ANN.

    ``None`` means no engine vector surface is reachable. An empty list is a
    real, authoritative answer: the engine ran the filtered plan and no entity
    qualified.

    ``capability_hierarchy`` (X-4) may inject an isolated ontology; omitting it
    resolves the bundled current hierarchy. See the module docstring for the
    push-down/post-filter split this triggers.
    """
    capability_hierarchy = _resolve_capability_hierarchy(capability_hierarchy)
    graph = getattr(engine, "graph", None)
    if graph is None:
        return None
    qvec = [float(x) for x in query_embedding]

    # X-4: a required capability with known ontology subtypes cannot be expressed
    # as a single exact `array_contains` Filter (that would miss a subtype-only
    # declaration), so it is excluded from push-down and re-checked locally.
    subsumed_caps: list[str] = []
    exact_caps = required_caps
    if required_caps:
        subsumed_caps = [
            c for c in required_caps if capability_hierarchy.descendants(c)
        ]
        if subsumed_caps:
            exact_caps = [c for c in required_caps if c not in subsumed_caps]

    # Release visibility is an ordered policy (stable < beta < edge), not an
    # exact equality predicate. Apply it to the bounded ANN candidate set while
    # the remaining exact predicates stay pushed into the native plan.
    needs_local_check = bool(subsumed_caps) or active_release_channel is not None
    filters = build_capability_filters(exact_caps, tenant, policy_tags)
    plan_k = max(int(k) * 4, 20) if needs_local_check else int(k)

    # Tier 1 — ONE unified cross-modal plan: Scan (optional) |> Filter* |> Rank |> Limit.
    query_unified = getattr(graph, "query_unified", None)
    if callable(query_unified):
        plan: list[dict[str, Any]] = []
        if label:
            plan.append({"Scan": {"label": label}})
        for f in filters:
            plan.append({"Filter": f})
        plan.append({"Rank": {"query": qvec}})
        plan.append({"Limit": {"k": plan_k}})
        try:
            rows = query_unified(plan) or []
            candidates = [
                (str(r["id"]), float(r.get("score") or 0.0))
                for r in rows
                if isinstance(r, dict) and r.get("id") is not None
            ]
            if not needs_local_check:
                return candidates
            return _local_capability_filter(
                graph,
                candidates,
                required_caps,
                tenant,
                policy_tags,
                capability_hierarchy,
                active_release_channel,
                k,
            )
        except Exception as e:  # noqa: BLE001 — unavailable unified planning -> stage 2
            logger.debug(
                "engine_capability_search: unified filtered plan unavailable, "
                "falling to native ANN + bounded post-filter: %s",
                e,
            )

    # Stage 2 — native unfiltered ANN, then a BOUNDED post-filter over the returned pool.
    semantic_search = getattr(graph, "semantic_search", None)
    if not callable(semantic_search):
        return None
    has_any_filter = bool(filters) or needs_local_check
    fetch_k = int(k) if not has_any_filter else max(int(k) * 4, 20)
    try:
        raw = semantic_search(qvec, fetch_k) or []
    except Exception as e:  # noqa: BLE001 — no engine ANN reachable at all
        logger.debug("engine_capability_search: native ANN unavailable: %s", e)
        return None

    candidates = [(str(nid), float(score)) for nid, score in raw if nid]
    if not has_any_filter:
        return candidates[: int(k)]

    return _local_capability_filter(
        graph,
        candidates,
        required_caps,
        tenant,
        policy_tags,
        capability_hierarchy,
        active_release_channel,
        k,
    )
