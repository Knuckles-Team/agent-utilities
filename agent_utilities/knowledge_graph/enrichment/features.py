"""Feature discovery via call-graph community detection (CONCEPT:EG-KG.storage.nonblocking-checkpoint Phase 2).

A *feature* is a cohesive cluster of code symbols that implement a capability
together. We build the resolved call graph and run the **epistemic-graph engine's
community detection** (the compute layer) to find these clusters — then optionally
name/summarise each via the LLM. Answers "how does feature X work" and "what are
the major features of this codebase".
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from .models import CodeEntity, EdgeRung, EnrichmentEdge, Feature

logger = logging.getLogger(__name__)

# Max ops per bulk_mutate call — bounds the per-request MsgPack payload while
# still amortising the socket round-trip over thousands of writes. (CONCEPT:EG-KG.compute.graph-compute-engine)
_COMMUNITY_BULK_CHUNK = 10_000

# (node_ids, edges) -> list of communities (each a list of node ids). Each edge is
# (source, target, confidence) -- confidence is the resolver's own per-edge score
# (CONCEPT:EG-KG.compute.type-scope-resolved-call tiers: 0.95 scoped / 0.90 same_file
# / 0.70 arity / 0.60 unique) when the primary index_repository path produced it,
# or None when it is genuinely unknown (the Python-side name-only fallback resolver
# computes no confidence at all, and never fabricates one -- EH-284/EH-274).
CommunityFn = Callable[
    [list[str], list[tuple[str, str, float | None]]], list[list[str]]
]


# A callee name resolving to MORE than this many symbols is ambiguous — a common
# method name like Java `toString`/`equals`/`hashCode`/`get*` — and carries no
# call-graph signal: name-only resolution edges it to EVERY same-named symbol, an
# N×M blow-up (egeria: `toString` is on 1,864 symbols → 6.4M spurious edges in 72s,
# and a community pass over them is catastrophic). Capping the fan-out drops that
# noise (egeria → 162k real edges in 2.1s) while keeping the precise calls.
# (CONCEPT:EG-KG.storage.nonblocking-checkpoint)
_MAX_CALL_FANOUT = 10


def resolve_call_edges(code: list[CodeEntity]) -> list[EnrichmentEdge]:
    """Resolve code→code CALLS edges by matching callee names to symbols.

    Calls whose name is ambiguous (>``_MAX_CALL_FANOUT`` candidate targets) are
    skipped — common names that explode the edge set without adding signal.
    """
    by_name: dict[str, list[str]] = {}
    for c in code:
        by_name.setdefault(c.name, []).append(c.id)
    edges: list[EnrichmentEdge] = []
    seen: set[tuple[str, str]] = set()
    for c in code:
        for callee in set(c.calls):
            targets = by_name.get(callee, [])
            if len(targets) > _MAX_CALL_FANOUT:
                continue  # ambiguous common name → no signal, skip the fan-out
            for tgt in targets:
                if tgt == c.id:
                    continue
                key = (c.id, tgt)
                if key not in seen:
                    seen.add(key)
                    edges.append(
                        EnrichmentEdge(
                            source=c.id,
                            target=tgt,
                            rel_type="CALLS",
                            # Pure name-only matching, no type/scope info at
                            # all -- still symbol resolution -> INFERRED. No
                            # confidence: this fallback computes none, and
                            # never fabricates one (EH-284/EH-274).
                            rung=EdgeRung.INFERRED,
                        )
                    )
    return edges


def cluster_features(
    code: list[CodeEntity],
    community_fn: CommunityFn,
    min_size: int = 3,
    *,
    call_edges: list[EnrichmentEdge] | None = None,
) -> list[Feature]:
    """Cluster code symbols into features via injected community detection.

    ``call_edges`` lets the caller pass already-resolved CALLS edges so the
    fan-out resolution isn't recomputed: the ingest pipeline needs the same edge
    set to WRITE the CALLS relationships, and resolving twice over a big repo is
    pure waste (~5s on egeria). Defaults to resolving here when omitted.

    Each edge's resolver ``confidence`` (EH-284) rides through to
    ``community_fn`` as the tuple's third element — ``e.confidence`` (EH-274:
    a first-class :class:`~.models.EnrichmentEdge` field, promoted off the
    pre-EH-274 ``props["confidence"]`` convention) when the primary
    ``index_repository`` resolver produced it, ``None`` when
    it did not (the Python-side name-only fallback in :func:`resolve_call_edges`
    computes no confidence at all; never fabricated here).
    """
    ids = [c.id for c in code]
    resolved = call_edges if call_edges is not None else resolve_call_edges(code)
    # `confidence` is EnrichmentEdge's own modelled field (EH-274) now, not
    # `props["confidence"]` -- promoted from the pre-EH-274 convention.
    edges = [(e.source, e.target, e.confidence) for e in resolved]
    if not ids:
        return []
    communities = community_fn(ids, edges)

    by_id = {c.id: c for c in code}
    features: list[Feature] = []
    for i, members in enumerate(communities):
        members = [m for m in members if m in by_id]
        if len(members) < min_size:
            continue
        patterns: list[str] = []
        for m in members:
            patterns.extend(by_id[m].patterns)
        # Provisional name from the most-connected / first member; LLM refines later.
        seed = by_id[members[0]].name
        features.append(
            Feature(
                id=f"feature:{i}:{seed}",
                name=f"{seed} cluster",
                member_ids=members,
                size=len(members),
                patterns=sorted(set(patterns)),
            )
        )
    return features


def _strip_confidence(
    edges: list[tuple[str, str, float | None]],
) -> list[tuple[str, str]]:
    """Plain (source, target) pairs for the ``CommunityDetectEphemeral`` wire
    method, which has no properties/weight slot at all (EH-284/EH-274) --
    extracted so ``make_community_fn``'s closure stays at its pre-EH-284
    complexity (the comprehension itself, not a branch, was the delta)."""
    return [(src, tgt) for src, tgt, _confidence in edges]


def _call_edge_properties(confidence: float | None) -> dict[str, Any]:
    """Properties for one CALLS edge loaded into the community-detection scratch
    tenant. ``confidence`` (EH-284) is attached under the repo-wide edge-quality
    convention key only when the resolver actually supplied one -- an absent key
    is exactly what the engine's own ``resolver_confidence_weight`` already
    treats as "no recorded confidence" (falls back to its documented uniform
    weight), so this never fabricates a value for edges that genuinely have none
    (the Python-side name-only fallback resolver)."""
    props: dict[str, Any] = {"relationship": "CALLS"}
    if confidence is not None:
        props["confidence"] = confidence
    return props


def make_community_fn(graph_compute: Any, resolution: float = 1.0) -> CommunityFn:
    """Engine-backed community detection over an isolated scratch tenant.

    Loads the call graph into the provided GraphComputeEngine (caller should pass
    a dedicated/ephemeral tenant) and runs the Rust community detection.
    """

    def _fn(
        node_ids: list[str], edges: list[tuple[str, str, float | None]]
    ) -> list[list[str]]:
        # Stateless path (preferred): hand the call graph to the engine INLINE so it
        # runs detection on an in-memory throwaway graph — NO bulk-load into a tenant,
        # NO per-job comm-tenant sprawl, NO comm checkpoint. This removes the dominant
        # cost of the community stage (the ~160k-edge bulk load) and the tenant churn
        # the GC/dedicated-engine work was compensating for. Falls back to the
        # tenant-load path below on any error or against an older engine. (KG-2.58)
        #
        # EH-284/EH-274: ``CommunityDetectEphemeral``'s wire method is
        # ``edges: Vec<(String, String)>`` (epistemic-graph
        # ``eg-types/src/protocol/method/method_02.rs``) with NO properties/weight
        # slot at all — the handler builds every ephemeral edge with
        # ``Vec::new()`` properties (``server/handlers/graph_ops/algorithms.rs``).
        # So confidence CANNOT reach this call without an engine-side wire-protocol
        # change; it is dropped here, not lost by an AU oversight. This is the path
        # actually taken whenever the engine advertises it (i.e. almost always in
        # production), so EH-284's confidence weighting is INERT on this branch
        # until that protocol gap is closed on the epistemic-graph side.
        ephemeral = getattr(graph_compute, "community_detect_ephemeral", None)
        if ephemeral is not None:
            try:
                return ephemeral(node_ids, _strip_confidence(edges), resolution)
            except Exception as e:  # noqa: BLE001 — degrade to the tenant-load path
                logger.debug(
                    "ephemeral community detect failed (%s); tenant-load fallback", e
                )

        # Load the call graph into the scratch tenant in ONE bulk pass instead of
        # a per-element add_node/add_edge round-trip each (a big repo is tens of
        # thousands of symbols → tens of thousands of socket round-trips). The
        # engine's ``batch_update`` now MsgPack-encodes properties (epistemic-graph
        # 9e3620c), so the batched load is read-compatible; this was reverted to
        # per-element only while that op stored unreadable bytes. Nodes are loaded
        # before edges so every edge endpoint exists. Falls back to per-element if
        # the engine has no bulk op or a batch fails. (CONCEPT:EG-KG.compute.graph-compute-engine)
        #
        # This path DOES reach the engine's persisted-graph ``edge_properties``
        # (``CommunityDetection { resolution }``, which EH-284's
        # ``resolver_confidence_weight`` reads) so, unlike the ephemeral path
        # above, ``confidence`` is attached here when the caller supplied one.
        bulk = getattr(graph_compute, "bulk_mutate", None) or getattr(
            graph_compute, "batch_update", None
        )
        loaded = False
        if bulk is not None:
            node_ops = [
                {
                    "op": "add_node",
                    "id": nid,
                    "properties": {"node_type": "Code"},
                }
                for nid in node_ids
            ]
            edge_ops = [
                {
                    "op": "add_edge",
                    "source": src,
                    "target": tgt,
                    "properties": _call_edge_properties(confidence),
                }
                for src, tgt, confidence in edges
            ]
            try:
                for ops in (node_ops, edge_ops):  # all nodes, THEN all edges
                    for i in range(0, len(ops), _COMMUNITY_BULK_CHUNK):
                        bulk(ops[i : i + _COMMUNITY_BULK_CHUNK])
                loaded = True
            except Exception as e:  # noqa: BLE001 - degrade to per-element load
                logger.debug("community batch load failed (%s); per-element", e)
        if not loaded:
            for nid in node_ids:
                graph_compute.add_node(nid, {"node_type": "Code"})
            for src, tgt, confidence in edges:
                graph_compute.add_edge(src, tgt, _call_edge_properties(confidence))
        try:
            return graph_compute.community_detection(resolution)
        except Exception:
            # Fall back to weakly-connected components if community detection
            # is unavailable for this build.
            try:
                return graph_compute.connected_components()
            except Exception:
                return []

    return _fn
