"""Feature discovery via call-graph community detection (CONCEPT:KG-2.8 Phase 2).

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

from .models import CodeEntity, EnrichmentEdge, Feature

logger = logging.getLogger(__name__)

# Max ops per bulk_mutate call — bounds the per-request MsgPack payload while
# still amortising the socket round-trip over thousands of writes. (CONCEPT:KG-2.16)
_COMMUNITY_BULK_CHUNK = 10_000

# (node_ids, edges) -> list of communities (each a list of node ids)
CommunityFn = Callable[[list[str], list[tuple[str, str]]], list[list[str]]]


def resolve_call_edges(code: list[CodeEntity]) -> list[EnrichmentEdge]:
    """Resolve code→code CALLS edges by matching callee names to symbols."""
    by_name: dict[str, list[str]] = {}
    for c in code:
        by_name.setdefault(c.name, []).append(c.id)
    edges: list[EnrichmentEdge] = []
    seen: set[tuple[str, str]] = set()
    for c in code:
        for callee in set(c.calls):
            for tgt in by_name.get(callee, []):
                if tgt == c.id:
                    continue
                key = (c.id, tgt)
                if key not in seen:
                    seen.add(key)
                    edges.append(
                        EnrichmentEdge(source=c.id, target=tgt, rel_type="CALLS")
                    )
    return edges


def cluster_features(
    code: list[CodeEntity],
    community_fn: CommunityFn,
    min_size: int = 3,
) -> list[Feature]:
    """Cluster code symbols into features via injected community detection."""
    ids = [c.id for c in code]
    edges = [(e.source, e.target) for e in resolve_call_edges(code)]
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


def make_community_fn(graph_compute: Any, resolution: float = 1.0) -> CommunityFn:
    """Engine-backed community detection over an isolated scratch tenant.

    Loads the call graph into the provided GraphComputeEngine (caller should pass
    a dedicated/ephemeral tenant) and runs the Rust community detection.
    """

    def _fn(node_ids: list[str], edges: list[tuple[str, str]]) -> list[list[str]]:
        # Load the call graph into the scratch tenant in ONE bulk pass instead of
        # a per-element add_node/add_edge round-trip each (a big repo is tens of
        # thousands of symbols → tens of thousands of socket round-trips). The
        # engine's ``batch_update`` now MsgPack-encodes properties (epistemic-graph
        # 9e3620c), so the batched load is read-compatible; this was reverted to
        # per-element only while that op stored unreadable bytes. Nodes are loaded
        # before edges so every edge endpoint exists. Falls back to per-element if
        # the engine has no bulk op or a batch fails. (CONCEPT:KG-2.16)
        bulk = getattr(graph_compute, "bulk_mutate", None) or getattr(
            graph_compute, "batch_update", None
        )
        loaded = False
        if bulk is not None:
            node_ops = [
                {"op": "add_node", "id": nid, "properties": {"type": "Code"}}
                for nid in node_ids
            ]
            edge_ops = [
                {
                    "op": "add_edge",
                    "source": src,
                    "target": tgt,
                    "properties": {"type": "CALLS"},
                }
                for src, tgt in edges
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
                graph_compute.add_node(nid, {"type": "Code"})
            for src, tgt in edges:
                graph_compute.add_edge(src, tgt, {"type": "CALLS"})
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
