#!/usr/bin/python
from __future__ import annotations

"""Synergy bundles + leverage ranking (CONCEPT:KG-2.7 / KG-2.5).

Two graph operations that turn a deduped, gap-analysed feature graph into a
prioritised work-list:

* :func:`synergy_bundles` — community-detect the feature graph (preferring the
  engine's Louvain/`community_detection`, local connected-components fallback) and
  flag any community that spans **≥2 pillars** (ORCH/KG/AHE/ECO/OS) as a synergy
  bundle, linking its members with `HAS_SYNERGY_WITH`. Cross-pillar clusters are
  where the novel combinations live — features that are individually known but
  *together* are new.
* :func:`rank_features` — score the **open** gaps by leverage
  ``source_count × (1 + centrality)`` (centrality from the engine's PageRank, local
  degree fallback) so the golden loop spends its budget on the highest-impact gaps
  first.

Edge reads are backend-portable via the `_rel` property marker (see gap_analysis);
duplicate (`SUPERSEDES`) edges are excluded from the synergy graph.

Concept: synergy-ranking
"""

from dataclasses import dataclass, field
from typing import Any

from ...models.knowledge_graph import RegistryEdgeType
from .dedup import iter_all_edges
from .gap_analysis import _FEATURE_TYPES, open_features

_EXCLUDED_RELS = {"SUPERSEDES", "SATISFIED_BY"}


@dataclass
class SynergyBundle:
    members: list[str]
    pillars: list[str]


@dataclass
class SynergyReport:
    communities: int = 0
    bundles: list[SynergyBundle] = field(default_factory=list)
    edges_written: int = 0


@dataclass
class RankedFeature:
    feature_id: str
    score: float
    source_count: int
    centrality: float


def _pillar_of(data: dict[str, Any]) -> str:
    """Derive a pillar tag (ORCH/KG/AHE/ECO/OS) from a node's concept ids."""
    if data.get("pillar"):
        return str(data["pillar"])
    for cid in data.get("concept_ids", []) or []:
        head = str(cid).split("-", 1)[0].upper()
        if head:
            return head
    return ""


def _feature_nodes(engine: Any, feature_types: tuple[str, ...]) -> dict[str, dict]:
    graph = getattr(engine, "graph", None)
    if graph is None:
        return {}
    try:
        node_iter = graph.nodes(data=True)
    except TypeError:  # pragma: no cover
        return {}
    wanted = {t.lower() for t in feature_types}  # case-insensitive (live labels)
    return {
        nid: data
        for nid, data in node_iter
        if isinstance(data, dict) and str(data.get("type", "")).lower() in wanted
    }


def _adjacency(engine: Any, ids: set[str]) -> dict[str, set[str]]:
    """Undirected feature-feature adjacency from non-duplicate edges.

    BATCHED: one bulk edge traversal (:func:`~assimilation.dedup.iter_all_edges`)
    instead of ``O(ids)`` per-node ``out_edges`` round-trips — the live-backend
    scaling fix; falls back to per-node when no bulk edge view exists.
    """
    adj: dict[str, set[str]] = {i: set() for i in ids}
    graph = getattr(engine, "graph", None)
    if graph is None:
        return adj

    def _link(src: str, dst: str, props: Any) -> None:
        if dst not in ids or src not in ids:
            return
        if isinstance(props, dict) and str(props.get("_rel", "")) in _EXCLUDED_RELS:
            return
        adj[src].add(dst)
        adj[dst].add(src)

    edges = iter_all_edges(graph)
    if edges is not None:  # bulk path
        for src, dst, props in edges:
            _link(src, dst, props)
        return adj
    for nid in ids:  # per-node fallback
        try:
            out = graph.out_edges(nid, data=True)
        except (TypeError, AttributeError):  # pragma: no cover
            continue
        for _s, dst, props in out:
            _link(nid, dst, props)
    return adj


def _connected_components(ids: set[str], adj: dict[str, set[str]]) -> list[list[str]]:
    seen: set[str] = set()
    comps: list[list[str]] = []
    for start in ids:
        if start in seen:
            continue
        stack, comp = [start], []
        seen.add(start)
        while stack:
            n = stack.pop()
            comp.append(n)
            for m in adj.get(n, ()):
                if m not in seen:
                    seen.add(m)
                    stack.append(m)
        comps.append(comp)
    return comps


def _communities(
    engine: Any, ids: set[str], adj: dict[str, set[str]]
) -> list[list[str]]:
    """Engine Louvain (filtered to features) if available, else components."""
    fn = getattr(engine, "community_detection", None)
    if callable(fn):
        try:
            raw = fn()
            scoped = [[n for n in c if n in ids] for c in raw]
            scoped = [c for c in scoped if len(c) >= 1]
            if scoped:
                return scoped
        except Exception:  # pragma: no cover - engine optional
            pass
    return _connected_components(ids, adj)


def synergy_bundles(
    engine: Any,
    *,
    feature_types: tuple[str, ...] = _FEATURE_TYPES,
    min_pillars: int = 2,
    write: bool = True,
) -> SynergyReport:
    """Flag cross-pillar feature communities as synergy bundles."""
    nodes = _feature_nodes(engine, feature_types)
    report = SynergyReport()
    if len(nodes) < 2:
        return report
    ids = set(nodes)
    adj = _adjacency(engine, ids)
    comms = _communities(engine, ids, adj)
    report.communities = len(comms)
    for comm in comms:
        if len(comm) < 2:
            continue
        pillars = sorted({p for p in (_pillar_of(nodes[n]) for n in comm) if p})
        if len(pillars) < min_pillars:
            continue
        report.bundles.append(SynergyBundle(members=sorted(comm), pillars=pillars))
        if write:
            ordered = sorted(comm)
            for i in range(len(ordered)):
                for j in range(i + 1, len(ordered)):
                    engine.link_nodes(
                        ordered[i],
                        ordered[j],
                        RegistryEdgeType.HAS_SYNERGY_WITH,
                        properties={"_rel": "HAS_SYNERGY_WITH", "concept": "KG-2.7"},
                    )
                    report.edges_written += 1
    return report


def _centrality(
    engine: Any, ids: set[str], adj: dict[str, set[str]]
) -> dict[str, float]:
    """Centrality over the feature subgraph.

    Defaults to fast feature-scoped degree centrality. The engine's global
    PageRank ranks features across the WHOLE graph (5k+ nodes) — too slow on a
    live backend to rank a few dozen features — so it is opt-in via
    ``ASSIMILATION_ENGINE_PAGERANK=1``.
    """
    import os

    if os.environ.get("ASSIMILATION_ENGINE_PAGERANK", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        fn = getattr(engine, "pagerank", None)
        if callable(fn):
            try:
                scores = {nid: float(s) for nid, s in fn() if nid in ids}
                if scores:
                    return scores
            except Exception:  # pragma: no cover - engine optional
                pass
    denom = float(max(1, len(ids) - 1))
    return {i: len(adj.get(i, ())) / denom for i in ids}


def rank_features(
    engine: Any,
    *,
    feature_ids: list[str] | None = None,
    feature_types: tuple[str, ...] = _FEATURE_TYPES,
) -> list[RankedFeature]:
    """Rank open gaps by leverage = ``source_count × (1 + centrality)``."""
    nodes = _feature_nodes(engine, feature_types)
    ids = (
        set(feature_ids)
        if feature_ids is not None
        else set(open_features(engine, feature_types=feature_types))
    )
    ids &= set(nodes)
    if not ids:
        return []
    adj = _adjacency(engine, ids)
    cent = _centrality(engine, ids, adj)
    ranked: list[RankedFeature] = []
    for fid in ids:
        srcs = nodes[fid].get("research_sources") or []
        source_count = max(1, len(srcs))
        c = float(cent.get(fid, 0.0))
        ranked.append(
            RankedFeature(
                feature_id=fid,
                score=round(source_count * (1.0 + c), 6),
                source_count=source_count,
                centrality=round(c, 6),
            )
        )
    ranked.sort(key=lambda r: (r.score, r.feature_id), reverse=True)
    return ranked


__all__ = [
    "SynergyBundle",
    "SynergyReport",
    "RankedFeature",
    "synergy_bundles",
    "rank_features",
]
