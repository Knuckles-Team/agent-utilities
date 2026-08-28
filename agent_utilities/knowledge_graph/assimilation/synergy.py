#!/usr/bin/python
from __future__ import annotations

"""Synergy bundles + leverage ranking (CONCEPT:AU-KG.query.vendor-agnostic-traversal / KG-2.5).

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

from agent_utilities.core.config import setting

from ...models.knowledge_graph import RegistryEdgeType
from .dedup import iter_all_edges, iter_typed_nodes
from .gap_analysis import _FEATURE_TYPES, _node_data_by_id, open_features

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
        namespace = str(cid).split(".", 1)[0].upper()
        parts = namespace.split("-")
        # Semantic ids are namespaced as AU-KG.*, AU-ORCH.*, EG-KG.*, etc.
        # The platform prefix is not the architectural pillar.
        head = parts[1] if len(parts) > 1 and parts[0] in {"AU", "EG"} else parts[0]
        if head:
            return head
    return ""


def _feature_nodes_full_scan_fallback(
    graph: Any, restrict_to: set[str], wanted: set[str]
) -> dict[str, dict]:
    """Filtered full-scan fallback when the per-id view returns nothing —
    correctness must never depend on the view supporting ``[id]``."""
    try:
        return {
            nid: data
            for nid, data in graph.nodes(data=True)
            if nid in restrict_to
            and isinstance(data, dict)
            and str(data.get("type", "")).lower() in wanted
        }
    except TypeError:  # pragma: no cover
        return {}


def _feature_nodes_scoped(
    graph: Any, restrict_to: set[str], wanted: set[str]
) -> dict[str, dict]:
    """SCOPED (CONCEPT:AU-KG.ingest.fetch-only-requested-ids): fetch only the
    requested ids per-id — avoids the whole-graph node pull that makes
    per-cohort synthesis O(graph) not O(cohort)."""
    scoped: dict[str, dict] = {}
    for nid in restrict_to:
        data = _node_data_by_id(graph, nid)
        if data is None:
            continue
        if str(data.get("type", "")).lower() in wanted:
            scoped[nid] = data
    if scoped or not restrict_to:
        return scoped
    return _feature_nodes_full_scan_fallback(graph, restrict_to, wanted)


def _feature_nodes(
    engine: Any,
    feature_types: tuple[str, ...],
    restrict_to: set[str] | None = None,
) -> dict[str, dict]:
    graph = getattr(engine, "graph", None)
    if graph is None:
        return {}
    wanted = {t.lower() for t in feature_types}  # case-insensitive (live labels)
    if restrict_to is not None:
        return _feature_nodes_scoped(graph, restrict_to, wanted)
    # Unrestricted: BOUNDED per-label fetch (CONCEPT:EG-KG.txn.per-graph-write-isolation/2.264) — never a
    # whole-graph ``GetNodes`` dump (refused as RESULT_TOO_LARGE at scale).
    return dict(iter_typed_nodes(graph, feature_types))


#: above this many ids, one bulk edge traversal amortizes better than per-node
#: round-trips; at/below it, per-node ``out_edges`` is BOUNDED — it touches only the
#: ids' own edges, never the whole-graph edge list (which on a 166K-node engine is a
#: gigabyte-scale payload that overloads the connection). (CONCEPT:AU-KG.ingest.fetch-only-requested-ids)
_ADJ_BULK_THRESHOLD = 1000


def _adjacency(engine: Any, ids: set[str]) -> dict[str, set[str]]:
    """Undirected feature-feature adjacency from non-duplicate edges.

    For a SMALL id set (a scoped/cohort pass) this fetches edges PER-NODE over just
    those ids — bounded work — instead of pulling every edge in the graph. For a
    large set it uses one bulk edge traversal (amortized). Falls back to per-node
    when no bulk edge view exists.
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

    # Bounded per-node path for scoped sets; bulk only when the set is large enough
    # to amortize a whole-graph edge pull.
    if len(ids) > _ADJ_BULK_THRESHOLD:
        edges = iter_all_edges(graph)
        if edges is not None:  # bulk path
            for src, dst, props in edges:
                _link(src, dst, props)
            return adj
    for nid in ids:  # bounded per-node (scoped) — only the ids' own edges
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


def _engine_communities(engine: Any, ids: set[str]) -> list[list[str]] | None:
    """Engine Louvain community_detection, filtered+scoped to ``ids``.

    None when the engine has no such method, it raises, or it yields nothing
    scoped — the caller falls back to local connected components.
    """
    fn = getattr(engine, "community_detection", None)
    if not callable(fn):
        return None
    try:
        raw = fn()
        scoped = [[n for n in c if n in ids] for c in raw]
        scoped = [c for c in scoped if len(c) >= 1]
    except Exception:  # noqa: BLE001 — optional engine algorithm has local fallback
        return None
    return scoped or None


def _communities(
    engine: Any, ids: set[str], adj: dict[str, set[str]]
) -> list[list[str]]:
    """Engine Louvain (filtered to features) if available, else components."""
    return _engine_communities(engine, ids) or _connected_components(ids, adj)


def _bundle_pillars(
    nodes: dict[str, dict], comm: list[str], min_pillars: int
) -> list[str] | None:
    """Sorted pillar set for one community, or None if below ``min_pillars``."""
    pillars = sorted({p for p in (_pillar_of(nodes[n]) for n in comm) if p})
    if len(pillars) < min_pillars:
        return None
    return pillars


def _write_synergy_edges(engine: Any, ordered: list[str]) -> int:
    """Pairwise HAS_SYNERGY_WITH edges across an ordered community. Returns
    the number of edges written."""
    written = 0
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            engine.link_nodes(
                ordered[i],
                ordered[j],
                RegistryEdgeType.HAS_SYNERGY_WITH,
                properties={
                    "_rel": "HAS_SYNERGY_WITH",
                    "concept": "AU-KG.query.vendor-agnostic-traversal",
                },
            )
            written += 1
    return written


def synergy_bundles(
    engine: Any,
    *,
    feature_types: tuple[str, ...] = _FEATURE_TYPES,
    min_pillars: int = 2,
    write: bool = True,
    restrict_to: set[str] | None = None,
) -> SynergyReport:
    """Flag cross-pillar feature communities as synergy bundles.

    ``restrict_to`` scopes detection to a specific feature set (e.g. one research
    cohort), so synergy among the cohort's own sources is found without an O(graph)
    pull (CONCEPT:AU-KG.ingest.fetch-only-requested-ids).
    """
    nodes = _feature_nodes(engine, feature_types, restrict_to=restrict_to)
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
        pillars = _bundle_pillars(nodes, comm, min_pillars)
        if pillars is None:
            continue
        report.bundles.append(SynergyBundle(members=sorted(comm), pillars=pillars))
        if write:
            report.edges_written += _write_synergy_edges(engine, sorted(comm))
    return report


def _engine_pagerank_centrality(engine: Any, ids: set[str]) -> dict[str, float] | None:
    """Engine PageRank centrality, gated by ``ASSIMILATION_ENGINE_PAGERANK=1``.

    None when disabled, unavailable, it raises, or yields nothing scoped —
    the caller falls back to local degree centrality.
    """
    if setting("ASSIMILATION_ENGINE_PAGERANK", "").lower() not in (
        "1",
        "true",
        "yes",
    ):
        return None
    fn = getattr(engine, "pagerank", None)
    if not callable(fn):
        return None
    try:
        scores = {nid: float(s) for nid, s in fn() if nid in ids}
    except Exception:  # noqa: BLE001 — optional engine algorithm has local fallback
        return None
    return scores or None


def _centrality(
    engine: Any, ids: set[str], adj: dict[str, set[str]]
) -> dict[str, float]:
    """Centrality over the feature subgraph.

    Defaults to fast feature-scoped degree centrality. The engine's global
    PageRank ranks features across the WHOLE graph (5k+ nodes) — too slow on a
    live backend to rank a few dozen features — so it is opt-in via
    ``ASSIMILATION_ENGINE_PAGERANK=1``.
    """
    scores = _engine_pagerank_centrality(engine, ids)
    if scores is not None:
        return scores
    denom = float(max(1, len(ids) - 1))
    return {i: len(adj.get(i, ())) / denom for i in ids}


def rank_features(
    engine: Any,
    *,
    feature_ids: list[str] | None = None,
    feature_types: tuple[str, ...] = _FEATURE_TYPES,
) -> list[RankedFeature]:
    """Rank open gaps by leverage = ``source_count × (1 + centrality)``.

    When ``feature_ids`` is given, node collection is SCOPED to those ids
    (CONCEPT:AU-KG.ingest.fetch-only-requested-ids) so ranking a cohort never pulls the whole graph.
    """
    scope = set(feature_ids) if feature_ids is not None else None
    nodes = _feature_nodes(engine, feature_types, restrict_to=scope)
    ids = (
        scope
        if scope is not None
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
