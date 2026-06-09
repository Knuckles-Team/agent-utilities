#!/usr/bin/python
from __future__ import annotations

"""Cross-source feature deduplication (CONCEPT:KG-2.7).

The same capability often appears in a paper, an OSS library, AND our own code —
three nodes for one idea. This collapses them in the graph so downstream gap
analysis, synergy, and plan synthesis see one feature with multi-source provenance.

Mechanism (graph-native, no LLM):
1. Collect embedded `Feature`/`Article`/`SDDFeature` nodes.
2. Compute pairwise cosine similarity — preferring the engine's batched
   ``compute_similarity_edges`` (one round-trip; KG-2.3 similarity-collapse) and
   falling back to a local numpy pass when unavailable (deterministic, testable).
3. Write a `SIMILAR_TO` edge (score property) for every pair ≥ ``similar_threshold``.
4. Union-find cluster the pairs ≥ ``dup_threshold``; in each cluster keep the
   highest-importance survivor and write ``survivor -[SUPERSEDES]-> duplicate``.

Idempotent: edges MERGE on write (re-running converges). Incremental: pass
``restrict_to`` (e.g. newly-ingested node ids) to only compare new features against
the existing set — O(new·N) instead of O(N²).

Concept: feature-dedup
"""

import math
from dataclasses import dataclass, field
from typing import Any

from ...models.knowledge_graph import RegistryEdgeType, RegistryNodeType

_DEFAULT_TYPES: tuple[str, ...] = (
    RegistryNodeType.SDD_FEATURE.value,
    RegistryNodeType.CAPABILITY.value,
    RegistryNodeType.ARTICLE.value,
)


@dataclass
class DedupReport:
    """Outcome of a dedup pass."""

    candidates: int = 0
    similar_pairs: int = 0
    clusters: int = 0
    duplicates_superseded: int = 0
    survivors: list[str] = field(default_factory=list)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def iter_all_edges(graph: Any) -> list[tuple[str, str, dict]] | None:
    """All ``(src, dst, props)`` edges via the graph's BULK edge view.

    The live engine exposes ``graph.edges`` as a single bulk traversal (one
    round-trip to the Rust daemon); using it replaces the assimilation stages'
    ``O(features)`` per-node ``out_edges``/``in_edges`` round-trips — the
    live-backend scaling fix. Returns ``None`` when no usable bulk view exists
    (minimal test doubles with only per-node ``out_edges``, or a view that yields
    no edge ``data``) so callers fall back to the per-node path with identical
    semantics.
    """
    view = getattr(graph, "edges", None)
    if view is None:
        return None
    try:
        seq = view(data=True) if callable(view) else view
        out: list[tuple[str, str, dict]] = []
        for e in seq:
            # Require (src, dst, props-dict); without edge data we can't classify
            # closing/excluded relationships, so bail to the per-node fallback.
            if (
                not isinstance(e, tuple | list)
                or len(e) < 3
                or not isinstance(e[2], dict)
            ):
                return None
            out.append((e[0], e[1], e[2]))
        return out
    except Exception:  # pragma: no cover - defensive; any view error → fallback
        return None


def _collect(engine: Any, node_types: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    """Map id → {vec, importance} for embedded nodes of the target types."""
    out: dict[str, dict[str, Any]] = {}
    graph = getattr(engine, "graph", None)
    if graph is None:
        return out
    try:
        node_iter = graph.nodes(data=True)
    except TypeError:  # pragma: no cover - non-standard graph
        return out
    # Case-insensitive: the live graph stores `type` as a capitalized label
    # ("Article"/"Concept") while our enum values are lowercase.
    wanted = {t.lower() for t in node_types}
    for nid, data in node_iter:
        if not isinstance(data, dict):
            continue
        if str(data.get("type", "")).lower() not in wanted:
            continue
        emb = data.get("embedding")
        if not emb:
            continue
        out[nid] = {
            "vec": list(emb),
            "importance": float(data.get("importance_score", 0.0) or 0.0),
        }
    return out


def _engine_pairs(engine: Any, ids: set[str], threshold: float):
    """Try the engine's batched all-pairs similarity; None if unavailable."""
    fn = getattr(engine, "compute_similarity_edges", None)
    if fn is None:
        gc = getattr(engine, "graph", None)
        fn = getattr(gc, "compute_similarity_edges", None) if gc is not None else None
    if not callable(fn):
        return None
    try:
        raw = fn(threshold)
    except Exception:  # pragma: no cover - engine optional
        return None
    return [(a, b, float(s)) for (a, b, s) in raw if a in ids and b in ids and a != b]


def _local_pairs(nodes: dict[str, dict[str, Any]], threshold: float):
    """Deterministic local all-pairs cosine (fallback / test path)."""
    items = list(nodes.items())
    pairs: list[tuple[str, str, float]] = []
    for i in range(len(items)):
        ai, av = items[i]
        for j in range(i + 1, len(items)):
            bj, bv = items[j]
            s = _cosine(av["vec"], bv["vec"])
            if s >= threshold:
                pairs.append((ai, bj, s))
    return pairs


def _clusters(ids: list[str], dup_pairs) -> list[list[str]]:
    """Union-find connected components over the duplicate pairs (size ≥ 2)."""
    parent = {n: n for n in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, _ in dup_pairs:
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    groups: dict[str, list[str]] = {}
    for n in ids:
        groups.setdefault(find(n), []).append(n)
    return [g for g in groups.values() if len(g) > 1]


def dedup_features(
    engine: Any,
    *,
    node_types: tuple[str, ...] = _DEFAULT_TYPES,
    similar_threshold: float = 0.83,
    dup_threshold: float = 0.93,
    restrict_to: set[str] | None = None,
    write: bool = True,
) -> DedupReport:
    """Link similar features and supersede duplicates across sources.

    Args:
        engine: the knowledge engine (needs ``graph.nodes(data=True)`` +
            ``link_nodes``; uses ``compute_similarity_edges`` when present).
        node_types: node types to dedup (default Feature/Article/SDDFeature).
        similar_threshold: cosine ≥ this → a `SIMILAR_TO` edge.
        dup_threshold: cosine ≥ this → treated as a duplicate (clustered + superseded).
        restrict_to: if given, only consider pairs touching these ids (incremental).
        write: persist edges (False = analysis-only / dry run).

    Returns:
        A :class:`DedupReport`.
    """
    nodes = _collect(engine, node_types)
    report = DedupReport(candidates=len(nodes))
    if len(nodes) < 2:
        return report
    ids = set(nodes)

    pairs = _engine_pairs(engine, ids, similar_threshold)
    if pairs is None:
        pairs = _local_pairs(nodes, similar_threshold)
    if restrict_to:
        pairs = [(a, b, s) for a, b, s in pairs if a in restrict_to or b in restrict_to]
    report.similar_pairs = len(pairs)

    if write:
        for a, b, s in pairs:
            engine.link_nodes(
                a,
                b,
                RegistryEdgeType.SIMILAR_TO,
                properties={"_rel": "SIMILAR_TO", "score": round(s, 6)},
            )

    dup_pairs = [(a, b, s) for a, b, s in pairs if s >= dup_threshold]
    clusters = _clusters(list(ids), dup_pairs)
    report.clusters = len(clusters)
    for cluster in clusters:
        survivor = max(cluster, key=lambda n: (nodes[n]["importance"], n))
        report.survivors.append(survivor)
        for dup in cluster:
            if dup == survivor:
                continue
            if write:
                engine.link_nodes(
                    survivor,
                    dup,
                    RegistryEdgeType.SUPERSEDES,
                    # `_rel` mirrors the edge label into properties so the lifecycle
                    # read path (gap_analysis.open_features) is backend-portable —
                    # out_edges/in_edges expose properties, not the rel label.
                    properties={
                        "_rel": "SUPERSEDES",
                        "reason": "duplicate",
                        "concept": "KG-2.7",
                    },
                )
            report.duplicates_superseded += 1
    return report


__all__ = ["DedupReport", "dedup_features"]
