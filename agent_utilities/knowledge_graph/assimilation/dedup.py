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
from .entity_resolution import resolve_entities

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
    # entropy-gated name-resolution fast-path (CONCEPT:AHE-3.69)
    name_resolved_pairs: int = 0
    low_entropy_skipped: int = 0
    # version-variant pairs LINKED (not merged) as VARIANT_OF (CONCEPT:AHE-3.70)
    variants_linked: int = 0
    # proposals applied from the engine ResolveCandidates escalation (CONCEPT:KG-2.260)
    engine_proposals: int = 0


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
            "name": str(
                data.get("name") or data.get("label") or data.get("title") or nid
            ),
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

    # Entropy-gated name-resolution fast-path (CONCEPT:AHE-3.69): merge entities
    # whose normalized names match exactly or fuzzy-match (MinHash/LSH Jaccard) —
    # LLM-free and embedding-independent, so it catches same-entity duplicates even
    # when their vectors disagree (cosine < dup_threshold). Generic low-entropy
    # names are deliberately NOT merged here; they stay on the embedding path.
    name_res = resolve_entities([(nid, str(nodes[nid]["name"])) for nid in sorted(ids)])
    report.name_resolved_pairs = len(name_res.merge_pairs)
    report.low_entropy_skipped = name_res.low_entropy
    name_dup_pairs: list[tuple[str, str, float]] = []
    for a, b, score, _tier in name_res.merge_pairs:
        if restrict_to and a not in restrict_to and b not in restrict_to:
            continue
        name_dup_pairs.append((a, b, score))
        if write:
            engine.link_nodes(
                a,
                b,
                RegistryEdgeType.SIMILAR_TO,
                properties={"_rel": "SIMILAR_TO", "score": round(score, 6)},
            )

    # Version-variant pairs are LINKED as VARIANT_OF, never merged (CONCEPT:AHE-3.70):
    # a base and its versioned sibling are distinct entities with a real relationship.
    for base, variant, score, _kind in name_res.variants:
        if restrict_to and base not in restrict_to and variant not in restrict_to:
            continue
        report.variants_linked += 1
        if write:
            engine.link_nodes(
                base,
                variant,
                RegistryEdgeType.VARIANT_OF,
                properties={
                    "_rel": "VARIANT_OF",
                    "concept": "AHE-3.70",
                    "score": round(score, 6),
                },
            )

    # Server-side escalation (CONCEPT:KG-2.260): when the engine exposes the native
    # ResolveCandidates op, escalate the ambiguous residual to it — embedding
    # similarity + clustering yields same_as (merge) AND extends (variant) proposals
    # the local name-only pass can't produce. Capability-gated + best-effort: a no-op
    # until the engine ships the op, so it never breaks the pre-deploy path.
    resolve_fn = getattr(engine, "resolve_candidates", None)
    if name_res.residual_ids and callable(resolve_fn):
        try:
            proposals = resolve_fn(0.8, dup_threshold, None) or []
        except Exception:  # noqa: BLE001 — escalation never breaks dedup
            proposals = []
        residual = set(name_res.residual_ids)
        for prop in proposals:
            members = [m for m in (prop.get("members") or []) if m in nodes]
            if len(members) < 2 or residual.isdisjoint(members):
                continue
            canonical = prop.get("canonical") or members[0]
            if prop.get("kind") == "extends":
                report.engine_proposals += 1
                for m in members:
                    if m != canonical and write:
                        engine.link_nodes(
                            canonical,
                            m,
                            RegistryEdgeType.VARIANT_OF,
                            properties={"_rel": "VARIANT_OF", "concept": "KG-2.260"},
                        )
            else:  # same_as → feed the duplicate clustering below
                score = float(prop.get("score", dup_threshold))
                for m in members:
                    if m != canonical:
                        report.engine_proposals += 1
                        name_dup_pairs.append((canonical, m, score))

    dup_pairs = [(a, b, s) for a, b, s in pairs if s >= dup_threshold] + name_dup_pairs
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
