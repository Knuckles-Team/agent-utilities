#!/usr/bin/python
from __future__ import annotations

"""Cross-source feature deduplication (CONCEPT:AU-KG.query.vendor-agnostic-traversal).

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
    # entropy-gated name-resolution fast-path (CONCEPT:AU-AHE.assimilation.merge-entities)
    name_resolved_pairs: int = 0
    low_entropy_skipped: int = 0
    # version-variant pairs LINKED (not merged) as VARIANT_OF (CONCEPT:AU-AHE.assimilation.transliteration-singularization-extend-ahe)
    variants_linked: int = 0
    # proposals applied from the engine ResolveCandidates escalation
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


def iter_typed_nodes(
    graph: Any, node_types: tuple[str, ...]
) -> list[tuple[str, dict[str, Any]]]:
    """``(id, data)`` for nodes whose ``type`` matches ``node_types`` — fetched via
    the engine's BOUNDED per-label index (``get_nodes_by_label``), NOT a whole-graph
    ``GetNodes`` dump.

    On a large multi-tenant engine a full node list is refused by the response guard
    (``RESULT_TOO_LARGE``, CONCEPT:EG-KG.ingest.resets-socket-so-assimilation) and resets the socket, so the assimilation
    collectors must scope their pull by label (CONCEPT:EG-KG.txn.per-graph-write-isolation/2.193). Falls back to a
    filtered whole-graph scan ONLY when no label index exists (a test-double dict/NX
    graph). Dedups by id so case-variant label buckets can't double-count a node.
    """
    wanted = {t.lower() for t in node_types}
    by_label = getattr(graph, "get_nodes_by_label", None)
    if callable(by_label):
        # Live labels are inconsistently cased ("article" vs "Concept"); try each.
        labels = {
            cased
            for t in node_types
            for cased in (t, t.lower(), t.capitalize(), t.upper(), t.title())
        }
        out: dict[str, dict[str, Any]] = {}
        for lbl in labels:
            try:
                rows = by_label(lbl, 0) or []  # limit 0 = all of THIS label (bounded)
            except Exception:  # noqa: BLE001 — try the next label casing
                continue
            for row in rows:
                if isinstance(row, list | tuple) and len(row) >= 2:
                    nid, data = str(row[0]), row[1]
                    if (
                        isinstance(data, dict)
                        and str(data.get("type", "")).lower() in wanted
                    ):
                        out[nid] = data
        return list(out.items())
    # No label index (minimal test double) → filtered whole-graph scan.
    try:
        node_iter = graph.nodes(data=True)
    except TypeError:  # pragma: no cover - non-standard graph
        return []
    return [
        (nid, data)
        for nid, data in node_iter
        if isinstance(data, dict) and str(data.get("type", "")).lower() in wanted
    ]


def _collect(engine: Any, node_types: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    """Map id → {vec, importance} for embedded nodes of the target types."""
    out: dict[str, dict[str, Any]] = {}
    graph = getattr(engine, "graph", None)
    if graph is None:
        return out
    # Case-insensitive label/type match; bounded per-label fetch (no whole-graph dump).
    for nid, data in iter_typed_nodes(graph, node_types):
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


def _restrict_pairs(pairs, restrict_to: set[str] | None):
    """Keep only pairs touching the incremental target set."""
    if not restrict_to:
        return pairs
    return [
        (a, b, score) for a, b, score in pairs if a in restrict_to or b in restrict_to
    ]


def _write_similarity_edges(engine: Any, pairs, write: bool) -> None:
    """Persist ``SIMILAR_TO`` links for a set of scored pairs."""
    if not write:
        return
    for a, b, score in pairs:
        engine.link_nodes(
            a,
            b,
            RegistryEdgeType.SIMILAR_TO,
            properties={"_rel": "SIMILAR_TO", "score": round(score, 6)},
        )


def _name_duplicate_pairs(name_resolution, restrict_to: set[str] | None):
    """Build duplicate pairs from the entropy-gated name resolver."""
    return [
        (a, b, score)
        for a, b, score, _tier in name_resolution.merge_pairs
        if not restrict_to or a in restrict_to or b in restrict_to
    ]


def _write_variant_links(
    engine: Any, variants, restrict_to: set[str] | None, write: bool
) -> int:
    """Persist selected name-resolution variants and return their count."""
    linked = 0
    for base, variant, score, _kind in variants:
        if restrict_to and base not in restrict_to and variant not in restrict_to:
            continue
        linked += 1
        if write:
            engine.link_nodes(
                base,
                variant,
                RegistryEdgeType.VARIANT_OF,
                properties={
                    "_rel": "VARIANT_OF",
                    "concept": "AU-AHE.assimilation.transliteration-singularization-extend-ahe",
                    "score": round(score, 6),
                },
            )
    return linked


def _engine_candidates(engine: Any, residual_ids: set[str], dup_threshold: float):
    """Fetch native ``ResolveCandidates`` proposals when available."""
    if not residual_ids:
        return []
    resolve_fn = getattr(engine, "resolve_candidates", None)
    if not callable(resolve_fn):
        return []
    try:
        return resolve_fn(0.8, dup_threshold, None) or []
    except Exception:  # noqa: BLE001 — escalation never breaks dedup
        return []


def _write_engine_variant_links(
    engine: Any, canonical: str, members: list[str], write: bool
) -> None:
    """Persist an engine ``extends`` proposal as ``VARIANT_OF`` links."""
    if not write:
        return
    for member in members:
        if member == canonical:
            continue
        engine.link_nodes(
            canonical,
            member,
            RegistryEdgeType.VARIANT_OF,
            properties={
                "_rel": "VARIANT_OF",
                "concept": "AU-KG.compute.when-exposes-native",
            },
        )


def _proposal_members(prop: dict[str, Any], nodes, residual: set[str]):
    """Return valid in-scope proposal members and its canonical id."""
    members = [member for member in (prop.get("members") or []) if member in nodes]
    if len(members) < 2 or residual.isdisjoint(members):
        return None
    return prop.get("canonical") or members[0], members


def _apply_engine_proposals(
    engine: Any,
    nodes: dict[str, dict[str, Any]],
    residual_ids: set[str],
    dup_threshold: float,
    write: bool,
) -> tuple[list[tuple[str, str, float]], int]:
    """Apply native variant proposals and return native same-as pairs."""
    residual = set(residual_ids)
    duplicate_pairs: list[tuple[str, str, float]] = []
    proposal_count = 0
    for prop in _engine_candidates(engine, residual, dup_threshold):
        parts = _proposal_members(prop, nodes, residual)
        if parts is None:
            continue
        canonical, members = parts
        if prop.get("kind") == "extends":
            proposal_count += 1
            _write_engine_variant_links(engine, canonical, members, write)
            continue
        score = float(prop.get("score", dup_threshold))
        same_as = [
            (canonical, member, score) for member in members if member != canonical
        ]
        duplicate_pairs.extend(same_as)
        proposal_count += len(same_as)
    return duplicate_pairs, proposal_count


def _supersede_cluster(
    engine: Any,
    cluster: list[str],
    nodes: dict[str, dict[str, Any]],
    write: bool,
) -> tuple[str, int]:
    """Choose a survivor and persist its ``SUPERSEDES`` links."""
    survivor = max(cluster, key=lambda node: (nodes[node]["importance"], node))
    superseded = 0
    for duplicate in cluster:
        if duplicate == survivor:
            continue
        if write:
            engine.link_nodes(
                survivor,
                duplicate,
                RegistryEdgeType.SUPERSEDES,
                # `_rel` mirrors the edge label into properties so the lifecycle
                # read path (gap_analysis.open_features) is backend-portable —
                # out_edges/in_edges expose properties, not the rel label.
                properties={
                    "_rel": "SUPERSEDES",
                    "reason": "duplicate",
                    "concept": "AU-KG.query.vendor-agnostic-traversal",
                },
            )
        superseded += 1
    return survivor, superseded


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
    pairs = _restrict_pairs(pairs, restrict_to)
    report.similar_pairs = len(pairs)
    _write_similarity_edges(engine, pairs, write)

    # Entropy-gated name-resolution fast-path (CONCEPT:AU-AHE.assimilation.merge-entities): merge entities
    # whose normalized names match exactly or fuzzy-match (MinHash/LSH Jaccard) —
    # LLM-free and embedding-independent, so it catches same-entity duplicates even
    # when their vectors disagree (cosine < dup_threshold). Generic low-entropy
    # names are deliberately NOT merged here; they stay on the embedding path.
    name_res = resolve_entities([(nid, str(nodes[nid]["name"])) for nid in sorted(ids)])
    report.name_resolved_pairs = len(name_res.merge_pairs)
    report.low_entropy_skipped = name_res.low_entropy
    name_dup_pairs = _name_duplicate_pairs(name_res, restrict_to)
    _write_similarity_edges(engine, name_dup_pairs, write)

    # Version-variant pairs are LINKED as VARIANT_OF, never merged (CONCEPT:AU-AHE.assimilation.transliteration-singularization-extend-ahe):
    # a base and its versioned sibling are distinct entities with a real relationship.
    report.variants_linked = _write_variant_links(
        engine, name_res.variants, restrict_to, write
    )

    # Server-side escalation: when the engine exposes the native
    # ResolveCandidates op, escalate the ambiguous residual to it — embedding
    # similarity + clustering yields same_as (merge) AND extends (variant) proposals
    # the local name-only pass can't produce. Capability-gated + best-effort: a no-op
    # until the engine ships the op, so it never breaks the pre-deploy path.
    engine_pairs, report.engine_proposals = _apply_engine_proposals(
        engine, nodes, name_res.residual_ids, dup_threshold, write
    )

    dup_pairs = [pair for pair in pairs if pair[2] >= dup_threshold]
    dup_pairs.extend(name_dup_pairs)
    dup_pairs.extend(engine_pairs)
    clusters = _clusters(list(ids), dup_pairs)
    report.clusters = len(clusters)
    for cluster in clusters:
        survivor, superseded = _supersede_cluster(engine, cluster, nodes, write)
        report.survivors.append(survivor)
        report.duplicates_superseded += superseded
    return report


__all__ = ["DedupReport", "dedup_features"]
