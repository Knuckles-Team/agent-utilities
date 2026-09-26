#!/usr/bin/python
from __future__ import annotations

"""Comparative feature / innovation matrix (CONCEPT:AU-KG.research.default-so-every-cycle).

The assimilation pass already turns ingested research (papers + codebases) into a
graph: every extracted feature is matched against our ecosystem ``Concept``
registry (``SATISFIED_BY`` = we already built it, ``RELATES_TO`` = novel-but-
relevant gap), cross-pillar communities are flagged as synergy bundles, and open
gaps are leverage-ranked. That graph *is* the comparative analysis — but it is
implicit, scattered across edges. This module **materializes** it into one
deliverable artifact:

* a ``FeatureMatrix`` of feature × source × coverage rows (covered / related /
  novel) carrying novelty, leverage and synergy partners,
* a rendered markdown report (the human view: what each source contributes, the
  novel gaps to implement, and — the headline — the **cross-source synergies**
  that combine individually-known ideas into novel unique implementations that
  surpass the individual sources), and
* a queryable ``feature_matrix`` graph node + a refreshable ``LiveArtifact``
  (KG-2.24), so both the gateway and MCP surfaces can read it.

It is pure assembly over the existing assimilation outputs — **no new scoring
math**: leverage is :func:`~assimilation.synergy.rank_features`
``source_count × (1 + centrality)``; novelty is the ``RELATES_TO`` edge's
``novelty``; synergy is membership in a :func:`~assimilation.synergy.synergy_bundles`
cross-pillar bundle. Wired default-ON into ``LoopController._run_assimilate`` so
the matrix is regenerated from the graph every cycle.

Concept: feature-matrix
"""

import json
from dataclasses import asdict
from typing import Any

from epistemic_graph.assimilation_derivation import pillar_of
from epistemic_graph.assimilation_matrix import (
    MAX_PERSISTED_ROWS,
    FeatureMatrix,
    FeatureMatrixRow,
    clip_feature_label,
    render_markdown,
)

from ...models.knowledge_graph import RegistryNodeType
from .dedup import iter_all_edges
from .gap_analysis import _CONCEPT_TYPES, _FEATURE_TYPES, _collect_rich, _rel_of
from .synergy import rank_features, synergy_bundles


def build_feature_matrix(
    engine: Any,
    *,
    feature_types: tuple[str, ...] = _FEATURE_TYPES,
    concept_types: tuple[str, ...] = _CONCEPT_TYPES,
    generated_at: str = "",
    restrict_to: set[str] | None = None,
) -> FeatureMatrix:
    """Assemble the comparative matrix from the post-assimilation graph.

    Reads the coverage edges (``SATISFIED_BY``/``RELATES_TO``), overlays leverage
    (:func:`rank_features`) and synergy membership (:func:`synergy_bundles`,
    ``write=False`` — the edges were already written by the assimilate pass, so we
    only READ here), and emits one row per feature.

    ``restrict_to`` scopes the WHOLE build to a feature set (e.g. one research
    cohort's sources, CONCEPT:AU-KG.ingest.fetch-only-requested-ids): collection is per-id, coverage is read
    per-id, and leverage/synergy are scoped — so a per-cohort matrix is O(cohort),
    not the O(graph) whole-graph pull that doesn't scale to 10× corpora.
    """
    features = _collect_rich(engine, feature_types, restrict_to=restrict_to)
    # Concept types are read by rank/synergy via the graph; kept in the signature
    # so callers can scope the registry the matrix is graded against.
    _ = concept_types

    covered: dict[str, tuple[str, float]] = {}
    related: dict[str, tuple[str, float]] = {}
    graph = getattr(engine, "graph", None)

    def _record(src: str, dst: str, props: Any) -> None:
        if src not in features:
            return
        rel = _rel_of(props)
        nov = (
            float(props.get("novelty", 0.0) or 0.0) if isinstance(props, dict) else 0.0
        )
        if rel == "SATISFIED_BY" and src not in covered:
            covered[src] = (str(dst), nov)
        elif rel == "RELATES_TO" and src not in related:
            related[src] = (str(dst), nov)

    # Coverage edges. SCOPED: read per-id out-edges over just the cohort's features
    # (no whole-graph edge pull). Unscoped: bulk traversal, per-node fallback on
    # test doubles / graphs with no bulk edge view (mirrors gap_analysis).
    edges = (
        None
        if restrict_to is not None
        else (iter_all_edges(graph) if graph is not None else None)
    )
    if edges is not None:  # bulk path — one traversal
        for src, dst, props in edges:
            _record(src, dst, props)
    elif graph is not None:  # per-node (scoped, or no bulk edge view)
        for fid in features:
            try:
                for _s, dst, props in graph.out_edges(fid, data=True):
                    _record(fid, dst, props)
            except (TypeError, AttributeError):  # pragma: no cover
                continue

    leverage = {
        r.feature_id: r.score
        for r in rank_features(
            engine,
            feature_ids=(list(restrict_to) if restrict_to is not None else None),
            feature_types=feature_types,
        )
    }

    syn = synergy_bundles(
        engine, feature_types=feature_types, write=False, restrict_to=restrict_to
    )
    partners: dict[str, list[str]] = {}
    bundle_pillars: dict[str, list[str]] = {}
    for bundle in syn.bundles:
        for member in bundle.members:
            others = [m for m in bundle.members if m != member]
            partners[member] = sorted(set(partners.get(member, [])) | set(others))
            bundle_pillars[member] = sorted(
                set(bundle_pillars.get(member, [])) | set(bundle.pillars)
            )

    rows: list[FeatureMatrixRow] = []
    source_index: dict[str, list[str]] = {}
    for fid, data in features.items():
        if fid in covered:
            coverage, concept_id, novelty = "covered", covered[fid][0], covered[fid][1]
        elif fid in related:
            coverage, concept_id, novelty = "related", related[fid][0], related[fid][1]
        else:
            coverage, concept_id, novelty = "novel", "", 1.0
        sources = [str(s) for s in (data.get("research_sources") or []) if s]
        rows.append(
            FeatureMatrixRow(
                feature_id=fid,
                name=str(data.get("name") or data.get("title") or fid),
                pillar=pillar_of(data),
                feature_type=str(data.get("type", "")),
                coverage=coverage,
                concept_id=concept_id,
                novelty_score=round(novelty, 4),
                leverage_score=round(float(leverage.get(fid, 0.0)), 4),
                sources=sources,
                synergy_partners=partners.get(fid, []),
                synergy_pillars=bundle_pillars.get(fid, []),
            )
        )
        for s in sources:
            source_index.setdefault(s, []).append(fid)

    rows.sort(key=lambda r: (r.leverage_score, r.feature_id), reverse=True)
    counts = {
        "total": len(rows),
        "covered": sum(1 for r in rows if r.coverage == "covered"),
        "related": sum(1 for r in rows if r.coverage == "related"),
        "novel": sum(1 for r in rows if r.coverage == "novel"),
        "bundles": len(syn.bundles),
        "sources": len(source_index),
    }
    return FeatureMatrix(
        rows=rows,
        bundles=syn.bundles,
        source_index=source_index,
        counts=counts,
        generated_at=generated_at,
    )


def materialize(
    engine: Any,
    matrix: FeatureMatrix,
    *,
    node_id: str = "feature_matrix:latest",
    write: bool = True,
) -> dict[str, Any]:
    """Persist the matrix as a queryable graph node + a refreshable LiveArtifact.

    The graph node (``RegistryNodeType.FEATURE_MATRIX``) carries the bounded
    top-``MAX_PERSISTED_ROWS`` rows + counts as JSON; the rendered markdown is the
    human view. The LiveArtifact (KG-2.24) is best-effort so the matrix is
    refreshable and reachable over the artifacts REST surface even where the
    artifact store is not configured.
    """
    bounded = [asdict(r) for r in matrix.rows[:MAX_PERSISTED_ROWS]]
    markdown = render_markdown(matrix)
    summary = {
        "node_id": node_id,
        "counts": matrix.counts,
        "generated_at": matrix.generated_at,
        "rows_persisted": len(bounded),
    }
    if not write:
        summary["markdown"] = markdown
        return summary

    try:
        engine.add_node(
            node_id,
            node_type=RegistryNodeType.FEATURE_MATRIX.value,
            properties={
                "counts": json.dumps(matrix.counts),
                "rows": json.dumps(bounded),
                "bundles": json.dumps(
                    [
                        {"members": b.members, "pillars": b.pillars}
                        for b in matrix.bundles
                    ]
                ),
                "generated_at": matrix.generated_at,
                "markdown": markdown,
                "concept": "AU-KG.research.default-so-every-cycle",
            },
        )
        summary["persisted"] = True
    except Exception as exc:  # noqa: BLE001 — materialization is best-effort
        summary["persisted"] = False
        summary["error"] = str(exc)

    if summary["persisted"]:
        _register_matrix_materialization(
            engine, node_id, matrix.rows[:MAX_PERSISTED_ROWS]
        )

    _persist_live_artifact(node_id, matrix, markdown, summary)
    summary["markdown"] = markdown
    return summary


def _register_matrix_materialization(
    engine: Any, node_id: str, rows: list[FeatureMatrixRow]
) -> None:
    """X-6 / Seam 3 (CONCEPT:EG-KG.epistemic.truth-maintenance): stamp the matrix
    node's real invalidation-dependency provenance and register it as a live
    engine-side TruthMaintenance materialization.

    The matrix is a materialized JOIN over the assimilated graph: each persisted
    row's ``feature_id`` (the base Feature/SDDFeature/Article node it summarizes)
    and, when present, the ``concept_id`` it was matched against (the
    ``SATISFIED_BY``/``RELATES_TO`` target) are the REAL base facts this row came
    from — never fabricated. A ``:DerivedFrom`` edge (``relationship_type=
    "DERIVED_FROM"``, the exact property key ``eg_epistemic::
    register_from_provenance`` reads) lands from the matrix node to each distinct
    one of those ids, then the matrix node is registered as a live TruthMaintenance
    materialization off that SAME provenance — so a change to any summarized
    feature or matched concept marks the whole matrix stale, prompting the next
    cycle's ``materialize`` call to refresh it. Best-effort: never gates the
    already-completed persist above (mirrors ``loop_controller.
    _register_derived_claim``'s posture).
    """
    deps: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for dep in (row.feature_id, row.concept_id):
            if dep and dep not in seen:
                seen.add(dep)
                deps.append(dep)
    for dep in deps:
        try:
            engine.add_edge(node_id, dep, "DERIVED_FROM")
        except Exception:  # noqa: BLE001 — provenance edges are best-effort
            pass
    try:
        engine.register_materialization(node_id)
    except Exception:  # noqa: BLE001 — TMS registration is best-effort
        pass


def _persist_live_artifact(
    node_id: str, matrix: FeatureMatrix, markdown: str, summary: dict[str, Any]
) -> None:
    """Best-effort KG-2.24 LiveArtifact so the matrix is refreshable + REST-reachable."""
    try:
        from ..live_artifacts.models import LiveArtifact
        from ..live_artifacts.store import get_live_artifact_store

        # Keep the artifact data within the KG-2.24 bounded-JSON envelope: counts +
        # the leverage-ranked novel-gap shortlist (the full set lives in the graph
        # node). validate_data() raises BoundedJSONError if it still overflows.
        gaps = matrix.novel_gaps()[:50]
        artifact = LiveArtifact(
            artifact_id=node_id,
            name="Comparative Feature / Innovation Matrix",
            data={
                "counts": matrix.counts,
                "novel_gaps": [
                    {
                        "feature_id": r.feature_id,
                        "name": clip_feature_label(r.name, 80),
                        "coverage": r.coverage,
                        "leverage": r.leverage_score,
                        "novelty": r.novelty_score,
                    }
                    for r in gaps
                ],
            },
            last_rendered=markdown,
            source_node_ids=[node_id],
        )
        artifact.validate_data()
        get_live_artifact_store().put(artifact)
        summary["live_artifact"] = True
    except Exception:  # noqa: BLE001 — artifact store optional / bounded-JSON guard
        summary["live_artifact"] = False


__all__ = [
    "FeatureMatrixRow",
    "FeatureMatrix",
    "build_feature_matrix",
    "render_markdown",
    "materialize",
    "MAX_PERSISTED_ROWS",
]
