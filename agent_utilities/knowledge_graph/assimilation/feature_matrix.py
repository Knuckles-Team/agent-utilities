#!/usr/bin/python
from __future__ import annotations

"""Comparative feature / innovation matrix (CONCEPT:KG-2.173).

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
from dataclasses import asdict, dataclass, field
from typing import Any

from ...models.knowledge_graph import RegistryNodeType
from .dedup import iter_all_edges
from .gap_analysis import _CONCEPT_TYPES, _FEATURE_TYPES, _collect_rich, _rel_of
from .synergy import SynergyBundle, _pillar_of, rank_features, synergy_bundles

#: max rows persisted in the graph node / live artifact (bounded; full set stays
#: derivable by re-running build). Keeps the node property + artifact within the
#: KG-2.24 bounded-JSON envelope.
MAX_PERSISTED_ROWS = 200


@dataclass
class FeatureMatrixRow:
    """One feature's place in the comparative matrix."""

    feature_id: str
    name: str
    pillar: str
    feature_type: str
    coverage: str  # "covered" | "related" | "novel"
    concept_id: str  # the SATISFIED_BY/RELATES_TO concept, "" when fully novel
    novelty_score: float
    leverage_score: float
    sources: list[str] = field(default_factory=list)
    synergy_partners: list[str] = field(default_factory=list)
    synergy_pillars: list[str] = field(default_factory=list)


@dataclass
class FeatureMatrix:
    """The materialized comparative analysis."""

    rows: list[FeatureMatrixRow] = field(default_factory=list)
    bundles: list[SynergyBundle] = field(default_factory=list)
    source_index: dict[str, list[str]] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    generated_at: str = ""

    def novel_gaps(self) -> list[FeatureMatrixRow]:
        """Open (non-covered) rows, leverage-ranked — the work-list to implement."""
        gaps = [r for r in self.rows if r.coverage != "covered"]
        gaps.sort(key=lambda r: (r.leverage_score, r.feature_id), reverse=True)
        return gaps


def build_feature_matrix(
    engine: Any,
    *,
    feature_types: tuple[str, ...] = _FEATURE_TYPES,
    concept_types: tuple[str, ...] = _CONCEPT_TYPES,
    generated_at: str = "",
) -> FeatureMatrix:
    """Assemble the comparative matrix from the post-assimilation graph.

    Reads the coverage edges (``SATISFIED_BY``/``RELATES_TO``) in one bulk pass,
    overlays leverage (:func:`rank_features`) and synergy membership
    (:func:`synergy_bundles`, ``write=False`` — the edges were already written by
    the assimilate pass, so we only READ here), and emits one row per feature.
    """
    features = _collect_rich(engine, feature_types)
    # Concept types are read by rank/synergy via the graph; kept in the signature
    # so callers can scope the registry the matrix is graded against.
    _ = concept_types

    # Coverage edges: bulk traversal on a live backend, per-node fallback on test
    # doubles / graphs with no bulk edge view (mirrors gap_analysis).
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

    edges = iter_all_edges(graph) if graph is not None else None
    if edges is not None:  # bulk path — one traversal
        for src, dst, props in edges:
            _record(src, dst, props)
    elif graph is not None:  # per-node fallback (no bulk edge view)
        for fid in features:
            try:
                for _s, dst, props in graph.out_edges(fid, data=True):
                    _record(fid, dst, props)
            except (TypeError, AttributeError):  # pragma: no cover
                continue

    leverage = {
        r.feature_id: r.score
        for r in rank_features(engine, feature_types=feature_types)
    }

    syn = synergy_bundles(engine, feature_types=feature_types, write=False)
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
                pillar=_pillar_of(data),
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


def render_markdown(matrix: FeatureMatrix) -> str:
    """Render the matrix as a comparative-analysis markdown report."""
    c = matrix.counts
    out: list[str] = []
    out.append("# Comparative Feature / Innovation Matrix")
    if matrix.generated_at:
        out.append(f"\n_Generated: {matrix.generated_at}_")
    out.append(
        f"\n**{c.get('total', 0)} features** across **{c.get('sources', 0)} sources** "
        f"— covered: {c.get('covered', 0)} · related: {c.get('related', 0)} · "
        f"novel: {c.get('novel', 0)} · synergy bundles: {c.get('bundles', 0)}\n"
    )

    out.append("## Feature × coverage\n")
    out.append(
        "| Feature | Pillar | Coverage | Concept | Novelty | Leverage | Sources |"
    )
    out.append("|---|---|---|---|---:|---:|---:|")
    for r in matrix.rows[:MAX_PERSISTED_ROWS]:
        out.append(
            f"| {_clip(r.name, 48)} | {r.pillar or '—'} | {r.coverage} | "
            f"{r.concept_id or '—'} | {r.novelty_score:.2f} | "
            f"{r.leverage_score:.2f} | {len(r.sources)} |"
        )

    gaps = matrix.novel_gaps()
    out.append("\n## Novel gaps to implement (leverage-ranked)\n")
    if gaps:
        for r in gaps[:50]:
            tag = "novel" if r.coverage == "novel" else f"related→{r.concept_id}"
            out.append(
                f"- **{_clip(r.name, 64)}** ({tag}) — leverage {r.leverage_score:.2f}, "
                f"novelty {r.novelty_score:.2f}, {len(r.sources)} source(s)"
            )
    else:
        out.append("_No open gaps — everything ingested is already covered._")

    out.append(
        "\n## Cross-source synergies → novel unique implementations\n"
        "_Cross-pillar bundles: ideas that are individually known but TOGETHER are "
        "new — the combine-to-surpass candidates._\n"
    )
    if matrix.bundles:
        names = {r.feature_id: r.name for r in matrix.rows}
        for i, b in enumerate(matrix.bundles, 1):
            members = ", ".join(_clip(names.get(m, m), 40) for m in b.members[:6])
            out.append(f"{i}. **[{' + '.join(b.pillars)}]** {members}")
    else:
        out.append("_No cross-pillar synergy bundles detected this cycle._")

    out.append("\n## Per-source contribution\n")
    if matrix.source_index:
        names = {r.feature_id: r.name for r in matrix.rows}
        for src, fids in sorted(
            matrix.source_index.items(), key=lambda kv: len(kv[1]), reverse=True
        )[:40]:
            sample = ", ".join(_clip(names.get(f, f), 32) for f in fids[:4])
            more = f" (+{len(fids) - 4} more)" if len(fids) > 4 else ""
            out.append(f"- `{_clip(src, 40)}` → {len(fids)} feature(s): {sample}{more}")
    else:
        out.append("_No source provenance recorded on the features._")

    return "\n".join(out) + "\n"


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
                "concept": "KG-2.173",
            },
        )
        summary["persisted"] = True
    except Exception as exc:  # noqa: BLE001 — materialization is best-effort
        summary["persisted"] = False
        summary["error"] = str(exc)

    _persist_live_artifact(node_id, matrix, markdown, summary)
    summary["markdown"] = markdown
    return summary


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
                        "name": _clip(r.name, 80),
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


def _clip(text: str, n: int) -> str:
    text = str(text).replace("\n", " ").replace("|", "/").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


__all__ = [
    "FeatureMatrixRow",
    "FeatureMatrix",
    "build_feature_matrix",
    "render_markdown",
    "materialize",
    "MAX_PERSISTED_ROWS",
]
