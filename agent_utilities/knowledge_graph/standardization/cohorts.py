#!/usr/bin/python
from __future__ import annotations

"""Consolidation candidate adapter (CONCEPT:AU-KG.ontology.populated-at-import-real-3).

Reads the three already-materialized graph structures that signal redundancy and
normalizes them into one :class:`CandidateGroup` shape the decision engine ranks:

  1. **Capability cohorts** — governed assets sharing the vendor-neutral
     ``capability`` tag from ``>= 2`` distinct sources/vendors (the egeria
     reconcile P22 capability-cohort signal, e.g. ServiceNow+ERPNext = ITSM,
     GitLab+GitHub = vcs). Computed natively from node properties so it works
     whether or not egeria has written cohort collection nodes.
  2. **Dedup SUPERSEDES clusters** — connected components over ``SUPERSEDES``
     edges (written by ``assimilation/dedup.py``), recovered with the *same*
     ``dedup._clusters`` union-find helper.
  3. **Synergy bundles** — components over ``HAS_SYNERGY_WITH`` edges (written by
     ``assimilation/synergy.py``).

Edge reads use the bulk ``iter_all_edges`` traversal (one round-trip; the
live-backend scaling fix) and fall back cleanly on minimal test doubles.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ...models.knowledge_graph import RegistryNodeType
from ..assimilation.dedup import _clusters, iter_all_edges
from .standards import applicable_standard

# The engine's own output node types — never scored/grouped as governed assets
# (else a re-run would treat its own recommendations as consolidatable, CONCEPT:AU-KG.ontology.populated-at-import-real-3).
_ENGINE_OUTPUT_TYPES = {
    RegistryNodeType.ENTERPRISE_STANDARD.value,
    RegistryNodeType.CONSOLIDATION_RECOMMENDATION.value,
    RegistryNodeType.DRIFT_ROLLUP.value,
}


@dataclass
class CandidateGroup:
    """A normalized set of assets that may be consolidatable.

    Attributes:
        members: KG node ids in the group.
        kind: ``retire_tool`` (cross-vendor capability cohort), ``merge_codebases``
            (dedup SUPERSEDES cluster), or ``build_consolidator`` (synergy bundle).
        capability: The shared vendor-neutral capability, when known.
        sources: Distinct source/vendor strings represented (redundancy breadth).
        origin: Which signal produced the group (``cohort``/``dedup``/``synergy``).
    """

    members: list[str]
    kind: str
    capability: str | None = None
    sources: set[str] = field(default_factory=set)
    origin: str = ""

    def key(self) -> str:
        """Stable identity for the group (sorted member set)."""
        return "|".join(sorted(self.members))


def _asset_source(data: dict[str, Any]) -> str:
    """Best-effort vendor/source identifier for cross-vendor cohort breadth."""
    for k in ("vendor", "source_system", "source", "tool", "domain"):
        v = data.get(k)
        if isinstance(v, str) and v and v != "system":
            return v.lower()
    return ""


def read_assets(engine: Any) -> dict[str, dict[str, Any]]:
    """Map ``id -> asset dict`` for every governed asset, enriched with link types.

    A governed asset is one that :func:`applicable_standard` routes to a standard.
    The returned dict mirrors the node ``data`` (so properties like ``capability``,
    ``owner`` are top-level) plus an ``id``, the resolved ``standard`` name, and a
    ``link_types`` set of its outgoing edge types (so
    :meth:`Interface.gaps_for` can evaluate link constraints).
    """
    graph = getattr(engine, "graph", None)
    if graph is None:
        return {}
    try:
        node_iter = list(graph.nodes(data=True))
    except TypeError:  # pragma: no cover - non-standard graph
        return {}

    assets: dict[str, dict[str, Any]] = {}
    for nid, data in node_iter:
        if not isinstance(data, dict):
            continue
        if str(data.get("type", "") or "").lower() in _ENGINE_OUTPUT_TYPES:
            continue
        std = applicable_standard(data)
        if std is None:
            continue
        asset = dict(data)
        asset["id"] = nid
        asset["standard"] = std
        asset.setdefault("type", data.get("type", ""))
        asset["link_types"] = set()
        assets[nid] = asset

    # Enrich outgoing link types in one bulk traversal where available.
    edges = iter_all_edges(graph)
    if edges is not None:
        for src, _dst, props in edges:
            if src in assets:
                rel = ""
                if isinstance(props, dict):
                    rel = str(props.get("_rel", "") or "").lower()
                if rel:
                    assets[src]["link_types"].add(rel)
    return assets


def _component_clusters(edges, rel_label: str) -> list[list[str]]:
    """Connected components over edges whose ``_rel`` marker equals ``rel_label``."""
    if not edges:
        return []
    pairs = []
    nodes: set[str] = set()
    for src, dst, props in edges:
        if not isinstance(props, dict):
            continue
        if str(props.get("_rel", "") or "").upper() == rel_label.upper():
            pairs.append((src, dst, 1.0))
            nodes.add(src)
            nodes.add(dst)
    return _clusters(sorted(nodes), pairs)


def capability_cohorts(
    assets: dict[str, dict[str, Any]], *, min_sources: int = 2
) -> list[CandidateGroup]:
    """Group governed assets by shared ``capability`` across ``>= min_sources``.

    This is the P22 capability-cohort signal computed natively: same capability,
    multiple distinct vendors/sources ⇒ redundant tooling that may collapse to a
    single platform.
    """
    by_cap: dict[str, list[str]] = defaultdict(list)
    cap_sources: dict[str, set[str]] = defaultdict(set)
    for nid, data in assets.items():
        cap = str(data.get("capability", "") or "").lower()
        if not cap:
            continue
        by_cap[cap].append(nid)
        src = _asset_source(data)
        if src:
            cap_sources[cap].add(src)

    groups: list[CandidateGroup] = []
    for cap, members in by_cap.items():
        sources = cap_sources.get(cap, set())
        if len(members) >= 2 and len(sources) >= min_sources:
            groups.append(
                CandidateGroup(
                    members=sorted(members),
                    kind="retire_tool",
                    capability=cap,
                    sources=set(sources),
                    origin="cohort",
                )
            )
    return groups


def read_cohorts(
    engine: Any,
    *,
    assets: dict[str, dict[str, Any]] | None = None,
    min_sources: int = 2,
) -> list[CandidateGroup]:
    """Return all consolidation candidate groups from the three redundancy signals.

    Args:
        engine: knowledge engine (``graph.nodes`` + bulk ``edges``).
        assets: optional pre-read governed-asset map (avoids a second scan).
        min_sources: minimum distinct vendors for a capability cohort.
    """
    assets = assets if assets is not None else read_assets(engine)
    groups = capability_cohorts(assets, min_sources=min_sources)

    graph = getattr(engine, "graph", None)
    edges = iter_all_edges(graph) if graph is not None else None

    for cluster in _component_clusters(edges, "SUPERSEDES"):
        groups.append(
            CandidateGroup(
                members=sorted(cluster),
                kind="merge_codebases",
                capability=_dominant_capability(cluster, assets),
                sources={_asset_source(assets[m]) for m in cluster if m in assets}
                - {""},
                origin="dedup",
            )
        )
    for cluster in _component_clusters(edges, "HAS_SYNERGY_WITH"):
        groups.append(
            CandidateGroup(
                members=sorted(cluster),
                kind="build_consolidator",
                capability=_dominant_capability(cluster, assets),
                sources={_asset_source(assets[m]) for m in cluster if m in assets}
                - {""},
                origin="synergy",
            )
        )
    return groups


def _dominant_capability(
    members: list[str], assets: dict[str, dict[str, Any]]
) -> str | None:
    """Most common capability among members present in the asset map."""
    counts: dict[str, int] = defaultdict(int)
    for m in members:
        cap = str(assets.get(m, {}).get("capability", "") or "").lower()
        if cap:
            counts[cap] += 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]


__all__ = [
    "CandidateGroup",
    "read_assets",
    "read_cohorts",
    "capability_cohorts",
]
