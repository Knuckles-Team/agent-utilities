#!/usr/bin/python
from __future__ import annotations

"""Auto gap analysis — the "stop rediscovering built features" engine (CONCEPT:KG-2.7).

The first evolution attempt repeatedly re-proposed features we had *already built*,
because "does this already exist?" was answered by re-reading. Here it is a graph
operation: every extracted feature is embedding-matched against our existing
``Concept`` nodes; a match above threshold writes a candidate
``feature -[SATISFIED_BY]-> concept`` edge. ``open_features`` is then the gap query —
features that are neither satisfied, superseded, nor closed by status — and that is
the *only* set the golden loop should propose against.

Backend-portable: closing edges are detected via the ``_rel`` property marker that
the assimilation engine stamps on its edges (``out_edges``/``in_edges`` expose
properties, not the relationship label), plus the node's own ``status`` field.

Concept: gap-analysis
"""

import re
from dataclasses import dataclass, field
from typing import Any

from ...models.knowledge_graph import RegistryEdgeType, RegistryNodeType
from .dedup import _cosine

# A concept id like KG-2.7 / AHE-3.12 / ORCH-1.3b / OS-5 / KG-2.20g.
_CONCEPT_ID_RE = re.compile(r"\b([A-Z]{2,6}-\d+(?:\.\d+[a-z]?|-\d+)?)\b")

_FEATURE_TYPES: tuple[str, ...] = (
    RegistryNodeType.SDD_FEATURE.value,
    RegistryNodeType.CAPABILITY.value,
    RegistryNodeType.ARTICLE.value,
)
_CONCEPT_TYPES: tuple[str, ...] = (RegistryNodeType.CONCEPT.value,)
# A feature is "closed" (excluded from the cycle) by any of these statuses…
_CLOSED_STATUS = {"satisfied", "implemented", "rejected", "superseded", "done"}
# …or by these incident closing edges (matched on the `_rel` property marker).
_CLOSING_OUT = {"SATISFIED_BY", "DERIVED_FROM_RESEARCH"}
_CLOSING_IN = {"SUPERSEDES"}


@dataclass
class GapReport:
    features: int = 0
    concepts: int = 0
    satisfied: int = 0  # candidate SATISFIED_BY edges written
    candidates: list[tuple[str, str, float]] = field(default_factory=list)


def _canonical_id(raw: str) -> str:
    """Normalize a concept id token: ``kg-2-14`` / ``kg-2.14`` → ``KG-2.14``."""
    s = str(raw).strip()
    m = re.match(r"^([A-Za-z]{2,6})-(\d+)[.\-](\d+[a-z]?)$", s)
    if m:
        return f"{m.group(1).upper()}-{m.group(2)}.{m.group(3)}"
    m2 = re.match(r"^([A-Za-z]{2,6})-(\d+[a-z]?)$", s)  # bare major, e.g. OS-5
    return f"{m2.group(1).upper()}-{m2.group(2)}" if m2 else s.upper()


def _concept_key(nid: str, data: dict[str, Any]) -> str | None:
    """The canonical concept id a *concept node* represents (id field or its key)."""
    for cand in (data.get("concept_id"), data.get("id"), nid, data.get("name", "")):
        if cand:
            key = _canonical_id(str(cand))
            if re.match(r"^[A-Z]{2,6}-\d", key):
                return key
    return None


def _feature_refs(nid: str, data: dict[str, Any]) -> list[str]:
    """Canonical concept ids a feature **declares as its own identity**, ordered.

    Sourced — in priority order — from the curated ``concept_ids`` property, then
    the feature's own id, then its name/title. **Body/abstract prose is NOT
    scanned**: a research plan that *cites* ``ORCH-1.0`` as related work is not the
    same capability as ORCH-1.0, so scraping body text mis-marks new research as
    already-built. Matching on declared identity keeps precision high.
    """
    refs: list[str] = []

    def _add(s: Any) -> None:
        for m in _CONCEPT_ID_RE.findall(str(s).upper()):
            c = _canonical_id(m)
            if re.match(r"^[A-Z]{2,6}-\d", c) and c not in refs:
                refs.append(c)

    cids = data.get("concept_ids")
    if isinstance(cids, list | tuple):
        for c in cids:
            _add(c)
    _add(nid)
    _add(data.get("name", ""))
    _add(data.get("title", ""))
    return refs


def _collect_rich(
    engine: Any, node_types: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    """id → full node ``data`` for the target types (case-insensitive label match)."""
    out: dict[str, dict[str, Any]] = {}
    graph = getattr(engine, "graph", None)
    if graph is None:
        return out
    try:
        node_iter = graph.nodes(data=True)
    except TypeError:  # pragma: no cover - non-standard graph
        return out
    wanted = {t.lower() for t in node_types}
    for nid, data in node_iter:
        if isinstance(data, dict) and str(data.get("type", "")).lower() in wanted:
            out[nid] = data
    return out


def auto_satisfy(
    engine: Any,
    *,
    feature_types: tuple[str, ...] = _FEATURE_TYPES,
    concept_types: tuple[str, ...] = _CONCEPT_TYPES,
    threshold: float = 0.85,
    restrict_to: set[str] | None = None,
    write: bool = True,
) -> GapReport:
    """Write candidate ``SATISFIED_BY`` edges for features matching built concepts.

    **Hybrid match (CONCEPT:KG-2.7).** A feature is matched to a built concept by,
    in order:

    1. **explicit id reference** — the feature names a concept id (``concept_ids``
       property or a ``CONCEPT:<ID>`` / bare-id marker in its id/title/body) that
       exists among the concept nodes → exact, score ``1.0``;
    2. **embedding fallback** — otherwise the best concept by cosine ≥ ``threshold``.

    Pure-cosine matching was found to recognize 0/21 known-built capabilities
    (a terse concept vs. a verbose spec sits below the embedding noise floor, and
    the argmax concept was wrong 71% of the time), so the id signal — which the
    specs already carry — leads, with embedding only for unreferenced features.

    Args:
        engine: knowledge engine (``graph.nodes(data=True)`` + ``link_nodes``).
        feature_types / concept_types: node types to match between.
        threshold: cosine ≥ this → satisfied (embedding fallback only).
        restrict_to: only evaluate these feature ids (incremental).
        write: persist edges (False = dry run).

    Returns:
        A :class:`GapReport` (the candidate ``(feature, concept, score)`` matches).
    """
    features = _collect_rich(engine, feature_types)
    concepts = _collect_rich(engine, concept_types)
    report = GapReport(features=len(features), concepts=len(concepts))
    if not features or not concepts:
        return report

    # Concept indices: canonical id → node id, and node id → embedding.
    concept_by_key: dict[str, str] = {}
    concept_vecs: list[tuple[str, list[float]]] = []
    for cid, cdata in concepts.items():
        key = _concept_key(cid, cdata)
        if key and key not in concept_by_key:
            concept_by_key[key] = cid
        emb = cdata.get("embedding")
        if emb:
            concept_vecs.append((cid, list(emb)))

    for fid, fdata in features.items():
        if restrict_to and fid not in restrict_to:
            continue
        best_cid: str | None = None
        best_s = 0.0
        via = ""
        # 1) explicit id reference (high precision)
        hit = next(
            (concept_by_key[r] for r in _feature_refs(fid, fdata) if r in concept_by_key),
            None,
        )
        if hit is not None:
            best_cid, best_s, via = hit, 1.0, "id"
        else:
            # 2) embedding fallback
            fvec = fdata.get("embedding")
            if fvec:
                fv = list(fvec)
                for cid, cvec in concept_vecs:
                    s = _cosine(fv, cvec)
                    if s > best_s:
                        best_cid, best_s = cid, s
                via = "embedding"
        if best_cid is not None and (via == "id" or best_s >= threshold):
            report.satisfied += 1
            report.candidates.append((fid, best_cid, round(best_s, 6)))
            if write:
                engine.link_nodes(
                    fid,
                    best_cid,
                    RegistryEdgeType.SATISFIED_BY,
                    properties={
                        "_rel": "SATISFIED_BY",
                        "score": round(best_s, 6),
                        "auto": True,
                        "concept": best_cid,
                        "match": via,
                    },
                )
    return report


def _rel_of(props: Any) -> str:
    return str(props.get("_rel", "")) if isinstance(props, dict) else ""


def is_closed(engine: Any, feature_id: str, status: str = "") -> bool:
    """True if ``feature_id`` is satisfied/superseded or closed by status.

    When ``status`` is not supplied, the node's stored ``status`` is consulted, so
    ``is_closed(engine, fid)`` is self-sufficient.
    """
    graph = getattr(engine, "graph", None)
    if not status and graph is not None:
        try:
            for nid, data in graph.nodes(data=True):
                if nid == feature_id and isinstance(data, dict):
                    status = str(data.get("status", ""))
                    break
        except TypeError:  # pragma: no cover
            pass
    if (status or "").lower() in _CLOSED_STATUS:
        return True
    if graph is None:
        return False
    try:
        for _s, _t, props in graph.out_edges(feature_id, data=True):
            if _rel_of(props) in _CLOSING_OUT:
                return True
        for _s, _t, props in graph.in_edges(feature_id, data=True):
            if _rel_of(props) in _CLOSING_IN:
                return True
    except (TypeError, AttributeError):  # pragma: no cover - non-standard graph
        return False
    return False


def open_features(
    engine: Any,
    *,
    feature_types: tuple[str, ...] = _FEATURE_TYPES,
) -> list[str]:
    """Return feature ids with no closing edge / closed status — the cycle's input.

    This is the durable, queryable answer to "what have we NOT already hit?" — the
    set the golden loop proposes against (everything else is excluded).
    """
    out: list[str] = []
    graph = getattr(engine, "graph", None)
    if graph is None:
        return out
    try:
        node_iter = graph.nodes(data=True)
    except TypeError:  # pragma: no cover
        return out
    wanted = {t.lower() for t in feature_types}  # case-insensitive (live labels)
    for nid, data in node_iter:
        if (
            not isinstance(data, dict)
            or str(data.get("type", "")).lower() not in wanted
        ):
            continue
        if not is_closed(engine, nid, str(data.get("status", "open"))):
            out.append(nid)
    return out


__all__ = ["GapReport", "auto_satisfy", "open_features", "is_closed"]
