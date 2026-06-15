"""Code → business-capability resolution (CONCEPT:KG-2.8).

Bridges code features (clusters of symbols discovered by the epistemic-graph
engine, enriched with LLM capability cards) to ArchiMate ``BusinessCapability``
nodes, emitting ``REALIZES`` edges. This is the link that lets a single query
answer "show all code implementing capability X" regardless of naming.

It deliberately lives in Python (not the Rust engine, which stays purely
syntactic). It is a pure function over value objects — no backend, no network —
so it is trivially testable and reusable.

Three modes, unified in one call (composable via flags):

1. **Match existing** (top-down): match a feature to an already-ingested
   ``BusinessCapability`` (from LeanIX/Archi) by semantic similarity, with a
   deterministic token-overlap fallback when no embedder is supplied.
2. **Mint bottom-up**: when nothing matches and ``mint_missing`` is set, derive a
   new provisional ``BusinessCapability`` from the feature (post-merger
   codebases whose EA catalog is sparse). Provisional capabilities can be pushed
   back to LeanIX/Archi via the :mod:`writeback` capability sink.
3. **Curated registry**: pass a curated capability list as ``registry`` and set
   ``mint_missing=False`` to match strictly against an authored catalog.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from .models import EnrichmentEdge, GraphNode

# Optional embedder: text -> vector. When absent we fall back to token overlap.
EmbedFn = Callable[[str], list[float]]

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "for",
    "in",
    "on",
    "with",
    "this",
    "that",
    "is",
    "are",
    "by",
    "as",
    "it",
    "from",
    "into",
    "via",
    "code",
    "module",
    "function",
    "class",
    "feature",
    "handles",
    "provides",
}


def _slug(text: str) -> str:
    """Stable, id-safe slug from arbitrary text."""
    return "-".join(_TOKEN_RE.findall(text.lower()))[:64] or "unnamed"


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOPWORDS}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _cosine(u: list[float], v: list[float]) -> float:
    if not u or not v or len(u) != len(v):
        return 0.0
    dot = sum(x * y for x, y in zip(u, v, strict=False))
    nu = sum(x * x for x in u) ** 0.5
    nv = sum(y * y for y in v) ** 0.5
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return dot / (nu * nv)


def _text_of(obj: Any) -> str:
    """Join the name + summary of a feature or capability for matching."""
    name = obj.get("name") if isinstance(obj, dict) else getattr(obj, "name", "")
    summary = (
        obj.get("summary") if isinstance(obj, dict) else getattr(obj, "summary", "")
    )
    return f"{name or ''} {summary or ''}".strip()


def _cap_id(cap: Any) -> str:
    val = cap.get("id") if isinstance(cap, dict) else getattr(cap, "id", "")
    return str(val) if val else ""


def _score(
    feature_text: str,
    cap_text: str,
    embed_fn: EmbedFn | None,
    _cache: dict[str, list[float]],
) -> float:
    """Similarity in [0, 1]: cosine over embeddings, else token Jaccard."""
    if embed_fn is not None:
        fv = _cache.get(feature_text)
        if fv is None:
            fv = _cache[feature_text] = embed_fn(feature_text)
        cv = _cache.get(cap_text)
        if cv is None:
            cv = _cache[cap_text] = embed_fn(cap_text)
        return _cosine(fv, cv)
    return _jaccard(_tokens(feature_text), _tokens(cap_text))


def resolve_realizes(
    features: list[Any],
    capabilities: list[Any] | None = None,
    *,
    registry: list[Any] | None = None,
    mint_missing: bool = True,
    match_threshold: float = 0.2,
    embed_fn: EmbedFn | None = None,
) -> tuple[list[GraphNode], list[EnrichmentEdge]]:
    """Resolve features to capabilities, returning (minted_nodes, REALIZES edges).

    ``capabilities`` and ``registry`` are lists of dict/attr objects exposing
    ``id``/``name``/``summary`` (e.g. existing ``BusinessCapability`` nodes and a
    curated catalog). Registry candidates are preferred on ties. Edges target
    ``capability:{id}`` ids; minted capabilities use ``capability:derived:{slug}``.
    """
    candidates: list[Any] = list(registry or []) + list(capabilities or [])
    minted: list[GraphNode] = []
    minted_slugs: set[str] = set()
    edges: list[EnrichmentEdge] = []
    cache: dict[str, list[float]] = {}

    for feat in features:
        feat_id = feat.get("id") if isinstance(feat, dict) else getattr(feat, "id", "")
        if not feat_id:
            continue
        feat_text = _text_of(feat)

        best_cap: Any = None
        best_score = match_threshold
        for cap in candidates:
            cap_id = _cap_id(cap)
            if not cap_id:
                continue
            s = _score(feat_text, _text_of(cap), embed_fn, cache)
            # Qualify at >= threshold; replace only on a strictly higher score so
            # earlier candidates (registry is listed first) win on ties.
            if s >= match_threshold and (best_cap is None or s > best_score):
                best_score = s
                best_cap = cap

        if best_cap is not None:
            target = _cap_id(best_cap)
            edges.append(
                EnrichmentEdge(source=feat_id, target=target, rel_type="REALIZES")
            )
            continue

        if not mint_missing:
            continue

        # Bottom-up mint: derive a provisional capability from the feature.
        name = (
            feat.get("name") if isinstance(feat, dict) else getattr(feat, "name", "")
        ) or feat_id
        slug = _slug(name)
        cap_node_id = f"capability:derived:{slug}"
        if slug not in minted_slugs:
            minted_slugs.add(slug)
            summary = (
                feat.get("summary")
                if isinstance(feat, dict)
                else getattr(feat, "summary", "")
            )
            minted.append(
                GraphNode(
                    id=cap_node_id,
                    type="BusinessCapability",
                    props={
                        "name": name,
                        "summary": summary or "",
                        "derived_from": "code",
                        "provisional": True,
                    },
                )
            )
        edges.append(
            EnrichmentEdge(source=feat_id, target=cap_node_id, rel_type="REALIZES")
        )

    return minted, edges
