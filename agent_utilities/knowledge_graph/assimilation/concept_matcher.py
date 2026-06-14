#!/usr/bin/python
from __future__ import annotations

"""Robust research → ecosystem-Concept matcher (CONCEPT:KG-2.75).

The first gap matcher (:func:`gap_analysis.auto_satisfy`) recognised a built
capability only when a feature *cited its concept id*, with a single weak cosine
fallback its own docstring measured at "0/21 known-built capabilities … argmax
wrong 71%". So **external research papers — which never cite our internal
``CONCEPT:`` ids — matched nothing**, and every paper looked like an open gap no
matter how much of the ecosystem was ingested.

This is the robust replacement: a multi-signal, defense-in-depth matcher that
decides, for each feature (research ``Article`` / ``sdd_feature`` / ``capability``)
against the ecosystem ``Concept`` registry, whether the feature's contribution is

* **covered** — we already built this capability (→ ``SATISFIED_BY`` edge), or
* **related** — relevant but novel (→ ``RELATES_TO`` edge; stays an open gap), or
* **unrelated**.

Stages (precision/recall layered):

1. **explicit id** — feature declares a concept id that exists → ``covered`` 1.0.
2. **embedding retrieval** — top-K concept candidates at a *recall* threshold
   (a candidate generator, NOT a decision — the old fallback's mistake).
3. **LLM judge** — an LLM adjudicates each recalled (feature, concept) pair
   (covered / related / unrelated + confidence + rationale); cached, bounded,
   degrades to a deterministic cosine verdict when no LLM is reachable.
4. **fusion** — combine cosine + verdict into a decision, a per-feature
   ``novelty_score`` (1 − best match strength) and ``coverage``.

All deps (``embed_fn``, ``llm_judge_fn``) are injectable so the logic is unit
testable without a live model. Idempotent on a durable graph (auto edges are
cleared and rewritten, never accumulated). Reuses the canonical id helpers,
``GapReport``, and the numpy cosine path from :mod:`gap_analysis`; the embedder
(``make_embed_fn``) and lite LLM (``make_lite_llm_fn``) from enrichment — no new
model is loaded and no env flag is added (config discipline).

Concept: concept-matcher
"""

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from ...models.knowledge_graph import RegistryEdgeType
from .dedup import _cosine, iter_all_edges
from .gap_analysis import (
    GapReport,
    _collect_rich,
    _concept_key,
    _feature_refs,
    _rel_of,
)
from .ingest import content_fingerprint

logger = logging.getLogger(__name__)

Verdict = Literal["covered", "related", "unrelated"]

# --- calibrated thresholds (module constants — config discipline, no env knobs) --
#: cosine ≥ this makes a concept a *candidate* the judge will adjudicate (recall).
RETRIEVAL_THRESHOLD = 0.45
#: concepts retrieved per feature (bounds judge cost).
TOP_K = 6
#: LLM "covered" verdict needs at least this confidence to close a feature.
JUDGE_ACCEPT = 0.6
#: deterministic fallback (no LLM): cosine ≥ this → covered; ≥ related → related.
COVERED_COSINE = 0.82
RELATED_COSINE = 0.6

# texts -> embeddings (batched); matches enrichment.semantic.EmbedFn
EmbedFn = Callable[[list[str]], list[list[float]]]
# prompt -> completion; matches enrichment.cards.LLMFn
LLMFn = Callable[[str], str]

_RELATES_TO = "RELATES_TO"  # raw label (enrichment convention; NOT a closing edge)


@dataclass
class Match:
    """One adjudicated (feature → concept) candidate."""

    concept_id: str
    cosine: float
    verdict: Verdict
    confidence: float
    score: float  # fused decision strength in [0, 1]
    method: str  # "id" | "llm_judge" | "cosine"
    rationale: str = ""


@dataclass
class FeatureMatch:
    """The matcher's decision for one feature."""

    feature_id: str
    decision: Verdict
    best: Match | None
    novelty_score: float  # 1 − best match strength (1.0 = fully novel)
    matches: list[Match] = field(default_factory=list)


@dataclass
class MatchReport(GapReport):
    """``GapReport`` + the richer matcher outcome (covered vs related vs novel)."""

    related: int = 0  # RELATES_TO edges written (novel-but-relevant)
    unrelated: int = 0
    used_llm: bool = False
    feature_matches: list[FeatureMatch] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# LLM judge
# --------------------------------------------------------------------------- #
_JUDGE_INSTRUCTION = (
    "You compare a RESEARCH ITEM against ONE capability already built in our "
    "system. Decide the relationship between the research item's core "
    "contribution and the existing capability.\n"
    "- covered: the research item's contribution is essentially the SAME "
    "capability we already have (we already built this).\n"
    "- related: same area / relevant, but the research item adds something "
    "genuinely novel beyond the existing capability.\n"
    "- unrelated: different topic.\n"
    'Reply with ONLY compact JSON: {"verdict":"covered|related|unrelated",'
    '"confidence":0.0-1.0,"why":"one short sentence"}'
)


def _judge_prompt(feature_text: str, concept_text: str) -> str:
    return (
        f"{_JUDGE_INSTRUCTION}\n\n"
        f"EXISTING CAPABILITY:\n{concept_text[:1200]}\n\n"
        f"RESEARCH ITEM:\n{feature_text[:2000]}\n\nJSON:"
    )


def _parse_judge(text: str) -> tuple[Verdict, float, str]:
    """Lenient parse of the judge's JSON; degrades safely to ``unrelated``."""
    if not text:
        return "unrelated", 0.0, ""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    raw = m.group(0) if m else text
    try:
        d = json.loads(raw)
        v = str(d.get("verdict", "")).strip().lower()
        if v not in ("covered", "related", "unrelated"):
            v = "unrelated"
        conf = float(d.get("confidence", 0.0))
        conf = max(0.0, min(1.0, conf))
        return v, conf, str(d.get("why", ""))[:240]  # type: ignore[return-value]
    except (json.JSONDecodeError, ValueError, TypeError):
        low = text.lower()
        for v in ("covered", "related", "unrelated"):
            if v in low:
                return v, 0.5, ""  # type: ignore[return-value]
        return "unrelated", 0.0, ""


def _feature_text(data: dict[str, Any]) -> str:
    parts = [
        str(data.get("name") or data.get("title") or ""),
        str(data.get("summary") or ""),
        str(data.get("content") or data.get("abstract") or ""),
    ]
    return " — ".join(p for p in parts if p).strip()


def _concept_text(data: dict[str, Any]) -> str:
    parts = [
        str(data.get("concept_id") or data.get("id") or ""),
        str(data.get("name") or ""),
        str(data.get("description") or data.get("content") or data.get("doc") or ""),
        str(data.get("pillar") or ""),
    ]
    return " — ".join(p for p in parts if p).strip()


# --------------------------------------------------------------------------- #
# matcher
# --------------------------------------------------------------------------- #
class ConceptMatcher:
    """Robust multi-signal feature → ecosystem-Concept matcher (CONCEPT:KG-2.75)."""

    def __init__(
        self,
        *,
        embed_fn: EmbedFn | None = None,
        llm_judge_fn: LLMFn | None = None,
        use_llm: bool = True,
        top_k: int = TOP_K,
        retrieval_threshold: float = RETRIEVAL_THRESHOLD,
        judge_accept: float = JUDGE_ACCEPT,
    ) -> None:
        self._embed_fn = embed_fn
        self._llm_fn = llm_judge_fn
        self._use_llm = use_llm
        self.top_k = top_k
        self.retrieval_threshold = retrieval_threshold
        self.judge_accept = judge_accept
        # per-instance judge cache: (feature_hash, concept_id) → (verdict, conf, why)
        self._judge_cache: dict[tuple[str, str], tuple[Verdict, float, str]] = {}

    # -- lazy deps (no model loaded until first real use) ------------------- #
    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self._embed_fn is None:
            from ..enrichment.semantic import make_embed_fn

            self._embed_fn = make_embed_fn()
        return self._embed_fn(texts)

    def _judge(self, feature_text: str, feature_hash: str, cid: str, ctext: str):
        key = (feature_hash, cid)
        if key in self._judge_cache:
            return self._judge_cache[key]
        if self._llm_fn is None:
            from ..enrichment.cards import make_lite_llm_fn

            self._llm_fn = make_lite_llm_fn()
        out = _parse_judge(self._llm_fn(_judge_prompt(feature_text, ctext)))
        self._judge_cache[key] = out
        return out

    # -- core decision for one feature ------------------------------------- #
    def match_feature(
        self,
        fid: str,
        fdata: dict[str, Any],
        *,
        concept_by_key: dict[str, str],
        concept_vecs: list[tuple[str, list[float]]],
        concept_text: dict[str, str],
        feature_vec: list[float] | None,
    ) -> FeatureMatch:
        # Stage 0 — explicit id (highest precision)
        for ref in _feature_refs(fid, fdata):
            cid = concept_by_key.get(ref)
            if cid:
                m = Match(cid, 1.0, "covered", 1.0, 1.0, "id", "declared concept id")
                return FeatureMatch(fid, "covered", m, 0.0, [m])

        # Stage 1 — embedding retrieval (recall: candidate generation only)
        candidates: list[tuple[str, float]] = []
        if feature_vec:
            candidates = _top_k_cosine(
                feature_vec, concept_vecs, self.top_k, self.retrieval_threshold
            )
        if not candidates:
            return FeatureMatch(fid, "unrelated", None, 1.0, [])

        ftext = _feature_text(fdata)
        fhash = content_fingerprint(ftext)
        matches: list[Match] = []
        # Stage 2 — adjudicate each candidate (LLM judge, else cosine fallback)
        for cid, cos in candidates:
            if self._use_llm:
                verdict, conf, why = self._judge(
                    ftext, fhash, cid, concept_text.get(cid, cid)
                )
                method = "llm_judge"
            else:
                verdict, conf, why = _cosine_verdict(cos)
                method = "cosine"
            # Stage 3 — fuse cosine retrieval signal with the verdict confidence.
            score = round(0.4 * cos + 0.6 * conf, 6) if verdict != "unrelated" else 0.0
            matches.append(Match(cid, round(cos, 6), verdict, conf, score, method, why))

        return _decide(fid, matches, self.judge_accept)

    # -- whole-graph pass (assimilate-stage entry; replaces auto_satisfy) --- #
    def satisfy(
        self,
        engine: Any,
        *,
        feature_types: tuple[str, ...],
        concept_types: tuple[str, ...],
        restrict_to: set[str] | None = None,
        write: bool = True,
        reconcile: bool = True,
    ) -> MatchReport:
        features = _collect_rich(engine, feature_types)
        concepts = _collect_rich(engine, concept_types)
        report = MatchReport(features=len(features), concepts=len(concepts))
        report.used_llm = self._use_llm
        if not features or not concepts:
            return report

        targets = set(features) if restrict_to is None else set(features) & restrict_to
        if write and reconcile and targets:
            _clear_auto(engine, targets, ("SATISFIED_BY", _RELATES_TO))

        concept_by_key, concept_vecs, concept_text = _build_concept_index(concepts)

        for fid, fdata in features.items():
            if restrict_to and fid not in restrict_to:
                continue
            fm = self.match_feature(
                fid,
                fdata,
                concept_by_key=concept_by_key,
                concept_vecs=concept_vecs,
                concept_text=concept_text,
                feature_vec=fdata.get("embedding"),
            )
            report.feature_matches.append(fm)
            if fm.best is None or fm.decision == "unrelated":
                report.unrelated += 1
                continue
            best = fm.best
            report.candidates.append((fid, best.concept_id, best.score))
            if fm.decision == "covered":
                report.satisfied += 1
                if write:
                    engine.link_nodes(
                        fid,
                        best.concept_id,
                        RegistryEdgeType.SATISFIED_BY,
                        properties={
                            "_rel": "SATISFIED_BY",
                            "score": best.score,
                            "auto": True,
                            "concept": best.concept_id,
                            "match": best.method,
                            "rationale": best.rationale,
                        },
                    )
            else:  # related (novel but relevant) — stays an open gap
                report.related += 1
                if write:
                    engine.link_nodes(
                        fid,
                        best.concept_id,
                        _RELATES_TO,
                        properties={
                            "_rel": _RELATES_TO,
                            "score": best.score,
                            "auto": True,
                            "concept": best.concept_id,
                            "match": best.method,
                            "rationale": best.rationale,
                            "novelty": fm.novelty_score,
                        },
                    )
        return report


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _build_concept_index(
    concepts: dict[str, dict[str, Any]],
) -> tuple[dict[str, str], list[tuple[str, list[float]]], dict[str, str]]:
    concept_by_key: dict[str, str] = {}
    concept_vecs: list[tuple[str, list[float]]] = []
    concept_text: dict[str, str] = {}
    for cid, cdata in concepts.items():
        key = _concept_key(cid, cdata)
        if key and key not in concept_by_key:
            concept_by_key[key] = cid
        emb = cdata.get("embedding")
        if emb:
            concept_vecs.append((cid, list(emb)))
        concept_text[cid] = _concept_text(cdata)
    return concept_by_key, concept_vecs, concept_text


def _top_k_cosine(
    fvec: list[float],
    concept_vecs: list[tuple[str, list[float]]],
    k: int,
    threshold: float,
) -> list[tuple[str, float]]:
    """Top-k concepts by cosine ≥ threshold. numpy matrix path, pure-Python fallback."""
    if not concept_vecs:
        return []
    try:
        import numpy as np

        cmat = np.asarray([v for _, v in concept_vecs], dtype=np.float32)
        norms = np.linalg.norm(cmat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        cmat = cmat / norms
        fv = np.asarray(fvec, dtype=np.float32)
        fnorm = float(np.linalg.norm(fv))
        if not fnorm:
            return []
        sims = cmat @ (fv / fnorm)
        cids = [c for c, _ in concept_vecs]
        idx = np.argsort(-sims)[:k]
        return [(cids[i], float(sims[i])) for i in idx if float(sims[i]) >= threshold]
    except Exception:  # noqa: BLE001 — numpy optional → pure-Python
        scored = [(cid, _cosine(fvec, cvec)) for cid, cvec in concept_vecs]
        scored.sort(key=lambda t: t[1], reverse=True)
        return [(c, s) for c, s in scored[:k] if s >= threshold]


def _cosine_verdict(cos: float) -> tuple[Verdict, float, str]:
    """Deterministic verdict from cosine alone (no-LLM degradation)."""
    if cos >= COVERED_COSINE:
        return "covered", cos, "high embedding similarity"
    if cos >= RELATED_COSINE:
        return "related", cos, "moderate embedding similarity"
    return "unrelated", 0.0, ""


def _decide(fid: str, matches: list[Match], judge_accept: float) -> FeatureMatch:
    """Fuse per-candidate verdicts into a single feature decision."""
    covered = [
        m for m in matches if m.verdict == "covered" and m.confidence >= judge_accept
    ]
    related = [m for m in matches if m.verdict == "related"]
    if covered:
        best = max(covered, key=lambda m: m.score)
        return FeatureMatch(fid, "covered", best, round(1.0 - best.score, 6), matches)
    if related:
        best = max(related, key=lambda m: m.score)
        # novel-but-relevant: novelty is high (related, not covered)
        return FeatureMatch(
            fid, "related", best, round(1.0 - 0.5 * best.score, 6), matches
        )
    return FeatureMatch(fid, "unrelated", None, 1.0, matches)


def _clear_auto(engine: Any, feature_ids: set[str], rels: tuple[str, ...]) -> int:
    """Remove prior auto-written edges of ``rels`` from ``feature_ids`` (idempotent)."""
    graph = getattr(engine, "graph", None)
    deleter = getattr(engine, "delete_edge", None)
    if graph is None or not callable(deleter):
        return 0
    pairs: list[tuple[str, str, str]] = []
    edges = iter_all_edges(graph)
    if edges is not None:
        for src, dst, props in edges:
            rel = _rel_of(props)
            if (
                src in feature_ids
                and rel in rels
                and isinstance(props, dict)
                and props.get("auto")
            ):
                pairs.append((src, dst, rel))
    else:
        for fid in feature_ids:
            try:
                for _s, dst, props in graph.out_edges(fid, data=True):
                    rel = _rel_of(props)
                    if isinstance(props, dict) and rel in rels and props.get("auto"):
                        pairs.append((fid, dst, rel))
            except (TypeError, AttributeError):  # pragma: no cover
                continue
    for src, dst, rel in pairs:
        try:
            deleter(src, dst, rel)
        except Exception:  # pragma: no cover - best-effort reconcile
            pass
    return len(pairs)


__all__ = ["ConceptMatcher", "Match", "FeatureMatch", "MatchReport", "Verdict"]
