#!/usr/bin/python
from __future__ import annotations

"""Reasoning-aware reranking of retrieved candidates.

CONCEPT:KG-2.6 — Retrieval Quality (reranking stage)

A second-stage reranker that reorders an over-fetched candidate pool by
query-relevance before it is capped to the context window, so the nodes that
actually drive multi-hop context assembly are the most relevant ones. Distilled
from the MemReranker research (``.specify/specs/research-evolution-20260606/``
plan b4-02): calibrated [0,1] five-level scores, instruction-awareness, and a
prior-blended ranking that fuses the first-stage retrieval score with a finer
relevance signal.

The scoring backend is pluggable via :class:`RerankScorer`. The default
:class:`LexicalRelevanceScorer` is deterministic and dependency-free (no model,
no network) so it is always available and unit-testable; a distilled
cross-encoder can be dropped in later behind the same protocol without touching
the retrieval path.

Concept: retrieval-reranking
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

_WORD = re.compile(r"[a-z0-9]+")
# A tiny stoplist so query/document overlap reflects content words, not glue.
_STOP = frozenset(
    "a an the of to in on for and or is are be was were with as at by from this that "
    "it its into about over under what which who whom how why when where".split()
)


def _content_tokens(text: str) -> list[str]:
    return [t for t in _WORD.findall((text or "").lower()) if t not in _STOP]


def _bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    return set(zip(tokens, tokens[1:], strict=False)) if len(tokens) > 1 else set()


def calibrate(score: float) -> float:
    """Snap a continuous [0,1] score to the nearest of five calibrated levels."""
    if score >= 0.875:
        return 1.0
    if score >= 0.625:
        return 0.75
    if score >= 0.375:
        return 0.5
    if score >= 0.125:
        return 0.25
    return 0.0


@runtime_checkable
class RerankScorer(Protocol):
    """Protocol for a (query, candidate-text) relevance scorer."""

    name: str

    def score(self, query: str, text: str, instruction: str = "") -> float:
        """Return a relevance score in [0, 1]."""
        ...


@dataclass
class LexicalRelevanceScorer:
    """Deterministic, dependency-free relevance scorer.

    Blends query→document token containment, Jaccard overlap, and phrase
    (bigram) overlap into a continuous [0,1] relevance, then folds in
    instruction-term coverage when an instruction is supplied (instruction
    awareness). No model or network — always available.

    CONCEPT:KG-2.6
    Concept: retrieval-reranking
    """

    name: str = "lexical_relevance"
    containment_weight: float = 0.55
    jaccard_weight: float = 0.25
    bigram_weight: float = 0.20
    instruction_weight: float = 0.30

    def score(self, query: str, text: str, instruction: str = "") -> float:
        q = _content_tokens(query)
        d = _content_tokens(text)
        if not q or not d:
            return 0.0
        qs, ds = set(q), set(d)
        inter = qs & ds
        containment = len(inter) / len(qs)
        jaccard = len(inter) / len(qs | ds)
        qb, db = _bigrams(q), _bigrams(d)
        bigram = (len(qb & db) / len(qb)) if qb else 0.0

        raw = (
            self.containment_weight * containment
            + self.jaccard_weight * jaccard
            + self.bigram_weight * bigram
        )

        if instruction.strip():
            it = set(_content_tokens(instruction))
            if it:
                coverage = len(it & ds) / len(it)
                raw = (
                    1.0 - self.instruction_weight
                ) * raw + self.instruction_weight * coverage

        return max(0.0, min(1.0, raw))


def _auto_scorer() -> RerankScorer:
    """Default rerank scorer: a neural cross-encoder when one is installed AND loads,
    else the deterministic lexical proxy (CONCEPT:KG-2.85). Auto-detection — no flag —
    so the best available cross-encoder is used natively. The neural scorer is *probed*
    once here; if the model can't load or score (offline / not cached / no GPU), we fall
    back to the zero-infra lexical scorer so retrieval never breaks.
    """
    from .neural_reranker import build_rerank_scorer

    scorer = build_rerank_scorer()
    if type(scorer).__name__ == "LexicalRelevanceScorer":
        return scorer
    try:
        scorer.score("probe", "probe")  # force model load + verify it scores
    except Exception:  # pragma: no cover - model absent/unloadable -> safe fallback
        return LexicalRelevanceScorer()
    return scorer


@dataclass
class ReasoningAwareReranker:
    """Reorder retrieved candidates by a prior-blended, calibrated relevance.

    For each candidate the final ranking score fuses the first-stage retrieval
    prior (e.g. the vector ``_score``, normalised across the pool) with the
    second-stage :class:`RerankScorer` relevance. Candidates are sorted by the
    continuous blend; each is annotated with ``_rerank_score`` (continuous) and
    ``_rerank_level`` (calibrated five-level) for downstream thresholding.

    CONCEPT:KG-2.6
    Concept: retrieval-reranking
    """

    scorer: RerankScorer = field(default_factory=_auto_scorer)
    prior_weight: float = 0.4
    prior_key: str = "_score"

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        text_fn: Callable[[dict[str, Any]], str] | None = None,
        instruction: str = "",
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return ``candidates`` reordered by blended relevance (descending).

        Args:
            query: The search query.
            candidates: Candidate node dicts (mutated in place with rerank annotations).
            text_fn: Extracts the scoreable text from a candidate (defaults to common keys).
            instruction: Optional task/intent string for instruction-aware scoring.
            top_k: If set, return only the top-k after reranking.
        """
        if len(candidates) <= 1:
            return candidates[:top_k] if top_k is not None else candidates

        text_of = text_fn or _default_text
        max_prior = max(
            (float(c.get(self.prior_key, 0.0) or 0.0) for c in candidates), default=0.0
        )

        scored: list[tuple[float, int, dict[str, Any]]] = []
        for idx, cand in enumerate(candidates):
            relevance = self.scorer.score(query, text_of(cand), instruction)
            prior = float(cand.get(self.prior_key, 0.0) or 0.0)
            prior_norm = (prior / max_prior) if max_prior > 0 else 0.0
            final = (
                self.prior_weight * prior_norm + (1.0 - self.prior_weight) * relevance
            )
            cand["_rerank_score"] = round(final, 6)
            cand["_rerank_level"] = calibrate(final)
            cand["_rerank_relevance"] = round(relevance, 6)
            # idx as a stable tiebreaker preserves first-stage order on ties.
            scored.append((final, -idx, cand))

        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        ordered = [c for _, _, c in scored]
        return ordered[:top_k] if top_k is not None else ordered


def _default_text(node: dict[str, Any]) -> str:
    parts = [
        str(node.get(k, ""))
        for k in ("content", "text", "summary", "name", "description")
    ]
    body = " ".join(p for p in parts if p)
    return body or str(node)
