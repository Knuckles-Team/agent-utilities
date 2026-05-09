#!/usr/bin/python
"""Hybrid Search Scorer.

CONCEPT:KG-2.37 — Hybrid Search Index

Provides weighted semantic+keyword search scoring adapted from contextplus's
hybrid search. Uses existing embedding infrastructure.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

from agent_utilities.models.knowledge_graph import HybridSearchConfig

logger = logging.getLogger(__name__)


def _split_compound_name(text: str) -> set[str]:
    """Split camelCase, PascalCase, and snake_case into tokens."""
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    spaced = re.sub(r"([A-Z])([A-Z][a-z])", r"\1 \2", spaced)
    tokens = re.split(r"[\s_\-]+", spaced.lower())
    return {t for t in tokens if len(t) > 1}


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    va, vb = np.array(a), np.array(b)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


class HybridSearchScorer:
    """Weighted hybrid semantic+keyword scoring engine.

    CONCEPT:KG-2.37 — Hybrid Search Index

    Scores documents via configurable blend of semantic similarity
    and keyword matching with phrase boost and symbol-specific scoring.

    Example::

        scorer = HybridSearchScorer()
        results = scorer.score_documents(
            query="spectral clustering",
            query_embedding=[0.1, 0.2, ...],
            documents=[{"id": "d1", "text": "...", "embedding": [...]}],
        )
    """

    def __init__(self, config: HybridSearchConfig | None = None):
        self.config = config or HybridSearchConfig()

    def _keyword_score(
        self,
        query: str,
        query_terms: set[str],
        doc_text: str,
        symbols: list[str] | None = None,
    ) -> tuple[float, list[str]]:
        """Compute keyword score with phrase boost."""
        if not query_terms:
            return 0.0, []

        doc_terms = _split_compound_name(doc_text)
        term_coverage = sum(1 for t in query_terms if t in doc_terms) / len(query_terms)

        matched_symbols: list[str] = []
        symbol_coverage = 0.0
        if symbols:
            all_sym_terms: set[str] = set()
            for sym in symbols:
                sym_terms = _split_compound_name(sym)
                if sym_terms & query_terms:
                    matched_symbols.append(sym)
                all_sym_terms.update(sym_terms)
            if query_terms:
                symbol_coverage = sum(
                    1 for t in query_terms if t in all_sym_terms
                ) / len(query_terms)

        phrase_boost = (
            self.config.phrase_boost
            if query.strip().lower() in doc_text.lower()
            else 0.0
        )
        score = min(1.0, term_coverage * 0.65 + symbol_coverage * 0.2 + phrase_boost)
        return score, matched_symbols

    def _combined_score(self, semantic: float, keyword: float) -> float:
        """Compute weighted combined score."""
        total = self.config.semantic_weight + self.config.keyword_weight
        if total <= 0:
            return max(semantic, 0.0)
        return min(
            1.0,
            (
                self.config.semantic_weight * max(semantic, 0)
                + self.config.keyword_weight * keyword
            )
            / total,
        )

    def score_documents(
        self,
        query: str,
        query_embedding: list[float],
        documents: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Score and rank documents using hybrid scoring.

        Each doc dict should have: id, text, embedding, symbols (optional).

        Returns:
            Sorted list with added: semantic_score, keyword_score,
            combined_score, matched_symbols.
        """
        query_terms = _split_compound_name(query)
        results: list[dict[str, Any]] = []

        for doc in documents:
            doc_emb = doc.get("embedding")
            sem_score = (
                _cosine_similarity(query_embedding, doc_emb)
                if doc_emb and query_embedding
                else 0.0
            )
            kw_score, matched = self._keyword_score(
                query, query_terms, doc.get("text", ""), doc.get("symbols", [])
            )
            combined = self._combined_score(sem_score, kw_score)

            if max(sem_score, 0) < self.config.min_semantic_score:
                continue
            if kw_score < self.config.min_keyword_score:
                continue
            if combined < self.config.min_combined_score:
                continue

            result = dict(doc)
            result.update(
                {
                    "semantic_score": round(sem_score, 4),
                    "keyword_score": round(kw_score, 4),
                    "combined_score": round(combined, 4),
                    "matched_symbols": matched,
                }
            )
            results.append(result)

        results.sort(
            key=lambda x: (
                x["combined_score"],
                x["keyword_score"],
                x["semantic_score"],
            ),
            reverse=True,
        )
        return results[: self.config.top_k]
