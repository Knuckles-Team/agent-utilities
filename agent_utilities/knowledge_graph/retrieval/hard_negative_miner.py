#!/usr/bin/python
from __future__ import annotations

"""CONCEPT:AU-KG.retrieval.hard-negative-mining — Hard Negative Mining for Retrieval Calibration.

Continuously improves retriever precision by mining challenging distractors.
Directly from BrowseComp-Plus (arXiv:2508.06600): query decomposition →
multi-retrieval → filter pipeline.

Gated behind ``KG_ENABLE_HARD_NEGATIVE_MINING`` env var (default: false).

See docs/pillars/2_epistemic_knowledge_graph/KG-2.3-Graph_Integrity_And_Retrieval.md
"""

import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from agent_utilities.core.config import setting

if TYPE_CHECKING:
    from .hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)

_MINING_ENABLED = setting("KG_ENABLE_HARD_NEGATIVE_MINING", False)


class HardNegative(BaseModel):
    """A document that matches sub-queries but not the full query."""

    doc_id: str = Field(description="KG node ID of the false positive")
    triggering_subquery: str = Field(description="Sub-query that retrieved it")
    original_query: str = Field(description="The full query it failed against")
    relevance_score: float = Field(default=0.0, description="Score from sub-query")
    reason: str = Field(default="", description="Why this is a false positive")


class HardNegativeMiner:
    """Mines hard negatives from query decomposition for retriever calibration.

    CONCEPT:AU-KG.retrieval.hard-negative-mining — Hard Negative Mining (BrowseComp-Plus)

    Uses the existing ``_decompose_query()`` in ``HybridRetriever`` to break
    complex queries into sub-queries, fetches results per sub-query, and
    identifies documents that match sub-queries but not the full query.

    Gated behind ``KG_ENABLE_HARD_NEGATIVE_MINING=true`` env var.

    Args:
        retriever: The HybridRetriever instance.
        penalty_factor: Score multiplier for known hard negatives (0–1).
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        penalty_factor: float = 0.5,
    ) -> None:
        self._retriever = retriever
        self._penalty_factor = max(0.0, min(1.0, penalty_factor))
        self._known_negatives: dict[str, set[str]] = {}  # query_hash → doc_ids

    @property
    def enabled(self) -> bool:
        """Whether hard negative mining is active."""
        return _MINING_ENABLED

    def mine(
        self,
        query: str,
        gold_doc_ids: set[str] | None = None,
        max_subtasks: int = 5,
        context_window: int = 10,
    ) -> list[HardNegative]:
        """Mine hard negatives for a given query.

        Process:
        1. Decompose query into sub-queries
        2. Retrieve results for each sub-query
        3. Retrieve results for the full query
        4. Documents in sub-query results but NOT in full-query results
           (or not in gold_doc_ids) are hard negatives

        Args:
            query: The complex query to mine negatives for.
            gold_doc_ids: Known correct document IDs (optional).
            max_subtasks: Maximum sub-queries for decomposition.
            context_window: Results per sub-query.

        Returns:
            List of HardNegative objects.
        """
        if not self.enabled:
            return []

        # Step 1: Decompose
        subqueries = self._retriever._decompose_query(query, max_subtasks=max_subtasks)
        if len(subqueries) <= 1:
            return []

        # Step 2: Retrieve for each sub-query
        subquery_results: dict[str, list[dict[str, Any]]] = {}
        for sq in subqueries:
            results = self._retriever.retrieve_hybrid(sq, context_window=context_window)
            subquery_results[sq] = results

        # Step 3: Retrieve for full query
        full_results = self._retriever.retrieve_hybrid(
            query, context_window=context_window
        )
        full_ids = {r.get("id", "") for r in full_results}

        # Step 4: Identify hard negatives
        negatives: list[HardNegative] = []
        seen: set[str] = set()

        for sq, results in subquery_results.items():
            for r in results:
                doc_id = r.get("id", "")
                if not doc_id or doc_id in seen:
                    continue

                is_negative = doc_id not in full_ids
                if gold_doc_ids:
                    is_negative = is_negative or doc_id not in gold_doc_ids

                if is_negative:
                    seen.add(doc_id)
                    negatives.append(
                        HardNegative(
                            doc_id=doc_id,
                            triggering_subquery=sq,
                            original_query=query,
                            relevance_score=r.get("_score", 0.0),
                            reason=f"Matched sub-query '{sq}' but not full query",
                        )
                    )

        # Cache for use in penalty application
        query_key = str(hash(query))
        self._known_negatives[query_key] = {n.doc_id for n in negatives}

        logger.info(
            "Mined %d hard negatives for query (from %d sub-queries)",
            len(negatives),
            len(subqueries),
        )
        return negatives

    def get_penalty_set(self, query: str) -> set[str]:
        """Get cached hard negative IDs for a query.

        Args:
            query: The query to look up.

        Returns:
            Set of document IDs to penalize, empty if none cached.
        """
        return self._known_negatives.get(str(hash(query)), set())

    def apply_penalties(
        self, results: list[dict[str, Any]], hard_negative_ids: set[str]
    ) -> list[dict[str, Any]]:
        """Apply score penalties to known hard negatives in results.

        Args:
            results: Retrieval results with ``_score`` fields.
            hard_negative_ids: IDs to penalize.

        Returns:
            Results with adjusted scores.
        """
        if not hard_negative_ids:
            return results

        for r in results:
            if r.get("id") in hard_negative_ids:
                r["_score"] = r.get("_score", 0.0) * self._penalty_factor
                r["_hard_negative"] = True

        results.sort(key=lambda x: x.get("_score", 0.0), reverse=True)
        return results
