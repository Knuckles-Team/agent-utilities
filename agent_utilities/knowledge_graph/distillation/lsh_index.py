#!/usr/bin/python
from __future__ import annotations

"""Locality-Sensitive Hashing Index for Knowledge Deduplication.

CONCEPT:AU-KG.ingest.knowledge-distillation — Knowledge Distillation Engine

Provides efficient approximate nearest-neighbor candidate generation using
random hyperplane LSH.  Designed for the deduplication pipeline where we need
to find similar IdeaBlock embeddings without O(n²) pairwise comparisons.

Adapted from Blockify's ``app/dedupe/lsh.py`` but reimplemented against the
certified native numeric kernel and integrated with our Pydantic models.

Example::

    index = LSHIndex(num_tables=8, hash_size=12, input_dim=768)
    index.index("block_1", embedding_1)
    index.index("block_2", embedding_2)
    candidates = index.query(embedding_1, k=10)
"""


import logging
import math
from typing import Any

from agent_utilities.numeric import NDArray, xp

logger = logging.getLogger(__name__)


class LSHIndex:
    """Locality-Sensitive Hashing index for approximate nearest-neighbor search.

    Uses random hyperplane hashing to partition the embedding space into
    buckets.  Multiple hash tables increase recall at a small memory cost.

    Args:
        num_tables: Number of independent hash tables (more = higher recall).
        hash_size: Number of hyperplanes per table (more = higher precision).
        input_dim: Dimensionality of input embeddings.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        num_tables: int = 8,
        hash_size: int = 12,
        input_dim: int = 768,
        seed: int = 42,
    ) -> None:
        self.num_tables = num_tables
        self.hash_size = hash_size
        self.input_dim = input_dim

        rng = xp.random.default_rng(seed)

        # Generate random hyperplanes for each table
        self._hyperplanes: list[NDArray] = [
            rng.standard_normal((hash_size, input_dim)) for _ in range(num_tables)
        ]

        # Hash tables: table_idx -> hash_key -> set of block IDs
        self._tables: list[dict[str, set[str]]] = [{} for _ in range(num_tables)]

        # Store embeddings for cosine re-ranking
        self._embeddings: dict[str, NDArray] = {}

    def _hash_vector(self, vector: NDArray, table_idx: int) -> str:
        """Compute the hash key for a vector in a specific table."""
        # Project every hyperplane in one native matrix multiplication.  The
        # previous per-hyperplane ``xp.dot`` loop paid one boundary crossing for
        # every bit in every table.
        projections_matrix = xp.matmul(
            self._hyperplanes[table_idx], [[value] for value in vector]
        )
        projections = [float(row[0]) for row in projections_matrix]
        return "".join("1" if projection > 0 else "0" for projection in projections)

    def index(self, block_id: str, embedding: list[float]) -> None:
        """Add a block embedding to all hash tables.

        Args:
            block_id: Unique identifier for the block.
            embedding: The embedding vector (must match ``input_dim``).
        """
        vec = [float(value) for value in embedding]
        if len(vec) != self.input_dim:
            logger.warning(
                "Embedding dim %d != expected %d for block %s, skipping",
                len(vec),
                self.input_dim,
                block_id,
            )
            return

        self._embeddings[block_id] = vec

        for table_idx in range(self.num_tables):
            key = self._hash_vector(vec, table_idx)
            if key not in self._tables[table_idx]:
                self._tables[table_idx][key] = set()
            self._tables[table_idx][key].add(block_id)

    def remove(self, block_id: str) -> None:
        """Remove a block from the index.

        Args:
            block_id: The block ID to remove.
        """
        if block_id not in self._embeddings:
            return

        vec = self._embeddings.pop(block_id)

        for table_idx in range(self.num_tables):
            key = self._hash_vector(vec, table_idx)
            bucket = self._tables[table_idx].get(key)
            if bucket:
                bucket.discard(block_id)
                if not bucket:
                    del self._tables[table_idx][key]

    def query(
        self,
        embedding: list[float],
        k: int = 10,
        exclude_id: str | None = None,
    ) -> list[tuple[str, float]]:
        """Find approximate nearest neighbors for a query embedding.

        Args:
            embedding: The query embedding vector.
            k: Maximum number of candidates to return.
            exclude_id: Optional block ID to exclude from results (self-match).

        Returns:
            List of ``(block_id, cosine_similarity)`` tuples sorted by
            descending similarity.
        """
        vec = [float(value) for value in embedding]
        if len(vec) != self.input_dim:
            return []

        # Collect candidates from all tables
        candidates: set[str] = set()
        for table_idx in range(self.num_tables):
            key = self._hash_vector(vec, table_idx)
            bucket = self._tables[table_idx].get(key, set())
            candidates.update(bucket)

        if exclude_id:
            candidates.discard(exclude_id)

        if not candidates:
            return []

        # Re-rank by cosine similarity
        scored: list[tuple[str, float]] = []
        vec_norm = math.sqrt(sum(value * value for value in vec))
        if vec_norm == 0:
            return []

        # All candidate projections are one native batch operation; only the
        # bounded score-to-id mapping remains in Python.
        candidate_ids = sorted(cid for cid in candidates if cid in self._embeddings)
        candidate_vectors = [self._embeddings[cid] for cid in candidate_ids]
        if candidate_vectors:
            dots = xp.matmul(candidate_vectors, [[value] for value in vec])
            for cid, row in zip(candidate_ids, dots, strict=True):
                cand_vec = self._embeddings[cid]
                cand_norm = math.sqrt(sum(value * value for value in cand_vec))
                if cand_norm:
                    scored.append((cid, float(row[0]) / (vec_norm * cand_norm)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def clear(self) -> None:
        """Remove all entries from the index."""
        self._embeddings.clear()
        for table in self._tables:
            table.clear()

    @property
    def size(self) -> int:
        """Number of indexed blocks."""
        return len(self._embeddings)

    def get_stats(self) -> dict[str, Any]:
        """Return diagnostic statistics about the index.

        Returns:
            Dict with index size, table count, and bucket distribution.
        """
        bucket_counts = []
        for table in self._tables:
            bucket_counts.append(len(table))

        return {
            "indexed_blocks": self.size,
            "num_tables": self.num_tables,
            "hash_size": self.hash_size,
            "input_dim": self.input_dim,
            "avg_buckets_per_table": (
                sum(bucket_counts) / len(bucket_counts) if bucket_counts else 0
            ),
        }
