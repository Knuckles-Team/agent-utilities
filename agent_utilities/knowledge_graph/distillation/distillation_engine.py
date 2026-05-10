#!/usr/bin/python
from __future__ import annotations

"""High-Level Knowledge Distillation Orchestrator.

CONCEPT:KG-2.23 — Knowledge Distillation Engine

Ties together IdeaBlock ingestion, embedding generation, semantic
deduplication, and KG persistence into a single cohesive pipeline.

Usage::

    from agent_utilities.knowledge_graph.distillation import DistillationEngine

    engine = DistillationEngine(kg_engine)
    blocks = engine.ingest_text(
        "Long document text...",
        source="research_paper",
        metadata={"title": "My Paper"},
    )
    result = engine.distill(iterations=3, base_threshold=0.65)
"""


import logging
import re
import time
import uuid
from typing import TYPE_CHECKING, Any

from .deduplicator import KnowledgeDeduplicator

if TYPE_CHECKING:
    from ..core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)

# Default chunk size (chars) — matches Blockify's recommendation
_DEFAULT_CHUNK_SIZE = 2000
_DEFAULT_CHUNK_OVERLAP = 200


def chunk_text(
    text: str,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    overlap: int = _DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split text into sentence-boundary-aware chunks with overlap.

    Splits at sentence boundaries (periods followed by whitespace or
    newlines) and ensures chunks respect the ``chunk_size`` limit with
    ``overlap`` character overlap between consecutive chunks.

    Args:
        text: The text to chunk.
        chunk_size: Target maximum characters per chunk.
        overlap: Characters of overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """
    if not text or len(text) <= chunk_size:
        return [text] if text else []

    # Split at sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if not sentences:
        return [text]

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_length = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        if current_length + sentence_len > chunk_size and current_chunk:
            # Emit current chunk
            chunks.append(" ".join(current_chunk))

            # Calculate overlap: keep last sentences that fit within overlap
            overlap_chunk: list[str] = []
            overlap_len = 0
            for s in reversed(current_chunk):
                if overlap_len + len(s) > overlap:
                    break
                overlap_chunk.insert(0, s)
                overlap_len += len(s)

            current_chunk = overlap_chunk
            current_length = overlap_len

        current_chunk.append(sentence)
        current_length += sentence_len

    # Final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


class DistillationEngine:
    """Orchestrates end-to-end knowledge distillation.

    Converts raw text into structured IdeaBlock nodes, embeds them,
    runs iterative deduplication, and persists results to the KG with
    full OWL alignment and provenance chains.

    Args:
        kg_engine: The ``IntelligenceGraphEngine`` instance.
        deduplicator: Optional pre-configured deduplicator.
        chunk_size: Character limit per chunk during ingestion.
        chunk_overlap: Overlap characters between chunks.
    """

    def __init__(
        self,
        kg_engine: IntelligenceGraphEngine | None = None,
        deduplicator: KnowledgeDeduplicator | None = None,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        self.kg_engine = kg_engine
        self.deduplicator = deduplicator or KnowledgeDeduplicator()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # In-memory block registry for distillation
        self._blocks: dict[str, dict[str, Any]] = {}
        self._distillation_history: list[dict[str, Any]] = []

    def ingest_text(
        self,
        text: str,
        source: str = "document",
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Ingest raw text by chunking and converting to IdeaBlock-like nodes.

        Each chunk is converted to a structured knowledge unit using LLM
        extraction.  Generated blocks are stored in-memory and optionally
        persisted to the KG.

        Args:
            text: The raw text to ingest.
            source: Source identifier for provenance.
            metadata: Optional metadata dict.

        Returns:
            List of created block dicts.
        """
        meta = metadata or {}
        chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)

        if not chunks:
            return []

        created_blocks: list[dict[str, Any]] = []
        embed_model = self._get_embed_model()

        for i, chunk_text_content in enumerate(chunks):
            block_id = f"ideablock:{uuid.uuid4().hex[:8]}"

            # Generate embedding
            embedding = None
            if embed_model:
                try:
                    embedding = embed_model.get_text_embedding(chunk_text_content)
                except Exception as e:
                    logger.warning("Failed to embed chunk %d: %s", i, e)

            block = {
                "id": block_id,
                "type": "idea_block",
                "name": meta.get("title", f"Block from {source}") + f" §{i + 1}",
                "description": chunk_text_content[:200],
                "critical_question": "",  # Will be populated by LLM extraction
                "trusted_answer": chunk_text_content,
                "tags": meta.get("tags", []),
                "keywords": [],
                "entities": [],
                "source": source,
                "source_document_id": meta.get("document_id"),
                "distillation_round": 0,
                "merged_from": [],
                "embedding": embedding,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "importance_score": 0.6,
            }

            self._blocks[block_id] = block
            created_blocks.append(block)

            # Persist to KG if available
            if self.kg_engine:
                self.kg_engine.graph.add_node(block_id, **block)
                if self.kg_engine.backend:
                    self.kg_engine._upsert_node(
                        "IdeaBlock",
                        block_id,
                        {
                            "id": block_id,
                            "name": block["name"],
                            "description": block["description"],
                            "critical_question": block["critical_question"],
                            "trusted_answer": block["trusted_answer"],
                            "source": source,
                            "timestamp": block["timestamp"],
                        },
                    )

        logger.info(
            "Ingested %d blocks from %d chunks (source=%s)",
            len(created_blocks),
            len(chunks),
            source,
        )
        return created_blocks

    def ingest_ideablock(
        self,
        question: str,
        answer: str,
        name: str = "",
        tags: list[str] | None = None,
        keywords: list[str] | None = None,
        entities: list[dict[str, str]] | None = None,
        source: str = "manual",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a single structured IdeaBlock directly.

        Args:
            question: The critical question this block answers.
            answer: The trusted, validated answer.
            name: Optional human-readable title.
            tags: Classification tags.
            keywords: Retrieval keywords.
            entities: Named entity references.
            source: Provenance source.
            metadata: Additional metadata.

        Returns:
            The created block dict.
        """
        block_id = f"ideablock:{uuid.uuid4().hex[:8]}"
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        block = {
            "id": block_id,
            "type": "idea_block",
            "name": name or question[:80],
            "description": answer[:200],
            "critical_question": question,
            "trusted_answer": answer,
            "tags": tags or [],
            "keywords": keywords or [],
            "entities": entities or [],
            "source": source,
            "source_document_id": (metadata or {}).get("document_id"),
            "distillation_round": 0,
            "merged_from": [],
            "embedding": None,
            "timestamp": ts,
            "importance_score": 0.7,
        }

        # Generate embedding
        embed_model = self._get_embed_model()
        if embed_model:
            try:
                embed_text = f"{question} {answer}"
                block["embedding"] = embed_model.get_text_embedding(embed_text)
            except Exception as e:
                logger.warning("Failed to embed IdeaBlock: %s", e)

        self._blocks[block_id] = block

        # Persist to KG
        if self.kg_engine:
            self.kg_engine.graph.add_node(block_id, **block)

        return block

    def distill(
        self,
        block_ids: list[str] | None = None,
        iterations: int | None = None,
        base_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Run iterative deduplication on IdeaBlocks.

        Args:
            block_ids: Optional list of block IDs to distill.
                If None, distills all registered blocks.
            iterations: Override the deduplicator's default iterations.
            base_threshold: Override the deduplicator's default threshold.

        Returns:
            Distillation result with stats and updated blocks.
        """
        # Select blocks to distill
        if block_ids:
            blocks = [self._blocks[bid] for bid in block_ids if bid in self._blocks]
        else:
            blocks = list(self._blocks.values())

        if len(blocks) < 2:
            return {
                "blocks": blocks,
                "stats": {
                    "starting_count": len(blocks),
                    "final_count": len(blocks),
                    "blocks_removed": 0,
                    "reduction_percent": 0.0,
                    "iterations_run": 0,
                },
                "rounds": [],
            }

        # Override deduplicator settings if provided
        dedup = self.deduplicator
        if iterations is not None:
            dedup.iterations = iterations
        if base_threshold is not None:
            dedup.base_threshold = base_threshold

        result = dedup.deduplicate(blocks)

        # Update internal registry
        self._blocks.clear()
        for block in result["blocks"]:
            self._blocks[block["id"]] = block

        # Persist merged blocks to KG with provenance
        if self.kg_engine:
            for block in result["blocks"]:
                merged_from = block.get("merged_from", [])
                if merged_from:
                    # This is a merged block — add with provenance
                    self.kg_engine.graph.add_node(block["id"], **block)
                    for source_id in merged_from:
                        self.kg_engine.link_nodes(
                            block["id"],
                            source_id,
                            "DISTILLED_FROM",
                            {"confidence": 1.0, "source": "distillation"},
                        )

        # Record history
        self._distillation_history.append(result["stats"])

        logger.info(
            "Distillation complete: %d → %d blocks (%.1f%% reduction)",
            result["stats"]["starting_count"],
            result["stats"]["final_count"],
            result["stats"]["reduction_percent"],
        )

        return result

    def get_stats(self) -> dict[str, Any]:
        """Return aggregate distillation statistics.

        Returns:
            Dict with current block count, total distillation rounds,
            and cumulative reduction metrics.
        """
        total_removed = sum(
            h.get("blocks_removed", 0) for h in self._distillation_history
        )
        return {
            "current_blocks": len(self._blocks),
            "total_distillation_rounds": len(self._distillation_history),
            "total_blocks_removed": total_removed,
            "history": self._distillation_history,
        }

    def get_blocks(self) -> list[dict[str, Any]]:
        """Return all registered blocks.

        Returns:
            List of block dicts.
        """
        return list(self._blocks.values())

    def _get_embed_model(self) -> Any:
        """Attempt to get the embedding model from the KG engine or create one."""
        # First try the KG engine's existing retriever
        if (
            self.kg_engine
            and hasattr(self.kg_engine, "hybrid_retriever")
            and self.kg_engine.hybrid_retriever.embed_model
        ):
            return self.kg_engine.hybrid_retriever.embed_model

        # Fallback: create one directly
        try:
            from agent_utilities.core.embedding_utilities import (
                create_embedding_model,
            )

            return create_embedding_model()
        except Exception as e:
            logger.warning("No embedding model available: %s", e)
            return None
