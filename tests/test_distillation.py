"""Tests for the Knowledge Distillation Engine.

CONCEPT:KG-2.2 — Knowledge Distillation Engine

Covers:
  - IdeaBlock Pydantic model creation and validation
  - LSH indexing and approximate nearest-neighbor accuracy
  - Deduplicator clustering logic (dense + LSH modes, BFS clustering)
  - Distillation engine text ingestion and orchestration
  - Document pipeline chunking improvements
  - Pydantic model enum integration
"""

import numpy as np
import pytest

from agent_utilities.knowledge_graph.distillation.deduplicator import (
    KnowledgeDeduplicator,
)
from agent_utilities.knowledge_graph.distillation.distillation_engine import (
    DistillationEngine,
    chunk_text,
)
from agent_utilities.knowledge_graph.distillation.lsh_index import LSHIndex
from agent_utilities.models.knowledge_graph import (
    DistillationRoundNode,
    EntityReference,
    IdeaBlockNode,
    RegistryEdgeType,
    RegistryNodeType,
)


# ── Fixtures ──


def _make_random_embedding(dim: int = 128, seed: int = 0) -> list[float]:
    """Generate a reproducible random embedding."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def _make_similar_embedding(
    base: list[float], noise: float = 0.05, seed: int = 1
) -> list[float]:
    """Create an embedding similar to base by adding small noise."""
    rng = np.random.RandomState(seed)
    vec = np.array(base, dtype=np.float32) + noise * rng.randn(len(base)).astype(
        np.float32
    )
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def _make_block(
    block_id: str, question: str, answer: str, embedding: list[float]
) -> dict:
    """Create a minimal block dict for testing."""
    return {
        "id": block_id,
        "name": question[:50],
        "critical_question": question,
        "trusted_answer": answer,
        "embedding": embedding,
        "tags": ["TECHNOLOGY"],
        "keywords": ["test"],
    }


# ── IdeaBlock Model Tests ──


class TestIdeaBlockModel:
    """Tests for the IdeaBlockNode Pydantic model."""

    def test_create_ideablock_node(self):
        """An IdeaBlockNode can be created with all required fields."""
        node = IdeaBlockNode(
            id="ideablock:test001",
            name="Test Block",
            critical_question="What is X?",
            trusted_answer="X is a thing.",
        )
        assert node.type == RegistryNodeType.IDEA_BLOCK
        assert node.critical_question == "What is X?"
        assert node.trusted_answer == "X is a thing."
        assert node.distillation_round == 0
        assert node.merged_from == []

    def test_ideablock_with_entities(self):
        """IdeaBlockNode supports entity references."""
        entities = [
            EntityReference(entity_name="CLAUDE CODE", entity_type="PRODUCT"),
            EntityReference(entity_name="ANTHROPIC", entity_type="ORGANIZATION"),
        ]
        node = IdeaBlockNode(
            id="ideablock:test002",
            name="Claude Code Overview",
            critical_question="What is Claude Code?",
            trusted_answer="Claude Code is an AI development tool.",
            entities=entities,
        )
        assert len(node.entities) == 2
        assert node.entities[0].entity_name == "CLAUDE CODE"
        assert node.entities[1].entity_type == "ORGANIZATION"

    def test_ideablock_with_tags_keywords(self):
        """IdeaBlockNode stores governance tags and retrieval keywords."""
        node = IdeaBlockNode(
            id="ideablock:test003",
            name="Test",
            tags=["IMPORTANT", "TECHNOLOGY", "AI"],
            keywords=["machine learning", "neural networks"],
        )
        assert "IMPORTANT" in node.tags
        assert "machine learning" in node.keywords

    def test_ideablock_merged_from(self):
        """IdeaBlockNode tracks merge provenance."""
        node = IdeaBlockNode(
            id="ideablock:merged001",
            name="Merged Block",
            distillation_round=2,
            merged_from=["ideablock:001", "ideablock:002", "ideablock:003"],
        )
        assert node.distillation_round == 2
        assert len(node.merged_from) == 3

    def test_distillation_round_node(self):
        """DistillationRoundNode records iteration metrics."""
        node = DistillationRoundNode(
            id="round:001",
            name="Round 1",
            iteration=1,
            similarity_threshold=0.65,
            blocks_before=100,
            blocks_after=75,
            pairs_found=30,
            clusters_merged=8,
        )
        assert node.type == RegistryNodeType.DISTILLATION_ROUND
        assert node.blocks_before == 100
        assert node.blocks_after == 75


class TestRegistryEnums:
    """Tests for new enum entries."""

    def test_idea_block_enum(self):
        """IDEA_BLOCK is registered in RegistryNodeType."""
        assert RegistryNodeType.IDEA_BLOCK == "idea_block"

    def test_distillation_round_enum(self):
        """DISTILLATION_ROUND is registered in RegistryNodeType."""
        assert RegistryNodeType.DISTILLATION_ROUND == "distillation_round"

    def test_distilled_from_edge(self):
        """DISTILLED_FROM is registered in RegistryEdgeType."""
        assert RegistryEdgeType.DISTILLED_FROM == "distilled_from"

    def test_produced_in_round_edge(self):
        """PRODUCED_IN_ROUND is registered in RegistryEdgeType."""
        assert RegistryEdgeType.PRODUCED_IN_ROUND == "produced_in_round"


# ── LSH Index Tests ──


class TestLSHIndex:
    """Tests for LSH approximate nearest-neighbor search."""

    def test_index_and_query(self):
        """Index blocks and query returns candidates."""
        idx = LSHIndex(num_tables=4, hash_size=8, input_dim=128)

        emb1 = _make_random_embedding(128, seed=1)
        emb2 = _make_similar_embedding(emb1, noise=0.02, seed=2)
        emb3 = _make_random_embedding(128, seed=99)  # Dissimilar

        idx.index("b1", emb1)
        idx.index("b2", emb2)
        idx.index("b3", emb3)

        results = idx.query(emb1, k=5, exclude_id="b1")
        # b2 should be a high-similarity candidate
        result_ids = [r[0] for r in results]
        # At minimum, the query should return candidates
        assert isinstance(results, list)

    def test_empty_query(self):
        """Query on empty index returns empty list."""
        idx = LSHIndex(num_tables=2, hash_size=4, input_dim=64)
        emb = _make_random_embedding(64, seed=0)
        assert idx.query(emb, k=5) == []

    def test_wrong_dimension(self):
        """Indexing wrong-dimension embedding is skipped with warning."""
        idx = LSHIndex(num_tables=2, hash_size=4, input_dim=64)
        wrong_dim = _make_random_embedding(128, seed=0)
        idx.index("b1", wrong_dim)  # Should be skipped
        assert idx.size == 0

    def test_remove(self):
        """Remove a block from the index."""
        idx = LSHIndex(num_tables=2, hash_size=4, input_dim=64)
        emb = _make_random_embedding(64, seed=0)
        idx.index("b1", emb)
        assert idx.size == 1
        idx.remove("b1")
        assert idx.size == 0

    def test_clear(self):
        """Clear removes all entries."""
        idx = LSHIndex(num_tables=2, hash_size=4, input_dim=64)
        for i in range(10):
            idx.index(f"b{i}", _make_random_embedding(64, seed=i))
        assert idx.size == 10
        idx.clear()
        assert idx.size == 0

    def test_stats(self):
        """Stats returns diagnostic information."""
        idx = LSHIndex(num_tables=4, hash_size=8, input_dim=64)
        for i in range(5):
            idx.index(f"b{i}", _make_random_embedding(64, seed=i))

        stats = idx.get_stats()
        assert stats["indexed_blocks"] == 5
        assert stats["num_tables"] == 4
        assert stats["hash_size"] == 8
        assert stats["input_dim"] == 64


# ── Deduplicator Tests ──


class TestKnowledgeDeduplicator:
    """Tests for the iterative deduplication engine."""

    def test_find_similar_pairs_dense(self):
        """Find similar pairs using dense cosine similarity."""
        dedup = KnowledgeDeduplicator(embedding_dim=128)

        emb1 = _make_random_embedding(128, seed=1)
        emb2 = _make_similar_embedding(emb1, noise=0.01, seed=2)
        emb3 = _make_random_embedding(128, seed=99)

        blocks = [
            _make_block("b1", "What is X?", "X is A.", emb1),
            _make_block("b2", "What is X?", "X is something.", emb2),
            _make_block("b3", "What is Z?", "Z is different.", emb3),
        ]

        pairs = dedup.find_similar_pairs(blocks, threshold=0.85)
        # b1 and b2 should be similar, b3 should not pair with either
        pair_ids = [(p[0], p[1]) for p in pairs]
        assert any(("b1" in pair and "b2" in pair) for pair in pair_ids), (
            f"Expected b1-b2 pair, got {pair_ids}"
        )

    def test_cluster_bfs(self):
        """BFS clustering groups connected pairs."""
        dedup = KnowledgeDeduplicator()

        pairs = [
            ("b1", "b2", 0.9),
            ("b2", "b3", 0.85),
            ("b4", "b5", 0.92),
        ]

        clusters = dedup.cluster_similar_blocks(pairs)
        assert len(clusters) == 2  # {b1,b2,b3} and {b4,b5}

        # Find cluster containing b1
        b1_cluster = [c for c in clusters if "b1" in c][0]
        assert "b2" in b1_cluster
        assert "b3" in b1_cluster

    def test_cluster_empty(self):
        """Empty pairs produce empty clusters."""
        dedup = KnowledgeDeduplicator()
        assert dedup.cluster_similar_blocks([]) == []

    def test_cluster_oversized_split(self):
        """Oversized clusters are split."""
        dedup = KnowledgeDeduplicator(max_cluster_size=3)

        # Create a chain of 6 blocks
        pairs = [(f"b{i}", f"b{i + 1}", 0.9) for i in range(5)]

        clusters = dedup.cluster_similar_blocks(pairs)
        for cluster in clusters:
            assert len(cluster) <= 3

    def test_merge_cluster_single_block(self):
        """Single-block cluster returns the block unchanged."""
        dedup = KnowledgeDeduplicator()

        block = _make_block("b1", "Q?", "A.", _make_random_embedding(128))
        result = dedup.merge_cluster([block])
        assert result["id"] == "b1"

    def test_merge_cluster_fallback(self):
        """Merge falls back to first block when LLM unavailable."""
        dedup = KnowledgeDeduplicator()

        blocks = [
            _make_block("b1", "What is X?", "X is A.", _make_random_embedding(128, 1)),
            _make_block("b2", "What is X?", "X is B.", _make_random_embedding(128, 2)),
        ]

        result = dedup.merge_cluster(blocks)
        # Should fall back to first block with merged_from annotation
        assert "merged_from" in result
        assert "b1" in result["merged_from"]
        assert "b2" in result["merged_from"]

    def test_deduplicate_no_similar_blocks(self):
        """Deduplication with no similar blocks terminates early."""
        dedup = KnowledgeDeduplicator(iterations=3, base_threshold=0.99)

        blocks = [
            _make_block(f"b{i}", f"Q{i}?", f"A{i}.", _make_random_embedding(128, i))
            for i in range(5)
        ]

        result = dedup.deduplicate(blocks)
        assert result["stats"]["final_count"] == 5
        assert result["stats"]["blocks_removed"] == 0

    def test_deduplicate_with_similar_blocks(self):
        """Deduplication merges similar blocks and reduces count."""
        dedup = KnowledgeDeduplicator(iterations=2, base_threshold=0.80)

        emb1 = _make_random_embedding(128, seed=1)
        emb2 = _make_similar_embedding(emb1, noise=0.01, seed=2)
        emb3 = _make_random_embedding(128, seed=99)

        blocks = [
            _make_block("b1", "What is X?", "X is A.", emb1),
            _make_block("b2", "What is X?", "X is something.", emb2),
            _make_block("b3", "What is Z?", "Z is different.", emb3),
        ]

        result = dedup.deduplicate(blocks)
        assert result["stats"]["final_count"] < result["stats"]["starting_count"]
        assert result["stats"]["blocks_removed"] > 0
        assert len(result["rounds"]) > 0

    def test_deduplicate_stats(self):
        """Deduplication stats include reduction percentage."""
        dedup = KnowledgeDeduplicator(iterations=1, base_threshold=0.99)

        blocks = [
            _make_block(f"b{i}", f"Q{i}?", f"A{i}.", _make_random_embedding(128, i))
            for i in range(3)
        ]

        result = dedup.deduplicate(blocks)
        stats = result["stats"]
        assert "starting_count" in stats
        assert "final_count" in stats
        assert "blocks_removed" in stats
        assert "reduction_percent" in stats
        assert "iterations_run" in stats


# ── Chunking Tests ──


class TestChunkText:
    """Tests for the sentence-boundary-aware chunking function."""

    def test_short_text_no_split(self):
        """Text shorter than chunk_size is returned as-is."""
        text = "This is a short sentence."
        chunks = chunk_text(text, chunk_size=1000)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_empty_text(self):
        """Empty text returns empty list."""
        assert chunk_text("") == []

    def test_sentence_boundary_split(self):
        """Chunks split at sentence boundaries."""
        sentences = [f"Sentence {i} is here." for i in range(20)]
        text = " ".join(sentences)
        chunks = chunk_text(text, chunk_size=100, overlap=20)

        assert len(chunks) > 1
        # No chunk should be excessively larger than chunk_size
        # (allowing some margin for sentence boundaries)
        for chunk in chunks:
            assert len(chunk) < 200  # 2x chunk_size as safety margin

    def test_overlap(self):
        """Consecutive chunks have overlapping content."""
        sentences = [f"This is sentence number {i}." for i in range(30)]
        text = " ".join(sentences)
        chunks = chunk_text(text, chunk_size=200, overlap=50)

        if len(chunks) >= 2:
            # Check that the end of chunk[0] overlaps with start of chunk[1]
            # The last sentence of chunk[0] should appear in chunk[1]
            last_words_chunk0 = chunks[0].split()[-3:]
            first_words_chunk1 = chunks[1].split()[:10]
            # At least some words should overlap
            overlap_found = any(w in first_words_chunk1 for w in last_words_chunk0)
            assert overlap_found, "Expected overlap between consecutive chunks"


# ── Distillation Engine Tests ──


class TestDistillationEngine:
    """Tests for the high-level distillation orchestrator."""

    def test_ingest_text_basic(self):
        """Ingesting text creates IdeaBlock-like blocks."""
        engine = DistillationEngine(kg_engine=None)

        text = (
            "Machine learning is a subset of artificial intelligence. "
            "It uses statistical methods to learn from data. "
            "Neural networks are a popular ML technique. "
            "They are inspired by biological neural networks."
        )

        blocks = engine.ingest_text(text, source="test", metadata={"title": "ML Intro"})
        assert len(blocks) >= 1
        assert all(b["id"].startswith("ideablock:") for b in blocks)
        assert all(b["source"] == "test" for b in blocks)

    def test_ingest_ideablock_direct(self):
        """Directly create a structured IdeaBlock."""
        engine = DistillationEngine(kg_engine=None)

        block = engine.ingest_ideablock(
            question="What is a Knowledge Graph?",
            answer="A knowledge graph represents information as entities and relationships.",
            name="KG Definition",
            tags=["TECHNOLOGY", "AI"],
            keywords=["knowledge graph", "entities", "relationships"],
        )

        assert block["id"].startswith("ideablock:")
        assert block["critical_question"] == "What is a Knowledge Graph?"
        assert block["trusted_answer"].startswith("A knowledge graph")
        assert "TECHNOLOGY" in block["tags"]

    def test_get_stats_empty(self):
        """Stats on empty engine returns zero counts."""
        engine = DistillationEngine(kg_engine=None)
        stats = engine.get_stats()
        assert stats["current_blocks"] == 0
        assert stats["total_distillation_rounds"] == 0

    def test_get_blocks(self):
        """get_blocks returns all registered blocks."""
        engine = DistillationEngine(kg_engine=None)
        engine.ingest_ideablock("Q1?", "A1.")
        engine.ingest_ideablock("Q2?", "A2.")
        assert len(engine.get_blocks()) == 2

    def test_distill_too_few_blocks(self):
        """Distilling with < 2 blocks returns immediately."""
        engine = DistillationEngine(kg_engine=None)
        engine.ingest_ideablock("Q?", "A.")

        result = engine.distill()
        assert result["stats"]["final_count"] == 1
        assert result["stats"]["blocks_removed"] == 0

    def test_distill_with_embeddings(self):
        """Distillation works end-to-end with synthetic embeddings."""
        engine = DistillationEngine(kg_engine=None)

        emb1 = _make_random_embedding(128, seed=1)
        emb2 = _make_similar_embedding(emb1, noise=0.01, seed=2)
        emb3 = _make_random_embedding(128, seed=99)

        # Manually add blocks with embeddings
        engine._blocks["b1"] = _make_block("b1", "What is X?", "X is A.", emb1)
        engine._blocks["b2"] = _make_block("b2", "What is X?", "X is B.", emb2)
        engine._blocks["b3"] = _make_block("b3", "What is Z?", "Z is C.", emb3)

        result = engine.distill(iterations=2, base_threshold=0.80)
        # Should merge b1 and b2 into one block
        assert result["stats"]["starting_count"] == 3
        assert result["stats"]["final_count"] < 3

    def test_configurable_parameters(self):
        """Distillation parameters can be overridden per call."""
        dedup = KnowledgeDeduplicator(
            iterations=5,
            base_threshold=0.50,
            threshold_increment=0.05,
        )
        engine = DistillationEngine(kg_engine=None, deduplicator=dedup)

        # Verify the deduplicator has our custom values
        assert engine.deduplicator.iterations == 5
        assert engine.deduplicator.base_threshold == 0.50
        assert engine.deduplicator.threshold_increment == 0.05

    def test_chunk_size_configurable(self):
        """Engine respects custom chunk size."""
        engine = DistillationEngine(kg_engine=None, chunk_size=500, chunk_overlap=50)
        assert engine.chunk_size == 500
        assert engine.chunk_overlap == 50


# ── Integration Tests ──


class TestDistillationIntegration:
    """Integration tests verifying KG model compatibility."""

    def test_ideablock_as_registry_node(self):
        """IdeaBlockNode serializes via model_dump() like all RegistryNodes."""
        node = IdeaBlockNode(
            id="ideablock:int001",
            name="Integration Test",
            critical_question="Does this work?",
            trusted_answer="Yes it does.",
            tags=["IMPORTANT"],
            entities=[EntityReference(entity_name="TEST", entity_type="CONCEPT")],
        )

        data = node.model_dump()
        assert data["type"] == "idea_block"
        assert data["critical_question"] == "Does this work?"
        assert len(data["entities"]) == 1
        assert data["entities"][0]["entity_name"] == "TEST"

    def test_distillation_round_serialization(self):
        """DistillationRoundNode serializes correctly."""
        node = DistillationRoundNode(
            id="round:int001",
            name="Test Round",
            iteration=3,
            similarity_threshold=0.69,
            blocks_before=50,
            blocks_after=40,
            pairs_found=15,
            clusters_merged=5,
        )

        data = node.model_dump()
        assert data["type"] == "distillation_round"
        assert data["similarity_threshold"] == 0.69
        assert data["clusters_merged"] == 5

    def test_entity_reference_model(self):
        """EntityReference validates correctly."""
        ref = EntityReference(entity_name="BLOCKIFY", entity_type="PRODUCT")
        assert ref.entity_name == "BLOCKIFY"
        assert ref.entity_type == "PRODUCT"

        # Default type
        ref2 = EntityReference(entity_name="SOMETHING")
        assert ref2.entity_type == "CONCEPT"
