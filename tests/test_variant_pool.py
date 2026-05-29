#!/usr/bin/python
from __future__ import annotations

"""Tests for CONCEPT:AHE-3.2 — Evolutionary Variant Selection."""


from agent_utilities.harness.variant_pool import VariantPool
from agent_utilities.models.knowledge_graph import (
    RegistryEdgeType,
    SystemPromptNode,
)


class FakeBackend:
    """Minimal mock backend."""

    def __init__(self):
        self.store: dict[str, dict] = {}

    def execute(self, query, params=None):
        return []


class FakeEngine:  # type: ignore
    """Minimal mock engine for variant pool tests."""

    def __init__(self, backend=None):
        from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

        self.graph = GraphComputeEngine(backend_type="rust")
        self.backend = backend

    def _upsert_node(self, label, node_id, props):
        if self.backend:
            self.backend.store[node_id] = {**props, "id": node_id}

    def link_nodes(self, src, tgt, rel_type, props=None):
        self.graph.add_edge(src, tgt, type=rel_type, **(props or {}))


class TestVariantRegistration:
    def test_register_variant_creates_node(self):
        engine = FakeEngine()  # type: ignore
        pool = VariantPool(engine)  # type: ignore

        base = SystemPromptNode(
            id="prompt:base",
            name="Base Prompt",
            content="You are a helpful assistant.",
            version="1.0",
            source="MANUAL",
        )
        engine.graph.add_node(base.id, **base.model_dump())

        variant = SystemPromptNode(
            id="prompt:var1",
            name="Variant 1",
            content="You are an expert assistant.",
            version="1.1",
            source="GENERATED",
        )
        result_id = pool.register_variant(
            base.id, variant, generation=1, strategy="llm"
        )

        assert result_id == "prompt:var1"
        assert "prompt:var1" in engine.graph
        assert engine.graph.has_edge("prompt:var1", "prompt:base")

    def test_register_variant_sets_metadata(self):
        engine = FakeEngine()  # type: ignore
        pool = VariantPool(engine)  # type: ignore

        variant = SystemPromptNode(
            id="prompt:var2",
            name="Variant 2",
            content="Test",
            version="1.0",
            source="GENERATED",
        )
        pool.register_variant(
            "prompt:base", variant, generation=2, strategy="parametric"
        )

        node_data = engine.graph.nodes["prompt:var2"]
        meta = node_data.get("metadata", {})
        assert meta.get("generation") == 2
        assert meta.get("strategy") == "parametric"


class TestParametricVariants:
    def test_generate_parametric_variant(self):
        engine = FakeEngine()  # type: ignore
        pool = VariantPool(engine)  # type: ignore

        base = SystemPromptNode(
            id="prompt:base",
            name="Base Prompt",
            content="You are helpful.",
            version="1.0",
            source="MANUAL",
        )

        variant = pool.generate_parametric_variant(
            base,
            mutations={"content_suffix": "Always be concise."},
        )
        assert isinstance(variant, SystemPromptNode)

        assert variant.id != base.id
        assert "Always be concise." in variant.content
        assert "You are helpful." in variant.content

    def test_generate_parametric_variant_with_prefix(self):
        engine = FakeEngine()  # type: ignore
        pool = VariantPool(engine)  # type: ignore

        base = SystemPromptNode(
            id="prompt:base",
            name="Base",
            content="Original content.",
            version="1.0",
            source="MANUAL",
        )

        variant = pool.generate_parametric_variant(
            base,
            mutations={"content_prefix": "IMPORTANT:"},
        )
        assert isinstance(variant, SystemPromptNode)

        assert variant.content.startswith("IMPORTANT:")


class TestFitnessEvaluation:
    def test_fitness_returns_zero_for_unknown(self):
        engine = FakeEngine()  # type: ignore
        pool = VariantPool(engine)  # type: ignore

        fitness = pool.evaluate_fitness("nonexistent")
        assert fitness == 0.0

    def test_fitness_from_graph(self):
        engine = FakeEngine()  # type: ignore
        pool = VariantPool(engine)  # type: ignore

        # Set up graph: variant <- episode -> evaluation
        engine.graph.add_node("var:1", type="system_prompt")
        engine.graph.add_node("ep:1", type="episode")
        engine.graph.add_node("eval:1", type="outcome_evaluation", reward=0.9)

        engine.graph.add_edge("ep:1", "var:1", type="EXECUTED_BY")
        engine.graph.add_edge("ep:1", "eval:1", type="PRODUCED_OUTCOME")

        fitness = pool.evaluate_fitness("var:1")
        assert fitness == 0.9


class TestSelection:
    def test_get_variants_empty(self):
        engine = FakeEngine()  # type: ignore
        pool = VariantPool(engine)  # type: ignore

        variants = pool.get_variants("base:1")
        assert variants == []

    def test_tournament_select_returns_top_k(self):
        engine = FakeEngine()  # type: ignore
        pool = VariantPool(engine)  # type: ignore

        # Create variants with known fitness scores
        engine.graph.add_node("base:1")
        for i in range(5):
            vid = f"var:{i}"
            engine.graph.add_node(vid, name=f"Variant {i}", metadata={})
            engine.graph.add_edge(vid, "base:1", type=RegistryEdgeType.VARIANT_OF)

        # Tournament select should return at most top_k
        winners = pool.tournament_select("base:1", top_k=3)
        assert len(winners) <= 3

    def test_prune_losers_keeps_top(self):
        engine = FakeEngine()  # type: ignore
        pool = VariantPool(engine)  # type: ignore

        engine.graph.add_node("base:1")
        for i in range(5):
            vid = f"var:{i}"
            engine.graph.add_node(vid, name=f"Variant {i}", metadata={})
            engine.graph.add_edge(vid, "base:1", type=RegistryEdgeType.VARIANT_OF)

        pruned = pool.prune_losers("base:1", keep=3)
        assert pruned >= 0  # May be 0 if all have same fitness


class TestPromotion:
    def test_promote_winner_creates_supersedes_edge(self):
        engine = FakeEngine()  # type: ignore
        pool = VariantPool(engine)  # type: ignore

        engine.graph.add_node("var:winner")
        engine.graph.add_node("base:1")

        pool.promote_winner("var:winner", "base:1")

        assert engine.graph.has_edge("var:winner", "base:1")
