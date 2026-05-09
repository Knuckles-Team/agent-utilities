"""Tests for CONCEPT:KG-2.43 — Structural Causal Reasoning Engine."""

import pytest

from agent_utilities.knowledge_graph.core.causal_reasoning import (
    CausalEdge,
    CausalFactor,
    CausalRelationType,
    CausalVerifier,
    CounterfactualGenerator,
    SpuriousnessDetector,
    StructuralCausalModel,
    trajectory_causal_alignment_score,
)


@pytest.fixture
def medical_scm():
    """Build a simple medical causal model: Smoking → Cancer, Genetics → Cancer."""
    scm = StructuralCausalModel()
    scm.add_factor(CausalFactor(id="smoking", name="Smoking", value="yes"))
    scm.add_factor(
        CausalFactor(id="genetics", name="Genetic Predisposition", value="high")
    )
    scm.add_factor(CausalFactor(id="cancer", name="Cancer", value="positive"))
    scm.add_factor(CausalFactor(id="treatment", name="Treatment", value="chemo"))
    scm.add_factor(CausalFactor(id="outcome", name="Outcome", value="remission"))

    scm.add_edge(
        CausalEdge(source_id="smoking", target_id="cancer", mechanism="carcinogens")
    )
    scm.add_edge(
        CausalEdge(source_id="genetics", target_id="cancer", mechanism="mutations")
    )
    scm.add_edge(
        CausalEdge(source_id="cancer", target_id="treatment", mechanism="diagnosis")
    )
    scm.add_edge(
        CausalEdge(source_id="treatment", target_id="outcome", mechanism="therapy")
    )
    return scm


class TestStructuralCausalModel:
    """Tests for SCM construction and operations."""

    def test_build_scm(self, medical_scm):
        assert medical_scm.factor_count == 5
        assert medical_scm.edge_count == 4

    def test_cycle_prevention(self, medical_scm):
        with pytest.raises(ValueError, match="cycle"):
            medical_scm.add_edge(
                CausalEdge(
                    source_id="outcome",
                    target_id="smoking",
                )
            )

    def test_do_intervention(self, medical_scm):
        mutilated = medical_scm.do_intervention("cancer", "negative")
        # After do(cancer=negative), cancer has no parents
        parents = list(mutilated.predecessors("cancer"))
        assert len(parents) == 0

    def test_causal_ancestors(self, medical_scm):
        ancestors = medical_scm.get_causal_ancestors("outcome")
        assert "treatment" in ancestors
        assert "cancer" in ancestors
        assert "smoking" in ancestors

    def test_causal_descendants(self, medical_scm):
        descendants = medical_scm.get_causal_descendants("smoking")
        assert "cancer" in descendants
        assert "outcome" in descendants

    def test_topological_order(self, medical_scm):
        order = medical_scm.topological_causal_order()
        assert order.index("smoking") < order.index("cancer")
        assert order.index("cancer") < order.index("treatment")

    def test_d_separation_collider(self):
        # Build a simple collider: A → C ← B
        scm = StructuralCausalModel()
        scm.add_factor(CausalFactor(id="A", name="A"))
        scm.add_factor(CausalFactor(id="B", name="B"))
        scm.add_factor(CausalFactor(id="C", name="C"))
        scm.add_edge(CausalEdge(source_id="A", target_id="C"))
        scm.add_edge(CausalEdge(source_id="B", target_id="C"))
        # A ⊥ B | {} (collider blocks)
        assert scm.is_d_separated("A", "B", set())
        # A ⊥̸ B | {C} (conditioning on collider opens path)
        assert not scm.is_d_separated("A", "B", {"C"})

    def test_d_separation_chain(self):
        # Chain: A → B → C
        scm = StructuralCausalModel()
        scm.add_factor(CausalFactor(id="A", name="A"))
        scm.add_factor(CausalFactor(id="B", name="B"))
        scm.add_factor(CausalFactor(id="C", name="C"))
        scm.add_edge(CausalEdge(source_id="A", target_id="B"))
        scm.add_edge(CausalEdge(source_id="B", target_id="C"))
        # A ⊥ C | {B} (chain blocked by B)
        assert scm.is_d_separated("A", "C", {"B"})
        # A ⊥̸ C | {} (chain unblocked)
        assert not scm.is_d_separated("A", "C", set())


class TestCausalVerifier:
    """Tests for causal verification protocol (MedCausalX §3.2)."""

    def test_valid_chain(self, medical_scm):
        verifier = CausalVerifier(medical_scm)
        steps = [
            {"cause": "smoking", "effect": "cancer"},
            {"cause": "cancer", "effect": "treatment"},
            {"cause": "treatment", "effect": "outcome"},
        ]
        result = verifier.verify_chain(steps)
        assert result.is_consistent
        assert result.consistency_score == 1.0

    def test_reversed_causality(self, medical_scm):
        verifier = CausalVerifier(medical_scm)
        steps = [{"cause": "cancer", "effect": "smoking"}]  # Reversed!
        result = verifier.verify_chain(steps)
        assert not result.is_consistent
        assert "Reversed" in result.violations[0]

    def test_no_causal_path(self, medical_scm):
        verifier = CausalVerifier(medical_scm)
        steps = [{"cause": "genetics", "effect": "outcome"}]  # Indirect only
        result = verifier.verify_chain(steps)
        assert "intermediaries" in result.violations[0] or len(result.violations) > 0

    def test_empty_chain(self, medical_scm):
        verifier = CausalVerifier(medical_scm)
        result = verifier.verify_chain([])
        assert result.is_consistent
        assert result.consistency_score == 1.0


class TestSpuriousnessDetector:
    """Tests for spurious correlation detection."""

    def test_detect_valid_edge(self, medical_scm):
        detector = SpuriousnessDetector(medical_scm)
        results = detector.detect_spurious_edges([("smoking", "cancer")])
        assert not results[0]["is_spurious"]

    def test_detect_missing_node(self, medical_scm):
        detector = SpuriousnessDetector(medical_scm)
        results = detector.detect_spurious_edges([("unknown", "cancer")])
        assert results[0]["is_spurious"]


class TestCounterfactualGenerator:
    """Tests for counterfactual generation (MedCausalX §3.1)."""

    def test_generate_counterfactuals(self, medical_scm):
        gen = CounterfactualGenerator(medical_scm)
        queries = gen.generate_counterfactuals("outcome")
        assert len(queries) > 0
        assert all(q.target_node == "outcome" for q in queries)

    def test_closest_ancestors_first(self, medical_scm):
        gen = CounterfactualGenerator(medical_scm)
        queries = gen.generate_counterfactuals("outcome", max_interventions=2)
        assert queries[0].intervention_node == "treatment"  # Closest

    def test_missing_target(self, medical_scm):
        gen = CounterfactualGenerator(medical_scm)
        queries = gen.generate_counterfactuals("nonexistent")
        assert len(queries) == 0


class TestTrajectoryAlignment:
    """Tests for trajectory-level causal alignment scoring."""

    def test_aligned_trajectory(self, medical_scm):
        steps = [
            {"cause": "smoking", "effect": "cancer"},
            {"cause": "cancer", "effect": "treatment"},
        ]
        score = trajectory_causal_alignment_score(steps, medical_scm)
        assert score == 1.0

    def test_misaligned_trajectory(self, medical_scm):
        steps = [
            {"cause": "outcome", "effect": "smoking"},  # Wrong direction
        ]
        score = trajectory_causal_alignment_score(steps, medical_scm)
        assert score < 1.0

    def test_empty_trajectory(self, medical_scm):
        score = trajectory_causal_alignment_score([], medical_scm)
        assert score == 1.0
