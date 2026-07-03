"""Tests for CONCEPT:KG-2.6 — Latent Space Anti-Collapse Regularizer."""

from agent_utilities.numeric import xp as np
import pytest

from agent_utilities.knowledge_graph.memory import (
    MemoryOptimizationEngine as LatentSpaceRegularizer,
)


@pytest.fixture
def regularizer():
    return LatentSpaceRegularizer(
        n_projections=5, collapse_threshold=0.1, significance_level=0.01
    )


class TestCollapseDetection:
    """Tests for SVD + SIGReg collapse detection."""

    def test_healthy_embeddings(self, regularizer):
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((100, 20))
        report = regularizer.detect_collapse(embeddings)
        assert report.effective_dim > 5
        assert report.collapse_ratio > 0.5  # Not dimensionally collapsed

    def test_collapsed_embeddings(self, regularizer):
        # All rows nearly identical → collapse via normality test
        base = np.ones((100, 20))
        noise = np.random.randn(100, 20) * 1e-10
        report = regularizer.detect_collapse(base + noise)
        # SIGReg normality test detects collapse (p < significance)
        assert report.normality_p_value < 0.05 or report.collapsed

    def test_partial_collapse(self, regularizer):
        rng = np.random.default_rng(42)
        # Only 3 dimensions vary, rest constant
        embeddings = np.ones((100, 20))
        embeddings[:, :3] = rng.standard_normal((100, 3))
        report = regularizer.detect_collapse(embeddings)
        assert report.effective_dim <= 5
        assert report.collapse_ratio < 0.5

    def test_insufficient_data(self, regularizer):
        report = regularizer.detect_collapse([[1.0, 2.0]])
        assert report.recommendation == "insufficient_data"

    def test_singular_values_reported(self, regularizer):
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((50, 10))
        report = regularizer.detect_collapse(embeddings)
        assert len(report.top_singular_values) > 0
        assert report.top_singular_values[0] == pytest.approx(1.0)  # Normalized


class TestDiversityMetrics:
    """Tests for embedding diversity measurement."""

    def test_isotropic_distribution(self, regularizer):
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((200, 10))
        metrics = regularizer.compute_diversity_metrics(embeddings)
        assert metrics.isotropy_score > 0.1
        assert metrics.participation_ratio > 3.0
        assert metrics.entropy > 0.0

    def test_collapsed_distribution(self, regularizer):
        # Very low variance → near-zero mean pairwise distance
        embeddings = np.ones((50, 10)) + np.random.randn(50, 10) * 1e-8
        metrics = regularizer.compute_diversity_metrics(embeddings)
        assert metrics.mean_pairwise_distance < 1e-5  # Nearly zero distance

    def test_mean_pairwise_distance(self, regularizer):
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((30, 5))
        metrics = regularizer.compute_diversity_metrics(embeddings)
        assert metrics.mean_pairwise_distance > 0.0


class TestDiversityPreservingConsolidation:
    """Tests for EWC + diversity preservation."""

    def test_basic_synthesis(self, regularizer):
        old = [1.0, 0.0, 0.0, 0.0, 0.0]
        new = [0.0, 1.0, 0.0, 0.0, 0.0]
        fisher = [0.5, 0.5, 0.1, 0.1, 0.1]
        all_emb = np.random.randn(20, 5)
        result = regularizer.diversity_preserving_consolidation(
            old,
            new,
            fisher,
            all_emb,
        )
        assert len(result) == 5
        # Result should be normalized
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 0.01

    def test_synthesis_respects_fisher(self, regularizer):
        old = [1.0, 0.0, 0.0]
        new = [0.0, 1.0, 0.0]
        fisher = [10.0, 0.0, 0.0]  # High Fisher on dim 0 → resist change
        all_emb = np.random.randn(20, 3)
        result = regularizer.diversity_preserving_consolidation(
            old,
            new,
            fisher,
            all_emb,
            lambda_param=0.9,
        )
        # Dim 0 should stay closer to old value (1.0) due to high Fisher
        result_arr = np.array(result)
        assert result_arr[0] > 0.3  # Should resist fully changing to 0


class TestPredictiveConsistency:
    """Tests for predictive consistency scoring."""

    def test_perfect_prediction(self):
        states = [[1.0, 0.0], [0.0, 1.0]]
        score = LatentSpaceRegularizer.predictive_consistency_score(states, states)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_poor_prediction(self):
        pred = [[1.0, 0.0], [0.0, 1.0]]
        obs = [[-1.0, 0.0], [0.0, -1.0]]
        score = LatentSpaceRegularizer.predictive_consistency_score(pred, obs)
        assert score < 0.1

    def test_empty_input(self):
        score = LatentSpaceRegularizer.predictive_consistency_score([], [])
        assert score == 1.0
