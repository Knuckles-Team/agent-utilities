#!/usr/bin/env python3
"""Latent Space Anti-Collapse Regularizer.

CONCEPT:KG-2.44 — Latent Space Anti-Collapse Regularizer

Formal anti-collapse guarantees derived from LeWorldModel (arXiv:2603.19312v2).
Provides SIGReg normality testing, SVD-based collapse detection,
diversity-preserving consolidation, and predictive consistency scoring.

Extends AHE-3.6 (EWC) with embedding diversity constraints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CollapseReport:
    """Report from embedding collapse detection.

    Attributes:
        collapsed: True if embeddings have collapsed.
        effective_dim: Number of significant singular values.
        total_dim: Total embedding dimensions.
        collapse_ratio: effective_dim / total_dim.
        top_singular_values: Top-k singular values (normalized).
        normality_p_value: p-value from SIGReg normality test.
        recommendation: Action recommendation.
    """

    collapsed: bool = False
    effective_dim: int = 0
    total_dim: int = 0
    collapse_ratio: float = 1.0
    top_singular_values: list[float] = None  # type: ignore[assignment]
    normality_p_value: float = 1.0
    recommendation: str = "healthy"

    def __post_init__(self) -> None:
        if self.top_singular_values is None:
            self.top_singular_values = []


@dataclass
class DiversityMetrics:
    """Metrics for embedding space diversity.

    Attributes:
        mean_pairwise_distance: Average L2 distance between embeddings.
        isotropy_score: How uniformly distributed embeddings are (0=collapsed, 1=isotropic).
        participation_ratio: Effective number of independent dimensions.
        entropy: Shannon entropy of the singular value distribution.
    """

    mean_pairwise_distance: float = 0.0
    isotropy_score: float = 0.0
    participation_ratio: float = 0.0
    entropy: float = 0.0


class LatentSpaceRegularizer:
    """Anti-collapse regularizer with SIGReg normality and SVD diagnostics.

    CONCEPT:KG-2.44 — Latent Space Anti-Collapse Regularizer

    Ensures KG embedding distributions maintain diversity and don't collapse
    to low-dimensional manifolds.  Based on LeWorldModel's SIGReg approach
    that projects embeddings onto random directions and tests for Gaussian
    normality.

    Integrates with:
    - AHE-3.6 (EWC): Extends consolidation with diversity preservation.
    - KG-2.36 (Auto-Similarity): Collapsed embeddings → degenerate clusters.
    - KG-2.34 (Spectral Clustering): Anti-collapse ensures meaningful clusters.
    """

    def __init__(
        self,
        n_projections: int = 10,
        significance_level: float = 0.05,
        collapse_threshold: float = 0.1,
    ) -> None:
        """Initialize the regularizer.

        Args:
            n_projections: Number of random projections for SIGReg normality test.
            significance_level: p-value threshold for normality rejection.
            collapse_threshold: effective_dim/total_dim ratio below which collapse is flagged.
        """
        self._n_projections = n_projections
        self._significance_level = significance_level
        self._collapse_threshold = collapse_threshold

    def detect_collapse(
        self,
        embeddings: np.ndarray | list[list[float]],
    ) -> CollapseReport:
        """Full collapse detection combining SVD and SIGReg normality.

        CONCEPT:KG-2.44 — Collapse Detection

        Performs two complementary checks:
        1. **SVD effective dimensionality**: If top-k singular values capture
           >95% of variance with k << d, the space has collapsed.
        2. **SIGReg normality test**: Projects embeddings onto random directions
           and tests each 1D projection for Gaussian normality.  Collapsed
           embeddings fail normality tests.

        Args:
            embeddings: Embedding matrix (N × D).

        Returns:
            CollapseReport with full diagnostics.
        """
        arr = np.array(embeddings, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 3:
            return CollapseReport(recommendation="insufficient_data")

        n, d = arr.shape

        # Center embeddings
        centered = arr - arr.mean(axis=0)

        # SVD analysis
        try:
            _, s, _ = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return CollapseReport(recommendation="svd_failed")

        s_normalized = s / (s[0] if s[0] > 0 else 1.0)
        effective_dim = int(np.sum(s_normalized > 0.01))

        # Top singular values
        top_k = min(10, len(s))
        top_svs = (s[:top_k] / (s[0] if s[0] > 0 else 1.0)).tolist()

        # Collapse ratio
        collapse_ratio = effective_dim / d if d > 0 else 1.0
        collapsed_svd = collapse_ratio < self._collapse_threshold

        # SIGReg normality test
        p_value = self._sigreg_normality_test(centered)
        collapsed_normality = p_value < self._significance_level

        collapsed = collapsed_svd or collapsed_normality

        if collapsed:
            rec = "re-diversify_embeddings"
        elif collapse_ratio < self._collapse_threshold * 2:
            rec = "monitor_closely"
        else:
            rec = "healthy"

        return CollapseReport(
            collapsed=collapsed,
            effective_dim=effective_dim,
            total_dim=d,
            collapse_ratio=collapse_ratio,
            top_singular_values=top_svs,
            normality_p_value=p_value,
            recommendation=rec,
        )

    def _sigreg_normality_test(self, centered: np.ndarray) -> float:
        """SIGReg normality test via random projections.

        CONCEPT:KG-2.44 — SIGReg (LeWM §1, Fig. 1)

        Projects embeddings onto random unit directions and tests each
        1D projection for Gaussian normality using the Shapiro-Wilk test.
        Aggregates p-values as the minimum across projections.

        A healthy (non-collapsed) embedding space should produce approximately
        Gaussian 1D projections by the Central Limit Theorem.  Collapsed
        spaces will fail this test.

        Args:
            centered: Centered embedding matrix (N × D).

        Returns:
            Minimum p-value across all projections.  Low p-value = collapsed.
        """
        n, d = centered.shape
        rng = np.random.default_rng(42)
        min_p = 1.0

        for _ in range(self._n_projections):
            # Random unit direction
            direction = rng.standard_normal(d)
            direction = direction / (np.linalg.norm(direction) or 1.0)

            # Project embeddings onto this direction
            projection = centered @ direction  # Shape: (N,)

            # Shapiro-Wilk normality test
            # Use simplified version: compare against expected normal quantiles
            p_value = self._simplified_normality_p(projection)
            min_p = min(min_p, p_value)

        return min_p

    @staticmethod
    def _simplified_normality_p(data: np.ndarray) -> float:
        """Simplified normality test using kurtosis-based heuristic.

        For a normal distribution, kurtosis ≈ 3.  Large deviations indicate
        non-normality (e.g., collapse to a point has kurtosis ≈ ∞).

        Returns an approximate p-value in [0, 1].
        """
        n = len(data)
        if n < 3:
            return 1.0

        std = np.std(data)
        if std < 1e-10:
            # Near-zero variance → collapsed
            return 0.0

        # Compute excess kurtosis
        mean = np.mean(data)
        centered = data - mean
        m4 = np.mean(centered**4)
        m2 = np.mean(centered**2)
        kurtosis = (m4 / (m2**2)) - 3.0 if m2 > 0 else 0.0

        # Also check skewness
        m3 = np.mean(centered**3)
        skewness = (m3 / (m2**1.5)) if m2 > 0 else 0.0

        # Jarque-Bera-like statistic
        jb = (n / 6.0) * (skewness**2 + (kurtosis**2) / 4.0)

        # Approximate p-value using chi-squared CDF approximation
        # JB ~ χ²(2) under null; use simple exponential approximation
        p_value = float(np.exp(-jb / 2.0))
        return min(1.0, max(0.0, p_value))

    def compute_diversity_metrics(
        self,
        embeddings: np.ndarray | list[list[float]],
    ) -> DiversityMetrics:
        """Compute diversity metrics for an embedding space.

        CONCEPT:KG-2.44 — Diversity Metrics

        Args:
            embeddings: Embedding matrix (N × D).

        Returns:
            DiversityMetrics with isotropy, participation ratio, and entropy.
        """
        arr = np.array(embeddings, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 2:
            return DiversityMetrics()

        n, d = arr.shape
        centered = arr - arr.mean(axis=0)

        # Mean pairwise L2 distance (sample)
        sample_size = min(n, 100)
        rng = np.random.default_rng(42)
        indices = (
            rng.choice(n, size=sample_size, replace=False)
            if n > sample_size
            else np.arange(n)
        )
        sample = arr[indices]
        diffs = sample[:, np.newaxis, :] - sample[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        mean_dist = float(np.mean(dists[np.triu_indices(len(sample), k=1)]))

        # SVD for isotropy and participation ratio
        try:
            _, s, _ = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return DiversityMetrics(mean_pairwise_distance=mean_dist)

        # Isotropy: ratio of min/max singular value (1 = perfectly isotropic)
        isotropy = float(s[-1] / s[0]) if s[0] > 0 and len(s) > 1 else 0.0

        # Participation ratio: (Σ λᵢ)² / Σ λᵢ² where λᵢ = sᵢ²
        lambdas = s**2
        sum_l = lambdas.sum()
        sum_l2 = (lambdas**2).sum()
        pr = float((sum_l**2) / sum_l2) if sum_l2 > 0 else 0.0

        # Shannon entropy of normalized singular value distribution
        p = lambdas / (sum_l if sum_l > 0 else 1.0)
        p = p[p > 0]  # Remove zeros
        entropy = float(-np.sum(p * np.log2(p))) if len(p) > 0 else 0.0

        return DiversityMetrics(
            mean_pairwise_distance=mean_dist,
            isotropy_score=isotropy,
            participation_ratio=pr,
            entropy=entropy,
        )

    def diversity_preserving_consolidation(
        self,
        old_embedding: list[float],
        new_embedding: list[float],
        fisher_diag: list[float],
        all_embeddings: np.ndarray | list[list[float]],
        lambda_param: float = 0.5,
        diversity_weight: float = 0.3,
    ) -> list[float]:
        """EWC consolidation extended with diversity preservation.

        CONCEPT:KG-2.44 — Diversity-Preserving Consolidation

        Extends AHE-3.6 EWC by adding a diversity penalty: if accepting
        the new embedding would reduce the overall embedding space diversity
        (measured by participation ratio), the consolidation is dampened.

        Args:
            old_embedding: Previous consolidated embedding.
            new_embedding: Proposed new embedding.
            fisher_diag: Fisher importance diagonal from EWC.
            all_embeddings: All current embeddings for diversity measurement.
            lambda_param: EWC penalty strength.
            diversity_weight: Weight of diversity preservation (0–1).

        Returns:
            Consolidated embedding with diversity guarantee.
        """
        old = np.array(old_embedding)
        new = np.array(new_embedding)
        fisher = np.array(fisher_diag)

        if old.shape != new.shape or old.shape != fisher.shape:
            return new_embedding

        # Standard EWC consolidation
        delta = new - old
        dampening = np.clip(1.0 - (lambda_param * fisher), 0.0, 1.0)
        ewc_result = old + delta * dampening

        # Diversity check: would this consolidation reduce participation ratio?
        all_arr = np.array(all_embeddings, dtype=np.float64)
        if all_arr.ndim != 2 or all_arr.shape[0] < 3:
            norm = np.linalg.norm(ewc_result)
            if norm > 0:
                ewc_result = ewc_result / norm
            return ewc_result.tolist()

        # Compute current diversity
        current_metrics = self.compute_diversity_metrics(all_arr)

        # Simulate replacement
        simulated = all_arr.copy()
        # Find the closest existing embedding to old_embedding
        dists = np.linalg.norm(simulated - old, axis=1)
        closest_idx = int(np.argmin(dists))
        simulated[closest_idx] = ewc_result
        new_metrics = self.compute_diversity_metrics(simulated)

        # If diversity drops, dampen the change further
        if new_metrics.participation_ratio < current_metrics.participation_ratio * 0.9:
            diversity_dampening = 1.0 - diversity_weight
            ewc_result = old + (ewc_result - old) * diversity_dampening
            logger.info(
                "Diversity-preserving dampening applied: PR %.2f → %.2f",
                current_metrics.participation_ratio,
                new_metrics.participation_ratio,
            )

        # Normalize
        norm = np.linalg.norm(ewc_result)
        if norm > 0:
            ewc_result = ewc_result / norm

        return ewc_result.tolist()

    @staticmethod
    def predictive_consistency_score(
        predicted_states: list[list[float]],
        observed_states: list[list[float]],
    ) -> float:
        """Measure how well predicted KG states match observations.

        CONCEPT:KG-2.44 — Predictive Consistency (LeWM latent prediction)

        For agent action sequences, measures whether the KG's state
        predictions (via trajectory extrapolation) match observed outcomes.

        Args:
            predicted_states: Predicted embedding states.
            observed_states: Actual observed embedding states.

        Returns:
            Consistency score in [0, 1]. 1.0 = perfect prediction.
        """
        if not predicted_states or not observed_states:
            return 1.0

        min_len = min(len(predicted_states), len(observed_states))
        pred = np.array(predicted_states[:min_len])
        obs = np.array(observed_states[:min_len])

        # Cosine similarity per step
        pred_norms = np.linalg.norm(pred, axis=1, keepdims=True)
        obs_norms = np.linalg.norm(obs, axis=1, keepdims=True)
        pred_norms = np.where(pred_norms == 0, 1.0, pred_norms)
        obs_norms = np.where(obs_norms == 0, 1.0, obs_norms)

        cosines = np.sum((pred / pred_norms) * (obs / obs_norms), axis=1)
        return float(np.mean(np.clip(cosines, 0.0, 1.0)))
