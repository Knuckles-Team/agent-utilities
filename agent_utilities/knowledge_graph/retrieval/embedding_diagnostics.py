#!/usr/bin/env python3
from __future__ import annotations

"""Embedding Alignment Diagnostics.

CONCEPT:KG-2.42 — Embedding Alignment Diagnostics

Multi-layer embedding quality analysis derived from MINER (arXiv:2605.06460v1).
Provides Centered Kernel Alignment (CKA), Alignment Ratio diagnostics,
adaptive sparse fusion, and continuous embedding health monitoring.
"""


import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CKAResult:
    """Result of Centered Kernel Alignment comparison.

    Attributes:
        cka_score: CKA similarity in [0, 1]. High = structurally similar spaces.
        alignment_ratio: AR = mean_cosine / CKA. High = directly usable.
        mean_cosine: Average pairwise cosine similarity between paired samples.
        needs_transformation: True if AR is low (structure present but misaligned).
    """

    cka_score: float = 0.0
    alignment_ratio: float = 0.0
    mean_cosine: float = 0.0
    needs_transformation: bool = False


@dataclass
class FusionResult:
    """Result of adaptive sparse multi-layer fusion.

    Attributes:
        fused_embeddings: The fused embedding matrix (N × D).
        layer_weights: Learned weight per input layer.
        active_dimensions: Number of non-masked dimensions per layer.
        sparsity_ratio: Fraction of dimensions masked out.
    """

    fused_embeddings: list[list[float]] = field(default_factory=list)
    layer_weights: list[float] = field(default_factory=list)
    active_dimensions: list[int] = field(default_factory=list)
    sparsity_ratio: float = 0.0


@dataclass
class EmbeddingHealthReport:
    """Health report for an embedding space.

    Attributes:
        effective_dimensionality: Number of significant singular values.
        total_dimensions: Total embedding dimensions.
        collapse_detected: True if effective dim << total dim.
        cka_vs_baseline: CKA score vs. the baseline snapshot.
        drift_severity: 'none', 'mild', 'severe'.
        recommendation: Suggested action (e.g., 're-embed', 'monitor').
    """

    effective_dimensionality: int = 0
    total_dimensions: int = 0
    collapse_detected: bool = False
    cka_vs_baseline: float = 1.0
    drift_severity: str = "none"
    recommendation: str = "healthy"


class EmbeddingDiagnostics:
    """Multi-layer embedding quality diagnostics engine.

    CONCEPT:KG-2.42 — Embedding Alignment Diagnostics

    Adapted from MINER's layerwise analysis framework.  Provides tools to
    measure, compare, and fuse embedding spaces from multiple sources
    (e.g., title vs content vs graph-structural embeddings).
    """

    @staticmethod
    def compute_cka(
        X: np.ndarray | list[list[float]],
        Y: np.ndarray | list[list[float]],
    ) -> CKAResult:
        """Compute linear Centered Kernel Alignment between two embedding spaces.

        CONCEPT:KG-2.42 — CKA Diagnostic (MINER §3, Eq. 1)

        CKA measures structural similarity between two representation spaces
        at the dataset level.  It is invariant to orthogonal transformations
        and isotropic scaling, making it robust to coordinate mismatches.

        Formula: CKA(X, Y) = ||X^T Y||_F^2 / (||X^T X||_F · ||Y^T Y||_F)

        Args:
            X: First embedding matrix (N samples × D dimensions).
            Y: Second embedding matrix (N samples × D' dimensions).

        Returns:
            CKAResult with cka_score, alignment_ratio, and guidance.
        """
        X_arr = np.array(X, dtype=np.float64)
        Y_arr = np.array(Y, dtype=np.float64)

        if X_arr.shape[0] != Y_arr.shape[0]:
            logger.warning(
                "CKA: sample count mismatch (%d vs %d)", X_arr.shape[0], Y_arr.shape[0]
            )
            return CKAResult()

        if X_arr.shape[0] < 2:
            return CKAResult(cka_score=1.0, alignment_ratio=1.0, mean_cosine=1.0)

        # Center the matrices
        X_c = X_arr - X_arr.mean(axis=0)
        Y_c = Y_arr - Y_arr.mean(axis=0)

        # Compute CKA
        cross = np.linalg.norm(X_c.T @ Y_c, "from") ** 2
        xx = np.linalg.norm(X_c.T @ X_c, "from")
        yy = np.linalg.norm(Y_c.T @ Y_c, "from")

        denom = xx * yy
        cka = float(cross / denom) if denom > 0 else 0.0
        cka = min(1.0, max(0.0, cka))

        # Compute mean pairwise cosine for Alignment Ratio
        mean_cos = EmbeddingDiagnostics._mean_pairwise_cosine(X_arr, Y_arr)

        ar = float(mean_cos / cka) if cka > 0 else 0.0
        needs_transform = ar < 0.5

        return CKAResult(
            cka_score=cka,
            alignment_ratio=ar,
            mean_cosine=mean_cos,
            needs_transformation=needs_transform,
        )

    @staticmethod
    def _mean_pairwise_cosine(X: np.ndarray, Y: np.ndarray) -> float:
        """Mean per-sample cosine similarity between paired rows."""
        # Truncate or pad to same dimensionality
        min_d = min(X.shape[1], Y.shape[1])
        X_t = X[:, :min_d]
        Y_t = Y[:, :min_d]

        x_norms = np.linalg.norm(X_t, axis=1, keepdims=True)
        y_norms = np.linalg.norm(Y_t, axis=1, keepdims=True)
        x_norms = np.where(x_norms == 0, 1.0, x_norms)
        y_norms = np.where(y_norms == 0, 1.0, y_norms)

        cosines = np.sum((X_t / x_norms) * (Y_t / y_norms), axis=1)
        return float(np.mean(cosines))

    @staticmethod
    def adaptive_sparse_fusion(
        embedding_layers: list[np.ndarray | list[list[float]]],
        performance_scores: list[float] | None = None,
        sparsity_target: float = 0.3,
    ) -> FusionResult:
        """Fuse multiple embedding layers with performance-adaptive neuron masking.

        CONCEPT:KG-2.42 — Adaptive Sparse Fusion (MINER §4.2)

        For each embedding layer, applies neuron-level masking to retain
        only the most informative dimensions, then aggregates via learned
        cross-layer weighted sum.

        Args:
            embedding_layers: List of embedding matrices (each N × D).
            performance_scores: Optional per-layer quality scores for weighting.
                If None, uses uniform weights.
            sparsity_target: Target fraction of dimensions to mask (0.0–1.0).

        Returns:
            FusionResult with fused embeddings and fusion metadata.
        """
        if not embedding_layers:
            return FusionResult()

        arrays = [np.array(layer, dtype=np.float64) for layer in embedding_layers]
        n_layers = len(arrays)
        n_samples = arrays[0].shape[0]

        # Determine common dimensionality (truncate to minimum)
        min_dim = min(a.shape[1] for a in arrays)
        arrays = [a[:, :min_dim] for a in arrays]

        # Compute per-layer weights
        if performance_scores and len(performance_scores) == n_layers:
            total = sum(performance_scores) or 1.0
            weights = [s / total for s in performance_scores]
        else:
            weights = [1.0 / n_layers] * n_layers

        # Neuron-level masking: for each layer, mask low-variance dimensions
        active_dims: list[int] = []
        masked_arrays: list[np.ndarray] = []

        for arr in arrays:
            variances = np.var(arr, axis=0)
            threshold_idx = int(min_dim * (1 - sparsity_target))
            threshold_idx = max(1, threshold_idx)

            # Keep top-variance dimensions
            top_indices = np.argsort(variances)[-threshold_idx:]
            mask = np.zeros(min_dim)
            mask[top_indices] = 1.0

            masked = arr * mask[np.newaxis, :]
            masked_arrays.append(masked)
            active_dims.append(int(np.sum(mask > 0)))

        # Weighted fusion
        fused = np.zeros((n_samples, min_dim))
        for i, (arr, w) in enumerate(zip(masked_arrays, weights, strict=False)):
            fused += w * arr

        # Normalize rows
        norms = np.linalg.norm(fused, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        fused = fused / norms

        total_possible = n_layers * min_dim
        total_active = sum(active_dims)
        actual_sparsity = (
            1.0 - (total_active / total_possible) if total_possible > 0 else 0.0
        )

        return FusionResult(
            fused_embeddings=fused.tolist(),
            layer_weights=weights,
            active_dimensions=active_dims,
            sparsity_ratio=actual_sparsity,
        )

    @staticmethod
    def embedding_health_check(
        current_embeddings: np.ndarray | list[list[float]],
        baseline_embeddings: np.ndarray | list[list[float]] | None = None,
        collapse_threshold: float = 0.1,
        drift_threshold: float = 0.7,
    ) -> EmbeddingHealthReport:
        """Continuous embedding health monitoring.

        CONCEPT:KG-2.42 — Embedding Health Monitor

        Monitors effective dimensionality (via SVD), detects collapse, and
        measures drift vs. a baseline snapshot using CKA.

        Args:
            current_embeddings: Current embedding matrix (N × D).
            baseline_embeddings: Optional baseline snapshot for drift detection.
            collapse_threshold: Ratio of effective_dim/total_dim below which
                collapse is flagged.
            drift_threshold: CKA score below which drift is flagged.

        Returns:
            EmbeddingHealthReport with diagnostics and recommendations.
        """
        arr = np.array(current_embeddings, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 2:
            return EmbeddingHealthReport(recommendation="insufficient_data")

        n, d = arr.shape

        # SVD-based effective dimensionality
        centered = arr - arr.mean(axis=0)
        try:
            _, s, _ = np.linalg.svd(centered, full_matrices=False)
            # Effective dimensionality: count singular values that contribute
            # meaningfully (> 1% of the max)
            s_normalized = s / (s[0] if s[0] > 0 else 1.0)
            effective_dim = int(np.sum(s_normalized > 0.01))
        except np.linalg.LinAlgError:
            effective_dim = d

        collapse = effective_dim / d < collapse_threshold if d > 0 else False

        # CKA drift vs baseline
        cka_baseline = 1.0
        drift = "none"
        if baseline_embeddings is not None:
            cka_result = EmbeddingDiagnostics.compute_cka(arr, baseline_embeddings)
            cka_baseline = cka_result.cka_score
            if cka_baseline < drift_threshold * 0.5:
                drift = "severe"
            elif cka_baseline < drift_threshold:
                drift = "mild"

        # Recommendation
        if collapse:
            rec = "re-diversify_embeddings"
        elif drift == "severe":
            rec = "re-embed_all_nodes"
        elif drift == "mild":
            rec = "monitor_closely"
        else:
            rec = "healthy"

        return EmbeddingHealthReport(
            effective_dimensionality=effective_dim,
            total_dimensions=d,
            collapse_detected=collapse,
            cka_vs_baseline=cka_baseline,
            drift_severity=drift,
            recommendation=rec,
        )
