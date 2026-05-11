#!/usr/bin/env python3
from __future__ import annotations

"""Knowledge Stability Engine.

CONCEPT:AHE-3.4 — Knowledge Stability Engine

Consolidated engine for maintaining embedding health across the Knowledge Graph.
Combines four previously separate capabilities into a single configurable engine:

1. **EWC Consolidation** (was ewc.py) — Fisher-proxy diagonal approximation
   to prevent catastrophic forgetting when adapting node embeddings.
2. **Temporal Drift Detection** (was drift_tracker.py) — Measures shift in
   node embeddings over time to warn against concept drift.
3. **Anti-Collapse Regularization** (was latent_space_regularizer.py) —
   SIGReg normality testing, SVD-based collapse detection, and
   diversity-preserving consolidation. Derived from LeWorldModel.
4. **Embedding Alignment Diagnostics** (was embedding_diagnostics.py) —
   CKA alignment, multi-layer fusion, and continuous health monitoring.
   Derived from MINER.

All four operate on the same ``list[list[float]]`` embedding matrices
using numpy and produce diagnostic reports. This consolidation enables
a single ``KnowledgeStabilityEngine`` to run the full diagnostic +
corrective pipeline in one call.

Integrates with:
    - KG-2.36 (Auto-Similarity): Collapsed embeddings → degenerate clusters
    - KG-2.34 (Spectral Clustering): Anti-collapse ensures meaningful clusters
    - KG-2.37 (Hybrid Search): Embedding quality affects retrieval

See docs/pillars/3_agentic_harness_engineering/AHE-3.6*.md
"""

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class DriftReport:
    """Report detailing detected knowledge drift."""

    node_id: str
    coefficient_of_variation: float
    cosine_shift: float
    has_drifted: bool


@dataclass
class CollapseReport:
    """Report from embedding collapse detection."""

    collapsed: bool = False
    effective_dim: int = 0
    total_dim: int = 0
    collapse_ratio: float = 1.0
    top_singular_values: list[float] = field(default_factory=list)
    normality_p_value: float = 1.0
    recommendation: str = "healthy"


@dataclass
class DiversityMetrics:
    """Metrics for embedding space diversity."""

    mean_pairwise_distance: float = 0.0
    isotropy_score: float = 0.0
    participation_ratio: float = 0.0
    entropy: float = 0.0


@dataclass
class CKAResult:
    """Result of Centered Kernel Alignment comparison."""

    cka_score: float = 0.0
    alignment_ratio: float = 0.0
    mean_cosine: float = 0.0
    needs_transformation: bool = False


@dataclass
class FusionResult:
    """Result of adaptive sparse multi-layer fusion."""

    fused_embeddings: list[list[float]] = field(default_factory=list)
    layer_weights: list[float] = field(default_factory=list)
    active_dimensions: list[int] = field(default_factory=list)
    sparsity_ratio: float = 0.0


@dataclass
class EmbeddingHealthReport:
    """Comprehensive health report for an embedding space."""

    effective_dimensionality: int = 0
    total_dimensions: int = 0
    collapse_detected: bool = False
    cka_vs_baseline: float = 1.0
    drift_severity: str = "none"
    recommendation: str = "healthy"


@dataclass
class StabilityReport:
    """Unified report from the full stability pipeline."""

    drift: DriftReport | None = None
    collapse: CollapseReport | None = None
    diversity: DiversityMetrics | None = None
    health: EmbeddingHealthReport | None = None
    overall_status: str = "healthy"


# ---------------------------------------------------------------------------
# EWC functions (from ewc.py)
# ---------------------------------------------------------------------------


def compute_fisher_diagonal_proxy(
    embeddings_history: list[list[float]],
) -> list[float]:
    """Compute a lightweight proxy for the Fisher Information diagonal.

    Uses inverse variance across historical embeddings. Low variance
    implies the dimension was consistently important for past representations.

    Args:
        embeddings_history: A list of historical embeddings for a node.

    Returns:
        A list of floats representing the diagonal Fisher proxy.
    """
    if len(embeddings_history) < 2:
        dim = len(embeddings_history[0]) if embeddings_history else 1536
        return [0.1] * dim

    arr = np.array(embeddings_history)
    variances = np.var(arr, axis=0)
    epsilon = 1e-6
    fisher_proxy = 1.0 / (variances + epsilon)
    max_val = np.max(fisher_proxy)
    if max_val > 0:
        fisher_proxy = fisher_proxy / max_val
    return fisher_proxy.tolist()


def apply_ewc_consolidation(
    old_embedding: list[float],
    new_embedding: list[float],
    fisher_diag: list[float],
    lambda_param: float = 0.5,
) -> list[float]:
    """Apply EWC penalty to consolidate an embedding update.

    Prevents catastrophic forgetting by resisting changes to dimensions
    that have a high Fisher importance score.

    Args:
        old_embedding: The previous (consolidated) embedding.
        new_embedding: The proposed new embedding.
        fisher_diag: The diagonal Fisher importance matrix.
        lambda_param: The EWC penalty strength (0.0 to 1.0).

    Returns:
        The consolidated new embedding.
    """
    if not old_embedding or not new_embedding or not fisher_diag:
        return new_embedding

    old_vec = np.array(old_embedding)
    new_vec = np.array(new_embedding)
    fisher_vec = np.array(fisher_diag)

    if old_vec.shape != new_vec.shape or old_vec.shape != fisher_vec.shape:
        logger.warning("Dimension mismatch in EWC consolidation. Bypassing EWC.")
        return new_embedding

    delta = new_vec - old_vec
    dampening = np.clip(1.0 - (lambda_param * fisher_vec), 0.0, 1.0)
    consolidated_vec = old_vec + delta * dampening

    norm = np.linalg.norm(consolidated_vec)
    if norm > 0:
        consolidated_vec = consolidated_vec / norm
    return consolidated_vec.tolist()


# ---------------------------------------------------------------------------
# Drift detection (from drift_tracker.py)
# ---------------------------------------------------------------------------


def calculate_cosine_distance(vec_a: list[float], vec_b: list[float]) -> float:
    """Calculate the cosine distance between two vectors."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    cosine_sim = np.dot(a, b) / (norm_a * norm_b)
    return 1.0 - cosine_sim


def check_knowledge_drift(
    node_id: str,
    historical_embeddings: list[list[float]],
    current_embedding: list[float],
    drift_threshold: float = 0.15,
) -> DriftReport:
    """Detect if a node's embedding has drifted significantly over time.

    Args:
        node_id: The ID of the node.
        historical_embeddings: List of past embeddings for this node.
        current_embedding: The node's current embedding.
        drift_threshold: The cosine distance threshold to trigger a drift warning.

    Returns:
        DriftReport containing the analysis metrics.
    """
    if not historical_embeddings or not current_embedding:
        return DriftReport(node_id, 0.0, 0.0, False)

    baseline = historical_embeddings[0]
    cosine_shift = calculate_cosine_distance(baseline, current_embedding)

    all_embs = np.array(historical_embeddings + [current_embedding])
    std_devs = np.std(all_embs, axis=0)
    means = np.mean(all_embs, axis=0)
    safe_means = np.where(means == 0, 1e-10, means)
    cv = np.mean(std_devs / np.abs(safe_means))

    has_drifted = bool(cosine_shift > drift_threshold)
    if has_drifted:
        logger.info(
            f"Knowledge drift detected for {node_id}: Shift={cosine_shift:.3f}, CV={cv:.3f}"
        )
    return DriftReport(node_id, cv, float(cosine_shift), has_drifted)


# ---------------------------------------------------------------------------
# KnowledgeStabilityEngine (consolidated from latent_space_regularizer.py
# and embedding_diagnostics.py)
# ---------------------------------------------------------------------------


class KnowledgeStabilityEngine:
    """Consolidated engine for embedding stability, diagnostics, and repair.

    CONCEPT:AHE-3.4 — Knowledge Stability Engine

    Provides:
    - ``detect_drift()`` — Temporal drift detection
    - ``detect_collapse()`` — SVD + SIGReg collapse detection
    - ``compute_diversity()`` — Embedding space diversity metrics
    - ``consolidate_ewc()`` — EWC with optional diversity preservation
    - ``compute_cka()`` — Centered Kernel Alignment between spaces
    - ``adaptive_sparse_fusion()`` — Multi-layer embedding fusion
    - ``health_check()`` — Full health report
    - ``run_full_pipeline()`` — Unified stability assessment
    """

    def __init__(
        self,
        n_projections: int = 10,
        significance_level: float = 0.05,
        collapse_threshold: float = 0.1,
        drift_threshold: float = 0.15,
    ) -> None:
        self._n_projections = n_projections
        self._significance_level = significance_level
        self._collapse_threshold = collapse_threshold
        self._drift_threshold = drift_threshold

    # --- Drift detection ---

    def detect_drift(
        self,
        node_id: str,
        historical_embeddings: list[list[float]],
        current_embedding: list[float],
        threshold: float | None = None,
    ) -> DriftReport:
        """Detect temporal knowledge drift for a node."""
        return check_knowledge_drift(
            node_id,
            historical_embeddings,
            current_embedding,
            threshold or self._drift_threshold,
        )

    # --- Collapse detection (from latent_space_regularizer.py) ---

    def detect_collapse(
        self,
        embeddings: np.ndarray | list[list[float]],
    ) -> CollapseReport:
        """Full collapse detection combining SVD and SIGReg normality."""
        arr = np.array(embeddings, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 3:
            return CollapseReport(recommendation="insufficient_data")

        n, d = arr.shape
        centered = arr - arr.mean(axis=0)

        try:
            _, s, _ = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return CollapseReport(recommendation="svd_failed")

        s_normalized = s / (s[0] if s[0] > 0 else 1.0)
        effective_dim = int(np.sum(s_normalized > 0.01))
        top_k = min(10, len(s))
        top_svs = (s[:top_k] / (s[0] if s[0] > 0 else 1.0)).tolist()
        collapse_ratio = effective_dim / d if d > 0 else 1.0
        collapsed_svd = collapse_ratio < self._collapse_threshold

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
        """SIGReg normality test via random projections."""
        n, d = centered.shape
        rng = np.random.default_rng(42)
        min_p = 1.0
        for _ in range(self._n_projections):
            direction = rng.standard_normal(d)
            direction = direction / (np.linalg.norm(direction) or 1.0)
            projection = centered @ direction
            p_value = self._simplified_normality_p(projection)
            min_p = float(min(min_p, p_value))
        return min_p

    @staticmethod
    def _simplified_normality_p(data: np.ndarray) -> float:
        """Simplified normality test using kurtosis-based heuristic."""
        n = len(data)
        if n < 3:
            return 1.0
        std = np.std(data)
        if std < 1e-10:
            return 0.0
        mean = np.mean(data)
        centered = data - mean
        m4 = np.mean(centered**4)
        m2 = np.mean(centered**2)
        kurtosis = (m4 / (m2**2)) - 3.0 if m2 > 0 else 0.0
        m3 = np.mean(centered**3)
        skewness = (m3 / (m2**1.5)) if m2 > 0 else 0.0
        jb = (n / 6.0) * (skewness**2 + (kurtosis**2) / 4.0)
        p_value = float(np.exp(-jb / 2.0))
        return min(1.0, max(0.0, p_value))

    # --- Diversity metrics ---

    def compute_diversity(
        self,
        embeddings: np.ndarray | list[list[float]],
    ) -> DiversityMetrics:
        """Compute diversity metrics for an embedding space."""
        arr = np.array(embeddings, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 2:
            return DiversityMetrics()

        n, d = arr.shape
        centered = arr - arr.mean(axis=0)

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

        try:
            _, s, _ = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return DiversityMetrics(mean_pairwise_distance=mean_dist)

        isotropy = float(s[-1] / s[0]) if s[0] > 0 and len(s) > 1 else 0.0
        lambdas = s**2
        sum_l = lambdas.sum()
        sum_l2 = (lambdas**2).sum()
        pr = float((sum_l**2) / sum_l2) if sum_l2 > 0 else 0.0
        p = lambdas / (sum_l if sum_l > 0 else 1.0)
        p = p[p > 0]
        entropy = float(-np.sum(p * np.log2(p))) if len(p) > 0 else 0.0

        return DiversityMetrics(
            mean_pairwise_distance=mean_dist,
            isotropy_score=isotropy,
            participation_ratio=pr,
            entropy=entropy,
        )

    # Backward-compatible alias for LatentSpaceRegularizer API
    compute_diversity_metrics = compute_diversity

    # --- EWC with diversity preservation ---

    def consolidate_ewc(
        self,
        old_embedding: list[float],
        new_embedding: list[float],
        fisher_diag: list[float],
        all_embeddings: np.ndarray | list[list[float]] | None = None,
        lambda_param: float = 0.5,
        diversity_weight: float = 0.3,
    ) -> list[float]:
        """EWC consolidation with optional diversity preservation.

        If ``all_embeddings`` is provided, applies diversity-preserving
        dampening to prevent participation ratio degradation.
        """
        if all_embeddings is None:
            return apply_ewc_consolidation(
                old_embedding, new_embedding, fisher_diag, lambda_param
            )

        old = np.array(old_embedding)
        new = np.array(new_embedding)
        fisher = np.array(fisher_diag)

        if old.shape != new.shape or old.shape != fisher.shape:
            return new_embedding

        delta = new - old
        dampening = np.clip(1.0 - (lambda_param * fisher), 0.0, 1.0)
        ewc_result = old + delta * dampening

        all_arr = np.array(all_embeddings, dtype=np.float64)
        if all_arr.ndim != 2 or all_arr.shape[0] < 3:
            norm = np.linalg.norm(ewc_result)
            if norm > 0:
                ewc_result = ewc_result / norm
            return ewc_result.tolist()

        current_metrics = self.compute_diversity(all_arr)
        simulated = all_arr.copy()
        dists = np.linalg.norm(simulated - old, axis=1)
        closest_idx = int(np.argmin(dists))
        simulated[closest_idx] = ewc_result
        new_metrics = self.compute_diversity(simulated)

        if new_metrics.participation_ratio < current_metrics.participation_ratio * 0.9:
            diversity_dampening = 1.0 - diversity_weight
            ewc_result = old + (ewc_result - old) * diversity_dampening
            logger.info(
                "Diversity-preserving dampening applied: PR %.2f → %.2f",
                current_metrics.participation_ratio,
                new_metrics.participation_ratio,
            )

        norm = np.linalg.norm(ewc_result)
        if norm > 0:
            ewc_result = ewc_result / norm
        return ewc_result.tolist()

    # Backward-compatible alias for LatentSpaceRegularizer API
    diversity_preserving_consolidation = consolidate_ewc

    # --- CKA diagnostics (from embedding_diagnostics.py) ---

    @staticmethod
    def compute_cka(
        X: np.ndarray | list[list[float]],
        Y: np.ndarray | list[list[float]],
    ) -> CKAResult:
        """Compute linear Centered Kernel Alignment between two embedding spaces."""
        X_arr = np.array(X, dtype=np.float64)
        Y_arr = np.array(Y, dtype=np.float64)

        if X_arr.shape[0] != Y_arr.shape[0]:
            logger.warning(
                "CKA: sample count mismatch (%d vs %d)", X_arr.shape[0], Y_arr.shape[0]
            )
            return CKAResult()
        if X_arr.shape[0] < 2:
            return CKAResult(cka_score=1.0, alignment_ratio=1.0, mean_cosine=1.0)

        X_c = X_arr - X_arr.mean(axis=0)
        Y_c = Y_arr - Y_arr.mean(axis=0)

        cross = np.linalg.norm(X_c.T @ Y_c) ** 2
        xx = np.linalg.norm(X_c.T @ X_c)
        yy = np.linalg.norm(Y_c.T @ Y_c)
        denom = xx * yy
        cka = float(cross / denom) if denom > 0 else 0.0
        cka = min(1.0, max(0.0, cka))

        mean_cos = KnowledgeStabilityEngine._mean_pairwise_cosine(X_arr, Y_arr)
        ar = float(mean_cos / cka) if cka > 0 else 0.0

        return CKAResult(
            cka_score=cka,
            alignment_ratio=ar,
            mean_cosine=mean_cos,
            needs_transformation=ar < 0.5,
        )

    @staticmethod
    def _mean_pairwise_cosine(X: np.ndarray, Y: np.ndarray) -> float:
        """Mean per-sample cosine similarity between paired rows."""
        min_d = min(X.shape[1], Y.shape[1])
        X_t = X[:, :min_d]
        Y_t = Y[:, :min_d]
        x_norms = np.linalg.norm(X_t, axis=1, keepdims=True)
        y_norms = np.linalg.norm(Y_t, axis=1, keepdims=True)
        x_norms = np.where(x_norms == 0, 1.0, x_norms)
        y_norms = np.where(y_norms == 0, 1.0, y_norms)
        cosines = np.sum((X_t / x_norms) * (Y_t / y_norms), axis=1)
        return float(np.mean(cosines))

    # --- Multi-layer fusion ---

    @staticmethod
    def adaptive_sparse_fusion(
        embedding_layers: list[np.ndarray | list[list[float]]],
        performance_scores: list[float] | None = None,
        sparsity_target: float = 0.3,
    ) -> FusionResult:
        """Fuse multiple embedding layers with performance-adaptive neuron masking."""
        if not embedding_layers:
            return FusionResult()

        arrays = [np.array(layer, dtype=np.float64) for layer in embedding_layers]
        n_layers = len(arrays)
        n_samples = arrays[0].shape[0]
        min_dim = min(a.shape[1] for a in arrays)
        arrays = [a[:, :min_dim] for a in arrays]

        if performance_scores and len(performance_scores) == n_layers:
            total = sum(performance_scores) or 1.0
            weights = [s / total for s in performance_scores]
        else:
            weights = [1.0 / n_layers] * n_layers

        active_dims: list[int] = []
        masked_arrays: list[np.ndarray] = []
        for arr in arrays:
            variances = np.var(arr, axis=0)
            threshold_idx = max(1, int(min_dim * (1 - sparsity_target)))
            top_indices = np.argsort(variances)[-threshold_idx:]
            mask = np.zeros(min_dim)
            mask[top_indices] = 1.0
            masked_arrays.append(arr * mask[np.newaxis, :])
            active_dims.append(int(np.sum(mask > 0)))

        fused = np.zeros((n_samples, min_dim))
        for arr, w in zip(masked_arrays, weights, strict=False):
            fused += w * arr
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

    # --- Health check ---

    def health_check(
        self,
        current_embeddings: np.ndarray | list[list[float]],
        baseline_embeddings: np.ndarray | list[list[float]] | None = None,
        drift_threshold: float = 0.7,
    ) -> EmbeddingHealthReport:
        """Continuous embedding health monitoring."""
        arr = np.array(current_embeddings, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 2:
            return EmbeddingHealthReport(recommendation="insufficient_data")

        n, d = arr.shape
        centered = arr - arr.mean(axis=0)

        try:
            _, s, _ = np.linalg.svd(centered, full_matrices=False)
            s_normalized = s / (s[0] if s[0] > 0 else 1.0)
            effective_dim = int(np.sum(s_normalized > 0.01))
        except np.linalg.LinAlgError:
            effective_dim = d

        collapse = effective_dim / d < self._collapse_threshold if d > 0 else False

        cka_baseline = 1.0
        drift = "none"
        if baseline_embeddings is not None:
            cka_result = self.compute_cka(arr, baseline_embeddings)
            cka_baseline = cka_result.cka_score
            if cka_baseline < drift_threshold * 0.5:
                drift = "severe"
            elif cka_baseline < drift_threshold:
                drift = "mild"

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

    # Backward-compatible static method for EmbeddingDiagnostics API
    @staticmethod
    def embedding_health_check(
        current_embeddings: np.ndarray | list[list[float]],
        baseline_embeddings: np.ndarray | list[list[float]] | None = None,
        collapse_threshold: float = 0.1,
        drift_threshold: float = 0.7,
    ) -> EmbeddingHealthReport:
        """Static backward-compatible wrapper for health_check()."""
        engine = KnowledgeStabilityEngine(collapse_threshold=collapse_threshold)
        return engine.health_check(
            current_embeddings, baseline_embeddings, drift_threshold
        )

    # --- Predictive consistency ---

    @staticmethod
    def predictive_consistency_score(
        predicted_states: list[list[float]],
        observed_states: list[list[float]],
    ) -> float:
        """Measure how well predicted KG states match observations."""
        if not predicted_states or not observed_states:
            return 1.0
        min_len = min(len(predicted_states), len(observed_states))
        pred = np.array(predicted_states[:min_len])
        obs = np.array(observed_states[:min_len])
        pred_norms = np.linalg.norm(pred, axis=1, keepdims=True)
        obs_norms = np.linalg.norm(obs, axis=1, keepdims=True)
        pred_norms = np.where(pred_norms == 0, 1.0, pred_norms)
        obs_norms = np.where(obs_norms == 0, 1.0, obs_norms)
        cosines = np.sum((pred / pred_norms) * (obs / obs_norms), axis=1)
        return float(np.mean(np.clip(cosines, 0.0, 1.0)))

    # --- Unified pipeline ---

    def run_full_pipeline(
        self,
        node_id: str,
        current_embedding: list[float],
        historical_embeddings: list[list[float]],
        all_embeddings: np.ndarray | list[list[float]] | None = None,
        baseline_embeddings: np.ndarray | list[list[float]] | None = None,
    ) -> StabilityReport:
        """Run the full stability assessment pipeline.

        Executes drift detection, collapse detection, diversity analysis,
        and health check in a single call.
        """
        drift = self.detect_drift(node_id, historical_embeddings, current_embedding)

        embeddings_for_analysis = (
            all_embeddings
            if all_embeddings is not None
            else historical_embeddings + [current_embedding]
        )
        collapse = self.detect_collapse(embeddings_for_analysis)
        diversity = self.compute_diversity(embeddings_for_analysis)
        health = self.health_check(embeddings_for_analysis, baseline_embeddings)

        # Determine overall status
        if collapse.collapsed or health.collapse_detected:
            overall = "critical"
        elif drift.has_drifted or health.drift_severity == "severe":
            overall = "degraded"
        elif health.drift_severity == "mild":
            overall = "warning"
        else:
            overall = "healthy"

        return StabilityReport(
            drift=drift,
            collapse=collapse,
            diversity=diversity,
            health=health,
            overall_status=overall,
        )
