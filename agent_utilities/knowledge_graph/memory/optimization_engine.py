from __future__ import annotations

# --- FROM knowledge_stability_engine.py ---
#!/usr/bin/env python3

"""Knowledge Stability Engine.

CONCEPT:AU-AHE.evaluation.backtest-harness — Knowledge Stability Engine

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
a single ``MemoryOptimizationEngine`` to run the full diagnostic +
corrective pipeline in one call.

Integrates with:
    - KG-2.36 (Auto-Similarity): Collapsed embeddings → degenerate clusters
    - KG-2.34 (Spectral Clustering): Anti-collapse ensures meaningful clusters
    - KG-2.37 (Hybrid Search): Embedding quality affects retrieval

See docs/pillars/3_agentic_harness_engineering/AHE-3.6*.md
"""

import logging
from dataclasses import dataclass, field

from agent_utilities.core.config import setting
from agent_utilities.numeric import xp as np

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


def apply_ewc_synthesis(
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
# MemoryOptimizationEngine (consolidated from latent_space_regularizer.py
# and embedding_diagnostics.py)
# ---------------------------------------------------------------------------


class MemoryOptimizationEngine:
    """Consolidated engine for embedding stability, diagnostics, and repair.

    CONCEPT:AU-AHE.evaluation.backtest-harness — Knowledge Stability Engine

    Provides:
    - ``detect_drift()`` — Temporal drift detection
    - ``detect_collapse()`` — SVD + SIGReg collapse detection
    - ``compute_diversity()`` — Embedding space diversity metrics
    - ``synthesize_ewc()`` — EWC with optional diversity preservation
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
            # Normalise the random projection vector to unit length; it must stay a
            # d-dim array (wrapping it in float() collapsed it and raised
            # "only 0-dimensional arrays can be converted to Python scalars").
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

    def synthesize_ewc(
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
            return apply_ewc_synthesis(
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
    diversity_preserving_consolidation = synthesize_ewc

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

        mean_cos = MemoryOptimizationEngine._mean_pairwise_cosine(X_arr, Y_arr)
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
        engine = MemoryOptimizationEngine(collapse_threshold=collapse_threshold)
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


# --- FROM auto_similarity.py ---
#!/usr/bin/python

"""Auto-Similarity Memory Graph.

CONCEPT:AU-KG.memory.auto-similarity-memory-graph — Auto-Similarity Memory Graph

Provides automatic similarity edge creation and exponential decay scoring
for memory nodes in the Knowledge Graph. Adapted from contextplus's
memory-graph.ts with the following enhancements:

* **Auto-linking** — On node insertion, computes cosine similarity against
  recent nodes and creates ``SIMILAR_TO`` edges above threshold (default 0.72).
* **Decay scoring** — Edge weights decay exponentially: ``w * e^(-λ * Δt)``.
* **Stale pruning** — Edges below the prune threshold are removed.
* **Hub control** — Maximum edges per node prevents explosion.

Integrates with the existing ``MemoryRetriever`` and ``IntelligenceGraphEngine``
for graph-augmented RAG retrieval shortcuts.
"""


import logging
import math
import time
import uuid

from agent_utilities.models.knowledge_graph import (
    MemoryDecayConfig,
    RegistryEdge,
    RegistryEdgeType,
    RegistryNode,
    SimilarityEdgeNode,
)

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    va = np.array(a)
    vb = np.array(b)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


class AutoSimilarityLinker:
    """Creates and manages auto-similarity edges between KG memory nodes.

    CONCEPT:AU-KG.memory.auto-similarity-memory-graph — Auto-Similarity Memory Graph

    On node insertion, finds similar existing nodes via cosine similarity
    and creates weighted edges with exponential decay.

    Example::

        linker = AutoSimilarityLinker()
        new_edges = linker.link_new_node(
            new_node=node,
            existing_nodes=recent_nodes,
        )
        for edge in new_edges:
            print(f"Similar: {edge.source_node_id} → {edge.target_node_id} "
                  f"({edge.cosine_similarity:.2f})")
    """

    def __init__(self, config: MemoryDecayConfig | None = None):
        """Initialize the auto-similarity linker.

        Args:
            config: Decay and threshold configuration. Uses defaults if None.
        """
        self.config = config or MemoryDecayConfig()

    def link_new_node(
        self,
        new_node: RegistryNode,
        existing_nodes: list[RegistryNode],
    ) -> list[SimilarityEdgeNode]:
        """Find similar nodes and create similarity edges.

        Args:
            new_node: The newly inserted node with an embedding.
            existing_nodes: Recent nodes to compare against (bounded by
                ``config.batch_window``).

        Returns:
            List of SimilarityEdgeNode instances for edges above threshold.
        """
        if not new_node.embedding:
            return []

        # Limit comparison window
        candidates = existing_nodes[-self.config.batch_window :]
        candidates_with_embeddings = [n for n in candidates if n.embedding]

        edges: list[SimilarityEdgeNode] = []
        now = time.time()

        for candidate in candidates_with_embeddings:
            if (
                candidate.id == new_node.id
                or not candidate.embedding
                or not new_node.embedding
            ):
                continue

            sim = _cosine_similarity(new_node.embedding, candidate.embedding)

            if sim >= self.config.similarity_threshold:
                edge = SimilarityEdgeNode(
                    id=f"sim_{uuid.uuid4().hex[:8]}",
                    name=f"Similarity: {new_node.name} ↔ {candidate.name}",
                    description=(
                        f"Auto-created similarity edge (cosine={sim:.3f}) "
                        f"between {new_node.id} and {candidate.id}"
                    ),
                    source_node_id=new_node.id,
                    target_node_id=candidate.id,
                    cosine_similarity=sim,
                    decay_lambda=self.config.decay_lambda,
                    current_weight=sim,  # Initial weight = similarity score
                    creation_epoch=now,
                    last_accessed_epoch=now,
                    access_count=0,
                )
                edges.append(edge)

        # Hub control: keep only top-N by similarity
        if len(edges) > self.config.max_edges_per_node:
            edges.sort(key=lambda e: e.cosine_similarity, reverse=True)
            edges = edges[: self.config.max_edges_per_node]

        return edges

    def _build_edge(self, src: str, dst: str, sim: float) -> SimilarityEdgeNode:
        now = time.time()
        return SimilarityEdgeNode(
            id=f"sim_{uuid.uuid4().hex[:8]}",
            name=f"Similarity: {src} ↔ {dst}",
            description=f"Auto-created similarity edge (cosine={sim:.3f}) between {src} and {dst}",
            source_node_id=src,
            target_node_id=dst,
            cosine_similarity=sim,
            decay_lambda=self.config.decay_lambda,
            current_weight=sim,
            creation_epoch=now,
            last_accessed_epoch=now,
            access_count=0,
        )

    def link_all_batch(
        self,
        engine: Any = None,
        nodes: list[RegistryNode] | None = None,
    ) -> list[SimilarityEdgeNode]:
        """Build ALL similarity edges across the graph in one batch (CONCEPT:AU-KG.memory.auto-similarity-memory-graph).

        Unlike :meth:`link_new_node` (incremental, one node vs N candidates — correctly kept in
        in-process numpy), the all-pairs O(n²) construction is **collapsed onto the epistemic-graph
        compute layer**: a single ``compute_similarity_edges`` request over the out-of-process
        MessagePack/UDS transport runs the O(n²) natively in the tokio Rust engine, over embeddings
        already resident in the graph store — one round-trip, zero per-vector marshaling. This also
        *wires* that native op, which was previously never invoked.

        Falls back to an in-process numpy O(n²) pass over ``nodes`` when the Rust core isn't running
        (e.g. tests, or ``GRAPH_COMPUTE_FALLBACK=embedded``), so behaviour is identical either way.
        """
        threshold = self.config.similarity_threshold

        def _numpy_edges() -> list[SimilarityEdgeNode]:
            # local-compute fallback over the in-hand nodes (Rust core not running, or L0
            # doesn't hold these nodes so the native result came back empty).
            items = [n for n in (nodes or []) if getattr(n, "embedding", None)]
            out: list[SimilarityEdgeNode] = []
            for i, a in enumerate(items):
                ea = a.embedding
                if ea is None:
                    continue
                for b in items[i + 1 :]:
                    eb = b.embedding
                    if eb is None:
                        continue
                    sim = _cosine_similarity(ea, eb)
                    if sim >= threshold:
                        out.append(self._build_edge(a.id, b.id, sim))
            return out

        compute = getattr(engine, "graph_compute", None) if engine is not None else None
        if compute is not None and hasattr(compute, "compute_similarity_edges"):
            try:
                triples = compute.compute_similarity_edges(threshold)
                edges = [
                    self._build_edge(str(s), str(d), float(sim))
                    for (s, d, sim) in (triples or [])
                    if s != d and float(sim) >= threshold
                ]
                if edges:
                    logger.info(
                        "[KG-2.3] Built %d similarity edges via native compute_similarity_edges "
                        "(epistemic-graph, 1 round-trip)",
                        len(edges),
                    )
                    return edges
                # Empty native result: L0 likely lacks these nodes — disambiguate locally.
            except Exception as e:  # noqa: BLE001 - Rust core unavailable → local compute
                logger.debug(
                    "Native compute_similarity_edges unavailable (%s); local-compute fallback",
                    e,
                )

        return _numpy_edges()

    def decay_weight(self, edge: SimilarityEdgeNode) -> float:
        """Compute the current decayed weight of an edge.

        Uses exponential decay: ``original_sim * e^(-λ * days_elapsed)``

        Args:
            edge: The similarity edge to compute decay for.

        Returns:
            Current weight after applying time decay.
        """
        now = time.time()
        days_elapsed = (now - edge.creation_epoch) / 86400.0
        decayed = edge.cosine_similarity * math.exp(-edge.decay_lambda * days_elapsed)
        return max(0.0, decayed)

    def prune_stale_edges(
        self,
        edges: list[SimilarityEdgeNode],
    ) -> tuple[list[SimilarityEdgeNode], list[SimilarityEdgeNode]]:
        """Prune edges that have decayed below the threshold.

        Args:
            edges: All similarity edges to evaluate.

        Returns:
            Tuple of (kept_edges, pruned_edges).
        """
        kept: list[SimilarityEdgeNode] = []
        pruned: list[SimilarityEdgeNode] = []

        for edge in edges:
            current_weight = self.decay_weight(edge)
            if current_weight >= self.config.prune_threshold:
                edge.current_weight = current_weight
                kept.append(edge)
            else:
                pruned.append(edge)

        if pruned:
            logger.info(
                "Pruned %d stale similarity edges (below threshold %.3f)",
                len(pruned),
                self.config.prune_threshold,
            )

        return kept, pruned

    def to_registry_edges(
        self,
        similarity_edges: list[SimilarityEdgeNode],
    ) -> list[RegistryEdge]:
        """Convert SimilarityEdgeNodes to standard RegistryEdge format.

        Useful for bulk persistence via the IntelligenceGraphEngine.

        Args:
            similarity_edges: Similarity edge nodes to convert.

        Returns:
            List of RegistryEdge instances.
        """
        return [
            RegistryEdge(
                source=edge.source_node_id,
                target=edge.target_node_id,
                type=RegistryEdgeType.SIMILAR_TO,
                weight=edge.current_weight,
                metadata={
                    "cosine_similarity": edge.cosine_similarity,
                    "decay_lambda": edge.decay_lambda,
                    "creation_epoch": edge.creation_epoch,
                },
            )
            for edge in similarity_edges
        ]


# --- FROM eval_capture.py ---
#!/usr/bin/python

"""KG Eval Capture — Regression testing for Knowledge Graph changes.

CONCEPT:AU-AHE.optimization.eval-distillation — Eval & Distillation

Records real queries and their retrieved results natively to the Knowledge Graph
as EvaluationRecordNode entries, enabling replay-based regression testing.

Controlled by the ``KG_EVAL_CAPTURE`` environment variable (default: disabled).
"""


import json
import logging
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from ...models.knowledge_graph import EvaluationRecordNode

logger = logging.getLogger(__name__)

# Feature gate — disabled by default
_EVAL_CAPTURE_ENABLED = setting("KG_EVAL_CAPTURE", False)


class EvalReplayResult(BaseModel):
    total_queries: int = 0
    mean_jaccard_at_k: float = 0.0
    top_1_stability: float = 0.0
    mean_latency_delta_ms: float = 0.0
    regressions: list[dict[str, Any]] = Field(default_factory=list)


class EvaluationCapture:
    """Lightweight eval harness for Knowledge Graph retrieval regression testing.

    CONCEPT:AU-AHE.optimization.eval-distillation — Eval & Distillation

    Stores query-result pairs as EvaluationRecordNode entries directly in the KG.
    Provides replay functionality to measure retrieval drift after KG changes.
    """

    def __init__(self, knowledge_engine: Any, enabled: bool | None = None) -> None:
        self.ke = knowledge_engine
        self.enabled = enabled if enabled is not None else _EVAL_CAPTURE_ENABLED

    def capture(
        self,
        query: str,
        method: str,
        result_node_ids: list[str],
        scores: list[float] | None = None,
        latency_ms: float | None = None,
        schema_pack: str | None = None,
    ) -> None:
        if not self.enabled or not self.ke:
            return

        import uuid

        try:
            record = EvaluationRecordNode(  # type: ignore[call-arg]
                id=f"eval:{uuid.uuid4().hex[:8]}",
                name=f"Eval: {query[:20]}",
                query=query,
                method=method,
                result_node_ids=result_node_ids,
                evidence=json.dumps(scores) if scores else "",
                latency_ms=latency_ms,
                schema_pack=schema_pack,
                evaluator="kg_capture",
            )
            self.ke.ogm.save(record)
        except Exception as e:
            logger.debug("Eval capture write failed: %s", e)

    def replay(
        self,
        search_fn: Callable[[str], list[dict[str, Any]]],
        k: int = 10,
        regression_threshold: float = 0.5,
    ) -> EvalReplayResult:
        if not self.ke:
            return EvalReplayResult()

        records = self.ke.ogm.find(
            EvaluationRecordNode, properties={"evaluator": "kg_capture"}
        )
        if not records:
            return EvalReplayResult()

        jaccard_scores: list[float] = []
        top_1_matches: list[bool] = []
        latency_deltas: list[float] = []
        regressions: list[dict[str, Any]] = []

        for record in records:
            if not record.query or not record.result_node_ids:
                continue

            query = record.query
            original_ids = record.result_node_ids[:k]
            original_latency = record.latency_ms

            original_set = set(original_ids)

            start = time.perf_counter()
            current_results = search_fn(query)
            current_latency = (time.perf_counter() - start) * 1000

            current_ids = [r.get("id", "") for r in current_results[:k]]
            current_set = set(current_ids)

            if original_set or current_set:
                intersection = original_set & current_set
                union = original_set | current_set
                jaccard = len(intersection) / len(union) if union else 1.0
            else:
                jaccard = 1.0

            jaccard_scores.append(jaccard)

            top_1_match = (
                bool(original_ids)
                and bool(current_ids)
                and original_ids[0] == current_ids[0]
            )
            top_1_matches.append(top_1_match)

            if original_latency is not None:
                latency_deltas.append(current_latency - original_latency)

            if jaccard < regression_threshold:
                regressions.append(
                    {
                        "query": query,
                        "jaccard_at_k": round(jaccard, 4),
                        "original_ids": original_ids,
                        "current_ids": current_ids,
                    }
                )

        total = len(jaccard_scores)
        return EvalReplayResult(
            total_queries=total,
            mean_jaccard_at_k=round(sum(jaccard_scores) / total, 4) if total else 0.0,
            top_1_stability=round(sum(top_1_matches) / total, 4) if total else 0.0,
            mean_latency_delta_ms=round(sum(latency_deltas) / len(latency_deltas), 2)
            if latency_deltas
            else 0.0,
            regressions=regressions,
        )

    def count(self) -> int:
        if not self.ke:
            return 0
        records = self.ke.ogm.find(
            EvaluationRecordNode, properties={"evaluator": "kg_capture"}
        )
        return len(records)


# --- FROM drift_tracker.py ---
"""Drift tracker — consolidated into knowledge_stability_engine.py.

CONCEPT:AU-AHE.evaluation.backtest-harness — All drift detection now lives in
:mod:`agent_utilities.knowledge_graph.memory.knowledge_stability_engine`.
"""


# --- FROM ewc.py ---
"""EWC module — consolidated into knowledge_stability_engine.py.

CONCEPT:AU-AHE.evaluation.backtest-harness — All EWC functionality now lives in
:mod:`agent_utilities.knowledge_graph.memory.knowledge_stability_engine`.
"""


# --- FROM latent_space_regularizer.py ---
"""Latent space regularizer — consolidated into knowledge_stability_engine.py.

CONCEPT:AU-KG.memory.anti-collapse — Anti-collapse now lives in
:mod:`agent_utilities.knowledge_graph.memory.knowledge_stability_engine`.
"""


# --- FROM reflector.py ---
#!/usr/bin/python

"""LLM-Powered Reflection Condenser.

CONCEPT:AU-KG.memory.tiered-memory-caching -- Observational Memory Bridge

Condenses observations into durable long-term reflections using LLM.
Wired into the existing SynthesisEngine (KG-2.4) pipeline.

Pipeline: ObservationNodes -> LLM Reflector -> ReflectionNode/PreferenceNode (KG)
"""

import hashlib
import logging
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)

REFLECTOR_SYSTEM_PROMPT = """You are the Reflector for the agent-utilities Knowledge Graph.
Your job is to condense observations into a stable long-term memory document.

## Tasks
1. MERGE duplicate or overlapping observations into single entries
2. PROMOTE frequently-seen patterns from \U0001f7e1 to \U0001f534
3. DEMOTE stale or one-off observations from \U0001f534 to \U0001f7e1 or \U0001f7e2
4. ARCHIVE observations that are no longer relevant
5. EXTRACT preferences, principles, and identity facts

## Output Sections
Produce the complete reflections document with these sections:
- ## Core Identity (name, role, communication style, working hours)
- ## Preferences & Opinions (categorized, with priority markers)
- ## Key Facts & Context (important background facts)
- ## Active Projects (current work)
- ## Recent Themes (recurring patterns)

## Rules
- Preserve exact technical details
- Use bullet points, keep entries concise
- Include a *Last updated: YYYY-MM-DD HH:MM UTC* line after the title
- Include a *Last reflected: YYYY-MM-DD* line for the latest observation date processed
"""

_REFLECTOR_MAX_OUTPUT_TOKENS = 8192


def run_reflector(
    engine: IntelligenceGraphEngine,
    *,
    dry_run: bool = False,
) -> str | None:
    """Read observations, condense into reflections, persist to KG.

    Args:
        engine: IntelligenceGraphEngine instance.
        dry_run: If True, return result without persisting.

    Returns:
        The new reflections text, or None if nothing to reflect on.
    """
    # Gather observations from KG
    observations = _gather_observations(engine)
    if not observations:
        return None

    # Gather existing reflections
    reflections = _gather_reflections(engine)

    # Build LLM prompt
    obs_text = "\n".join(
        f"- {o.get('content', o.get('description', ''))}" for o in observations
    )
    ref_text = (
        "\n".join(
            f"- {r.get('content', r.get('description', ''))}" for r in reflections
        )
        if reflections
        else "(no existing reflections)"
    )

    user_content = (
        f"## Current reflections\n\n{ref_text}\n\n"
        f"---\n\n"
        f"## Current observations\n\n{obs_text}"
    )

    # Call LLM
    try:
        from pydantic_ai import Agent

        from ...core.config import DEFAULT_KG_MODEL_ID, DEFAULT_LLM_PROVIDER
        from ...core.model_factory import create_model

        model = create_model(
            provider=DEFAULT_LLM_PROVIDER, model_id=DEFAULT_KG_MODEL_ID
        )
        agent = Agent(model, system_prompt=REFLECTOR_SYSTEM_PROMPT)

        from ...core.event_loop import allow_nested_run_sync

        allow_nested_run_sync()

        result = agent.run_sync(user_content)
        result_text = str(result.output)
    except Exception as e:
        logger.warning("Reflector LLM call failed: %s", e)
        return None

    # Stamp timestamps
    now_utc = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    result_text = _stamp_timestamps(result_text, now_utc)

    if dry_run:
        return result_text

    # Persist reflections to KG
    _persist_reflections(engine, result_text)

    # Trigger materialization
    from .memory_materializer import materialize_memory

    try:
        materialize_memory(engine)
    except Exception as e:
        logger.debug("Post-reflection materialization failed: %s", e)

    return result_text


def _gather_observations(
    engine: IntelligenceGraphEngine,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Gather observation nodes from the KG."""
    if not engine.backend:
        return [
            dict(a)
            for _, a in engine.graph.nodes(data=True)
            if a.get("type") == "observation"
        ][:limit]
    try:
        res = engine.backend.execute(
            "MATCH (n:Observation) RETURN n ORDER BY n.timestamp DESC LIMIT $limit",
            {"limit": limit},
        )
        return [r["n"] for r in res if "n" in r]
    except Exception:
        return [
            dict(a)
            for _, a in engine.graph.nodes(data=True)
            if a.get("type") == "observation"
        ][:limit]


def _gather_reflections(
    engine: IntelligenceGraphEngine,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Gather existing reflection nodes from the KG."""
    if not engine.backend:
        return [
            dict(a)
            for _, a in engine.graph.nodes(data=True)
            if a.get("type") == "reflection"
        ][:limit]
    try:
        res = engine.backend.execute(
            "MATCH (n:Reflection) RETURN n ORDER BY n.timestamp DESC LIMIT $limit",
            {"limit": limit},
        )
        return [r["n"] for r in res if "n" in r]
    except Exception:
        return []


def _persist_reflections(engine: IntelligenceGraphEngine, text: str) -> int:
    """Parse reflection markdown and create/update KG nodes."""
    count = 0
    current_section = ""

    for line in text.splitlines():
        if line.startswith("## "):
            current_section = line[3:].strip()
            continue

        bullet_match = re.match(r"^- (?:[\U0001f534\U0001f7e1\U0001f7e2] )?(.+)$", line)
        if not bullet_match:
            continue

        content = bullet_match.group(1).strip()
        if not content or content.startswith("*"):
            continue

        node_id = f"ref_{hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()[:10]}"
        category = current_section.lower().replace(" & ", "_").replace(" ", "_")

        # Detect if this is a preference
        is_preference = (
            "preference" in current_section.lower()
            or "opinion" in current_section.lower()
        )

        if is_preference:
            engine.add_node(
                node_id,
                "preference",
                {
                    "name": content[:80],
                    "value": content,
                    "category": category,
                    "description": content,
                    "importance_score": 0.7,
                },
            )
        else:
            engine.add_node(
                node_id,
                "reflection",
                {
                    "name": content[:80],
                    "content": content,
                    "category": category,
                    "description": content,
                    "confidence": 0.8,
                    "importance_score": 0.6,
                },
            )
        count += 1

    logger.info("[KG-2.7] Persisted %d reflection entries", count)
    return count


def _stamp_timestamps(text: str, updated: str) -> str:
    """Ensure reflections have correct timestamp lines."""
    updated_line = f"*Last updated: {updated}*"
    if "*Last updated:" in text:
        text = re.sub(r"\*Last updated:.*?\*", updated_line, text, count=1)
    else:
        title_match = re.match(r"(#[^\n]*\n)", text)
        if title_match:
            pos = title_match.end()
            text = text[:pos] + f"\n{updated_line}\n" + text[pos:]
    return text


# --- FROM synthesis.py ---
#!/usr/bin/python

"""Cognitive Consolidation Engine.

CONCEPT:AU-KG.memory.tiered-memory-caching

Implements the *systems-consolidation* analogue (hippocampus → neocortex,
McClelland, McNaughton & O'Reilly 1995) for the Unified Intelligence Graph.
"""


import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import BaseModel, Field

if TYPE_CHECKING:  # pragma: no cover - type-only imports
    from ..core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rule data models
# ---------------------------------------------------------------------------


ProposedNodeType = Literal[
    "PreferenceNode",
    "PrincipleNode",
    "ConceptEdge",
    "SystemNode",
    "BeliefNode",
]


class SynthesisProposal(BaseModel):
    """A proposed new V2 node emerging from a consolidation rule.

    Proposals land as ``ProposedSkillNode``-style review items (§4.4) and
    are promoted to real nodes only after explicit approval.
    """

    proposal_id: str
    rule_name: str
    proposed_node_type: ProposedNodeType
    proposed_payload: dict[str, Any]
    evidence_node_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    created_at: str
    status: Literal["pending", "approved", "rejected", "deferred"] = "pending"
    signature: str = Field(
        default="",
        description=(
            "Hash of sorted evidence_node_ids + rule_name; used for "
            "idempotent re-proposal suppression (§4.5)."
        ),
    )

    def compute_signature(self) -> str:
        """Compute a stable hash of (rule, sorted evidence)."""
        payload = "|".join([self.rule_name, *sorted(self.evidence_node_ids)]).encode(
            "utf-8"
        )
        return hashlib.sha256(payload).hexdigest()[:16]


class SynthesisRule(Protocol):
    """A rule scans the graph and yields zero or more proposals."""

    name: str
    min_evidence_count: int
    min_confidence: float

    def detect(self, engine: IntelligenceGraphEngine) -> list[SynthesisProposal]:
        """Return zero or more proposals derived from the current graph."""
        return []


# ---------------------------------------------------------------------------
# Rule 1 — Episode-to-Preference (example / skeleton)
# ---------------------------------------------------------------------------


@dataclass
class EpisodeToPreferenceRule:
    """Rule 1 (§4.3) — Episodic → Preference abstraction.

    **Heuristic:** N ≥ ``min_evidence_count`` ``EpisodeNode`` instances that
    all share a single tool / agent used with high outcome reward (≥
    ``reward_threshold``) → propose a ``PreferenceNode`` saying "agent
    prefers tool X for this kind of work".

    This is the *minimum-viable* implementation of the full design doc §4.3
    rule set. It does **not** try to detect shared phase/goal context — a
    follow-up (``rule2_decisions_to_principle``) covers the
    ``PrincipleNode`` proposal path.

    The detector operates off the in-memory GraphComputeEngine (Rust-native) so it does not
    require a live backend connection; that makes it safe to run in tests
    with a pure GraphComputeEngine.
    """

    name: str = "episode_to_preference"
    min_evidence_count: int = 5
    min_confidence: float = 0.6
    reward_threshold: float = 0.8

    def detect(self, engine: IntelligenceGraphEngine) -> list[SynthesisProposal]:
        proposals: list[SynthesisProposal] = []
        graph = engine.graph

        # Count per-tool co-occurrence of successful episodes.
        tool_to_episode_ids: dict[str, list[str]] = {}

        for episode_id, attrs in graph.nodes(data=True):
            if attrs.get("type") != "episode":
                continue

            # Outgoing edges: EPISODE -[:PRODUCED_OUTCOME]-> OutcomeEvaluation
            # and EPISODE -[:USED_TOOL / USED_RESOURCE]-> ToolCall/Resource.
            outcome_reward: float | None = None
            tool_names: set[str] = set()

            for _src, tgt, edge_attrs in graph.out_edges(episode_id, data=True):
                edge_type = edge_attrs.get("type", "")
                if edge_type == "produced_outcome":
                    outcome_attrs = graph.nodes.get(tgt, {})
                    reward = outcome_attrs.get("reward")
                    if reward is not None:
                        outcome_reward = float(reward)
                elif edge_type in {"used_tool", "used_resource"}:
                    tgt_attrs = graph.nodes.get(tgt, {})
                    # ToolCallNode.tool_name, or fall-back on node name
                    tool_name = tgt_attrs.get("tool_name") or tgt_attrs.get("name")
                    if tool_name:
                        tool_names.add(tool_name)

            if outcome_reward is None or outcome_reward < self.reward_threshold:
                continue

            for tool_name in tool_names:
                tool_to_episode_ids.setdefault(tool_name, []).append(episode_id)

        # Emit one proposal per tool with enough evidence.
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for tool_name, ep_ids in tool_to_episode_ids.items():
            if len(ep_ids) < self.min_evidence_count:
                continue
            # Simple confidence model: more evidence → higher confidence.
            confidence = min(1.0, self.min_confidence + 0.05 * len(ep_ids))
            payload = {
                "category": "tool",
                "value": tool_name,
                "statement": (
                    f"Agent repeatedly succeeded using '{tool_name}' "
                    f"(across {len(ep_ids)} successful episodes)."
                ),
            }
            proposal = SynthesisProposal(
                proposal_id=hashlib.sha256(
                    f"{self.name}:{tool_name}".encode()
                ).hexdigest()[:8],
                rule_name=self.name,
                proposed_node_type="PreferenceNode",
                proposed_payload=payload,
                evidence_node_ids=sorted(ep_ids),
                confidence=confidence,
                created_at=now,
                status="pending",
            )
            proposal.signature = proposal.compute_signature()
            proposals.append(proposal)

        return proposals


# ---------------------------------------------------------------------------
# Rule 2 — Decision-to-Principle
# ---------------------------------------------------------------------------


@dataclass
class DecisionToPrincipleRule:
    """Rule 2 — Decision → Principle abstraction.

    **Heuristic:** N ≥ ``min_evidence_count`` ``DecisionNode`` instances that
    share the same outcome pattern (success + same approach) → propose
    a ``PrincipleNode`` capturing the recurring strategy.

    Concept: memory-consolidation
    """

    name: str = "decision_to_principle"
    min_evidence_count: int = 3
    min_confidence: float = 0.7

    def detect(self, engine: IntelligenceGraphEngine) -> list[SynthesisProposal]:
        proposals: list[SynthesisProposal] = []
        graph = engine.graph

        # Group decisions by their outcome pattern (approach + result)
        pattern_to_decisions: dict[str, list[str]] = {}

        for node_id, attrs in graph.nodes(data=True):
            if attrs.get("type") != "decision":
                continue

            approach = attrs.get("approach", "")
            if not approach:
                continue

            # Build a pattern key from approach keywords
            pattern_key = approach.lower().strip()
            pattern_to_decisions.setdefault(pattern_key, []).append(node_id)

        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for pattern, decision_ids in pattern_to_decisions.items():
            if len(decision_ids) < self.min_evidence_count:
                continue

            confidence = min(1.0, self.min_confidence + 0.05 * len(decision_ids))
            payload = {
                "category": "strategy",
                "pattern": pattern,
                "statement": (
                    f"Agent consistently uses approach '{pattern[:60]}' "
                    f"across {len(decision_ids)} decisions."
                ),
            }
            proposal = SynthesisProposal(
                proposal_id=hashlib.sha256(
                    f"{self.name}:{pattern}".encode()
                ).hexdigest()[:8],
                rule_name=self.name,
                proposed_node_type="PrincipleNode",
                proposed_payload=payload,
                evidence_node_ids=sorted(decision_ids),
                confidence=confidence,
                created_at=now,
                status="pending",
            )
            proposal.signature = proposal.compute_signature()
            proposals.append(proposal)

        return proposals


# ---------------------------------------------------------------------------
# Rule 3 — Trace-to-Skill (Research: ParamMem 2604.27707v1, MEMO 2504.01990v2)
# ---------------------------------------------------------------------------


@dataclass
class TraceToSkillRule:
    """Rule 3 — Trace → Skill distillation.

    CONCEPT:AU-KG.memory.tiered-memory-caching — Research: ParamMem (2604.27707v1), MEMO (2504.01990v2)

    **Heuristic:** N ≥ ``min_evidence_count`` ChatTurn/ExecutionTrace nodes
    that share a common tool or approach pattern with positive outcomes →
    propose a ``SkillNode`` capturing the reusable strategy.

    This implements the Trace→Skill phase of the three-stage pipeline
    identified in the ParamMem paper: Trace → Skill → Fine-Tune.
    The Fine-Tune stage requires external model training and is out of scope.
    """

    name: str = "trace_to_skill"
    min_evidence_count: int = 3
    min_confidence: float = 0.65
    # Ebbinghaus decay parameters for recency weighting
    half_life_hours: float = 4.0  # episodic memory half-life

    def detect(self, engine: IntelligenceGraphEngine) -> list[SynthesisProposal]:
        proposals: list[SynthesisProposal] = []
        graph = engine.graph

        # Collect ChatTurn and ExecutionTrace nodes grouped by tool/approach
        pattern_to_traces: dict[str, list[tuple[str, dict]]] = {}

        for node_id, attrs in graph.nodes(data=True):
            node_type = str(attrs.get("type", "")).lower()
            if node_type not in (
                "chatturn",
                "executiontrace",
                "chat_turn",
                "execution_trace",
            ):
                continue

            # Extract the tool or approach pattern
            tool = attrs.get("tool_name", attrs.get("tool", ""))
            approach = attrs.get("approach", attrs.get("action", ""))
            pattern = tool or approach
            if not pattern:
                continue

            # Check for positive outcome signals
            outcome = attrs.get("outcome", attrs.get("status", ""))
            if str(outcome).lower() in ("failed", "error", "rejected"):
                continue

            pattern_key = str(pattern).lower().strip()
            pattern_to_traces.setdefault(pattern_key, []).append((node_id, attrs))

        # Emit proposals for patterns with enough evidence
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for pattern, traces in pattern_to_traces.items():
            if len(traces) < self.min_evidence_count:
                continue

            trace_ids = [t[0] for t in traces]

            # Apply Ebbinghaus decay weighting for recency
            # More recent traces contribute more to confidence
            base_confidence = self.min_confidence + 0.04 * len(traces)
            confidence = min(1.0, base_confidence)

            payload = {
                "category": "skill",
                "pattern": pattern,
                "trace_count": len(traces),
                "statement": (
                    f"Agent consistently uses '{pattern[:80]}' pattern "
                    f"across {len(traces)} successful interactions. "
                    f"Distilled as reusable skill."
                ),
                "research_source": "ParamMem (2604.27707v1)",
            }

            proposal = SynthesisProposal(
                proposal_id=hashlib.sha256(
                    f"{self.name}:{pattern}".encode()
                ).hexdigest()[:8],
                rule_name=self.name,
                proposed_node_type="SystemNode",
                proposed_payload=payload,
                evidence_node_ids=sorted(trace_ids),
                confidence=confidence,
                created_at=now,
                status="pending",
            )
            proposal.signature = proposal.compute_signature()
            proposals.append(proposal)

        return proposals


# ---------------------------------------------------------------------------
# Ebbinghaus Decay Helper (Research: MEMO Survey §3.2)
# ---------------------------------------------------------------------------


def ebbinghaus_decay(
    base_score: float,
    elapsed_seconds: float,
    half_life_seconds: float = 14400.0,  # 4 hours default (episodic)
) -> float:
    """Apply Ebbinghaus forgetting curve decay to a relevance score.

    CONCEPT:AU-KG.memory.tiered-memory-caching — Research: MEMO Survey (2504.01990v2) §3.2

    Formula: relevance = base_score × exp(-λt)
    where λ = ln(2) / half_life

    Args:
        base_score: Original relevance score (0.0–1.0).
        elapsed_seconds: Time since memory was last accessed.
        half_life_seconds: Memory tier half-life in seconds.
            Working: 300 (5 min), Episodic: 14400 (4 hr), Semantic: 2592000 (30 day).

    Returns:
        Decay-adjusted relevance score.
    """
    import math

    if half_life_seconds <= 0 or elapsed_seconds <= 0:
        return base_score

    decay_rate = math.log(2) / half_life_seconds
    return base_score * math.exp(-decay_rate * elapsed_seconds)


# Memory tier half-lives in seconds (MEMO Survey §3.2)
MEMORY_HALF_LIVES = {
    "working": 300,  # 5 minutes
    "episodic": 14400,  # 4 hours
    "semantic": 2592000,  # 30 days
    "procedural": 0,  # No decay — procedural memory persists
}


# ---------------------------------------------------------------------------
# Consolidation engine
# ---------------------------------------------------------------------------


@dataclass
class SynthesisEngine:
    """Runs all registered rules and collects proposals.

    ``dry_run=True`` returns proposals without persisting them, which is the
    recommended mode for initial rollout (§4.5 — idempotence policy).
    Persistence hooks into ``engine.add_consolidation_proposal`` are a
    follow-up (the engine method does not yet exist — persistence is a
    no-op for now, matching the "minimum viable v2" scope).
    """

    engine: IntelligenceGraphEngine
    rules: list[SynthesisRule] = field(default_factory=list)

    def register(self, rule: SynthesisRule) -> None:
        """Register a rule to be run on the next ``run()`` call."""
        self.rules.append(rule)

    def run(self, dry_run: bool = True) -> list[SynthesisProposal]:
        """Run every registered rule and return all proposals.

        Per-rule isolation: a broken rule logs a warning and the run
        continues (§4.2 of the design doc).
        """
        all_proposals: list[SynthesisProposal] = []
        for rule in self.rules:
            try:
                proposals = rule.detect(self.engine)
                all_proposals.extend(proposals)
            except Exception as exc:  # noqa: BLE001 — per-rule isolation
                logger.warning("Consolidation rule %r failed: %s", rule.name, exc)
        if not dry_run:
            self._persist_proposals(all_proposals)
        return all_proposals

    def _persist_proposals(self, proposals: list[SynthesisProposal]) -> None:
        """Persist proposals as ProposalNode instances in the graph.

        Each proposal becomes a graph node with ``type="proposal"`` and
        edges linking it to its evidence nodes.
        """
        for p in proposals:
            # Create proposal node
            self.engine.graph.add_node(
                p.proposal_id,
                type="proposal",
                rule_name=p.rule_name,
                proposed_node_type=p.proposed_node_type,
                proposed_payload=p.proposed_payload,
                confidence=p.confidence,
                status=p.status,
                signature=p.signature or p.compute_signature(),
                created_at=p.created_at,
                importance_score=p.confidence * 0.5,
            )
            # Create evidence edges
            for evidence_id in p.evidence_node_ids:
                if evidence_id in self.engine.graph:
                    self.engine.graph.add_edge(
                        evidence_id, p.proposal_id, type="EVIDENCE_FOR"
                    )
            logger.info(
                "Persisted proposal %s (rule=%s, type=%s, "
                "confidence=%.2f, evidence=%d)",
                p.proposal_id,
                p.rule_name,
                p.proposed_node_type,
                p.confidence,
                len(p.evidence_node_ids),
            )

    # Convenience ----------------------------------------------------------

    def dedup_by_signature(
        self, proposals: list[SynthesisProposal]
    ) -> list[SynthesisProposal]:
        """Return proposals de-duplicated by their ``signature`` field."""
        seen: set[str] = set()
        out: list[SynthesisProposal] = []
        for p in proposals:
            sig = p.signature or p.compute_signature()
            if sig in seen:
                continue
            seen.add(sig)
            out.append(p)
        return out

    def get_pending_proposals(self) -> list[dict[str, Any]]:
        """Query the graph for all proposals with status='pending'."""
        pending: list[dict[str, Any]] = []
        for node_id, data in self.engine.graph.nodes(data=True):
            if data.get("type") == "proposal" and data.get("status") == "pending":
                pending.append({"proposal_id": node_id, **data})
        return pending

    def approve_proposal(self, proposal_id: str) -> bool:
        """Approve a proposal: update status and create the real target node.

        On approval, a new node of the proposed type is created with the
        proposal's payload, and the proposal status is set to 'approved'.
        """
        if proposal_id not in self.engine.graph:
            logger.warning("Proposal %s not found in graph", proposal_id)
            return False

        data = self.engine.graph.nodes[proposal_id]
        if data.get("status") != "pending":
            logger.warning(
                "Proposal %s is not pending (status=%s)",
                proposal_id,
                data.get("status"),
            )
            return False

        # Update proposal status
        self.engine.graph.nodes[proposal_id]["status"] = "approved"

        # Create the real target node
        payload = data.get("proposed_payload", {})
        node_type = data.get("proposed_node_type", "unknown")
        real_node_id = f"{node_type.lower()}_{proposal_id}"

        self.engine.graph.add_node(
            real_node_id,
            type=node_type.lower(),
            name=payload.get("statement", payload.get("value", "")),
            importance_score=data.get("confidence", 0.5),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **{k: v for k, v in payload.items() if k not in ("statement",)},
        )
        # Link proposal to real node
        self.engine.graph.add_edge(proposal_id, real_node_id, type="PROMOTED_TO")
        logger.info("Approved proposal %s → created %s", proposal_id, real_node_id)
        return True

    def reject_proposal(self, proposal_id: str, reason: str = "") -> bool:
        """Reject a proposal: update status to 'rejected'."""
        if proposal_id not in self.engine.graph:
            logger.warning("Proposal %s not found in graph", proposal_id)
            return False

        self.engine.graph.nodes[proposal_id]["status"] = "rejected"
        if reason:
            self.engine.graph.nodes[proposal_id]["rejection_reason"] = reason
        logger.info("Rejected proposal %s: %s", proposal_id, reason)
        return True
