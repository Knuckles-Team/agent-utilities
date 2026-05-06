"""Temporal Knowledge Drift Tracker.

CONCEPT:AHE-3.6 — Temporal Drift
Measures the shift in node embeddings over time to warn against
'concept drift' or stale memories.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class DriftReport:
    """Report detailing the detected knowledge drift.

    CONCEPT:AHE-3.6
    """

    def __init__(
        self,
        node_id: str,
        coefficient_of_variation: float,
        cosine_shift: float,
        has_drifted: bool,
    ):
        self.node_id = node_id
        self.coefficient_of_variation = coefficient_of_variation
        self.cosine_shift = cosine_shift
        self.has_drifted = has_drifted


def calculate_cosine_distance(vec_a: list[float], vec_b: list[float]) -> float:
    """Calculate the cosine distance between two vectors.

    CONCEPT:AHE-3.6
    """
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

    CONCEPT:AHE-3.6

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

    # Calculate shift from the oldest known (baseline) embedding
    baseline = historical_embeddings[0]
    cosine_shift = calculate_cosine_distance(baseline, current_embedding)

    # Calculate coefficient of variation across all embeddings
    all_embs = np.array(historical_embeddings + [current_embedding])
    std_devs = np.std(all_embs, axis=0)
    means = np.mean(all_embs, axis=0)

    # Avoid division by zero
    safe_means = np.where(means == 0, 1e-10, means)
    cv = np.mean(std_devs / np.abs(safe_means))

    has_drifted = bool(cosine_shift > drift_threshold)
    if has_drifted:
        logger.info(
            f"Knowledge drift detected for {node_id}: Shift={cosine_shift:.3f}, CV={cv:.3f}"
        )

    return DriftReport(node_id, float(cv), float(cosine_shift), has_drifted)
