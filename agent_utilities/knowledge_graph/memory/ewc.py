"""Elastic Weight Consolidation (EWC++) Module.

CONCEPT:AHE-3.6 — EWC++
Implements a lightweight Fisher-proxy diagonal approximation to prevent
catastrophic forgetting when adapting node embeddings.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_fisher_diagonal_proxy(
    embeddings_history: list[list[float]],
) -> list[float]:
    """Compute a lightweight proxy for the Fisher Information diagonal.

    CONCEPT:AHE-3.6

    Instead of calculating second-order derivatives (which require
    a neural network and loss function), this uses the inverse of the
    variance across historical embeddings. High variance implies the
    dimension is highly plastic and not critical for past tasks. Low variance
    implies the dimension was consistently important for past representations.

    Args:
        embeddings_history: A list of historical embeddings for a node.

    Returns:
        A list of floats representing the diagonal Fisher proxy.
    """
    if len(embeddings_history) < 2:
        # Not enough history to establish importance; return uniform low penalty
        dim = len(embeddings_history[0]) if embeddings_history else 1536
        return [0.1] * dim

    arr = np.array(embeddings_history)
    variances = np.var(arr, axis=0)

    # Proxy: Inverse of variance. Small variance -> high Fisher importance
    # Add epsilon to prevent division by zero
    epsilon = 1e-6
    fisher_proxy = 1.0 / (variances + epsilon)

    # Normalize the proxy to have a max value of 1.0 for numerical stability
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

    CONCEPT:AHE-3.6

    This prevents catastrophic forgetting by resisting changes to
    dimensions that have a high Fisher importance score.

    L_ewc = L_new + (lambda / 2) * F * (theta - theta_star)^2
    In embedding space proxy, we pull the new embedding back towards
    the old embedding proportionally to the Fisher importance.

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

    # Ensure dimensions match
    if old_vec.shape != new_vec.shape or old_vec.shape != fisher_vec.shape:
        logger.warning("Dimension mismatch in EWC consolidation. Bypassing EWC.")
        return new_embedding

    # Delta between new and old
    delta = new_vec - old_vec

    # The higher the fisher_vec value, the more we dampen the delta
    # Dampening factor = 1.0 - (lambda * fisher_importance)
    # Clip between 0.0 and 1.0
    dampening = np.clip(1.0 - (lambda_param * fisher_vec), 0.0, 1.0)

    consolidated_delta = delta * dampening
    consolidated_vec = old_vec + consolidated_delta

    # Re-normalize if necessary (assuming embeddings should be unit vectors)
    norm = np.linalg.norm(consolidated_vec)
    if norm > 0:
        consolidated_vec = consolidated_vec / norm

    return consolidated_vec.tolist()
