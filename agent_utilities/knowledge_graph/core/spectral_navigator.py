#!/usr/bin/python
from __future__ import annotations

"""Spectral Cluster Navigator.

CONCEPT:KG-2.5 — Spectral Cluster Navigator

Provides tuning-free spectral clustering using the normalized Laplacian
eigengap heuristic for automatic k-selection. Adapted from contextplus's
clustering.ts with OWL ontology integration and financial regime detection.

Uses ``numpy`` + ``scipy.sparse.linalg.eigsh`` for eigendecomposition
(both existing dependencies). Clusters auto-map to ``skos:Concept``
nodes with ``broader``/``narrower`` edges for OWL-transitive hierarchies.
"""


import logging
import uuid
from dataclasses import dataclass, field

import numpy as np

from agent_utilities.models.knowledge_graph import (
    SpectralClusterNode,
)

logger = logging.getLogger(__name__)


@dataclass
class ClusterResult:
    """Result of spectral clustering on a set of embedding vectors.

    Attributes:
        cluster_id: Unique identifier for this cluster.
        label: Human-readable label for the cluster.
        indices: Indices of members in the original input array.
        centroid: Mean embedding of cluster members.
        coherence: Average pairwise cosine similarity within the cluster.
    """

    cluster_id: str = ""
    label: str = ""
    indices: list[int] = field(default_factory=list)
    centroid: list[float] = field(default_factory=list)
    coherence: float = 0.0


class SpectralClusterNavigator:
    """Tuning-free spectral clustering with OWL integration.

    CONCEPT:KG-2.5 — Spectral Cluster Navigator

    Performs spectral clustering using the normalized Laplacian and
    eigengap heuristic. No hyperparameter tuning is needed — the
    optimal k is selected automatically from eigenvalue gaps.

    Supports hierarchical recursive clustering for large datasets
    and financial regime detection via domain-specific embeddings.

    Example::

        navigator = SpectralClusterNavigator()
        vectors = [[0.1, 0.9], [0.2, 0.8], [0.9, 0.1], [0.8, 0.2]]
        clusters = navigator.cluster(vectors, max_k=10)
        for c in clusters:
            print(f"Cluster {c.label}: {len(c.indices)} members")
    """

    def __init__(self, min_cluster_size: int = 2, max_depth: int = 3):
        """Initialize the spectral cluster navigator.

        Args:
            min_cluster_size: Minimum members to form a cluster.
            max_depth: Maximum hierarchy depth for recursive clustering.
        """
        self._min_cluster_size = min_cluster_size
        self._max_depth = max_depth

    @staticmethod
    def _cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
        """Build the cosine similarity affinity matrix.

        Normalizes each vector to unit length then computes dot products.
        Clips to [0, 1] to ensure non-negative affinity.
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normalized = vectors / norms
        similarity = normalized @ normalized.T
        return np.clip(similarity, 0.0, 1.0)

    @staticmethod
    def _normalized_laplacian(affinity: np.ndarray) -> np.ndarray:
        """Compute the symmetric normalized Laplacian.

        L_sym = I - D^{-1/2} @ W @ D^{-1/2}

        where W is the affinity matrix and D is the degree matrix.
        """
        n = affinity.shape[0]
        np.fill_diagonal(affinity, 0.0)  # No self-loops
        degree = affinity.sum(axis=1)
        degree_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
        D_inv_sqrt = np.diag(degree_inv_sqrt)
        laplacian = np.eye(n) - D_inv_sqrt @ affinity @ D_inv_sqrt
        return laplacian

    @staticmethod
    def _eigengap_k(eigenvalues: np.ndarray, max_k: int) -> int:
        """Select optimal k using the eigengap heuristic.

        Finds the largest gap between consecutive eigenvalues in the
        range [2, max_k]. Returns at least 2 clusters.
        """
        if len(eigenvalues) < 3:
            return min(2, len(eigenvalues))

        # Compute gaps between consecutive sorted eigenvalues
        sorted_vals = np.sort(eigenvalues)
        upper = min(max_k, len(sorted_vals) - 1)
        if upper < 2:
            return 2

        gaps = np.diff(sorted_vals[1 : upper + 1])
        if len(gaps) == 0:
            return 2

        best_k = int(np.argmax(gaps)) + 2  # +2 because we skip eigenvalue 0
        return max(2, min(best_k, max_k))

    @staticmethod
    def _cluster_coherence(vectors: np.ndarray, indices: list[int]) -> float:
        """Compute mean pairwise cosine similarity within a cluster."""
        if len(indices) < 2:
            return 1.0

        cluster_vecs = vectors[indices]
        norms = np.linalg.norm(cluster_vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normalized = cluster_vecs / norms

        # Mean pairwise similarity (upper triangle only)
        sims = normalized @ normalized.T
        n = len(indices)
        upper_sum = (sims.sum() - n) / 2  # Subtract diagonal
        pair_count = n * (n - 1) / 2
        return float(upper_sum / pair_count) if pair_count > 0 else 1.0

    def cluster(
        self,
        vectors: list[list[float]] | np.ndarray,
        max_k: int = 10,
        domain: str = "general",
    ) -> list[ClusterResult]:
        """Perform spectral clustering with automatic k-selection.

        Args:
            vectors: List of embedding vectors (all same dimensionality).
            max_k: Maximum number of clusters to consider.
            domain: Domain context for cluster labeling.

        Returns:
            List of ClusterResult objects, one per discovered cluster.
        """
        arr = np.array(vectors, dtype=np.float64)
        n = arr.shape[0]

        if n < 2:
            return [
                ClusterResult(
                    cluster_id=f"sc_{uuid.uuid4().hex[:8]}",
                    label=f"{domain}_singleton",
                    indices=list(range(n)),
                    centroid=arr[0].tolist() if n > 0 else [],
                    coherence=1.0,
                )
            ]

        # 1. Build affinity matrix
        affinity = self._cosine_similarity_matrix(arr)

        # 2. Compute normalized Laplacian
        laplacian = self._normalized_laplacian(affinity)

        # 3. Eigendecomposition (smallest eigenvalues)
        num_eigs = min(max_k + 1, n)
        try:
            from scipy.sparse.linalg import eigsh

            eigenvalues, eigenvectors = eigsh(laplacian, k=num_eigs, which="SM")
        except Exception:
            # Fallback to dense eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
            eigenvalues = eigenvalues[:num_eigs]
            eigenvectors = eigenvectors[:, :num_eigs]

        # 4. Eigengap k-selection
        k = self._eigengap_k(eigenvalues, max_k)

        # 5. k-means on eigenvector embedding
        spectral_embedding = eigenvectors[:, :k]

        # Normalize rows for k-means stability
        row_norms = np.linalg.norm(spectral_embedding, axis=1, keepdims=True)
        row_norms = np.where(row_norms == 0, 1.0, row_norms)
        spectral_embedding = spectral_embedding / row_norms

        labels = self._kmeans(spectral_embedding, k)

        # 6. Build cluster results
        results: list[ClusterResult] = []
        for cluster_idx in range(k):
            member_indices = [i for i, lbl in enumerate(labels) if lbl == cluster_idx]
            if len(member_indices) < self._min_cluster_size:
                continue

            centroid = arr[member_indices].mean(axis=0).tolist()
            coherence = self._cluster_coherence(arr, member_indices)

            results.append(
                ClusterResult(
                    cluster_id=f"sc_{uuid.uuid4().hex[:8]}",
                    label=f"{domain}_cluster_{cluster_idx}",
                    indices=member_indices,
                    centroid=centroid,
                    coherence=coherence,
                )
            )

        # Sort by size descending
        results.sort(key=lambda c: len(c.indices), reverse=True)
        return results

    @staticmethod
    def _kmeans(data: np.ndarray, k: int, max_iters: int = 100) -> list[int]:
        """Simple k-means implementation without sklearn dependency.

        Uses k-means++ initialization for better convergence.
        """
        n = data.shape[0]
        if k >= n:
            return list(range(n))

        # k-means++ initialization
        rng = np.random.default_rng(42)
        centroids = np.empty((k, data.shape[1]))
        centroids[0] = data[rng.integers(n)]

        for c in range(1, k):
            dists = np.array(
                [
                    min(np.linalg.norm(data[i] - centroids[j]) ** 2 for j in range(c))
                    for i in range(n)
                ]
            )
            total = dists.sum()
            if total == 0:
                centroids[c] = data[rng.integers(n)]
            else:
                probs = dists / total
                centroids[c] = data[rng.choice(n, p=probs)]

        # Iterative assignment and update
        labels = [0] * n
        for _ in range(max_iters):
            # Assign
            new_labels = []
            for i in range(n):
                dists = [np.linalg.norm(data[i] - centroids[j]) for j in range(k)]
                new_labels.append(int(np.argmin(dists)))

            if new_labels == labels:
                break
            labels = new_labels

            # Update centroids
            for j in range(k):
                members = [i for i, lbl in enumerate(labels) if lbl == j]
                if members:
                    centroids[j] = data[members].mean(axis=0)

        return labels

    def cluster_to_kg_nodes(
        self,
        clusters: list[ClusterResult],
        domain: str = "general",
    ) -> list[SpectralClusterNode]:
        """Convert cluster results to KG-persistable SpectralClusterNodes.

        Args:
            clusters: Results from ``cluster()``.
            domain: Domain context for the clusters.

        Returns:
            List of SpectralClusterNode Pydantic models ready for KG persistence.
        """
        nodes: list[SpectralClusterNode] = []
        for cluster in clusters:
            node = SpectralClusterNode(
                id=cluster.cluster_id,
                name=cluster.label,
                description=(
                    f"Spectral cluster with {len(cluster.indices)} members, "
                    f"coherence={cluster.coherence:.3f}"
                ),
                cluster_label=cluster.label,
                member_count=len(cluster.indices),
                coherence_score=cluster.coherence,
                centroid_embedding=cluster.centroid,
                domain=domain,
            )
            nodes.append(node)
        return nodes

    def detect_financial_regimes(
        self,
        price_embeddings: list[list[float]] | np.ndarray,
        max_regimes: int = 5,
    ) -> list[ClusterResult]:
        """Detect market regimes via spectral clustering on price embeddings.

        CONCEPT:KG-2.5 — Financial regime detection extension.

        Applies spectral clustering to financial time-series embeddings
        to discover distinct market regimes (bull/bear/sideways/volatile).
        Each regime maps to a FIBO-aligned ``SpectralClusterNode`` with
        ``domain='financial'``.

        Args:
            price_embeddings: Embedding vectors representing time windows of price data.
            max_regimes: Maximum number of regimes to discover.

        Returns:
            List of ClusterResult objects representing detected regimes.
        """
        clusters = self.cluster(price_embeddings, max_k=max_regimes, domain="financial")
        # Relabel with financial semantics based on cluster ordering
        regime_labels = [
            "regime_dominant",
            "regime_secondary",
            "regime_tertiary",
            "regime_quaternary",
            "regime_minor",
        ]
        for i, cluster in enumerate(clusters):
            if i < len(regime_labels):
                cluster.label = f"financial_{regime_labels[i]}"
        return clusters
