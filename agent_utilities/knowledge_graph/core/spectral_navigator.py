#!/usr/bin/python
from __future__ import annotations

"""Spectral Cluster Navigator.

CONCEPT:AU-KG.compute.spectral-cluster-navigator — Spectral Cluster Navigator

Provides tuning-free spectral clustering using the normalized Laplacian
eigengap heuristic for automatic k-selection. Adapted from contextplus's
clustering.ts with OWL ontology integration and financial regime detection.

Eigendecomposition runs on the ``agent_utilities.numeric`` kernel shim
(``xp.eigsh`` — ``scipy.sparse.linalg.eigsh`` smallest-magnitude eigenpairs,
now a native ``epistemic_graph.numeric`` export; no numpy/scipy dependency).
Clusters auto-map to ``skos:Concept`` nodes with ``broader``/``narrower`` edges
for OWL-transitive hierarchies.
"""


import logging
import math
import uuid
from dataclasses import dataclass, field

from agent_utilities.models.knowledge_graph import (
    SpectralClusterNode,
)
from agent_utilities.numeric import xp

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

    CONCEPT:AU-KG.compute.spectral-cluster-navigator — Spectral Cluster Navigator

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
    def _cosine_similarity_matrix(vectors: list[list[float]]) -> list[list[float]]:
        """Build the cosine similarity affinity matrix.

        Normalizes each vector to unit length then computes dot products.
        Clips to [0, 1] to ensure non-negative affinity.
        """
        normalized = []
        for vector in vectors:
            norm = math.sqrt(sum(value * value for value in vector))
            normalized.append(
                [value / norm for value in vector] if norm else [0.0] * len(vector)
            )
        transpose = [list(column) for column in zip(*normalized, strict=True)]
        raw = xp.matmul(normalized, transpose)
        return [
            [max(0.0, min(1.0, float(row[column]))) for column in range(len(row))]
            for row in raw
        ]

    @staticmethod
    def _normalized_laplacian(affinity: list[list[float]]) -> list[list[float]]:
        """Compute the symmetric normalized Laplacian.

        L_sym = I - D^{-1/2} @ W @ D^{-1/2}

        where W is the affinity matrix and D is the degree matrix.
        """
        n = len(affinity)
        weights = [
            [
                0.0 if row == column else float(affinity[row][column])
                for column in range(n)
            ]
            for row in range(n)
        ]
        degree = [float(value) for value in xp.sum(weights, axis=1)]
        scales = [1.0 / math.sqrt(value) if value > 0 else 0.0 for value in degree]
        diagonal = [
            [scales[row] if row == column else 0.0 for column in range(n)]
            for row in range(n)
        ]
        normalized = xp.matmul(xp.matmul(diagonal, weights), diagonal)
        return [
            [
                (1.0 if row == column else 0.0) - float(values[column])
                for column in range(n)
            ]
            for row, values in enumerate(normalized)
        ]

    @staticmethod
    def _eigengap_k(eigenvalues: list[float], max_k: int) -> int:
        """Select optimal k using the eigengap heuristic.

        Finds the largest gap between consecutive eigenvalues in the
        range [2, max_k]. Returns at least 2 clusters.
        """
        if len(eigenvalues) < 3:
            return min(2, len(eigenvalues))

        # Compute gaps between consecutive sorted eigenvalues
        sorted_vals = sorted(float(value) for value in eigenvalues)
        upper = min(max_k, len(sorted_vals) - 1)
        if upper < 2:
            return 2

        gaps = [
            sorted_vals[index + 1] - sorted_vals[index] for index in range(1, upper)
        ]
        if len(gaps) == 0:
            return 2

        best_k = max(range(len(gaps)), key=lambda index: gaps[index]) + 2
        return max(2, min(best_k, max_k))

    @staticmethod
    def _cluster_coherence(vectors: list[list[float]], indices: list[int]) -> float:
        """Compute mean pairwise cosine similarity within a cluster."""
        if len(indices) < 2:
            return 1.0

        cluster_vecs = [vectors[index] for index in indices]
        normalized = []
        for vector in cluster_vecs:
            norm = math.sqrt(sum(value * value for value in vector))
            normalized.append(
                [value / norm for value in vector] if norm else [0.0] * len(vector)
            )
        transpose = [list(column) for column in zip(*normalized, strict=True)]
        similarities = xp.matmul(normalized, transpose)
        n = len(indices)
        pair_count = n * (n - 1) / 2
        upper_sum = sum(
            float(similarities[row][column])
            for row in range(n)
            for column in range(row + 1, n)
        )
        return float(upper_sum / pair_count) if pair_count else 1.0

    def cluster(
        self,
        vectors: list[list[float]],
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
        arr = [[float(value) for value in vector] for vector in vectors]
        n = len(arr)
        if arr and (not arr[0] or any(len(row) != len(arr[0]) for row in arr)):
            raise ValueError("cluster() requires a non-empty rectangular vector matrix")

        if n < 2:
            return [
                ClusterResult(
                    cluster_id=f"sc_{uuid.uuid4().hex}",
                    label=f"{domain}_singleton",
                    indices=list(range(n)),
                    centroid=arr[0] if n > 0 else [],
                    coherence=1.0,
                )
            ]

        # 1. Build affinity matrix
        affinity = self._cosine_similarity_matrix(arr)

        # 2. Compute normalized Laplacian
        laplacian = self._normalized_laplacian(affinity)

        # 3. Eigendecomposition (smallest eigenvalues)
        num_eigs = min(max_k + 1, n)
        if num_eigs >= n:
            # The native sparse contract requires k < n; use the explicit
            # dense path when the requested spectrum covers the whole matrix.
            eigenvalues, eigenvectors = xp.linalg.eigh(laplacian)
            eigenvalues = eigenvalues[:num_eigs]
            eigenvectors = [row[:num_eigs] for row in eigenvectors]
        else:
            try:
                eigenvalues, eigenvectors = xp.eigsh(laplacian, num_eigs)
            except xp.LinAlgError:
                # The native dense decomposition is the explicit small-matrix path.
                eigenvalues, eigenvectors = xp.linalg.eigh(laplacian)
                eigenvalues = eigenvalues[:num_eigs]
                eigenvectors = [row[:num_eigs] for row in eigenvectors]

        # 4. Eigengap k-selection
        k = self._eigengap_k(eigenvalues, max_k)

        # 5. k-means on eigenvector embedding
        raw_spectral_embedding = [row[:k] for row in eigenvectors]

        # Normalize rows for k-means stability
        spectral_embedding = []
        for row in raw_spectral_embedding:
            norm = math.sqrt(sum(value * value for value in row))
            spectral_embedding.append(
                [value / norm for value in row] if norm else [0.0] * len(row)
            )

        labels = self._kmeans(spectral_embedding, k)

        # 6. Build cluster results
        results: list[ClusterResult] = []
        for cluster_idx in range(k):
            member_indices = [i for i, lbl in enumerate(labels) if lbl == cluster_idx]
            if len(member_indices) < self._min_cluster_size:
                continue

            centroid = [
                sum(arr[index][dimension] for index in member_indices)
                / len(member_indices)
                for dimension in range(len(arr[0]))
            ]
            coherence = self._cluster_coherence(arr, member_indices)

            results.append(
                ClusterResult(
                    cluster_id=f"sc_{uuid.uuid4().hex}",
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
    def _kmeans(data: list[list[float]], k: int, max_iters: int = 100) -> list[int]:
        """Run bounded native k-means without a Python numeric implementation."""
        n = len(data)
        if k >= n:
            return list(range(n))

        labels, _centroids = xp.kmeans(data, k, max_iters, 42)
        return [int(label) for label in labels]

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
        price_embeddings: list[list[float]],
        max_regimes: int = 5,
    ) -> list[ClusterResult]:
        """Detect market regimes via spectral clustering on price embeddings.

        CONCEPT:AU-KG.compute.spectral-cluster-navigator — Financial regime detection extension.

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
