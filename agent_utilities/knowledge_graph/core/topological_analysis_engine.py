"""Topological Analysis Engine — Consolidated Graph Topology Facade.

CONCEPT:KG-2.5/2.15/2.34/2.35 — Topological Analysis Engine

Provides a single entry point for all graph-topology operations:
- Community detection via Louvain partitioning (KG-2.5)
- Topological analogy matching across domains (KG-2.15)
- Spectral cluster navigation with auto-k selection (KG-2.34)
- Symbol blast-radius analysis for impact scoring (KG-2.35)

All sub-modules remain as separate files for modularity; this engine
provides a unified API and lazy initialization.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class TopologicalAnalysisEngine:
    """Consolidated topological analysis engine.

    CONCEPT:KG-2.5/2.15/2.34/2.35 — Topological Analysis Engine

    Unifies community detection, analogy matching, spectral clustering,
    and blast-radius analysis into a single facade.

    Usage::

        engine = TopologicalAnalysisEngine(graph)

        # Community detection
        communities = engine.detect_communities()

        # Analogy matching
        analogies = engine.find_analogies("security", "finance")

        # Spectral clustering
        clusters = engine.build_spectral_clusters(nodes)

        # Blast radius
        report = engine.analyze_blast_radius("MyClass", project_root="/path")

    Args:
        graph: The GraphComputeEngine knowledge graph.
    """

    def __init__(self, graph: Any | None = None) -> None:
        self._graph = graph
        self._community_detector: Any = None
        self._analogy_engine: Any = None
        self._spectral_navigator: Any = None
        self._blast_analyzer: Any = None

    # --- Community Detection (KG-2.5) ---

    def detect_communities(self) -> list[set[str]]:
        """Detect emergent communities using Louvain partitioning.

        Returns:
            List of sets, each containing node IDs in a community.
        """
        if self._graph is None:
            return []

        from .topological_partition import detect_communities

        return detect_communities(self._graph)

    def persist_stable_communities(self, engine: Any) -> int:
        """Detect and persist stable communities into the Cypher backend.

        Args:
            engine: The KnowledgeGraphEngine instance.

        Returns:
            Number of communities persisted.
        """
        from .topological_partition import persist_stable_communities

        return persist_stable_communities(engine)

    # --- Topological Analogy (KG-2.15) ---

    def find_analogous_subgraphs(
        self,
        target_subgraph: Any,
        threshold: float = 0.8,
    ) -> list[Any]:
        """Find subgraphs analogous to the target subgraph.

        Uses VF2 exact subgraph isomorphism combined with vectorized
        EncPI embeddings for deep structural/semantic analogy matching.

        Args:
            target_subgraph: A GraphComputeEngine query subgraph.
            threshold: Semantic similarity threshold (0.0–1.0).

        Returns:
            List of AnalogyMatchNode instances.
        """
        if self._graph is None:
            return []

        if self._analogy_engine is None:
            from .analogy_engine import TopologicalAnalogyEngine

            self._analogy_engine = TopologicalAnalogyEngine(self._graph)

        return self._analogy_engine.find_analogous_subgraphs(
            target_subgraph, threshold=threshold
        )

    # --- Spectral Clustering (KG-2.34) ---

    def build_spectral_clusters(
        self,
        vectors: list[list[float]],
        max_k: int = 10,
        domain: str = "default",
    ) -> list[Any]:
        """Build spectral clusters from embedding vectors.

        Uses normalized Laplacian eigengap heuristic for automatic
        k-selection (no hyperparameter tuning needed).

        Args:
            vectors: List of embedding vectors.
            max_k: Maximum number of clusters to consider.
            domain: Domain label for the cluster results.

        Returns:
            List of ClusterResult instances.
        """
        if self._spectral_navigator is None:
            from .spectral_navigator import SpectralClusterNavigator

            self._spectral_navigator = SpectralClusterNavigator()

        return self._spectral_navigator.cluster(vectors, max_k=max_k, domain=domain)

    def cluster_to_kg_nodes(
        self,
        cluster_results: list[Any],
        domain: str = "default",
    ) -> list[Any]:
        """Convert cluster results to SpectralClusterNode instances.

        Args:
            cluster_results: Output from build_spectral_clusters.
            domain: Domain label.

        Returns:
            List of SpectralClusterNode instances.
        """
        if self._spectral_navigator is None:
            from .spectral_navigator import SpectralClusterNavigator

            self._spectral_navigator = SpectralClusterNavigator()

        return self._spectral_navigator.cluster_to_kg_nodes(
            cluster_results, domain=domain
        )

    # --- Blast Radius Analysis (KG-2.35) ---

    def analyze_blast_radius(
        self,
        symbol_name: str,
        root_dir: str = ".",
        definition_file: str | None = None,
        symbol_type: str = "function",
    ) -> Any:
        """Analyze the blast radius of a Python symbol.

        Args:
            symbol_name: The symbol name (function, class, variable).
            root_dir: Root directory of the codebase to scan.
            definition_file: Optional file where the symbol is defined.
            symbol_type: Type of symbol (function, class, variable).

        Returns:
            BlastRadiusNode with usage counts and impact score.
        """
        from .blast_radius import BlastRadiusAnalyzer

        if self._blast_analyzer is None:
            self._blast_analyzer = BlastRadiusAnalyzer(root_dir=root_dir)

        return self._blast_analyzer.analyze(
            symbol_name, definition_file=definition_file, symbol_type=symbol_type
        )

    # --- Unified Analysis ---

    def run_full_topology_analysis(
        self,
        min_community_size: int = 3,
    ) -> dict[str, Any]:
        """Run a complete topological analysis of the graph.

        Combines community detection and basic stats.

        Args:
            min_community_size: Minimum community size to report.

        Returns:
            Dict with community, cluster, and graph stats.
        """
        report: dict[str, Any] = {
            "total_nodes": 0,
            "total_edges": 0,
            "communities": [],
            "community_count": 0,
        }

        if self._graph is None:
            return report

        report["total_nodes"] = self._graph.number_of_nodes()
        report["total_edges"] = self._graph.number_of_edges()

        communities = self.detect_communities()
        significant = [c for c in communities if len(c) >= min_community_size]
        report["communities"] = [
            {"size": len(c), "sample_nodes": list(c)[:5]} for c in significant
        ]
        report["community_count"] = len(significant)

        return report
