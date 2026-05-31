from __future__ import annotations

"""Pure In-Memory Graph Backend (CONCEPT:OS-5.0).

Zero-dependency, zero-disk backend using GraphComputeEngine (Rust/epistemic-graph).
Ideal for testing, edge devices, ephemeral containers,
and as the default lightweight backend.

Implements the full GraphBackend ABC with optional
JSON serialization for persistence.
"""


import json
import logging
from typing import Any

from .base import GraphBackend

logger = logging.getLogger(__name__)


class EpistemicGraphBackend(GraphBackend):
    """Pure in-memory graph backend using GraphComputeEngine (Rust-native).

    This is the lightest-weight backend: zero disk, zero external
    dependencies beyond the compiled graph engine. All data lives in
    process memory and is lost on shutdown unless explicitly saved
    via ``save_to_json()``.

    Use cases:
        - Unit testing (fast, deterministic)
        - Edge compute (minimal footprint)
        - Ephemeral containers (no persistent storage needed)
        - Development/prototyping
    """

    def __init__(self) -> None:
        from ..core.graph_compute import GraphComputeEngine

        self._graph = GraphComputeEngine(backend_type="rust")
        self._embeddings: dict[str, list[float]] = {}
        self._node_counter = 0
        logger.info(
            "EpistemicGraphBackend initialized (GraphComputeEngine, pure in-memory)"
        )

    @property
    def graph(self) -> Any:
        """Direct access to the underlying GraphComputeEngine."""
        return self._graph

    # --- GraphBackend ABC Implementation ---

    def execute(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a query against the in-memory graph.

        For the memory backend, queries are simple key lookups
        or graph operations rather than Cypher.
        Supports basic patterns:
          - MATCH (n {id: $id}) → node lookup
          - MATCH (n:Label) → label filter
        """
        params = params or {}

        # Simple node lookup by ID
        if "id" in params:
            node_id = params["id"]
            if self._graph.has_node(node_id):
                data = self._graph._get_node_properties(node_id)
                data["id"] = node_id
                return [data]
            return []

        # Label filter
        if "label" in params:
            label = params["label"]
            results = []
            for nid in self._graph._get_all_nodes():
                data = self._graph._get_node_properties(nid)
                if data.get("label") == label:
                    entry = dict(data)
                    entry["id"] = nid
                    results.append(entry)
            return results

        # Return all nodes
        results = []
        for nid in self._graph._get_all_nodes():
            data = self._graph._get_node_properties(nid)
            entry = dict(data)
            entry["id"] = nid
            results.append(entry)
        return results

    def execute_batch(
        self, query: str, batch: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Execute batch operations."""
        results = []
        for params in batch:
            results.extend(self.execute(query, params))
        return results

    def create_schema(self) -> None:
        """Initialize schema metadata representation."""
        # Simple schema metadata tracking for in-memory backend
        self._schema_created = True

    def add_embedding(self, node_id: str, embedding: list[float]) -> None:
        """Store an embedding vector for a node."""
        self._embeddings[node_id] = embedding

    def semantic_search(
        self, query_embedding: list[float], n_results: int = 5
    ) -> list[dict[str, Any]]:
        """Cosine similarity search over stored embeddings."""
        if not self._embeddings:
            return []

        import numpy as np

        query_vec = np.array(query_embedding)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        scores: list[tuple[str, float]] = []
        for node_id, emb in self._embeddings.items():
            emb_vec = np.array(emb)
            emb_norm = np.linalg.norm(emb_vec)
            if emb_norm == 0:
                continue
            similarity = float(np.dot(query_vec, emb_vec) / (query_norm * emb_norm))
            scores.append((node_id, similarity))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for node_id, score in scores[:n_results]:
            if self._graph.has_node(node_id):
                data = self._graph._get_node_properties(node_id)
                data["id"] = node_id
                data["_similarity"] = score
                results.append(data)

        return results

    def prune(self, criteria: dict[str, Any]) -> None:
        """Prune nodes matching criteria."""
        to_remove = []
        for nid in self._graph._get_all_nodes():
            data = self._graph._get_node_properties(nid)
            match = all(data.get(k) == v for k, v in criteria.items())
            if match:
                to_remove.append(nid)

        for nid in to_remove:
            self._graph.remove_node(nid)
            self._embeddings.pop(nid, None)

        logger.info("Pruned %d nodes", len(to_remove))

    def close(self) -> None:
        """Reset the in-memory graph."""
        from ..core.graph_compute import GraphComputeEngine

        self._graph = GraphComputeEngine(backend_type="rust")
        self._embeddings.clear()

    # --- Extended API ---

    def health_check(self) -> bool:
        """Always healthy for in-memory backend."""
        return True

    def get_stats(self) -> dict[str, Any]:
        """Return graph statistics."""
        return {
            "backend": "memory",
            "nodes": self._graph.node_count(),
            "edges": self._graph.edge_count(),
            "embeddings": len(self._embeddings),
        }

    # --- Node/Edge Operations ---

    def add_node(self, node_id: str, label: str = "", **properties: Any) -> None:
        """Add a node to the graph."""
        props = {"label": label, **properties}
        self._graph.add_node(node_id, props)
        self._node_counter += 1

    def add_edge(
        self,
        source: str,
        target: str,
        rel_type: str = "",
        **properties: Any,
    ) -> None:
        """Add an edge between two nodes."""
        props = {"rel_type": rel_type, **properties}
        self._graph.add_edge(source, target, props)

    # --- Persistence ---

    def save_to_json(self, path: str) -> None:
        """Serialize the graph to a JSON file."""
        graph_json = self._graph.to_json()
        data = json.loads(graph_json)
        data["_embeddings"] = {k: v for k, v in self._embeddings.items()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("Graph saved to %s", path)

    def load_from_json(self, path: str) -> None:
        """Deserialize the graph from a JSON file."""
        with open(path) as f:
            data = json.load(f)

        embeddings = data.pop("_embeddings", {})
        self._graph.from_json(json.dumps(data))
        self._embeddings = embeddings
        logger.info(
            "Graph loaded from %s (%d nodes, %d edges)",
            path,
            self._graph.node_count(),
            self._graph.edge_count(),
        )
