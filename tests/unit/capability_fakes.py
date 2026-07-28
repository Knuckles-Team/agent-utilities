"""Shared owner-shaped graph engine doubles for capability tests."""

from __future__ import annotations

from typing import Any

from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine


class TypedGraphEngine:
    """Typed engine authority backed by one real isolated graph transport."""

    def __init__(self) -> None:
        self.graph = GraphComputeEngine(backend_type="rust")
        self.graph_compute = self.graph
        self.backend: Any = None
        self.node_writes: list[tuple[str, str, dict[str, Any]]] = []
        self.edge_writes: list[tuple[str, str, str, dict[str, Any]]] = []

    def add_node(
        self,
        node_id: str,
        node_type: str,
        properties: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> None:
        props = dict(properties or {})
        self.node_writes.append((node_id, str(node_type), props))
        self.graph.add_node(node_id, {"node_type": str(node_type), **props})

    def add_edge(
        self,
        source: str,
        target: str,
        rel_type: str,
        **properties: Any,
    ) -> None:
        props = dict(properties)
        self.edge_writes.append((source, target, rel_type, props))
        self.graph.add_edge(
            source,
            target,
            {"relationship": rel_type, **props},
        )
