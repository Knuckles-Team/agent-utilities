from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from agent_utilities.knowledge_graph.core.owl_bridge import OWLBridge


class _NativeBackend:
    typed_mutation_support = "native"
    cypher_support = "native"

    def __init__(self) -> None:
        self.edges: list[tuple[str, str, dict[str, Any]]] = []
        self.cypher_calls: list[tuple[str, dict[str, Any]]] = []

    def add_edge(self, source: str, target: str, /, **properties: Any) -> None:
        self.edges.append((source, target, dict(properties)))

    def execute(self, query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        self.cypher_calls.append((query, params))
        return []


def test_backfeed_uses_typed_edge_for_native_backend() -> None:
    """Native backfeed must avoid the unsupported comma-pattern edge write."""
    backend = _NativeBackend()
    bridge = OWLBridge(
        graph=SimpleNamespace(nodes=()), owl_backend=None, backend=backend
    )
    engine = SimpleNamespace(backend=object())

    assert (
        bridge._backfeed_one_inference(
            engine,
            {
                "subject": "source:1",
                "predicate": "dependsOn",
                "object": "target:1",
                "inference_type": "transitive_closure",
            },
        )
        is True
    )

    assert backend.edges == [
        (
            "source:1",
            "target:1",
            {
                "inferred": True,
                "inferred_from": "owl_reasoner",
                "inference_type": "transitive_closure",
                "relationship": "DEPENDSON",
            },
        )
    ]
    assert backend.cypher_calls == []
