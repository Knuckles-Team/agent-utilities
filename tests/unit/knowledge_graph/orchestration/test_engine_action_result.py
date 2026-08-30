"""Regression tests for the shared orchestration action-result write seam."""

from __future__ import annotations

from typing import Any

from agent_utilities.knowledge_graph.orchestration.engine_ahe import AHEMixin


class _Graph:
    def __init__(self, events: list[tuple[Any, ...]]) -> None:
        self.events = events
        self.nodes: dict[str, dict[str, Any]] = {"trace-1": {}}

    def add_node(self, node_id: str, **properties: Any) -> None:
        self.events.append(("graph.add_node", node_id, properties))
        self.nodes[node_id] = properties

    def __contains__(self, node_id: str) -> bool:
        return node_id in self.nodes

    def add_edge(self, source_id: str, target_id: str, **properties: Any) -> None:
        self.events.append(("graph.add_edge", source_id, target_id, properties))


class _Engine(AHEMixin):
    def __init__(self, *, backend: object | None) -> None:
        self.events: list[tuple[Any, ...]] = []
        self.graph = _Graph(self.events)
        self.backend = backend

    def _serialize_node(self, node: Any, label: str | None = None) -> dict[str, Any]:
        self.events.append(("serialize", label))
        return {
            "id": node.id,
            "name": node.name,
            "textual_gradient": node.textual_gradient,
            "timestamp": node.timestamp,
        }

    def _upsert_node(self, label: str, node_id: str, data: dict[str, Any]) -> None:
        self.events.append(("upsert", label, node_id, data))

    def link_nodes(self, source_id: str, target_id: str, relationship: str) -> None:
        self.events.append(("link", source_id, target_id, relationship))


def test_generate_critique_preserves_action_result_write_order_and_payload() -> None:
    engine = _Engine(backend=object())

    critique_id = engine.generate_critique("trace-1", "Use the broader context")

    assert critique_id.startswith("crit:")
    assert [event[0] for event in engine.events] == [
        "serialize",
        "graph.add_node",
        "serialize",
        "upsert",
        "link",
        "graph.add_edge",
    ]
    assert engine.events[1][1] == critique_id
    assert engine.events[1][2]["textual_gradient"] == "Use the broader context"
    assert engine.events[3][1:3] == ("Critique", critique_id)
    assert engine.events[3][3]["textual_gradient"] == "Use the broader context"
    assert engine.events[4][1:] == (
        "trace-1",
        critique_id,
        "GENERATED_CRITIQUE",
    )
    assert engine.events[5][1:] == (
        "trace-1",
        critique_id,
        {"relationship": "GENERATED_CRITIQUE"},
    )


def test_generate_critique_without_backend_keeps_compute_edge_only() -> None:
    engine = _Engine(backend=None)

    critique_id = engine.generate_critique("trace-1", "Keep the trace grounded")

    assert [event[0] for event in engine.events] == [
        "serialize",
        "graph.add_node",
        "graph.add_edge",
    ]
    assert engine.events[-1][2] == critique_id
