"""The inline EG coloring boundary preserves AU's conflict schedule contract."""

import pytest

from agent_utilities.knowledge_graph.core.formal_reasoning_core import (
    chromatic_number_upper_bound,
    chromatic_schedule,
)
from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
from agent_utilities.knowledge_graph.core.graph_primitives import PyGraph


def _conflicts():
    graph = PyGraph()
    a, b, c, isolate = [graph.add_node(node) for node in ("a", "b", "c", "d")]
    graph.add_edge(a, b, {})
    graph.add_edge(b, c, {})
    graph.add_edge(a, c, {})
    assert isolate == 3
    return graph


def test_native_coloring_preserves_schedule_and_upper_bound():
    calls = []

    def native(node_ids, edges):
        calls.append((node_ids, edges))
        return [["a", 0], ["b", 1], ["c", 2], ["d", 0]]

    graph = _conflicts()
    assert chromatic_schedule(graph, native_coloring=native) == {
        "a": 0,
        "b": 1,
        "c": 2,
        "d": 0,
    }
    assert chromatic_number_upper_bound(graph, native_coloring=native) == 3
    assert calls == [(["a", "b", "c", "d"], [("a", "b"), ("b", "c"), ("a", "c")])] * 2


def test_native_invalid_coloring_fails_without_local_fallback(monkeypatch):
    graph = _conflicts()
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.formal_reasoning_core.rx.graph_greedy_color",
        lambda _graph: pytest.fail("local coloring ran"),
    )
    with pytest.raises(ValueError, match="violates the conflict graph"):
        chromatic_schedule(
            graph,
            native_coloring=lambda _nodes, _edges: [
                ["a", 0],
                ["b", 0],
                ["c", 1],
                ["d", 0],
            ],
        )


def test_native_error_propagates():
    graph = _conflicts()

    def failed(_nodes, _edges):
        raise RuntimeError("EG unavailable")

    with pytest.raises(RuntimeError, match="EG unavailable"):
        chromatic_schedule(graph, native_coloring=failed)


def test_graph_facade_delegates_inline_request():
    calls = []

    class Graph:
        def graph_color_ephemeral(self, node_ids, edges):
            calls.append((node_ids, edges))
            return [["a", 0], ["b", 1]]

    compute = GraphComputeEngine.__new__(GraphComputeEngine)
    compute._client = type("Client", (), {"graph": Graph()})()
    assert compute.graph_color_ephemeral(["a", "b"], [("a", "b")]) == [
        ["a", 0],
        ["b", 1],
    ]
    assert calls == [(["a", "b"], [("a", "b")])]
