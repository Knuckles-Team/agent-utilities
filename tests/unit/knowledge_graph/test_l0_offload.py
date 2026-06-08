"""Plan 08 Synergy 4: graph traversal offloads to the Rust L0 compute tier.

`find_path` must issue a single L0 `get_shortest_path` call (compiled Rust)
rather than a Python BFS over per-edge L0 calls. The Python reimplementation
survives only as a fallback when L0 is unavailable.
"""

from __future__ import annotations

import types

from agent_utilities.knowledge_graph.orchestration.engine_query import QueryMixin


def test_find_path_offloads_to_l0_single_call():
    calls = {"shortest": 0, "successors": 0}

    class FakeGCE:
        def has_node(self, n):
            return True

        def get_shortest_path(self, s, t):
            calls["shortest"] += 1
            return [s, "mid", t]

        def get_successors(self, n):  # the Python BFS path — must NOT be used
            calls["successors"] += 1
            return []

    fake = types.SimpleNamespace(graph=FakeGCE())
    result = QueryMixin.find_path(fake, "a", "b")  # type: ignore[arg-type]

    assert result == ["a", "mid", "b"]
    assert calls["shortest"] == 1  # used the compiled L0 traversal
    assert calls["successors"] == 0  # did NOT fall into the Python BFS


def test_find_path_falls_back_to_bfs_when_l0_unavailable():
    adjacency = {"a": ["b"], "b": ["c"]}

    class FakeGCE:
        def has_node(self, n):
            return n in {"a", "b", "c"}

        def get_shortest_path(self, s, t):
            raise RuntimeError("L0 compute unavailable")

        def get_successors(self, n):
            return adjacency.get(n, [])

    fake = types.SimpleNamespace(graph=FakeGCE())
    result = QueryMixin.find_path(fake, "a", "c")  # type: ignore[arg-type]
    assert result == ["a", "b", "c"]  # BFS fallback still works


def test_find_path_missing_nodes_returns_empty():
    class FakeGCE:
        def has_node(self, n):
            return False

        def get_shortest_path(self, s, t):
            raise AssertionError("should not be called")

        def get_successors(self, n):
            return []

    fake = types.SimpleNamespace(graph=FakeGCE())
    assert QueryMixin.find_path(fake, "x", "y") == []  # type: ignore[arg-type]
