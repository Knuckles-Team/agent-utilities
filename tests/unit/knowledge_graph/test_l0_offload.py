"""Plan 08 Synergy 4: graph traversal offloads to the Rust-native compute engine.

`find_path` must issue a single L0 `get_shortest_path` call (compiled Rust)
rather than a Python BFS over per-edge L0 calls. The Python BFS
reimplementation has since been retired outright (CONCEPT:AU-P0-2 — the same
"no hidden Python fallback with different limits or semantics" shift
`EpistemicGraphBackend` made, enforced there by
`tests/unit/test_native_cypher_routing.py`'s ``retired`` tuple, which lists
`get_successors`/`get_predecessors` among the names that must not resurface):
`QueryMixin.find_path`'s own docstring is now explicit that "transport and
capability errors propagate instead of selecting a hidden Python
implementation with different limits or semantics." A failed L0 call must
raise, not silently degrade to a BFS with different depth/cycle semantics.
"""

from __future__ import annotations

import types
from typing import cast

import pytest

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
    result = QueryMixin.find_path(cast(QueryMixin, fake), "a", "b")

    assert result == ["a", "mid", "b"]
    assert calls["shortest"] == 1  # used the compiled L0 traversal
    assert calls["successors"] == 0  # did NOT fall into the Python BFS


def test_find_path_falls_back_to_bfs_when_l0_unavailable():
    """Name kept for node-id stability (this test is tracked by id elsewhere);
    behavior updated: a failed L0 call now RAISES instead of falling back to a
    Python BFS with different limits/semantics (see module docstring,
    CONCEPT:AU-P0-2). There is no more BFS fallback to test."""
    calls = {"successors": 0}

    class FakeGCE:
        def has_node(self, n):
            return n in {"a", "b", "c"}

        def get_shortest_path(self, s, t):
            raise RuntimeError("L0 compute unavailable")

        def get_successors(self, n):  # the retired Python BFS path — must NOT be used
            calls["successors"] += 1
            return {"a": ["b"], "b": ["c"]}.get(n, [])

    fake = types.SimpleNamespace(graph=FakeGCE())
    with pytest.raises(RuntimeError, match="L0 compute unavailable"):
        QueryMixin.find_path(cast(QueryMixin, fake), "a", "c")
    assert calls["successors"] == 0  # did NOT fall into the retired Python BFS


def test_find_path_missing_nodes_returns_empty():
    class FakeGCE:
        def has_node(self, n):
            return False

        def get_shortest_path(self, s, t):
            raise AssertionError("should not be called")

        def get_successors(self, n):
            return []

    fake = types.SimpleNamespace(graph=FakeGCE())
    assert QueryMixin.find_path(cast(QueryMixin, fake), "x", "y") == []
