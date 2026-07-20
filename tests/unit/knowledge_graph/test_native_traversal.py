"""Native graph traversal remains one bounded engine operation."""

from __future__ import annotations

import types

import pytest

from agent_utilities.knowledge_graph.orchestration.engine_query import QueryMixin


def test_find_path_uses_one_native_shortest_path_call():
    calls = {"shortest": 0}

    class FakeGraph:
        def has_node(self, node_id):
            return True

        def get_shortest_path(self, source, target):
            calls["shortest"] += 1
            return [source, "mid", target]

        def get_successors(self, node_id):
            raise AssertionError("find_path must not run a Python traversal")

    target = types.SimpleNamespace(graph=FakeGraph())
    assert QueryMixin.find_path(target, "a", "b") == ["a", "mid", "b"]
    assert calls == {"shortest": 1}


def test_find_path_propagates_native_capability_errors():
    class FakeGraph:
        def has_node(self, node_id):
            return True

        def get_shortest_path(self, source, target):
            raise RuntimeError("native traversal unavailable")

    target = types.SimpleNamespace(graph=FakeGraph())
    with pytest.raises(RuntimeError, match="native traversal unavailable"):
        QueryMixin.find_path(target, "a", "b")


def test_find_path_returns_empty_when_an_endpoint_is_missing():
    class FakeGraph:
        def has_node(self, node_id):
            return False

        def get_shortest_path(self, source, target):
            raise AssertionError("missing endpoints must not reach the engine")

    target = types.SimpleNamespace(graph=FakeGraph())
    assert QueryMixin.find_path(target, "missing", "also-missing") == []
