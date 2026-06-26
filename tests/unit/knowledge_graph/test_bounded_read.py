"""Unit tests for bounded node iteration (CONCEPT:KG-2.261)."""

from __future__ import annotations

from agent_utilities.knowledge_graph.core.bounded_read import iter_nodes_by_types


class _EngineGraph:
    """Engine-like graph: exposes get_nodes_by_label (bounded) and would EXPLODE on a
    full nodes(data=True) pull — proving the bounded path never falls back to it."""

    def __init__(self, by_label: dict[str, list]):
        self._by_label = by_label  # label -> [[id, data], ...]

    def get_nodes_by_label(self, label: str, limit: int = 0) -> list:
        return self._by_label.get(label, [])

    def nodes(self, data: bool = False):
        raise AssertionError("full graph.nodes() pull must NOT happen on the engine")


class _LocalGraph:
    """Small in-memory graph with NO bounded fetch → full iteration is correct."""

    def __init__(self, nodes: dict[str, dict]):
        self._n = nodes

    def nodes(self, data: bool = False):
        return list(self._n.items()) if data else list(self._n)


def test_engine_graph_uses_bounded_label_fetch_not_full_scan():
    g = _EngineGraph(
        {
            "Team": [["team:1", {"type": "team", "name": "A"}]],
            "team": [["team:2", {"type": "team", "name": "B"}]],  # casing variant
        }
    )
    out = dict(iter_nodes_by_types(g, "team"))
    assert set(out) == {"team:1", "team:2"}  # both casings, deduped
    assert all(d["type"] == "team" for d in out.values())


def test_engine_empty_type_does_not_full_scan():
    """A legitimately-empty type must return empty WITHOUT a full-graph pull."""
    g = _EngineGraph({})  # nothing of the type; nodes() would assert
    assert (
        dict(iter_nodes_by_types(g, "nonexistent")) == {}
    )  # no exception = no full scan


def test_local_graph_full_iteration():
    g = _LocalGraph(
        {
            "a": {"type": "team", "name": "A"},
            "b": {"type": "policy", "name": "B"},
            "c": {"type": "team", "name": "C"},
        }
    )
    out = dict(iter_nodes_by_types(g, "team"))
    assert set(out) == {"a", "c"}


def test_multiple_types():
    g = _EngineGraph(
        {
            "Team": [["t1", {"type": "team"}]],
            "Policy": [["p1", {"type": "policy"}]],
        }
    )
    out = dict(iter_nodes_by_types(g, "team", "policy"))
    assert set(out) == {"t1", "p1"}


def test_enum_type_value_resolved():
    class _NT:
        value = "team"

    g = _EngineGraph({"team": [["t1", {"type": "team"}]]})
    out = dict(iter_nodes_by_types(g, _NT()))
    assert set(out) == {"t1"}
