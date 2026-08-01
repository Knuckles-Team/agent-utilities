"""Unit tests for bounded node iteration (CONCEPT:AU-KG.ingest.never-scan-whole-graph)."""

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
            "Team": [["team:1", {"node_type": "team", "name": "A"}]],
            "team": [["team:2", {"node_type": "team", "name": "B"}]],  # casing variant
        }
    )
    out = dict(iter_nodes_by_types(g, "team"))
    assert set(out) == {"team:1", "team:2"}  # both casings, deduped
    assert all(d["node_type"] == "team" for d in out.values())


def test_engine_empty_type_does_not_full_scan():
    """A legitimately-empty type must return empty WITHOUT a full-graph pull."""
    g = _EngineGraph({})  # nothing of the type; nodes() would assert
    assert (
        dict(iter_nodes_by_types(g, "nonexistent")) == {}
    )  # no exception = no full scan


def test_local_graph_full_iteration():
    g = _LocalGraph(
        {
            "a": {"node_type": "team", "name": "A"},
            "b": {"node_type": "policy", "name": "B"},
            "c": {"node_type": "team", "name": "C"},
        }
    )
    out = dict(iter_nodes_by_types(g, "team"))
    assert set(out) == {"a", "c"}


def test_multiple_types():
    g = _EngineGraph(
        {
            "Team": [["t1", {"node_type": "team"}]],
            "Policy": [["p1", {"node_type": "policy"}]],
        }
    )
    out = dict(iter_nodes_by_types(g, "team", "policy"))
    assert set(out) == {"t1", "p1"}


def test_enum_type_value_resolved():
    class _NT:
        value = "team"

    g = _EngineGraph({"team": [["t1", {"node_type": "team"}]]})
    out = dict(iter_nodes_by_types(g, _NT()))
    assert set(out) == {"t1"}


def test_legacy_bare_type_key_still_matches_on_the_bounded_path():
    """A row written before the ``node_type`` convention existed (e.g. a
    pre-AU-P1-4 legacy ``:MediaAsset`` node, which only ever carried a bare
    ``type`` property) must still be found by a type-filtered scan — a scan
    keyed on ``node_type`` alone silently returns nothing for such rows, which
    is exactly why ``MediaStore.migrate_legacy_assets_bulk``'s sweep for
    ``type == 'MediaAsset'`` nodes previously found zero matches."""
    g = _EngineGraph(
        {"MediaAsset": [["media:1", {"type": "MediaAsset", "name": "legacy"}]]}
    )
    out = dict(iter_nodes_by_types(g, "MediaAsset"))
    assert set(out) == {"media:1"}


def test_legacy_bare_type_key_still_matches_on_the_local_full_scan_path():
    g = _LocalGraph(
        {
            "a": {"type": "MediaAsset", "name": "legacy"},
            "b": {"node_type": "team", "name": "current"},
        }
    )
    out = dict(iter_nodes_by_types(g, "MediaAsset"))
    assert set(out) == {"a"}


def test_node_type_wins_over_a_stray_bare_type_when_both_present():
    """``node_type`` is the CURRENT key — it must win over a legacy bare
    ``type`` a row might still (incorrectly) carry alongside it."""
    g = _LocalGraph({"a": {"node_type": "team", "type": "policy"}})
    out = dict(iter_nodes_by_types(g, "team"))
    assert set(out) == {"a"}
    assert dict(iter_nodes_by_types(g, "policy")) == {}
