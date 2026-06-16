"""A label-scoped MATCH uses the engine's bounded labeled fetch, not a full scan.

CONCEPT:KG-2.51 — `_exec_node_match` previously materialized the WHOLE graph
(`_get_all_nodes_with_properties`) for any `MATCH (n:Label) … LIMIT k`; it now
pushes the label (and the LIMIT, for a pure read) down to the engine.
"""

from __future__ import annotations

from typing import Any

from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
    EpistemicGraphBackend,
)


class _FakeGraph:
    def __init__(self) -> None:
        self.by_label_calls: list[tuple[str, int]] = []
        self.full_scan_calls = 0

    def get_nodes_by_label(
        self, label: str, limit: int
    ) -> list[tuple[str, dict[str, Any]]]:
        self.by_label_calls.append((label, limit))
        return [("a1", {"type": "Agent", "name": "A"})]

    def _get_all_nodes_with_properties(self) -> list[tuple[str, dict[str, Any]]]:
        self.full_scan_calls += 1
        return [("z1", {"type": "Other"})]

    def has_node(self, _nid: str) -> bool:
        return False

    def _get_node_properties(self, _nid: str) -> dict[str, Any]:
        return {}


def _backend(g: _FakeGraph) -> EpistemicGraphBackend:
    b = EpistemicGraphBackend.__new__(EpistemicGraphBackend)
    b._graph = g
    b._embeddings = {}
    return b


def test_label_query_pushes_label_and_limit_down() -> None:
    g = _FakeGraph()
    b = _backend(g)
    rows = b.execute("MATCH (n:Agent) RETURN n LIMIT 5")
    assert g.by_label_calls == [("Agent", 5)]  # pure read -> LIMIT pushed down
    assert g.full_scan_calls == 0  # NO full-graph scan
    assert rows  # returned the agent


def test_label_query_with_where_fetches_label_unbounded() -> None:
    g = _FakeGraph()
    b = _backend(g)
    b.execute("MATCH (n:Agent) WHERE n.name = 'A' RETURN n LIMIT 5")
    # WHERE could drop matches, so don't push the LIMIT — but still scope to label.
    assert g.by_label_calls == [("Agent", 0)]
    assert g.full_scan_calls == 0


def test_bare_match_without_label_still_full_scans() -> None:
    g = _FakeGraph()
    b = _backend(g)
    b.execute("MATCH (n) RETURN n LIMIT 3")
    assert g.by_label_calls == []
    assert g.full_scan_calls == 1
