"""Hardening (CONCEPT:ORCH-1.39): an unscoped query must NOT silently return the whole graph.

This was the `graph_context list` "garbage" over-match: an unparsed WHERE fell through to a
legacy reader that returned every node. It now returns [] unless KG_ALLOW_FULL_SCAN is set.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
    EpistemicGraphBackend,
)


class _FakeGraph:
    def __init__(self):
        self._nodes = {
            "ctx:s1:a": {"type": "ContextBlob", "session_id": "s1", "content": "x"},
            "doc:9": {"type": "Document", "content": "unrelated textbook chunk"},
        }

    def has_node(self, nid):
        return nid in self._nodes

    def _get_all_nodes(self):
        return list(self._nodes)

    def _get_node_properties(self, nid):
        return dict(self._nodes.get(nid, {}))


def _backend():
    b = object.__new__(EpistemicGraphBackend)  # bypass engine-connecting __init__
    b._graph = _FakeGraph()
    return b


@pytest.mark.concept("ORCH-1.39")
def test_unscoped_query_returns_empty_not_all_nodes(monkeypatch):
    monkeypatch.delenv("KG_ALLOW_FULL_SCAN", raising=False)
    b = _backend()
    # No id, no label → must be empty (NOT the whole graph).
    assert b._legacy_execute({}) == []


@pytest.mark.concept("ORCH-1.39")
def test_id_lookup_still_precise(monkeypatch):
    monkeypatch.delenv("KG_ALLOW_FULL_SCAN", raising=False)
    b = _backend()
    rows = b._legacy_execute({"id": "ctx:s1:a"})
    assert len(rows) == 1 and rows[0]["id"] == "ctx:s1:a"


@pytest.mark.concept("ORCH-1.39")
def test_explicit_opt_in_allows_full_scan(monkeypatch):
    monkeypatch.setenv("KG_ALLOW_FULL_SCAN", "true")
    b = _backend()
    assert len(b._legacy_execute({})) == 2  # opt-in returns all
