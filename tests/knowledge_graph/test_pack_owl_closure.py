from __future__ import annotations

"""Tests for pack-driven OWL closure (the "free value-add").

CONCEPT:KG-2.36 — Pack-Driven OWL Closure
"""


from unittest.mock import MagicMock

import networkx as nx
import pytest

from agent_utilities.knowledge_graph.core.owl_bridge import OWLBridge
from agent_utilities.models.schema_pack import OwlObjectProperty, SchemaPack
from agent_utilities.models.schema_packs import get_schema_pack


@pytest.fixture(autouse=True)
def _no_active_engine(monkeypatch):
    # The downfeed step re-embeds via the active engine; keep it None in tests.
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    monkeypatch.setattr(IntelligenceGraphEngine, "get_active", classmethod(lambda cls: None))


def _has_edge(g, u, v, rel):
    data = g.get_edge_data(u, v) or {}
    return any(e.get("type") == rel for e in data.values())


def _bridge(graph, pack):
    return OWLBridge(graph=graph, owl_backend=MagicMock(), backend=None, schema_pack=pack)


def _research_graph():
    g = nx.MultiDiGraph()
    for n in ("A", "B", "C", "P", "Q"):
        g.add_node(n, type="concept")
    g.add_edge("A", "B", type="supports_belief")
    g.add_edge("B", "C", type="supports_belief")
    g.add_edge("P", "Q", type="cites_source")
    return g


def test_transitive_supports_chain_inferred():
    g = _research_graph()
    bridge = _bridge(g, get_schema_pack("research-state"))
    bridge._downfeed_inferences(bridge._python_reasoning())
    assert _has_edge(g, "A", "C", "supports_belief")  # A→B→C ⇒ A→C


def test_inverse_citation_edge_inferred():
    g = _research_graph()
    bridge = _bridge(g, get_schema_pack("research-state"))
    bridge._downfeed_inferences(bridge._python_reasoning())
    assert _has_edge(g, "Q", "P", "cited_by_paper")  # P cites Q ⇒ Q cited_by P


def test_reasoning_is_a_fixpoint():
    g = _research_graph()
    bridge = _bridge(g, get_schema_pack("research-state"))
    bridge._downfeed_inferences(bridge._python_reasoning())
    # Second pass must add nothing new (idempotent closure).
    second = bridge._downfeed_inferences(bridge._python_reasoning())
    assert second == 0


def test_core_pack_makes_no_pack_inferences():
    g = _research_graph()
    bridge = _bridge(g, get_schema_pack("core"))
    bridge._downfeed_inferences(bridge._python_reasoning())
    assert not _has_edge(g, "A", "C", "supports_belief")
    assert not _has_edge(g, "Q", "P", "cited_by_paper")


def test_symmetric_pack_property():
    g = nx.MultiDiGraph()
    g.add_node("X", type="concept")
    g.add_node("Y", type="concept")
    g.add_edge("X", "Y", type="weakens")
    pack = SchemaPack(
        name="sym",
        owl_object_properties=[OwlObjectProperty(edge_type="weakens", symmetric=True)],
    )
    bridge = _bridge(g, pack)
    bridge._downfeed_inferences(bridge._python_reasoning())
    assert _has_edge(g, "Y", "X", "weakens")
