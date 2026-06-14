#!/usr/bin/python
"""Unit tests for the background OWL-RL + SHACL closure job (CONCEPT:KG-2.6).

Covers:
  * The full promote → reason → downfeed → SHACL path with a fake OWL backend
    (so it runs without the heavy ``owlready2`` extra — only rdflib/pyshacl, which
    are core deps, are exercised). Asserts implied edges are materialized back
    into the graph and the summary shape is correct.
  * The graceful-degradation paths (no engine graph, OWL backend unavailable) —
    every one returns a structured no-op summary and never raises.
"""

from __future__ import annotations

import networkx as nx
import pytest

from agent_utilities.knowledge_graph.maintenance import owl_closure


class _FakeOWLBackend:
    """A no-op OWL backend that satisfies OWLBridge.run_cycle's contract.

    ``run_cycle(lightweight=True)`` does its inference in Python (RDFS+/OWL-RL over
    the networkx graph) and only uses the backend for promote/clear/close — so this
    fake is enough to exercise the real closure end-to-end without owlready2.
    """

    def clear(self) -> None:  # noqa: D401
        pass

    def promote(self, nodes):
        return len(nodes)

    def promote_edges(self, edges):
        return len(edges)

    def reason(self):
        return []

    def close(self) -> None:
        pass


class _Engine:
    def __init__(self, graph, backend=None):
        self.graph = graph
        self.backend = backend


@pytest.fixture
def patched_backend(monkeypatch):
    """Patch create_owl_backend at its import site (resolved lazily in run_closure)."""
    import agent_utilities.knowledge_graph.backends.owl as owlmod

    monkeypatch.setattr(owlmod, "create_owl_backend", lambda *a, **k: _FakeOWLBackend())
    return owlmod


def _transitive_graph() -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for n in ("symbol:A", "symbol:B", "symbol:C"):
        g.add_node(n, type="symbol", importance_score=0.9)
    # depends_on is a transitive property → A->B->C should imply A->C.
    g.add_edge("symbol:A", "symbol:B", type="depends_on")
    g.add_edge("symbol:B", "symbol:C", type="depends_on")
    return g


def test_run_closure_materializes_inferred_edges(patched_backend):
    g = _transitive_graph()
    summary = owl_closure.run_closure(_Engine(g), limit=2000)

    assert summary["status"] == "completed"
    # 3 nodes + 2 edges promoted.
    assert summary["promoted"] >= 5
    # The transitive closure A->C is materialized back into the graph.
    assert summary["inferred_edges"] >= 1
    assert g.has_edge("symbol:A", "symbol:C")
    inferred = [
        d
        for _, _, d in g.edges(data=True)
        if d.get("type") == "depends_on" and d.get("inferred")
    ]
    assert inferred, "expected an inferred=True depends_on edge"


def test_run_closure_summary_shape(patched_backend):
    summary = owl_closure.run_closure(_Engine(_transitive_graph()))

    # The documented contract: exactly these governance-summary keys are present.
    for key in ("promoted", "inferred_edges", "conforms", "violations"):
        assert key in summary
    assert isinstance(summary["promoted"], int)
    assert isinstance(summary["inferred_edges"], int)
    assert isinstance(summary["conforms"], bool)
    assert isinstance(summary["violations"], list)
    # Default ontology promotes valid symbol nodes → governance shapes conform.
    assert summary["conforms"] is True


def test_run_closure_runs_shacl_validation(patched_backend, monkeypatch):
    """The closure must invoke SHACL governance validation on the live path."""
    calls: list[object] = []

    from agent_utilities.knowledge_graph.core import shacl_validator

    real_validate = shacl_validator.SHACLValidator.validate

    def _spy(self, data_graph, shapes_path, ont_graph=None):
        calls.append(shapes_path)
        return real_validate(self, data_graph, shapes_path, ont_graph)

    monkeypatch.setattr(shacl_validator.SHACLValidator, "validate", _spy)

    summary = owl_closure.run_closure(_Engine(_transitive_graph()))
    assert summary["status"] == "completed"
    assert calls, "SHACL validation was never invoked"
    # Validated against the governance shapes file.
    assert str(calls[0]).endswith("governance.shapes.ttl")


def test_run_closure_no_engine_graph_is_noop():
    summary = owl_closure.run_closure(object())
    assert summary["status"] == "skipped"
    assert summary == {
        "promoted": 0,
        "inferred_edges": 0,
        "conforms": True,
        "violations": [],
        "status": "skipped",
        "reason": "engine has no graph",
    }


def test_run_closure_none_engine_is_noop():
    summary = owl_closure.run_closure(None)
    assert summary["status"] == "skipped"
    assert summary["promoted"] == 0
    assert summary["inferred_edges"] == 0


def test_run_closure_backend_unavailable_degrades(monkeypatch):
    """A missing owlready2 (backend construction raises) degrades to a clean skip."""
    import agent_utilities.knowledge_graph.backends.owl as owlmod

    def _boom(*a, **k):
        raise ImportError("owlready2 not installed")

    monkeypatch.setattr(owlmod, "create_owl_backend", _boom)
    summary = owl_closure.run_closure(_Engine(_transitive_graph()))
    assert summary["status"] == "skipped"
    assert summary["reason"] == "owl backend unavailable"
    assert summary["conforms"] is True
    assert summary["violations"] == []


def test_run_closure_reasoning_error_never_raises(patched_backend, monkeypatch):
    """A failure inside the reasoning cycle returns status=error, never raises."""
    from agent_utilities.knowledge_graph.core import owl_bridge

    def _boom(self, *a, **k):
        raise RuntimeError("reasoner exploded")

    monkeypatch.setattr(owl_bridge.OWLBridge, "run_cycle", _boom)
    summary = owl_closure.run_closure(_Engine(_transitive_graph()))
    assert summary["status"] == "error"
    assert "reasoner exploded" in summary["reason"]
    # Best-effort keys still present.
    for key in ("promoted", "inferred_edges", "conforms", "violations"):
        assert key in summary
