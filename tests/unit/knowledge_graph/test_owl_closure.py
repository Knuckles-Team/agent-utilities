#!/usr/bin/python
"""Unit tests for the background OWL-RL + SHACL closure job (CONCEPT:KG-2.6).

Covers:
  * The full reason → downfeed → SHACL path. Reasoning is engine-native first
    (``client.rdf.owl_reason``, CONCEPT:KG-2.242) with a pure-Python RDFS+ closure
    last-resort, so the closure needs NO owlready2 backend (``owl_backend=None``);
    over a plain networkx graph the Python transitive closure runs. Asserts implied
    edges are materialized back into the graph and the summary shape is correct.
  * The graceful-degradation path (no engine graph) returns a structured no-op
    summary and never raises.
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


def test_run_closure_materializes_inferred_edges():
    g = _transitive_graph()
    summary = owl_closure.run_closure(_Engine(g), limit=2000)

    assert summary["status"] == "completed"
    # CONCEPT:KG-2.242 — engine-native closure needs no owlready2 promotion, so
    # `promoted` is 0; over a plain networkx graph the Python RDFS+ transitive
    # closure runs and materializes the implied edge back into the graph.
    assert summary["promoted"] == 0
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


def test_run_closure_runs_without_owlready2(monkeypatch):
    """CONCEPT:KG-2.242 — the closure reasons engine-native (or via the Python
    last-resort) and must NOT require owlready2: even if create_owl_backend would
    raise, the closure never calls it and still completes."""
    import agent_utilities.knowledge_graph.backends.owl as owlmod

    def _boom(*a, **k):
        raise ImportError("owlready2 not installed")

    # Patched to explode — proving the closure path never constructs an owl backend.
    monkeypatch.setattr(owlmod, "create_owl_backend", _boom)
    g = _transitive_graph()
    summary = owl_closure.run_closure(_Engine(g))
    assert summary["status"] == "completed"
    # Python RDFS+ closure still materializes the transitive edge.
    assert g.has_edge("symbol:A", "symbol:C")


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
