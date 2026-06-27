"""OntologyLifecycle.delete → physical engine retract (CONCEPT:KG-2.266).

KG-2.265 unloaded an ontology by dropping the registry record only; KG-2.266 wires it
to the engine's ``remove_triples`` retract op so the axioms physically leave the
engine RDF dataset. These tests attach a fake engine whose ``graph_compute`` records
``remove_triples`` calls.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.ontology.lifecycle import (
    OntologyLifecycle,
    reset_registry,
)

pytestmark = pytest.mark.concept("KG-2.266")

PETS_TTL = """
@prefix : <http://example.org/pets#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
<http://example.org/pets> a owl:Ontology .
:Dog a owl:Class .
:Cat a owl:Class .
:Dog rdfs:subClassOf :Animal .
"""


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_registry()
    yield
    reset_registry()


class _FakeGraphCompute:
    def __init__(self):
        self.added = []
        self.removed = []

    def add_triples(self, turtle=None, ntriples=None):
        self.added.append(turtle)
        return {"triples": 3}

    def remove_triples(self, turtle=None, ntriples=None):
        self.removed.append(turtle)
        return {"removed": 3}


class _FakeEngine:
    def __init__(self):
        self.graph_compute = _FakeGraphCompute()


def test_delete_retracts_axioms_from_engine():
    engine = _FakeEngine()
    lc = OntologyLifecycle(engine=engine)
    lc.load(PETS_TTL, source_type="text")

    res = lc.delete("http://example.org/pets")
    assert res["status"] == "ok"
    assert res["axioms_retracted_from_engine"] is True
    # the engine actually received a remove_triples call with the stored turtle
    assert engine.graph_compute.removed, "remove_triples was never called"
    assert "Dog" in engine.graph_compute.removed[0]
    assert res["retractions"][0]["retracted_from_engine"] is True
    assert "retracted" in res["engine_note"]


def test_delete_reports_gap_when_retract_unavailable():
    class _NoRetract:
        graph_compute = type("GC", (), {"add_triples": lambda self, **k: {}})()

    lc = OntologyLifecycle(engine=_NoRetract())
    lc.load(PETS_TTL, source_type="text")
    res = lc.delete("http://example.org/pets")
    assert res["status"] == "ok"
    assert res["axioms_retracted_from_engine"] is False
    assert "no engine retract surface" in res["engine_note"]


def test_delete_no_engine_is_registry_only():
    lc = OntologyLifecycle(engine=None)
    lc.load(PETS_TTL, source_type="text")
    res = lc.delete("http://example.org/pets")
    assert res["axioms_retracted_from_engine"] is False
    assert res["engine_note"] == "no engine attached"
