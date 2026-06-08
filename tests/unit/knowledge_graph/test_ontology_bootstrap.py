#!/usr/bin/python
"""Tests for the self-bootstrapping ontology agent (b7-05).

CONCEPT:KG-2.2
"""

import pytest

from agent_utilities.knowledge_graph.core.ontology_bootstrap import (
    BootstrapResult,
    OntologyBootstrapper,
)

pytestmark = pytest.mark.concept("KG-2.2")


def test_bootstrap_derives_classes_and_typed_properties():
    samples = [
        {
            "id": "p1",
            "type": "Product",
            "price": 9.99,
            "name": "Widget",
            "in_stock": True,
        },
        {
            "id": "p2",
            "type": "Product",
            "price": 19.0,
            "name": "Gadget",
            "in_stock": False,
        },
    ]
    b = OntologyBootstrapper()
    res = b.bootstrap(samples)
    assert isinstance(res, BootstrapResult)
    classes = {c.name for c in res.classes}
    assert classes == {"Product"}
    props = {p.name: p.datatype for p in res.properties}
    assert props["price"] == "xsd:decimal"
    assert props["in_stock"] == "xsd:boolean"
    assert props["name"] == "xsd:string"


def test_plateau_stops_early():
    # First record introduces all elements; the rest add nothing → plateau.
    base = {"id": "x", "type": "Doc", "title": "t"}
    samples = [dict(base, id=f"d{i}") for i in range(20)]
    b = OntologyBootstrapper(plateau_patience=3)
    res = b.bootstrap(samples)
    assert res.plateaued is True
    assert res.samples_seen < 20  # stopped before exhausting


def test_to_turtle_emits_class_and_property_triples():
    b = OntologyBootstrapper()
    b.observe({"id": "p1", "type": "Product", "price": 9.99})
    ttl = b.to_turtle()
    assert ":Product a rdfs:Class ." in ttl
    assert (
        ":price a rdf:Property ; rdfs:domain :Product ; rdfs:range xsd:decimal ." in ttl
    )


def test_grounded_triples_only_explicit_values():
    b = OntologyBootstrapper()
    rec = {
        "id": "p1",
        "type": "Product",
        "price": "9.99 USD",
        "note": "",
        "missing": None,
    }
    triples = b.grounded_triples(rec)
    # rdf:type + price only; empty/None dropped (anti-hallucination)
    preds = {p for _, p, _ in triples}
    assert "rdf:type" in preds and "price" in preds
    assert "note" not in preds and "missing" not in preds
    # unit-normalised value
    price = next(o for s, p, o in triples if p == "price")
    assert price == "9.99"


def test_grounded_triples_requires_subject():
    b = OntologyBootstrapper()
    assert (
        b.grounded_triples({"type": "Product", "price": 1}) == []
    )  # no id → no triples
