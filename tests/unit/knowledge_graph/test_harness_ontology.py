"""Harness-foundry ontology wiring (CONCEPT:KG-2.107).

The harness, its edits, isolated variants, the 9-dim taxonomy, and RL pathologies
are modelled as OWL so HarnessX's "operational mirror" (an admitted heuristic)
becomes formal reasoning + SHACL gating.
"""

from __future__ import annotations

from pathlib import Path

import rdflib

from agent_utilities.knowledge_graph.core.owl_bridge import (
    HARNESS_INVERSE_EDGES,
    PROMOTABLE_EDGE_TYPES,
    PROMOTABLE_NODE_TYPES,
)

_TTL = (
    Path(__file__).resolve().parents[3]
    / "agent_utilities"
    / "knowledge_graph"
    / "ontology_harness.ttl"
)
_KG = rdflib.Namespace("http://knuckles.team/kg#")


def test_harness_classes_in_ontology():
    g = rdflib.Graph()
    g.parse(str(_TTL), format="turtle")
    for cls in (
        "Harness",
        "Processor",
        "HarnessDimension",
        "HarnessEdit",
        "HarnessVariant",
        "HarnessPathology",
    ):
        assert (_KG[cls], rdflib.RDF.type, rdflib.OWL.Class) in g, cls
    # Inverse + scope properties declared.
    assert (_KG["mitigatesPathology"], rdflib.OWL.inverseOf, _KG["mitigatedBy"]) in g
    assert (_KG["variantOf"], rdflib.OWL.inverseOf, _KG["hasVariant"]) in g


def test_harness_types_promotable():
    for n in (
        "harness",
        "processor",
        "harness_dimension",
        "harness_edit",
        "harness_variant",
        "harness_pathology",
    ):
        assert n in PROMOTABLE_NODE_TYPES, n
    for e in (
        "targets_dimension",
        "has_variant",
        "variant_of",
        "applies_edit",
        "exhibits_pathology",
        "mitigates_pathology",
        "mitigated_by",
        "causes_regression",
        "confirms_fix",
    ):
        assert e in PROMOTABLE_EDGE_TYPES, e


def test_harness_inverses_declared():
    # Reasoning materialises variant↔base and edit↔pathology both ways.
    assert HARNESS_INVERSE_EDGES == {
        "has_variant": "variant_of",
        "mitigates_pathology": "mitigated_by",
    }
