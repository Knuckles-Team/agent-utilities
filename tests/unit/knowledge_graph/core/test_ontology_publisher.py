"""AU-BOUNDARY-R030.7: the typed declaration publisher uses pack compilation."""

from __future__ import annotations

import pytest
import rdflib

from agent_utilities.knowledge_graph.core.ontology_publisher import (
    OntologyDeclarationError,
    publish_declaration,
)


def _declaration() -> dict:
    return {
        "connector": "fixture-mcp",
        "resources": [{"name": "Widget", "label": "Widget"}],
        "provenance": {"integrity": {"hash": "0" * 64}},
    }


@pytest.mark.spec("AU-BOUNDARY-R030.7")
def test_publishes_fixture_declaration_as_turtle() -> None:
    ttl = publish_declaration(_declaration())
    graph = rdflib.Graph()
    graph.parse(data=ttl, format="turtle")
    assert len(graph) > 0
    assert "Widget" in ttl


@pytest.mark.spec("AU-BOUNDARY-R030.7")
def test_refuses_invalid_declaration() -> None:
    bad = _declaration()
    bad["unexpected"] = True
    with pytest.raises(OntologyDeclarationError):
        publish_declaration(bad)
    with pytest.raises(OntologyDeclarationError):
        publish_declaration({"resources": []})
