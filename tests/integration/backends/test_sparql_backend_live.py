"""Live SPARQL parity against a throwaway Apache Jena Fuseki.

CONCEPT:KG-2.7 — Optional SPARQL/RDF capability. The LPG backends correctly
*refuse* SPARQL (asserted in the conformance suite); this asserts the positive
side: the Fuseki backend actually executes SPARQL 1.1 update + query end-to-end
against a real server, so the enterprise profile's RDF/ontology-publish path is
trustworthy.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.knowledge_graph.backends import create_backend

pytestmark = [pytest.mark.integration, pytest.mark.live]


@pytest.fixture()
def fuseki_backend(ephemeral_fuseki: dict[str, Any]) -> Any:
    backend = create_backend(
        backend_type="jena_fuseki",
        jena_fuseki_url=ephemeral_fuseki["url"],
        dataset=ephemeral_fuseki["dataset"],
    )
    if backend is None:
        pytest.skip("jena_fuseki backend unavailable (install agent-utilities[fuseki])")
    try:
        yield backend
    finally:
        backend.close()


def test_sparql_update_then_query_roundtrip(fuseki_backend: Any) -> None:
    assert fuseki_backend.supports_sparql is True

    fuseki_backend.execute_sparql_update(
        "INSERT DATA { "
        "<urn:agent:router> <urn:prop:name> 'Expert Router' . "
        "<urn:agent:router> <urn:prop:score> '0.98' . "
        "}"
    )

    rows = fuseki_backend.execute_sparql_query(
        "SELECT ?name WHERE { <urn:agent:router> <urn:prop:name> ?name }"
    )
    names = [
        (
            r.get("name", {}).get("value")
            if isinstance(r.get("name"), dict)
            else r.get("name")
        )
        for r in rows
    ]
    assert "Expert Router" in names
