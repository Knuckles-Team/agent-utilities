from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_utilities.knowledge_graph.ontology.lifecycle import (
    OntologyError,
    OntologyLifecycle,
)


class _Typed:
    def __init__(self, payload):
        self.payload = payload

    def model_dump(self, *, mode: str):
        assert mode == "json"
        return self.payload


class _GraphSchema:
    def __init__(self):
        self.attached = []
        self.detached = []

    def graph_schema_attach(self, source_id, **payload):
        self.attached.append((source_id, payload))
        return _Typed(
            {
                "schema_version": 2,
                "graph": "default",
                "composed_digest": "digest:2",
                "graph_version": 4,
                "changed": True,
            }
        )

    def graph_schema_detach(self, source_id):
        self.detached.append(source_id)
        return _Typed(
            {
                "schema_version": 3,
                "graph": "default",
                "composed_digest": "digest:3",
                "graph_version": 5,
                "changed": True,
            }
        )

    def graph_schema_list(self):
        source_id = self.attached[-1][0]
        return _Typed(
            {
                "schema_version": 2,
                "graph": "default",
                "core_catalog_digest": "core",
                "composed_digest": "digest:2",
                "core_sources": [],
                "dynamic_sources": [
                    {"source_id": source_id, "origin": {"origin": "admin"}}
                ],
            }
        )


def test_lifecycle_delegates_exact_body_to_graphschema() -> None:
    schema = _GraphSchema()
    lifecycle = OntologyLifecycle(SimpleNamespace(graph_compute=schema))

    receipt = lifecycle.load(
        "@prefix : <urn:test:> . :A a <http://www.w3.org/2002/07/owl#Class> .",
        source_type="text",
        iri="urn:test:ontology",
        version="1.0.0",
    )

    source_id, payload = schema.attached[0]
    assert source_id.startswith("admin:ontology:")
    assert payload["ontology_ttl"].startswith("@prefix")
    assert payload.get("shapes_ttl") is None
    assert receipt["composed_digest"] == "digest:2"


def test_lifecycle_lists_metadata_without_exporting_bodies() -> None:
    schema = _GraphSchema()
    lifecycle = OntologyLifecycle(schema)
    lifecycle.load(
        "<urn:a> <urn:p> <urn:b> .", source_type="text", iri="urn:x", version="1"
    )

    listing = lifecycle.list_ontologies()
    assert listing["count"] == 1
    assert "ontology_ttl" not in listing["ontologies"][0]
    with pytest.raises(OntologyError, match="metadata-only"):
        lifecycle.get("urn:x", version="1", serialize=True)


def test_lifecycle_detaches_exact_version_and_has_no_offline_fallback() -> None:
    schema = _GraphSchema()
    lifecycle = OntologyLifecycle(schema)
    lifecycle.delete("urn:x", version="1")
    assert schema.detached[0].startswith("admin:ontology:")

    with pytest.raises(OntologyError, match="GraphSchema authority"):
        OntologyLifecycle().list_ontologies()


def test_lifecycle_refuses_legacy_local_semantic_modes() -> None:
    lifecycle = OntologyLifecycle(_GraphSchema())
    assert not hasattr(lifecycle, "validate")
    with pytest.raises(OntologyError, match="explicit iri and version"):
        lifecycle.load("<urn:a> <urn:p> <urn:b> .", source_type="text")
    with pytest.raises(OntologyError, match="fetch remote content"):
        lifecycle.load(
            "https://example.invalid/ontology.ttl",
            source_type="url",
            iri="urn:x",
            version="1",
        )
