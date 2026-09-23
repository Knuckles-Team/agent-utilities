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


class _AttachRefused(_GraphSchema):
    def graph_schema_attach(self, source_id, **payload):
        raise RuntimeError("GraphSchema composition rejected")


def test_lifecycle_attach_failure_propagates_without_an_active_restatement() -> None:
    """EH-367 (D-OBC-2): an EG attach failure is raised, never folded into
    an ``active``/``loaded_to_engine`` status restated from the request."""
    lifecycle = OntologyLifecycle(_AttachRefused())
    with pytest.raises(RuntimeError, match="composition rejected"):
        lifecycle.load(
            "<urn:a> <urn:p> <urn:b> .", source_type="text", iri="urn:x", version="1"
        )
    with pytest.raises(OntologyError, match="inactive"):
        lifecycle.load(
            "<urn:a> <urn:p> <urn:b> .",
            source_type="text",
            iri="urn:x",
            version="1",
            activate=False,
        )

    receipt = OntologyLifecycle(_GraphSchema()).load(
        "<urn:a> <urn:p> <urn:b> .", source_type="text", iri="urn:x", version="1"
    )
    assert "active" not in receipt


def test_lifecycle_untyped_engine_receipt_fails_loudly() -> None:
    class _Untyped(_GraphSchema):
        def graph_schema_attach(self, source_id, **payload):
            return {"changed": True}

    with pytest.raises(OntologyError, match="untyped result"):
        OntologyLifecycle(_Untyped()).load(
            "<urn:a> <urn:p> <urn:b> .", source_type="text", iri="urn:x", version="1"
        )


@pytest.mark.parametrize(
    "kwargs",
    [{"category": "domain"}, {"tags": ["x"]}],
)
def test_lifecycle_refuses_arguments_it_cannot_honour(kwargs) -> None:
    lifecycle = OntologyLifecycle(_GraphSchema())
    with pytest.raises(OntologyError, match="local-registry metadata"):
        lifecycle.load(
            "<urn:a> <urn:p> <urn:b> .",
            source_type="text",
            iri="urn:x",
            version="1",
            **kwargs,
        )


def test_lifecycle_refuses_a_graph_or_tenant_it_cannot_scope_to() -> None:
    with pytest.raises(OntologyError, match="cannot scope"):
        OntologyLifecycle(_GraphSchema(), graph_name="tenant:other")
    with pytest.raises(OntologyError, match="verified graph session"):
        OntologyLifecycle(_GraphSchema(), tenant="tenant:other")

    scoped = []

    class _Scoping(_GraphSchema):
        def for_graph(self, graph_name):
            scoped.append(graph_name)
            return self

    OntologyLifecycle(_Scoping(), graph_name="tenant:mine")
    assert scoped == ["tenant:mine"]
