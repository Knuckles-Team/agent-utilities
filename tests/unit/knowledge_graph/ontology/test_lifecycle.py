"""Unit tests for hosted-ontology lifecycle CRUD (CONCEPT:AU-KG.ontology.manage-arbitrary).

Exercises the engine-free service core: load → list → get → update → delete,
plus validate rejecting a malformed ontology. No live engine required — these
cover parsing, validation, counting, versioning, and registry bookkeeping.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.ontology.lifecycle import (
    OntologyLifecycle,
    reset_registry,
)

PETS_TTL = """@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex: <http://example.org/pets#> .
<http://example.org/pets> a owl:Ontology .
ex:Animal a owl:Class .
ex:Dog a owl:Class ; rdfs:subClassOf ex:Animal .
ex:Cat a owl:Class ; rdfs:subClassOf ex:Animal .
ex:hasOwner a owl:ObjectProperty .
ex:name a owl:DatatypeProperty .
"""

PETS_TTL_V2 = """@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex: <http://example.org/pets#> .
<http://example.org/pets> a owl:Ontology .
ex:Animal a owl:Class .
ex:Dog a owl:Class ; rdfs:subClassOf ex:Animal .
ex:Cat a owl:Class ; rdfs:subClassOf ex:Animal .
ex:Bird a owl:Class ; rdfs:subClassOf ex:Animal .
ex:hasOwner a owl:ObjectProperty .
"""


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_registry()
    yield
    reset_registry()


@pytest.fixture
def lc():
    return OntologyLifecycle(engine=None)


def test_load_lists_get_returns_classes(lc):
    # load
    result = lc.load(PETS_TTL, source_type="text")
    assert result["status"] == "ok"
    assert result["idempotent"] is False
    onto = result["ontology"]
    assert onto["iri"] == "http://example.org/pets"
    assert onto["n_classes"] == 3
    assert onto["n_properties"] == 2
    assert onto["active"] is True
    # a record minus turtle is returned to callers
    assert "turtle" not in onto

    # list shows it
    listed = lc.list_ontologies()
    assert listed["count"] == 1
    assert listed["ontologies"][0]["iri"] == "http://example.org/pets"

    # get returns its classes/properties
    got = lc.get("http://example.org/pets")["ontology"]
    assert "http://example.org/pets#Dog" in got["classes"]
    assert "http://example.org/pets#hasOwner" in got["properties"]

    # get with serialize=True returns turtle
    serialized = lc.get("http://example.org/pets", serialize=True)["ontology"]
    assert "ex:Dog" in serialized["turtle"] or "pets#Dog" in serialized["turtle"]


def test_load_is_idempotent_on_iri_version(lc):
    first = lc.load(PETS_TTL, source_type="text")
    second = lc.load(PETS_TTL, source_type="text")
    assert first["idempotent"] is False
    assert second["idempotent"] is True
    assert lc.list_ontologies()["count"] == 1


def test_load_from_file_path(lc, tmp_path):
    p = tmp_path / "pets.ttl"
    p.write_text(PETS_TTL)
    result = lc.load(str(p), source_type="file")
    assert result["status"] == "ok"
    assert result["ontology"]["n_classes"] == 3


def test_update_supersedes_prior_version(lc):
    lc.load(PETS_TTL, source_type="text", version="1.0.0")
    upd = lc.update(
        PETS_TTL_V2, iri="http://example.org/pets", version="2.0.0", source_type="text"
    )
    assert upd["status"] == "ok"
    assert upd["superseded_prior"] is True
    assert upd["ontology"]["n_classes"] == 4

    # both versions are retained (bi-temporal history); only v2 active
    listed = lc.list_ontologies()
    assert listed["count"] == 2
    active = lc.list_ontologies(active_only=True)
    assert active["count"] == 1
    assert active["ontologies"][0]["version"] == "2.0.0"

    # get with no version targets the newest
    got = lc.get("http://example.org/pets")["ontology"]
    assert got["version"] == "2.0.0"


def test_delete_removes_from_hosted_set(lc):
    lc.load(PETS_TTL, source_type="text")
    res = lc.delete("http://example.org/pets")
    assert res["status"] == "ok"
    assert res["removed"][0]["iri"] == "http://example.org/pets"
    # engine gap is reported honestly (no engine attached here)
    assert res["axioms_retracted_from_engine"] is False
    assert lc.list_ontologies()["count"] == 0

    # deleting again is a clean not-found
    assert "error" in lc.delete("http://example.org/pets")


def test_activate_deactivate_toggles_reasoning_flag(lc):
    lc.load(PETS_TTL, source_type="text")
    off = lc.set_active("http://example.org/pets", active=False)
    assert off["ontology"]["active"] is False
    assert lc.list_ontologies(active_only=True)["count"] == 0
    on = lc.set_active("http://example.org/pets", active=True)
    assert on["ontology"]["active"] is True


def test_validate_rejects_malformed_without_hosting(lc):
    bad = lc.validate("this is @@ not <<< valid turtle", source_type="text")
    assert bad["valid"] is False
    assert bad["errors"]
    # nothing was hosted
    assert lc.list_ontologies()["count"] == 0


def test_validate_accepts_wellformed_without_hosting(lc):
    ok = lc.validate(PETS_TTL, source_type="text")
    assert ok["valid"] is True
    assert ok["summary"]["n_classes"] == 3
    # validate does not commit
    assert lc.list_ontologies()["count"] == 0


def test_load_rejects_empty_ontology(lc):
    empty = "@prefix ex: <http://example.org/x#> .\n"
    result = lc.load(empty, source_type="text")
    assert result["status"] == "rejected"
    assert result["valid"] is False
