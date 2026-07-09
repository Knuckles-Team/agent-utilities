"""Tests for the deterministic ZERO-LLM connector-manifest generator (C5).

Builds a throwaway connector layout on disk (ontology ttl + mcp_source_presets.json +
a2a.json) and proves the generator:

  * projects those artifacts into a valid :class:`ConnectorManifest`,
  * is **deterministic** — same input → byte-identical manifest incl. a stable
    integrity hash (with a pinned timestamp + pinned signing secret),
  * detects the ontology-source slug from the ttl's ``owl:Ontology`` IRI,
  * leaves the crosswalk + policy residue as ``review_todos`` (no LLM guessing).
"""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "generate_connector_manifests",
    Path(__file__).resolve().parents[3] / "scripts" / "generate_connector_manifests.py",
)
gen = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(gen)

_ONTOLOGY = """\
@prefix : <http://knuckles.team/kg#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

<http://knuckles.team/kg/acme> a owl:Ontology ;
    rdfs:label "Acme" ;
    owl:imports <http://knuckles.team/kg> .

:Order a owl:Class ; rdfs:label "Order" .
:Person a owl:Class ; rdfs:label "Person" .
:placedBy a owl:ObjectProperty ; rdfs:label "placed by" ;
    rdfs:domain :Order ; rdfs:range :Person .
:email a owl:DatatypeProperty ; rdfs:label "email" ; rdfs:range xsd:string .
:total a owl:DatatypeProperty ; rdfs:label "total" ; rdfs:range xsd:decimal .
"""

_PRESETS = """\
{
  "_comment": "test",
  "acme-orders": {
    "server": "acme-api", "tool": "acme_orders", "action": "list",
    "records_path": "data", "id_field": "id", "title_field": "name",
    "text_field": "notes", "updated_field": "updated_at",
    "pagination": "page", "doc_type": "order"
  }
}
"""

_A2A = """\
{"name": "acme-agent", "capabilities": [
  {"id": "sync_orders", "name": "Sync Orders", "description": "Pull orders"}
]}
"""


@pytest.fixture
def connector_root(tmp_path: Path) -> Path:
    root = tmp_path / "acme-api"
    mod = root / "acme_api"
    (mod / "ontology").mkdir(parents=True)
    (mod / "connectors").mkdir(parents=True)
    (mod / "ontology" / "acme.ttl").write_text(_ONTOLOGY)
    (mod / "connectors" / "mcp_source_presets.json").write_text(_PRESETS)
    (root / "a2a.json").write_text(_A2A)
    return root


_NOW = datetime(2026, 7, 9, tzinfo=UTC)


def test_build_manifest_projects_all_artifacts(connector_root: Path):
    m = gen.build_manifest(connector_root, now=_NOW)
    assert m.connector == "acme-api"
    assert m.ontology_source == "acme"  # detected from the ttl's owl:Ontology IRI
    assert {r.name for r in m.resources} == {"Order", "Person"}
    order = next(r for r in m.resources if r.name == "Order")
    assert order.relations[0].name == "placedBy"
    assert order.relations[0].target == "Person"
    # datatype fields flowed into schema_mappings
    assert m.schema_mappings["Order"].fields["total"] == "xsd:decimal"
    # a2a capability → action
    assert m.actions[0].id == "sync_orders"
    # preset → sync + identity + watermark event
    assert m.sync[0].preset == "acme-orders"
    assert m.identity.id_field["order"] == "id"
    assert any(e.name == "acme-orders.updated" for e in m.events)
    # signed with the generator signer id
    assert m.provenance.signer == "ontology-manifest-generator"
    assert len(m.provenance.integrity.hash) == 64


def test_build_manifest_is_deterministic(connector_root: Path, monkeypatch):
    monkeypatch.setenv("AGENT_UTILITIES_TOKEN_SECRET", "pinned-secret")
    m1 = gen.build_manifest(connector_root, now=_NOW)
    m2 = gen.build_manifest(connector_root, now=_NOW)
    assert gen._to_yaml(m1) == gen._to_yaml(
        m2
    )  # byte-identical, incl. hash + signature
    assert m1.provenance.integrity.hash == m2.provenance.integrity.hash
    assert m1.provenance.signature == m2.provenance.signature


def test_pii_and_crosswalk_left_as_review_todos(connector_root: Path):
    m = gen.build_manifest(connector_root, now=_NOW)
    # PII heuristic flagged the :email field, and flagged it for review (never auto-enforced).
    assert "email" in m.policy.pii_fields.get(
        "Order", []
    ) or "email" in m.policy.pii_fields.get("Person", [])
    todo_text = "\n".join(m.review_todos)
    assert "ontology_class" in todo_text  # crosswalk residue
    assert "rls" in todo_text or "row-level-security" in todo_text  # policy residue


def test_generator_offline_no_network_symbols():
    """Guard the ZERO-LLM / offline contract: the generator source imports no HTTP/LLM client."""
    src = (Path(gen.__file__)).read_text()
    for banned in (
        "requests",
        "httpx",
        "openai",
        "anthropic",
        "urllib.request",
        "aiohttp",
    ):
        assert banned not in src, f"generator must be offline; found {banned!r}"
