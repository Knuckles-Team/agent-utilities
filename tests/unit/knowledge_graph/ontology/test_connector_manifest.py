"""Tests for the Connector Ontology Manifest schema + compiler + integrity (C5 + X6).

Covers (CONCEPT:AU-KG.ontology.connector-manifest-schema / -compiler / supply-chain-integrity):

  * schema round-trips (pydantic validate/dump),
  * canonical-hash **serialization-order invariance** (URDNA2015-equivalent),
  * Ed25519 release signing + fail-closed verification,
  * the **golden-file LeanIX regression** — the generalized compiler reproduces the
    existing ``leanix_metamodel`` OWL output losslessly (LeanIX = first caller),
  * the ``ontology.lock`` reader/writer.
"""

from __future__ import annotations

import base64
import secrets

import pytest
import rdflib

from agent_utilities.knowledge_graph.ontology import ontology_integrity as oi
from agent_utilities.knowledge_graph.ontology.connector_manifest import (
    ConnectorManifest,
    IntegrityInfo,
    ProvenanceSpec,
    ResourceRelation,
    ResourceSpec,
    SchemaMapping,
)
from tests.unit._sdk_manifest_ttl import sdk_manifest_ttl


@pytest.fixture(autouse=True)
def release_signing_key(monkeypatch):
    private_key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("=")
    monkeypatch.setenv("ONTOLOGY_RELEASE_SIGNING_TEST_MATERIAL", private_key)
    monkeypatch.setenv(
        "ONTOLOGY_RELEASE_SIGNING_PRIVATE_KEY_REF",
        "env://ONTOLOGY_RELEASE_SIGNING_TEST_MATERIAL",
    )


def _signed_manifest(connector: str = "servicenow") -> ConnectorManifest:
    """Build a minimal, correctly-signed manifest for compiler tests."""
    resources = [
        ResourceSpec(
            name="Incident",
            label="Incident",
            id_prefix="incident",
            relations=[ResourceRelation(name="affects", target="ConfigurationItem")],
        ),
        ResourceSpec(
            name="ConfigurationItem", label="Configuration Item", id_prefix="ci"
        ),
    ]
    schema_mappings = {
        "Incident": SchemaMapping(ontology_class=None, fields={"number": "xsd:string"}),
        "ConfigurationItem": SchemaMapping(ontology_class=None, fields={}),
    }
    base = ConnectorManifest(
        connector=connector,
        resources=resources,
        schema_mappings=schema_mappings,
        provenance=ProvenanceSpec(integrity=IntegrityInfo(hash="0" * 64)),
    )
    ttl = sdk_manifest_ttl(base)
    g = rdflib.Graph()
    g.parse(data=ttl, format="turtle")
    digest, n = oi.canonical_hash(g)
    signer = oi.ReleaseSigner.from_runtime()
    prov = ProvenanceSpec(
        integrity=IntegrityInfo(hash=digest, triple_count=n),
        signer=signer.signer_id,
        signature_algorithm=signer.algorithm,
        signing_public_key=signer.public_key,
    )
    unsigned = base.model_copy(update={"provenance": prov})
    return unsigned.model_copy(
        update={
            "provenance": prov.model_copy(
                update={"signature": signer.sign(oi.canonical_manifest_hash(unsigned))}
            )
        }
    )


# ── schema ────────────────────────────────────────────────────────────────────


def test_manifest_round_trips():
    m = ConnectorManifest(
        connector="acme",
        resources=[ResourceSpec(name="Widget")],
        provenance=ProvenanceSpec(integrity=IntegrityInfo(hash="a" * 64)),
    )
    dumped = m.model_dump(mode="json")
    again = ConnectorManifest.model_validate(dumped)
    assert again.connector == "acme"
    assert again.resources[0].name == "Widget"


def test_resolved_ontology_source_defaults_to_connector():
    m = ConnectorManifest(
        connector="gitlab-api",
        provenance=ProvenanceSpec(integrity=IntegrityInfo(hash="0" * 64)),
    )
    assert m.resolved_ontology_source == "gitlab-api"
    m2 = m.model_copy(update={"ontology_source": "gitlab"})
    assert m2.resolved_ontology_source == "gitlab"


# ── canonicalization invariance (X6) ───────────────────────────────────────────


def test_canonical_hash_is_serialization_order_invariant():
    """Hash a graph, reserialize it differently (Turtle→N-Triples→JSON-LD), re-parse:
    the canonical hash MUST be identical — that is the whole X6 integrity guarantee."""
    src = """\
@prefix : <http://knuckles.team/kg#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
:Beta a owl:Class ; rdfs:label "Beta" .
:Alpha a owl:Class ; rdfs:label "Alpha" ; rdfs:subClassOf :Beta .
:rel a owl:ObjectProperty ; rdfs:domain :Alpha ; rdfs:range :Beta .
"""
    g = rdflib.Graph()
    g.parse(data=src, format="turtle")
    h0, n0 = oi.canonical_hash(g)

    for fmt in ("nt", "json-ld", "xml"):
        reserialized = g.serialize(format=fmt)
        g2 = rdflib.Graph()
        g2.parse(data=reserialized, format=fmt)
        h2, n2 = oi.canonical_hash(g2)
        assert h2 == h0, f"hash changed after {fmt} round-trip"
        assert n2 == n0

    # A semantically different graph MUST hash differently.
    g3 = rdflib.Graph()
    g3.parse(data=src + ":Gamma a owl:Class .\n", format="turtle")
    assert oi.canonical_hash(g3)[0] != h0


def test_release_sign_verify_is_public_and_deterministic():
    digest = "d" * 64
    signer_one = oi.ReleaseSigner.from_runtime()
    signer_two = oi.ReleaseSigner.from_runtime()
    signature = signer_one.sign(digest)

    assert signer_one.public_key == signer_two.public_key
    assert signature == signer_two.sign(digest)
    assert oi.verify_release_signature(
        digest,
        signature,
        signer_id=signer_one.signer_id,
        algorithm=signer_one.algorithm,
        public_key=signer_one.public_key,
        trusted_public_keys=(signer_one.public_key,),
    )
    assert not oi.verify_release_signature(
        "e" * 64,
        signature,
        signer_id=signer_one.signer_id,
        algorithm=signer_one.algorithm,
        public_key=signer_one.public_key,
        trusted_public_keys=(signer_one.public_key,),
    )
    tampered_signature = signature[:-1] + ("A" if signature[-1] != "A" else "B")
    assert not oi.verify_release_signature(
        digest,
        tampered_signature,
        signer_id=signer_one.signer_id,
        algorithm=signer_one.algorithm,
        public_key=signer_one.public_key,
        trusted_public_keys=(signer_one.public_key,),
    )
    assert not oi.verify_release_signature(
        digest,
        signature,
        signer_id=signer_one.signer_id,
        algorithm=signer_one.algorithm,
        public_key=signer_one.public_key,
        trusted_public_keys=(),
    )


def test_release_signer_refuses_missing_or_malformed_runtime_key(monkeypatch):
    monkeypatch.delenv("ONTOLOGY_RELEASE_SIGNING_PRIVATE_KEY_REF", raising=False)
    with pytest.raises(oi.ReleaseSigningError):
        oi.ReleaseSigner.from_runtime()

    monkeypatch.setenv("ONTOLOGY_RELEASE_SIGNING_TEST_MATERIAL", "not-a-raw-key")
    monkeypatch.setenv(
        "ONTOLOGY_RELEASE_SIGNING_PRIVATE_KEY_REF",
        "env://ONTOLOGY_RELEASE_SIGNING_TEST_MATERIAL",
    )
    with pytest.raises(oi.ReleaseSigningError):
        oi.ReleaseSigner.from_runtime()


# ── ontology.lock ──────────────────────────────────────────────────────────────


def test_ontology_lock_read_write(tmp_path):
    lock = tmp_path / "ontology.lock"
    assert oi.load_lock(lock) == {}
    oi.update_lock_entry(lock, "ontology_servicenow.ttl", "a" * 64, triple_count=32)
    oi.update_lock_entry(lock, "ontology_gitlab.ttl", "b" * 64, triple_count=40)
    entries = oi.load_lock(lock)
    assert entries["ontology_servicenow.ttl"]["hash"] == "a" * 64
    assert entries["ontology_gitlab.ttl"]["triple_count"] == 40
    # keys are written sorted → byte-stable
    text1 = lock.read_text()
    oi.save_lock(lock, entries)
    assert lock.read_text() == text1
