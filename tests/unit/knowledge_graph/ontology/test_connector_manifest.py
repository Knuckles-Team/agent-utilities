"""Tests for the Connector Ontology Manifest schema + compiler + integrity (C5 + X6).

Covers (CONCEPT:AU-KG.ontology.connector-manifest-schema / -compiler / supply-chain-integrity):

  * schema round-trips (pydantic validate/dump),
  * the canonical hash comes from EG's ``OntologyInspect`` (its serialization-order
    invariance is pinned in EG; AU parses no RDF — EH-471),
  * Ed25519 release signing + fail-closed verification,
  * the **golden-file LeanIX regression** — the generalized compiler reproduces the
    existing ``leanix_metamodel`` OWL output losslessly (LeanIX = first caller),
  * the ``ontology.lock`` reader/writer.
"""

from __future__ import annotations

import base64
import secrets

import pytest

from agent_utilities.knowledge_graph.ontology import ontology_integrity as oi
from agent_utilities.knowledge_graph.ontology.connector_manifest import (
    ConnectorManifest,
    IntegrityInfo,
    ProvenanceSpec,
    ResourceRelation,
    ResourceSpec,
    SchemaMapping,
)
from agent_utilities.knowledge_graph.ontology.leanix_metamodel import (
    compile_leanix_metamodel,
    export_leanix_ttl,
)
from agent_utilities.knowledge_graph.ontology.manifest_compiler import (
    compile_manifest,
    export_manifest_ttl,
    manifest_from_leanix_spec,
)

pytestmark = pytest.mark.usefixtures("stub_canonical_ttl_hash")

# A LeanIX slice reused by the golden test (mirrors test_leanix_metamodel.META_MODEL).
META_MODEL = {
    "factSheets": {
        "Application": {
            "fields": {
                "displayName": {"type": "STRING"},
                "businessCriticality": {"type": "SINGLE_SELECT"},
            },
            "relations": {
                "relApplicationToITComponent": {"targetFactSheetType": "ITComponent"},
            },
        },
        "ITComponent": {"fields": {"release": {"type": "STRING"}}, "relations": {}},
        "DataCenter": {"fields": {"region": {"type": "STRING"}}, "relations": {}},
    }
}


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
    spec = compile_manifest(base)
    ttl = export_manifest_ttl(spec, source=base.resolved_ontology_source)
    digest, n = oi.canonical_ttl_hash(ttl)
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


# ── compiler ───────────────────────────────────────────────────────────────────


def test_compile_manifest_projects_classes_relations_fields():
    base = ConnectorManifest(
        connector="acme",
        resources=[
            ResourceSpec(
                name="Order",
                relations=[ResourceRelation(name="placedBy", target="Person")],
            ),
            ResourceSpec(name="Person"),
        ],
        schema_mappings={
            "Order": SchemaMapping(
                ontology_class="BusinessObject", fields={"total": "xsd:decimal"}
            ),
        },
        provenance=ProvenanceSpec(integrity=IntegrityInfo(hash="0" * 64)),
    )
    spec = compile_manifest(base)
    classes = {c.local: c for c in spec.classes}
    assert set(classes) == {"Order", "Person"}
    assert classes["Order"].parent == "BusinessObject"
    op = {p.local: p for p in spec.object_properties}
    assert op["placedBy"].domain == "Order"
    assert op["placedBy"].range == "Person"
    assert op["placedBy"].lpg_rel_type == "PLACED_BY"
    dtp = {d.local: d for d in spec.datatype_properties}
    assert dtp["total"].range == "xsd:decimal"


# ── golden-file LeanIX regression (LeanIX = first caller of the generalized compiler) ──


def _structure(view) -> tuple:
    """Classes + subClassOf, object-property domain/range and datatype-property
    ranges of an EG ``OntologyInspect`` view (labels/comments are cosmetic)."""
    return (
        sorted((c.iri, tuple(c.parents)) for c in view.classes),
        sorted(
            (p.iri, tuple(p.domains), tuple(p.ranges)) for p in view.object_properties
        ),
        sorted((p.iri, tuple(p.ranges)) for p in view.datatype_properties),
    )


def test_generalized_compiler_reproduces_leanix_ontology_losslessly(engine_graph):
    """The generalized manifest compiler reproduces the OWL graph the existing
    ``leanix_metamodel`` produces — same classes, subClassOf, object-property
    domain/range, and datatype-property ranges (labels/comments are cosmetic).
    EG reads both documents; AU parses no RDF (EH-471)."""
    lx_spec = compile_leanix_metamodel(META_MODEL)
    golden = engine_graph.ontology_inspect([export_leanix_ttl(lx_spec)])

    manifest = manifest_from_leanix_spec(lx_spec)
    generated = engine_graph.ontology_inspect(
        [export_manifest_ttl(compile_manifest(manifest), source="leanix")]
    )
    assert golden.classes, "the golden LeanIX ontology declares no classes"
    assert _structure(golden) == _structure(generated), (
        "generalized compiler diverged from leanix_metamodel output"
    )


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
