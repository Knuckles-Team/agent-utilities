"""Tests for the Connector Ontology Manifest schema + compiler + integrity (C5 + X6).

Covers (CONCEPT:AU-KG.ontology.connector-manifest-schema / -compiler / supply-chain-integrity):

  * schema round-trips (pydantic validate/dump),
  * canonical-hash **serialization-order invariance** (URDNA2015-equivalent),
  * HMAC sign/verify + fail-closed (unsigned / unknown-signer / tampered),
  * the compiler's **anti-sprawl refusal** and **fail-closed signature check** in
    ``apply_manifest``,
  * the **golden-file LeanIX regression** — the generalized compiler reproduces the
    existing ``leanix_metamodel`` OWL output losslessly (LeanIX = first caller),
  * the ``ontology.lock`` reader/writer.
"""

from __future__ import annotations

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
from agent_utilities.knowledge_graph.ontology.leanix_metamodel import (
    compile_leanix_metamodel,
    export_leanix_ttl,
)
from agent_utilities.knowledge_graph.ontology.manifest_compiler import (
    AntiSprawlError,
    SignatureVerificationError,
    apply_manifest,
    compile_manifest,
    export_manifest_ttl,
    manifest_from_leanix_spec,
)

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
    g = rdflib.Graph()
    g.parse(data=ttl, format="turtle")
    digest, n = oi.canonical_hash(g)
    prov = ProvenanceSpec(
        integrity=IntegrityInfo(hash=digest, triple_count=n),
        signer=oi.DEFAULT_SIGNER_ID,
        signature=oi.sign(digest, signer_id=oi.DEFAULT_SIGNER_ID),
    )
    return base.model_copy(update={"provenance": prov})


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


# ── sign / verify fail-closed (X6) ──────────────────────────────────────────────


def test_sign_verify_roundtrip_and_fail_closed():
    digest = "d" * 64
    sig = oi.sign(digest)
    assert oi.verify(digest, sig, signer_id=oi.DEFAULT_SIGNER_ID)
    # unsigned → False
    assert not oi.verify(digest, None, signer_id=oi.DEFAULT_SIGNER_ID)
    assert not oi.verify(digest, "", signer_id=oi.DEFAULT_SIGNER_ID)
    # unknown signer (not in allowlist) → False
    assert not oi.verify(digest, sig, signer_id="mallory")
    assert not oi.verify(
        digest, sig, signer_id=oi.DEFAULT_SIGNER_ID, allowlist=("someone-else",)
    )
    # tampered signature → False
    assert not oi.verify(
        digest,
        sig[:-1] + ("0" if sig[-1] != "0" else "1"),
        signer_id=oi.DEFAULT_SIGNER_ID,
    )
    # tampered hash (signature no longer matches) → False
    assert not oi.verify("e" * 64, sig, signer_id=oi.DEFAULT_SIGNER_ID)


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


def test_apply_manifest_writes_when_signed_and_wired(tmp_path, monkeypatch):
    from agent_utilities.knowledge_graph.core import owl_bridge

    monkeypatch.setattr(owl_bridge, "DYNAMIC_PROMOTABLE_NODE_TYPES", set())
    m = _signed_manifest()
    target = (
        tmp_path / "ontology_servicenow.ttl"
    )  # exists=False, but source is federated-wired
    result = apply_manifest(m, ttl_path=target, dry_run=False)
    assert target.exists()
    assert result["canonical_hash"] == m.provenance.integrity.hash
    assert {"incident", "configurationitem"} <= owl_bridge.DYNAMIC_PROMOTABLE_NODE_TYPES


def test_apply_manifest_fail_closed_on_bad_signature(tmp_path):
    m = _signed_manifest()
    tampered = m.model_copy(
        update={"provenance": m.provenance.model_copy(update={"signature": "0" * 64})}
    )
    target = tmp_path / "ontology_servicenow.ttl"
    try:
        apply_manifest(tampered, ttl_path=target, dry_run=False)
        raise AssertionError("expected SignatureVerificationError")
    except SignatureVerificationError:
        pass
    assert not target.exists()  # nothing emitted on failure


def test_apply_manifest_fail_closed_on_tampered_hash(tmp_path):
    m = _signed_manifest()
    tampered = m.model_copy(
        update={
            "provenance": m.provenance.model_copy(
                update={"integrity": IntegrityInfo(hash="f" * 64)}
            )
        }
    )
    target = tmp_path / "ontology_servicenow.ttl"
    try:
        apply_manifest(tampered, ttl_path=target, dry_run=False)
        raise AssertionError("expected SignatureVerificationError on hash mismatch")
    except SignatureVerificationError:
        pass
    assert not target.exists()


def test_apply_manifest_anti_sprawl_refusal(tmp_path):
    """A brand-new, un-imported source with no existing ttl must be refused loudly."""
    m = _signed_manifest(connector="totally-new-unwired-source")
    target = tmp_path / "ontology_totally-new-unwired-source.ttl"  # does not exist
    try:
        apply_manifest(m, ttl_path=target, dry_run=False)
        raise AssertionError("expected AntiSprawlError")
    except AntiSprawlError as exc:
        assert "owl:imports" in str(exc)
    assert not target.exists()


# ── golden-file LeanIX regression (LeanIX = first caller of the generalized compiler) ──


def test_generalized_compiler_reproduces_leanix_ontology_losslessly():
    """The generalized manifest compiler reproduces the OWL graph the existing
    ``leanix_metamodel`` produces — same classes, subClassOf, object-property
    domain/range, and datatype-property ranges (labels/comments are cosmetic)."""
    lx_spec = compile_leanix_metamodel(META_MODEL)
    golden_ttl = export_leanix_ttl(lx_spec)

    manifest = manifest_from_leanix_spec(lx_spec)
    gen_ttl = export_manifest_ttl(compile_manifest(manifest), source="leanix")

    def _structural(ttl: str) -> rdflib.Graph:
        g = rdflib.Graph()
        g.parse(data=ttl, format="turtle")
        ont_subjects = set(
            g.subjects(
                predicate=rdflib.RDF.type,
                object=rdflib.URIRef("http://www.w3.org/2002/07/owl#Ontology"),
            )
        )
        out = rdflib.Graph()
        for s, p, o in g:
            if s in ont_subjects or p in (rdflib.RDFS.comment, rdflib.RDFS.label):
                continue
            out.add((s, p, o))
        return out

    h_gold, n_gold = oi.canonical_hash(_structural(golden_ttl))
    h_gen, n_gen = oi.canonical_hash(_structural(gen_ttl))
    assert n_gold == n_gen and n_gold > 0
    assert h_gold == h_gen, "generalized compiler diverged from leanix_metamodel output"


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
