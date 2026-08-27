"""Tests for CA-23's ontology-pack pipeline (CONCEPT:AU-KG.ontology.pack-pipeline).

Covers: the full load->compile->(shacl)->apply->publish->verify composition, the
fail-closed apply_manifest path (hash tamper -> rejection, nothing written/published),
the fail-closed SHACL path when an instance sample is supplied, idempotent re-publish,
and that ``apply_manifest``'s existing hash/signature re-verification is exercised (not
bypassed) by this pipeline -- CA-23's acceptance gate #5.
"""

from __future__ import annotations

import base64
import secrets
from unittest.mock import MagicMock

import pytest

from agent_utilities.knowledge_graph.backends.sparql.jena_fuseki_backend import (
    JenaFusekiBackend,
)
from agent_utilities.knowledge_graph.ontology import ontology_integrity as oi
from agent_utilities.knowledge_graph.ontology.connector_manifest import (
    ConnectorManifest,
    IntegrityInfo,
    ProvenanceSpec,
    ResourceRelation,
    ResourceSpec,
    SchemaMapping,
)
from agent_utilities.knowledge_graph.ontology.manifest_compiler import (
    compile_manifest,
    export_manifest_ttl,
)
from agent_utilities.knowledge_graph.ontology.pack_pipeline import (
    PackRejected,
    count_graph_triples,
    run_pack,
)


@pytest.fixture(autouse=True)
def release_signing_key(monkeypatch):
    private_key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("=")
    monkeypatch.setenv("ONTOLOGY_RELEASE_SIGNING_TEST_MATERIAL", private_key)
    monkeypatch.setenv(
        "ONTOLOGY_RELEASE_SIGNING_PRIVATE_KEY_REF",
        "env://ONTOLOGY_RELEASE_SIGNING_TEST_MATERIAL",
    )


def _signed_manifest(connector: str = "servicenow") -> ConnectorManifest:
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
    import rdflib

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


def _mock_fuseki(count_response: int = 0) -> MagicMock:
    """A JenaFusekiBackend-shaped mock: upload_graph is a no-op recorder,
    execute_sparql_query answers a fixed COUNT(*) row."""
    fuseki = MagicMock(spec=JenaFusekiBackend)
    fuseki.execute_sparql_query.return_value = [{"c": str(count_response)}]
    return fuseki


# ── happy path: apply -> publish -> verify ──────────────────────────────────────


def test_run_pack_applies_publishes_and_verifies_equal_counts(tmp_path, monkeypatch):
    from agent_utilities.knowledge_graph.core import owl_bridge

    monkeypatch.setattr(owl_bridge, "DYNAMIC_PROMOTABLE_NODE_TYPES", set())
    m = _signed_manifest()
    target = tmp_path / "ontology_servicenow.ttl"
    fuseki = _mock_fuseki(count_response=m.provenance.integrity.triple_count)

    result = run_pack(
        m,
        ttl_path=target,
        fuseki=fuseki,
        trusted_public_keys=(str(m.provenance.signing_public_key),),
    )

    assert result.rejected is False
    assert target.exists()
    assert result.published is True
    fuseki.upload_graph.assert_called_once()
    assert result.fuseki_graph_uri == "http://knuckles.team/kg/servicenow"
    assert result.counts_equal is True
    assert result.local_triple_count == m.provenance.integrity.triple_count
    assert result.r2rml_turtle is not None
    assert result.r2rml_triples_map_count == 2


def test_run_pack_without_fuseki_skips_publish(tmp_path, monkeypatch):
    from agent_utilities.knowledge_graph.core import owl_bridge

    monkeypatch.setattr(owl_bridge, "DYNAMIC_PROMOTABLE_NODE_TYPES", set())
    m = _signed_manifest()
    target = tmp_path / "ontology_servicenow.ttl"

    result = run_pack(
        m,
        ttl_path=target,
        fuseki=None,
        trusted_public_keys=(str(m.provenance.signing_public_key),),
    )

    assert result.rejected is False
    assert result.published is False
    assert result.fuseki_triple_count is None
    assert result.counts_equal is None


# ── fail-closed: hash/signature re-verification is exercised, not bypassed ─────


def test_run_pack_rejects_tampered_hash_before_publish(tmp_path, monkeypatch):
    from agent_utilities.knowledge_graph.core import owl_bridge

    monkeypatch.setattr(owl_bridge, "DYNAMIC_PROMOTABLE_NODE_TYPES", set())
    m = _signed_manifest()
    tampered = m.model_copy(
        update={
            "provenance": m.provenance.model_copy(
                update={"integrity": IntegrityInfo(hash="f" * 64)}
            )
        }
    )
    target = tmp_path / "ontology_servicenow.ttl"
    fuseki = _mock_fuseki()

    result = run_pack(
        tampered,
        ttl_path=target,
        fuseki=fuseki,
        trusted_public_keys=(str(m.provenance.signing_public_key),),
    )

    assert result.rejected is True
    assert result.rejection_stage == "apply"
    assert (
        "hash" in result.rejection_reason.lower()
        or "SignatureVerificationError" in (result.rejection_reason or "")
    )
    assert not target.exists()
    fuseki.upload_graph.assert_not_called()
    # negative half of P5: COUNT(*) on the pack's graph is 0 -- nothing was published,
    # so a caller reading Fuseki afterwards must see 0, never a fabricated count.
    assert (
        count_graph_triples(_mock_fuseki(0), "http://knuckles.team/kg/servicenow") == 0
    )


def test_run_pack_rejects_bad_signature_before_publish(tmp_path, monkeypatch):
    from agent_utilities.knowledge_graph.core import owl_bridge

    monkeypatch.setattr(owl_bridge, "DYNAMIC_PROMOTABLE_NODE_TYPES", set())
    m = _signed_manifest()
    tampered = m.model_copy(
        update={"provenance": m.provenance.model_copy(update={"signature": "0" * 64})}
    )
    target = tmp_path / "ontology_servicenow.ttl"
    fuseki = _mock_fuseki()

    result = run_pack(
        tampered,
        ttl_path=target,
        fuseki=fuseki,
        trusted_public_keys=(str(m.provenance.signing_public_key),),
    )

    assert result.rejected is True
    assert result.rejection_stage == "apply"
    assert not target.exists()
    fuseki.upload_graph.assert_not_called()


def test_run_pack_raise_on_reject(tmp_path):
    m = _signed_manifest()
    tampered = m.model_copy(
        update={"provenance": m.provenance.model_copy(update={"signature": "0" * 64})}
    )
    target = tmp_path / "ontology_servicenow.ttl"
    with pytest.raises(PackRejected):
        run_pack(
            tampered,
            ttl_path=target,
            raise_on_reject=True,
            trusted_public_keys=(str(m.provenance.signing_public_key),),
        )


# ── fail-closed: SHACL gate, when an instance sample is supplied ───────────────


def test_run_pack_rejects_on_shacl_violation_before_apply(tmp_path):
    pytest.importorskip("pyshacl")
    from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

    m = _signed_manifest()
    target = tmp_path / "ontology_servicenow.ttl"
    fuseki = _mock_fuseki()

    sample = GraphComputeEngine()
    sample.add_node("bad_agent", {"node_type": "agent"})  # missing required :name

    result = run_pack(
        m,
        ttl_path=target,
        fuseki=fuseki,
        instance_sample=sample,
        trusted_public_keys=(str(m.provenance.signing_public_key),),
    )

    assert result.rejected is True
    assert result.rejection_stage == "shacl"
    assert "bad_agent" in result.shacl_violated_shapes
    assert not target.exists()
    fuseki.upload_graph.assert_not_called()


def test_run_pack_passes_with_conforming_instance_sample(tmp_path, monkeypatch):
    pytest.importorskip("pyshacl")
    from agent_utilities.knowledge_graph.core import owl_bridge
    from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

    monkeypatch.setattr(owl_bridge, "DYNAMIC_PROMOTABLE_NODE_TYPES", set())
    m = _signed_manifest()
    target = tmp_path / "ontology_servicenow.ttl"
    fuseki = _mock_fuseki(count_response=m.provenance.integrity.triple_count)

    sample = GraphComputeEngine()
    sample.add_node("good_agent", {"node_type": "agent", "name": "Planner"})

    result = run_pack(
        m,
        ttl_path=target,
        fuseki=fuseki,
        instance_sample=sample,
        trusted_public_keys=(str(m.provenance.signing_public_key),),
    )

    assert result.rejected is False
    assert result.published is True


# ── idempotent re-publish ────────────────────────────────────────────────────────


def test_republishing_identical_pack_is_a_safe_noop(tmp_path, monkeypatch):
    from agent_utilities.knowledge_graph.core import owl_bridge

    monkeypatch.setattr(owl_bridge, "DYNAMIC_PROMOTABLE_NODE_TYPES", set())
    m = _signed_manifest()
    target = tmp_path / "ontology_servicenow.ttl"
    fuseki = _mock_fuseki(count_response=m.provenance.integrity.triple_count)

    first = run_pack(
        m,
        ttl_path=target,
        fuseki=fuseki,
        trusted_public_keys=(str(m.provenance.signing_public_key),),
    )
    second = run_pack(
        m,
        ttl_path=target,
        fuseki=fuseki,
        trusted_public_keys=(str(m.provenance.signing_public_key),),
    )

    assert first.canonical_hash == second.canonical_hash
    assert first.counts_equal is True
    assert second.counts_equal is True
    assert fuseki.upload_graph.call_count == 2  # both calls succeed; no error/drift


# ── count_graph_triples ──────────────────────────────────────────────────────────


def test_count_graph_triples_parses_the_count_binding():
    fuseki = _mock_fuseki(count_response=42)
    assert count_graph_triples(fuseki, "http://example/graph") == 42


def test_count_graph_triples_returns_zero_for_absent_graph():
    fuseki = _mock_fuseki(count_response=0)
    assert count_graph_triples(fuseki, "http://example/nonexistent") == 0


def test_count_graph_triples_raises_on_query_error():
    fuseki = MagicMock(spec=JenaFusekiBackend)
    fuseki.execute_sparql_query.return_value = [{"error": "HTTP 500"}]
    with pytest.raises(RuntimeError):
        count_graph_triples(fuseki, "http://example/graph")
