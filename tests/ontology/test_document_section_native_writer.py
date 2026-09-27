"""Native document sections carry only verified, EG-compatible tenant scope."""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.core.session import (
    GraphSession,
    SessionRequiredError,
    suspend_session,
    use_session,
)
from agent_utilities.knowledge_graph.ingestion import envelope_ingest
from agent_utilities.knowledge_graph.ontology.document_processing import (
    DocumentProcessor,
    SectionTreeConfig,
    _carrier_tenant_scope,
)
from agent_utilities.security.actor_identity import ActorType
from agent_utilities.security.brain_context import ActorContext


def _session(tenant: str) -> GraphSession:
    return GraphSession(
        actor=ActorContext(
            actor_id="document-writer",
            actor_type=ActorType.AUTOMATED_SERVICE,
            tenant_id=tenant,
            authenticated=True,
        ),
        tenant=tenant,
        scopes=frozenset({"kg:read", "kg:write"}),
        graph="documents",
    )


def test_opaque_scope_matches_eg_carrier_digest_contract() -> None:
    expected = (
        "carrier-tenant:"
        "1f853bfd417851fcffe7a7cba25ba7f37cf151f1a4f28c29a80f9ccf1bd24bf4"
    )
    assert _carrier_tenant_scope("tenant-a") == expected
    assert _carrier_tenant_scope(expected) == expected


def test_native_section_tree_is_one_tenant_stamped_envelope(monkeypatch) -> None:
    envelopes = []

    def capture(_engine, envelope):
        envelopes.append(envelope)
        return {"status": "success"}

    monkeypatch.setattr(envelope_ingest, "ingest_envelope", capture)
    processor = DocumentProcessor(engine=object(), embed_fn=lambda texts: [None] * len(texts))
    with use_session(_session("tenant-a")):
        result = processor.process(
            "# Guide\n\n## Installation\n\nInstall it.\n",
            document_id="doc:guide",
            section_tree=SectionTreeConfig(thin=False),
            persist=True,
            metadata={"tenant_scope": "attacker-chosen"},
        )

    assert result.persisted
    assert len(envelopes) == 1
    payload = envelopes[0].typed_payload
    assert payload is not None
    sections = [
        row for row in payload["_nodes"] if row.get("type") == "DocumentSection"
    ]
    assert len(sections) == len(result.section_nodes)
    assert all(row["node_type"] == "DocumentSection" for row in sections)
    assert all(
        row["tenant_scope"] == _carrier_tenant_scope("tenant-a") for row in sections
    )
    roots = [row for row in sections if row["parent_id"] is None]
    assert len(roots) == 1
    assert any(row["parent_id"] == roots[0]["id"] for row in sections)
    assert all(row["document_id"] == "doc:guide" for row in sections)
    assert all(row["external_access"] for row in sections)
    assert any(edge["relationship"] == "HAS_SUBSECTION" for edge in payload["_links"])
    assert all(
        row["section_tree_version"] == payload["section_tree_version"]
        for row in sections
    )
    assert envelopes[0].source_version.endswith(payload["section_tree_version"])

    # A later rewrite with fewer sections gets a new active-tree marker. The
    # served reader must compare against it before legacy rows are backfilled.
    with use_session(_session("tenant-a")):
        processor.process(
            "# Guide\n\nBody only.\n",
            document_id="doc:guide",
            section_tree=SectionTreeConfig(thin=False),
            persist=True,
        )
    assert envelopes[1].typed_payload is not None
    assert (
        envelopes[1].typed_payload["section_tree_version"]
        != payload["section_tree_version"]
    )
    assert envelopes[1].idempotency_key != envelopes[0].idempotency_key


def test_native_section_writer_refuses_missing_verified_session(monkeypatch) -> None:
    monkeypatch.setattr(
        envelope_ingest,
        "ingest_envelope",
        lambda *_args: pytest.fail("native ingest reached without verified session"),
    )
    processor = DocumentProcessor(engine=object(), embed_fn=lambda texts: [None] * len(texts))
    with suspend_session(), pytest.raises(SessionRequiredError):
        processor.process(
            "# Guide\n\nBody.\n",
            document_id="doc:guide",
            section_tree=True,
            persist=True,
        )
