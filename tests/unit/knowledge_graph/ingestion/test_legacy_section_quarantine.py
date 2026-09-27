"""Legacy Section inventory never treats row payload as migration authority."""

from __future__ import annotations

import hashlib

import pytest

from agent_utilities.knowledge_graph.core.session import (
    GraphSession,
    SessionRequiredError,
    suspend_session,
    use_session,
)
from agent_utilities.knowledge_graph.ingestion import envelope_ingest
from agent_utilities.knowledge_graph.ingestion.legacy_section_quarantine import (
    AttestedSectionSource,
    replay_attested_legacy_sections,
    sample_legacy_section_quarantine,
)
from agent_utilities.knowledge_graph.ontology.document_processing import (
    DocumentProcessor,
)
from agent_utilities.protocols.source_connectors.base import ExternalAccess
from agent_utilities.security.actor_identity import ActorType
from agent_utilities.security.brain_context import ActorContext


def _session(*, write: bool = False) -> GraphSession:
    return GraphSession(
        actor=ActorContext(
            actor_id="migration-auditor",
            actor_type=ActorType.AUTOMATED_SERVICE,
            tenant_id="verified-tenant",
            authenticated=True,
        ),
        tenant="verified-tenant",
        scopes=frozenset({"kg:read", "kg:write"} if write else {"kg:read"}),
        graph="verified-graph",
    )


class _GovernedReader:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def query_cypher(self, query, params, *, session):
        self.calls.append((query, params, session))
        return self.rows

    def add_node(self, *_args, **_kwargs):
        pytest.fail("legacy inventory attempted a graph write")


def test_bounded_quarantine_report_does_not_adopt_row_tenant() -> None:
    reader = _GovernedReader(
        [
            {"id": "s-1", "document_id": "doc-a", "tenant_scope": "attacker"},
            {"id": "s-2", "document_id": "doc-a", "tenant_id": "other"},
            {"id": "s-3", "document_id": "doc-b"},
        ]
    )
    with use_session(_session()):
        report = sample_legacy_section_quarantine(reader, limit=2)
    assert report.document_ids == ("doc-a",)
    assert report.sampled_sections == 2
    assert report.possibly_more
    assert not report.complete
    assert report.state == "quarantined_requires_attested_source_replay"
    query, params, session = reader.calls[0]
    assert "MATCH (s:Section)" in query
    assert params == {"after_id": "", "limit": 3}
    assert session.tenant == "verified-tenant"
    assert "tenant" not in params


def test_missing_document_identity_is_reported_without_migration() -> None:
    reader = _GovernedReader([{"id": "s-1", "document_id": ""}])
    with use_session(_session()):
        report = sample_legacy_section_quarantine(reader, after_id="s-0")
    assert report.document_ids == ()
    assert report.unattributed_sections == 1
    assert report.last_seen_id == "s-1"


def test_no_verified_session_refuses_before_query() -> None:
    reader = _GovernedReader([{"id": "s-1", "document_id": "doc-a"}])
    with suspend_session(), pytest.raises(SessionRequiredError):
        sample_legacy_section_quarantine(reader)
    assert reader.calls == []


def test_malformed_authorized_page_fails_closed() -> None:
    reader = _GovernedReader([{"document_id": "doc-a"}])
    with use_session(_session()), pytest.raises(ValueError, match="section identity"):
        sample_legacy_section_quarantine(reader)


class _Resolver:
    def __init__(self, source):
        self.source = source
        self.calls = []

    def resolve(self, document_id, *, session):
        self.calls.append((document_id, session))
        return self.source


def _source(*, tenant: str = "verified-tenant", digest: str | None = None):
    text = "# Trusted Guide\n\n## Install\n\nTrusted instructions.\n"
    return AttestedSectionSource(
        document_id="doc-a",
        tenant=tenant,
        source_ref="source://doc-a",
        text=text,
        content_sha256=digest or hashlib.sha256(text.encode()).hexdigest(),
        external_access=ExternalAccess.public(),
        connector="verified-source",
    )


def test_replay_uses_attested_source_and_native_document_writer(monkeypatch) -> None:
    envelopes = []

    def capture(_engine, envelope):
        envelopes.append(envelope)
        return {"status": "success"}

    monkeypatch.setattr(envelope_ingest, "ingest_envelope", capture)
    processor = DocumentProcessor(engine=object(), embed_fn=lambda texts: [None] * len(texts))
    resolver = _Resolver(_source())
    with use_session(_session(write=True)):
        report = replay_attested_legacy_sections(
            ["doc-a", "doc-a"], resolver=resolver, processor=processor
        )

    assert report.replayed_document_ids == ("doc-a",)
    assert report.quarantined_document_ids == ()
    assert len(resolver.calls) == len(envelopes) == 1
    assert resolver.calls[0][1].tenant == "verified-tenant"
    envelope = envelopes[0]
    assert envelope.source_acl == ExternalAccess.public()
    assert envelope.tenant == "verified-tenant"
    assert envelope.typed_payload is not None
    assert any(
        row.get("type") == "DocumentSection"
        and row.get("document_id") == "doc-a"
        and row.get("section_tree_version")
        for row in envelope.typed_payload["_nodes"]
    )


@pytest.mark.parametrize("source", [None, _source(tenant="other"), _source(digest="bad")])
def test_unattested_or_mismatched_source_stays_quarantined(source, monkeypatch) -> None:
    monkeypatch.setattr(
        envelope_ingest,
        "ingest_envelope",
        lambda *_args: pytest.fail("unattested source reached native write"),
    )
    processor = DocumentProcessor(engine=object(), embed_fn=lambda texts: [None] * len(texts))
    with use_session(_session(write=True)):
        report = replay_attested_legacy_sections(
            ["doc-a"], resolver=_Resolver(source), processor=processor
        )
    assert report.replayed_document_ids == ()
    assert report.quarantined_document_ids == ("doc-a",)
