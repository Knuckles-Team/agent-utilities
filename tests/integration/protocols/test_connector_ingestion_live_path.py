"""Live-path test: connector → ingestion engine → KG (CONCEPT:AU-ECO.connector.document-source-framework + KG-2.7/2.50).

Wire-First verification — exercises the *real* ``IngestionEngine.ingest`` path with
``ContentType.CONNECTOR`` (not just the connector in isolation) and asserts the
side effects: Document + Chunk nodes written through the backend, contextual
enrichment applied, HAS_CHUNK edges created, and the checkpoint recorded so a
second run is incremental. Fully offline (temp dir + recording backend, no LLM).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_utilities.knowledge_graph.ingestion.change_envelope import ChangeEnvelope
from agent_utilities.knowledge_graph.ingestion.engine import (
    ContentType,
    IngestionEngine,
    IngestionManifest,
)
from agent_utilities.protocols.source_connectors import (
    CheckpointedBatch,
    ConnectorCheckpoint,
    ExternalAccess,
    PollConnector,
    SourceDocument,
)


class _RecordingBackend:
    """A duck-typed graph backend that records nodes/edges."""

    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.edges: list[tuple] = []

    def add_node(self, node_id, **props):
        self.nodes[node_id] = props

    def add_edge(self, source, target, rel_type=None, **props):
        self.edges.append((source, target, rel_type))


class _GovernedPollConnector(PollConnector):
    provider = "Synthetic"

    def configure(self, **config):
        self.dry_run = bool(config.get("dry_run", False))
        self.last_envelopes: list[ChangeEnvelope] = []
        self.plan_calls = 0

    def health_check(self) -> bool:
        return True

    def plan(self, _checkpoint=None) -> dict:
        self.plan_calls += 1
        return {
            "status": "planned",
            "dry_run": True,
            "counts": {"entities": 1, "documents": 1},
            "profile_digest": "synthetic-digest",
        }

    def poll(self, checkpoint=None) -> CheckpointedBatch:
        governed_id = "doc:synthetic-governed:opaque"
        access = ExternalAccess.quarantined()
        self.last_envelopes = [
            ChangeEnvelope(
                connector="synthetic_governed",
                source_instance="synthetic-source",
                source_object_id=governed_id,
                source_version="synthetic-version",
                typed_payload={"id": governed_id, "type": "Document"},
                source_acl=access,
                retention="P30D",
                provenance={"privacy_gate": True},
                checkpoint="synthetic-version",
            )
        ]
        return CheckpointedBatch(
            documents=[
                SourceDocument(
                    id="opaque-document-id",
                    source_uri="external-source://synthetic/opaque-document-id",
                    title="Synthetic governed document",
                    text="Synthetic governed content for embedding.",
                    metadata={"governed_entity_id": governed_id},
                    external_access=access,
                    updated_at="synthetic-version",
                )
            ],
            checkpoint=ConnectorCheckpoint(
                has_more=False,
                watermark="synthetic-watermark",
                state={"versions": {governed_id: "synthetic-version"}},
            ),
        )


class _SyntheticDocumentProcessor:
    def __init__(self, *_args, **_kwargs):
        pass

    def process(self, _text, *, document_id, **_kwargs):
        return SimpleNamespace(
            document_id=document_id,
            chunk_count=0,
            edges=[],
            access_synced=True,
        )


@pytest.mark.integration
@pytest.mark.concept("AU-ECO.connector.document-source-framework")
@pytest.mark.asyncio
async def test_filesystem_connector_ingestion_live_path(tmp_path):
    (tmp_path / "a.md").write_text(
        "# Title A\nalpha content about graphs and ontologies. " * 6
    )
    (tmp_path / "b.txt").write_text("beta content discussing retrieval. " * 6)

    backend = _RecordingBackend()
    engine = IngestionEngine(kg_engine=None, backend=backend)
    manifest = IngestionManifest(
        content_type=ContentType.CONNECTOR,
        source_uri="filesystem",
        metadata={
            "connector_config": {"root": str(tmp_path)},
            "connector_id": "fs-live-test",
            "contextual": True,
        },
    )

    result = await engine.ingest(manifest)
    assert result.status == "success"
    assert result.details["documents"] == 2

    docs = [n for n in backend.nodes.values() if n.get("type") == "Document"]
    chunks = [n for n in backend.nodes.values() if n.get("type") == "Chunk"]
    assert len(docs) == 2
    assert len(chunks) >= 2
    # KG-2.50 contextual enrichment ran on the connector path.
    assert all(c.get("context") for c in chunks)
    # HAS_CHUNK / CHUNK_OF edges materialized.
    rels = {e[2] for e in backend.edges}
    assert "HAS_CHUNK" in rels and "CHUNK_OF" in rels

    # Second run is incremental (checkpoint advanced → nothing new).
    result2 = await engine.ingest(manifest)
    assert result2.status == "success"
    assert result2.details["documents"] == 0


@pytest.mark.integration
@pytest.mark.concept("AU-ECO.connector.external-permission-sync")
@pytest.mark.asyncio
async def test_connector_ingestion_syncs_external_acl(tmp_path):
    (tmp_path / "secret.md").write_text("restricted content " * 8)

    backend = _RecordingBackend()
    engine = IngestionEngine(kg_engine=None, backend=backend)
    manifest = IngestionManifest(
        content_type=ContentType.CONNECTOR,
        source_uri="filesystem",
        metadata={
            "connector_config": {"root": str(tmp_path), "public": False},
            "connector_id": "fs-acl-test",
        },
    )
    result = await engine.ingest(manifest)
    assert result.status == "success"
    assert result.details["acl_synced"] == 1

    docs = {
        node_id: node
        for node_id, node in backend.nodes.items()
        if node.get("type") == "Document"
    }
    chunks = {
        node_id: node
        for node_id, node in backend.nodes.items()
        if node.get("type") == "Chunk"
    }
    assert len(docs) == 1
    assert chunks
    doc_id, doc = next(iter(docs.items()))
    access = doc["external_access"]
    assert access["is_public"] is False
    assert all(chunk["external_access"] == access for chunk in chunks.values())

    from agent_utilities.knowledge_graph.ontology.permissioning import get_company_brain

    acl = get_company_brain().permissions.get_acl(doc_id)
    assert acl is not None
    assert all(
        get_company_brain().permissions.get_acl(chunk_id) is not None
        for chunk_id in chunks
    )
    # Persisted nodes contain only an abstract connector URI, never the local root.
    assert str(tmp_path) not in repr(backend.nodes)


@pytest.mark.integration
@pytest.mark.concept("AU-KG.ingest.change-envelope")
@pytest.mark.asyncio
async def test_governed_connector_applies_envelope_before_embedding(
    monkeypatch,
):
    connector = _GovernedPollConnector()
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ontology.document_processing.DocumentProcessor",
        _SyntheticDocumentProcessor,
    )
    monkeypatch.setattr(
        "agent_utilities.protocols.source_connectors.build_connector",
        lambda _source_type, _config: connector,
    )
    applied: list[ChangeEnvelope] = []

    def _ingest(_engine, envelope, *, backend=None):
        del backend
        applied.append(envelope)
        return {"status": "success"}

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.envelope_ingest.ingest_envelope",
        _ingest,
    )
    backend = _RecordingBackend()
    engine = IngestionEngine(kg_engine=None, backend=backend)
    manifest = IngestionManifest(
        content_type=ContentType.CONNECTOR,
        source_uri="synthetic_governed",
        metadata={
            "connector_id": "synthetic-governed-live",
            "contextual": False,
        },
    )

    result = await engine.ingest(manifest)

    assert result.status == "success"
    assert result.details["envelopes"] == 1
    assert result.details["envelopes_failed"] == 0
    assert result.details["documents"] == 1
    assert result.details["checkpoint_advanced"] is True
    assert len(applied) == 1
    assert applied[0].provenance["privacy_gate"] is True


@pytest.mark.integration
@pytest.mark.concept("AU-KG.ingest.change-envelope")
@pytest.mark.asyncio
async def test_governed_connector_failure_blocks_document_and_checkpoint(
    monkeypatch,
):
    connector = _GovernedPollConnector()
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ontology.document_processing.DocumentProcessor",
        _SyntheticDocumentProcessor,
    )
    monkeypatch.setattr(
        "agent_utilities.protocols.source_connectors.build_connector",
        lambda _source_type, _config: connector,
    )
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.envelope_ingest.ingest_envelope",
        lambda _engine, _envelope, *, backend=None: {"status": "failed"},
    )
    backend = _RecordingBackend()
    engine = IngestionEngine(kg_engine=None, backend=backend)
    manifest = IngestionManifest(
        content_type=ContentType.CONNECTOR,
        source_uri="synthetic_governed",
        metadata={
            "connector_id": "synthetic-governed-failure",
            "contextual": False,
        },
    )

    result = await engine.ingest(manifest)

    assert result.status == "partial"
    assert result.details["envelopes_failed"] == 1
    assert result.details["documents"] == 0
    assert result.details["documents_failed"] == 1
    assert result.details["checkpoint_advanced"] is False


@pytest.mark.integration
@pytest.mark.concept("AU-KG.ingest.change-envelope")
@pytest.mark.asyncio
async def test_governed_connector_dry_run_never_reaches_persistence(monkeypatch):
    connector = _GovernedPollConnector(dry_run=True)
    monkeypatch.setattr(
        "agent_utilities.protocols.source_connectors.build_connector",
        lambda _source_type, _config: connector,
    )

    def _unexpected(*_args, **_kwargs):
        raise AssertionError("dry-run reached persistence")

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.envelope_ingest.ingest_envelope",
        _unexpected,
    )
    engine = IngestionEngine(kg_engine=None, backend=_RecordingBackend())
    manifest = IngestionManifest(
        content_type=ContentType.CONNECTOR,
        source_uri="synthetic_governed",
        metadata={
            "connector_id": "synthetic-governed-plan",
            "connector_config": {"dry_run": True},
        },
    )

    result = await engine.ingest(manifest)

    assert result.status == "success"
    assert result.details["dry_run"] is True
    assert result.details["checkpoint_advanced"] is False
    assert result.details["plan"]["counts"] == {"entities": 1, "documents": 1}
    assert connector.plan_calls == 1
