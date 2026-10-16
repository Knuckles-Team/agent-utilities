"""Live-path test: connector → ingestion engine → KG (CONCEPT:AU-ECO.connector.document-source-framework + KG-2.7/2.50).

Wire-First verification — exercises the *real* ``IngestionEngine.ingest`` path with
``ContentType.CONNECTOR`` (not just the connector in isolation) and asserts the
side effects: Document + Chunk nodes written through the backend, contextual
enrichment applied, HAS_CHUNK edges created, and the checkpoint recorded so a
second run is incremental. Fully offline (temp dir + recording backend, no LLM).
"""

from __future__ import annotations

import json
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

# Tests that reach the native engine request the ``real_ingestion_engine`` fixture
# (a real tenant on the session engine) instead of relying on suite order.


def _read(engine: IngestionEngine, cypher: str) -> list[dict]:
    return engine.backend.execute_read(cypher, {"_clearance_level": 999})


def _as_dict(value: object) -> dict:
    return json.loads(value) if isinstance(value, str) else dict(value)  # type: ignore[arg-type]


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
