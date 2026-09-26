"""AU polled documents use EG's accepted checkpoint through the SDK seam."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from agent_connector_sdk.manifest.model import (
    ConnectorManifest,
    IntegrityInfo,
    PermissionsSpec,
    ProvenanceSpec,
    SchemaMapping,
)
from agent_connector_sdk.testing.sinks import InMemorySink

from agent_utilities.knowledge_graph.ingestion.engine import (
    ContentType,
    IngestionEngine,
    IngestionManifest,
    IngestionResult,
)
from agent_utilities.protocols.source_connectors.base import (
    ExternalAccess,
    PollConnector,
    SourceDocument,
)
from agent_utilities.protocols.source_connectors.checkpoint import (
    CheckpointedBatch,
    ConnectorCheckpoint,
)


class _TwoPageConnector(PollConnector):
    source_type = "document-agent"

    def __init__(self, *, public: bool = True, marked: bool = False) -> None:
        self.seen: list[ConnectorCheckpoint | None] = []
        self.public = public
        self.marked = marked
        super().__init__()

    def poll(self, checkpoint: ConnectorCheckpoint | None = None) -> CheckpointedBatch:
        self.seen.append(checkpoint)
        if checkpoint is None:
            return CheckpointedBatch(
                documents=[
                    SourceDocument(
                        id="one",
                        text="First",
                        title="One",
                        external_access=(
                            ExternalAccess(
                                is_public=True,
                                markings=["restricted"] if self.marked else [],
                            )
                            if self.public
                            else None
                        ),
                    )
                ],
                checkpoint=ConnectorCheckpoint(
                    has_more=True, cursor="next", watermark="version-1"
                ),
            )
        assert checkpoint.cursor == "next"
        return CheckpointedBatch(
            documents=[
                SourceDocument(
                    id="two",
                    text="Second",
                    title="Two",
                    external_access=ExternalAccess.public(),
                )
            ],
            checkpoint=ConnectorCheckpoint(has_more=False, watermark="version-2"),
        )


def _manifest() -> IngestionManifest:
    declared = ConnectorManifest(
        connector="document-agent",
        schema_mappings={
            "Document": SchemaMapping(
                ontology_class="Document",
                fields={
                    name: name
                    for name in (
                        "text",
                        "title",
                        "doc_type",
                        "metadata",
                        "external_access",
                    )
                },
            )
        },
        permissions=PermissionsSpec(acl_fields=["external_access"]),
        provenance=ProvenanceSpec(integrity=IntegrityInfo(hash="a" * 64)),
    )
    return IngestionManifest(
        content_type=ContentType.CONNECTOR,
        source_uri="document-agent",
        metadata={
            "connector_manifest": declared,
            "document_mapping_key": "Document",
            "provider_contract_sha256": "a" * 64,
            "provider_server": "document-agent",
            "provider_tool": "poll-documents",
            "max_pages": 2,
        },
    )


@pytest.mark.asyncio
async def test_declared_connector_commits_each_provider_page() -> None:
    connector = _TwoPageConnector()
    sink = InMemorySink()
    client = SimpleNamespace(
        changes=object(), supports=lambda name: name == "SourceIngest"
    )
    compute = SimpleNamespace(client=client)
    compute.for_graph = lambda _: compute
    engine = object.__new__(IngestionEngine)
    engine.kg = compute
    session = SimpleNamespace(graph="main", tenant="tenant-1")

    with (
        patch(
            "agent_utilities.knowledge_graph.core.session.current_session",
            return_value=session,
        ),
        patch(
            "agent_utilities.knowledge_graph.core.session.resolve_session",
            return_value=session,
        ),
        patch(
            "agent_connector_sdk.ingest.transport.EpistemicGraphIngestTransport",
            return_value=sink,
        ) as transport,
    ):
        result = await engine._ingest_declared_document_source(
            _manifest(), connector, "document-agent", "instance-1"
        )

    assert result.status == "success"
    assert result.details["pages"] == result.details["documents"] == 2
    assert len(connector.seen) == 2
    assert connector.seen[0] is None
    assert connector.seen[1] is not None and connector.seen[1].cursor == "next"
    assert transport.call_args.args == (client,)
    status = await sink.source_status("document-agent", "instance-1")
    assert status.accepted_checkpoint is not None
    assert status.accepted_checkpoint.position["watermark"] == "version-2"


@pytest.mark.asyncio
async def test_declared_connector_requires_verified_session_before_poll() -> None:
    connector = _TwoPageConnector()
    engine = object.__new__(IngestionEngine)
    engine.kg = SimpleNamespace(client=SimpleNamespace(changes=object()))
    with patch(
        "agent_utilities.knowledge_graph.core.session.current_session",
        return_value=None,
    ):
        result = await engine._ingest_declared_document_source(
            _manifest(), connector, "document-agent", "instance-1"
        )
    assert result.status == "failed"
    assert result.error == "native source ingest failed (SourceContractError)"
    assert connector.seen == []


@pytest.mark.asyncio
@pytest.mark.parametrize(("public", "marked"), [(False, False), (True, True)])
async def test_declared_connector_holds_restricted_page_before_checkpoint(
    public: bool, marked: bool
) -> None:
    connector = _TwoPageConnector(public=public, marked=marked)
    sink = InMemorySink()
    client = SimpleNamespace(
        changes=object(), supports=lambda name: name == "SourceIngest"
    )
    compute = SimpleNamespace(client=client)
    compute.for_graph = lambda _: compute
    engine = object.__new__(IngestionEngine)
    engine.kg = compute
    session = SimpleNamespace(graph="main", tenant="tenant-1")
    with (
        patch(
            "agent_utilities.knowledge_graph.core.session.current_session",
            return_value=session,
        ),
        patch(
            "agent_utilities.knowledge_graph.core.session.resolve_session",
            return_value=session,
        ),
        patch(
            "agent_connector_sdk.ingest.transport.EpistemicGraphIngestTransport",
            return_value=sink,
        ),
    ):
        result = await engine._ingest_declared_document_source(
            _manifest(), connector, "document-agent", "instance-1"
        )
    assert result.status == "failed"
    assert result.error == "native source ingest failed (SourceContractError)"
    status = await sink.source_status("document-agent", "instance-1")
    assert status.accepted_checkpoint is None


@pytest.mark.asyncio
async def test_connector_adaptor_routes_declared_poll_to_sdk_path() -> None:
    connector = _TwoPageConnector()
    engine = object.__new__(IngestionEngine)
    result = IngestionResult(manifest=_manifest(), status="success")
    with (
        patch(
            "agent_utilities.protocols.source_connectors.build_connector",
            return_value=connector,
        ),
        patch.object(engine, "_connector_dry_run", return_value=None),
        patch.object(
            engine,
            "_ingest_declared_document_source",
            new_callable=AsyncMock,
            return_value=result,
        ) as native,
    ):
        actual = await engine._ingest_connector(_manifest())
    assert actual is result
    native.assert_awaited_once()
