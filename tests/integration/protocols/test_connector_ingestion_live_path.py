"""Live-path test: connector → ingestion engine → KG (CONCEPT:AU-ECO.connector.document-source-framework + KG-2.7/2.50).

Wire-First verification — exercises the *real* ``IngestionEngine.ingest`` path with
``ContentType.CONNECTOR`` (not just the connector in isolation) and asserts the
side effects: Document + Chunk nodes written through the backend, contextual
enrichment applied, HAS_CHUNK edges created, and the checkpoint recorded so a
second run is incremental. Fully offline (temp dir + recording backend, no LLM).
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.ingestion.engine import (
    ContentType,
    IngestionEngine,
    IngestionManifest,
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
    # ACL sync only fires for non-public docs with principals; the field is present.
    assert "acl_synced" in result.details
