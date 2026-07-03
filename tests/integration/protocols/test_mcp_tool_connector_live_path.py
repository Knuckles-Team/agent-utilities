"""Live-path test: MCP-backed source → entry point → ingestion → KG (CONCEPT:KG-2.59).

Wire-First verification for the ``mcp_tool`` source: exercises the *existing*
entry-point seam — ``kg.ontology.run_connector`` (the facade both the
``source_connector`` MCP tool and the ``/connector/run`` REST route call) and
the ``ContentType.CONNECTOR`` ingestion adaptor — against an in-process FastMCP
server with canned sql-mcp / objectstore-mcp envelopes. Fully offline: the fake
server is handed to the connector via the injected ``client`` target (the same
in-memory FastMCP client pattern the fleet repos test with), no fleet package
is imported, and no child process is spawned.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastmcp import FastMCP

from agent_utilities.knowledge_graph.ingestion.engine import (
    ContentType,
    IngestionEngine,
    IngestionManifest,
)
from agent_utilities.knowledge_graph.ontology import build_ontology_system


class _RecordingBackend:
    """A duck-typed graph backend that records nodes/edges."""

    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.edges: list[tuple] = []

    def add_node(self, node_id, **props):
        self.nodes[node_id] = props

    def add_edge(self, source, target, rel_type=None, **props):
        self.edges.append((source, target, rel_type))


def make_sql_server() -> FastMCP:
    """A fake sql-mcp: keyset-paginated sql_query over a 3-row table."""
    table = [
        {
            "id": 1,
            "title": "Graphs",
            "body": "alpha content about graphs. " * 6,
            "updated_at": "2026-01-01",
        },
        {
            "id": 2,
            "title": "Ontologies",
            "body": "beta content on ontologies. " * 6,
            "updated_at": "2026-02-01",
        },
        {
            "id": 3,
            "title": "Retrieval",
            "body": "gamma content on retrieval. " * 6,
            "updated_at": "2026-03-01",
        },
    ]
    server = FastMCP("fake-sql-mcp")

    @server.tool
    def sql_query(action: str, params_json: str = "{}", connection: str = "") -> dict:
        p = json.loads(params_json)
        bound = p.get("params") or {}
        matched = [r for r in table if r["id"] > bound.get("after", 0)]
        if bound.get("since") is not None:
            matched = [r for r in matched if r["updated_at"] > bound["since"]]
        page = matched[: int(p.get("max_rows", 2))]
        cols = ["id", "title", "body", "updated_at"]
        return {
            "columns": cols,
            "rows": [[r[c] for c in cols] for r in page],
            "row_count": len(page),
            "truncated": len(matched) > len(page),
        }

    return server


def make_objectstore_server() -> FastMCP:
    """A fake objectstore-mcp: single-page objects list + text-mode get."""
    store = {"kb/a.md": "object alpha content. " * 8, "kb/b.md": "object beta. " * 8}
    server = FastMCP("fake-objectstore-mcp")

    def objects_impl(action: str, params_json: str = "{}") -> dict:
        p = json.loads(params_json)
        if action == "list":
            return {
                "bucket": p["bucket"],
                "objects": [
                    {"key": k, "size": len(v), "last_modified": "2026-01-01T00:00:00Z"}
                    for k, v in sorted(store.items())
                ],
                "prefixes": [],
                "next_token": None,
                "truncated": False,
            }
        if action == "get":
            return {
                "bucket": p["bucket"],
                "key": p["key"],
                "size": len(store[p["key"]]),
                "encoding": "text",
                "content": store[p["key"]],
            }
        raise ValueError(f"unknown action {action!r}")

    server.tool(objects_impl, name="objects")
    return server


@pytest.mark.integration
@pytest.mark.concept("KG-2.59")
@pytest.mark.asyncio
async def test_mcp_tool_sql_source_through_run_connector_entry_point(
    tmp_path, monkeypatch
):
    """The facade the source_connector MCP tool + /connector/run route call."""
    # Isolate the DeltaManifest checkpoint store (global SQLite otherwise).
    monkeypatch.setenv("AGENT_UTILITIES_DATA_DIR", str(tmp_path))
    backend = _RecordingBackend()
    ontology = build_ontology_system(SimpleNamespace(backend=backend))

    result = await ontology.run_connector(
        "mcp_tool",
        {
            "preset": "sql-query",
            "client": make_sql_server(),
            "params": {
                "sql": "SELECT id, title, body, updated_at FROM articles "
                "WHERE id > :after ORDER BY id",
                "params": {"after": 0},
                "max_rows": 2,
            },
            "cursor_record_field": "id",
            "text_field": "body",
            "title_field": "title",
            "updated_field": "updated_at",
        },
        connector_id="mcp-tool-sql-live-test",
    )

    assert result["status"] == "success"
    assert result["documents"] == 3
    assert result["checkpoint_advanced"] is True

    docs = [n for n in backend.nodes.values() if n.get("type") == "Document"]
    chunks = [n for n in backend.nodes.values() if n.get("type") == "Chunk"]
    assert len(docs) == 3
    assert {d.get("name") for d in docs} == {"Graphs", "Ontologies", "Retrieval"}
    assert len(chunks) >= 3
    rels = {e[2] for e in backend.edges}
    assert "HAS_CHUNK" in rels and "CHUNK_OF" in rels


@pytest.mark.integration
@pytest.mark.concept("KG-2.59")
@pytest.mark.asyncio
async def test_mcp_tool_objectstore_source_incremental_reingest(tmp_path, monkeypatch):
    """ContentType.CONNECTOR adaptor: list+get sweep, then an incremental re-run."""
    # Isolate the DeltaManifest checkpoint store (global SQLite otherwise).
    monkeypatch.setenv("AGENT_UTILITIES_DATA_DIR", str(tmp_path))

    backend = _RecordingBackend()
    engine = IngestionEngine(kg_engine=None, backend=backend)
    server = make_objectstore_server()
    manifest = IngestionManifest(
        content_type=ContentType.CONNECTOR,
        source_uri="mcp_tool",
        metadata={
            "connector_config": {
                "preset": "objectstore-prefix",
                "client": server,
                "params": {"bucket": "docs", "prefix": "kb/"},
            },
            "connector_id": "mcp-tool-objectstore-live-test",
        },
    )

    result = await engine.ingest(manifest)
    assert result.status == "success"
    assert result.details["documents"] == 2
    docs = [n for n in backend.nodes.values() if n.get("type") == "Document"]
    assert len(docs) == 2

    # Second run resumes from the stored checkpoint: nothing newer than the
    # watermark, so the re-poll ingests nothing (ECO-4.26 incrementality).
    result2 = await engine.ingest(
        IngestionManifest(
            content_type=ContentType.CONNECTOR,
            source_uri="mcp_tool",
            metadata={
                "connector_config": {
                    "preset": "objectstore-prefix",
                    "client": server,
                    "params": {"bucket": "docs", "prefix": "kb/"},
                },
                "connector_id": "mcp-tool-objectstore-live-test",
            },
        )
    )
    assert result2.status == "success"
    assert result2.details["documents"] == 0
