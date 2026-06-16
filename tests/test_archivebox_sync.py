"""Tests for the ArchiveBox source (preset + delta sync) — CONCEPT:KG-2.7."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import agent_utilities.knowledge_graph.core.source_sync as ss
from agent_utilities.protocols.source_connectors.connectors.mcp_tool import (
    MCP_TOOL_PRESETS,
)


def test_archivebox_preset_shape():
    p = MCP_TOOL_PRESETS["archivebox"]
    assert p["server"] == "archivebox-api"
    assert p["tool"] == "archivebox_core" and p["action"] == "get_snapshots"
    assert p["pagination"] == "page" and p["page_kind"] == "offset"
    assert p["records_path"] == "results"


def test_sync_archivebox_ingests_new_snapshots(monkeypatch):
    monkeypatch.setenv("ARCHIVEBOX_URL", "http://archivebox:8000")
    monkeypatch.setattr(ss, "_read_watermark", lambda b, s: None)
    written: dict[str, str] = {}
    monkeypatch.setattr(
        ss, "_write_watermark", lambda b, s, w: written.__setitem__(s, w)
    )

    docs = [
        SimpleNamespace(
            metadata={"url": "https://a.io/post"}, updated_at="2026-06-10T00:00:00Z"
        ),
        SimpleNamespace(
            metadata={"url": "https://b.io/post"}, updated_at="2026-06-12T00:00:00Z"
        ),
        SimpleNamespace(metadata={"url": "ftp://skip"}, updated_at=None),  # non-http
    ]
    fake_conn = SimpleNamespace(poll_all=lambda: docs)
    monkeypatch.setattr(
        "agent_utilities.protocols.source_connectors.registry.build_connector",
        lambda *a, **k: fake_conn,
    )

    seen_manifests = {}

    async def fake_batch(self, manifests):
        seen_manifests["m"] = manifests
        from agent_utilities.knowledge_graph.ingestion.engine import IngestionResult

        return [
            IngestionResult(manifest=m, status="success") for m in manifests
        ]

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.engine.IngestionEngine.ingest_batch",
        fake_batch,
    )

    engine = MagicMock()
    engine.backend = MagicMock()
    res = ss.sync_source(engine, "archivebox", mode="delta")

    assert res["status"] == "ok" and res["source"] == "archivebox"
    assert res["snapshots_seen"] == 3
    assert res["documents_ingested"] == 2  # only the two http snapshots
    # only http urls become DOCUMENT manifests
    assert {m.source_uri for m in seen_manifests["m"]} == {
        "https://a.io/post",
        "https://b.io/post",
    }
    assert written["archivebox"] == "2026-06-12T00:00:00Z"  # max updated_at


def test_sync_archivebox_skips_when_unconfigured(monkeypatch):
    monkeypatch.delenv("ARCHIVEBOX_URL", raising=False)
    res = ss.sync_source(MagicMock(), "archivebox", mode="delta")
    assert res["status"] == "skipped"
