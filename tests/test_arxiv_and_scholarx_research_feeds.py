"""arXiv native connector + ScholarX MCP fallback — CONCEPT:AU-KG.ingest.arxiv-feed-connector / research-connector-presets (KG-7.3)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import agent_utilities.core.config as cfg_module
import agent_utilities.knowledge_graph.core.source_sync as ss


@pytest.fixture(autouse=True)
def _source_sync_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ontology.connector_manifest_gate.precheck_source",
        lambda _source: {"checked": True, "ok": True},
    )


# ── _sync_arxiv delta handler ────────────────────────────────────────────────


def test_sync_arxiv_skips_when_unconfigured(monkeypatch):
    monkeypatch.setattr(cfg_module.config, "kg_arxiv_categories", "")
    res = ss.sync_source(MagicMock(), "arxiv", mode="delta")
    assert res["status"] == "skipped"


def test_sync_arxiv_gates_and_watermarks(monkeypatch):
    monkeypatch.setattr(cfg_module.config, "kg_arxiv_categories", "cs.AI, cs.LG")
    monkeypatch.setattr(ss, "_read_envelope_watermark", lambda *a, **k: None)
    monkeypatch.setattr(
        "agent_utilities.automation.feed_sources.upsert_feed_source",
        lambda *a, **k: "feed:arxiv:fixture",
    )
    committed: list[dict] = []
    monkeypatch.setattr(
        ss,
        "_ingest_graph_slice_via_envelope",
        lambda *a, **k: committed.append(k) or {"status": "success"},
    )

    docs = [
        SimpleNamespace(updated_at="2026-01-01T00:00:00Z"),
        SimpleNamespace(updated_at="2026-01-03T00:00:00Z"),
        SimpleNamespace(updated_at=None),
    ]
    fake_conn = SimpleNamespace(poll_all=lambda *a, **kw: docs)
    connector_configs: list[dict] = []

    def build_connector(_source_type, config):
        connector_configs.append(config)
        return fake_conn

    monkeypatch.setattr(
        "agent_utilities.protocols.source_connectors.registry.build_connector",
        build_connector,
    )

    class FakeRunner:
        def __init__(self, engine=None, config=None, connector="", source_instance=""):
            pass

        def run_gated_ingest(self, docs):
            return SimpleNamespace(
                ingested=3, relevant=0, marginal=0, research=3, skipped=0, failed=0
            )

    monkeypatch.setattr(
        "agent_utilities.automation.worldmodel_pipeline.WorldModelPipelineRunner",
        FakeRunner,
    )

    engine = MagicMock()
    engine.backend = MagicMock()
    res = ss.sync_source(engine, "arxiv", mode="delta")

    assert res["status"] == "ok" and res["source"] == "arxiv"
    assert res["details"]["items_seen"] == 3
    assert res["details"]["research"] == 3
    assert committed == [{"checkpoint": "2026-01-03T00:00:00Z"}]
    assert connector_configs == [
        {"categories": ["cs.AI", "cs.LG"], "max_results": 50}
    ]


def test_unknown_mcp_tool_preset_downgrades_to_skip_not_crash(monkeypatch):
    """A configured-but-uninstalled preset (e.g. freshrss-agent/scholarx not

    pip-installed alongside agent-utilities) must skip cleanly, not crash the
    connector_sync task (CONCEPT:AU-KG.ingest.research-connector-presets).
    """
    monkeypatch.setenv("FRESHRSS_URL", "http://freshrss.arpa")

    def _boom(_source_type, _config):
        raise ValueError("Unknown mcp_tool preset 'freshrss'. Available: sql-table")

    monkeypatch.setattr(
        "agent_utilities.protocols.source_connectors.registry.build_connector", _boom
    )
    monkeypatch.setattr(
        "agent_utilities.automation.feed_sources.upsert_feed_source",
        lambda *a, **k: "feed:freshrss:fixture",
    )

    res = ss.sync_source(MagicMock(), "freshrss", mode="delta")
    assert res["status"] == "skipped"


# ── ScholarX MCP fallback (feed_sources.py) ──────────────────────────────────


def test_scholarx_mcp_configured_detects_server(monkeypatch):
    from agent_utilities.automation import feed_sources

    monkeypatch.setattr(
        "agent_utilities.protocols.source_connectors.connectors.mcp_tool._load_mcp_config",
        lambda: {"scholarx-mcp": {}},
    )
    assert feed_sources._scholarx_mcp_configured() is True

    monkeypatch.setattr(
        "agent_utilities.protocols.source_connectors.connectors.mcp_tool._load_mcp_config",
        lambda: {},
    )
    assert feed_sources._scholarx_mcp_configured() is False


def test_scholarx_feed_documents_falls_back_to_mcp_when_package_missing(monkeypatch):
    from agent_utilities.automation import feed_sources
    from agent_utilities.protocols.source_connectors.base import SourceDocument

    monkeypatch.setattr(feed_sources, "_scholarx_mcp_configured", lambda: True)

    raw_doc = SourceDocument(
        id="2601.00090",
        source_uri="mcp-tool://scholarx-mcp/sx_search/2601.00090",
        title="A paper on agents",
        text="We study self-improving agent harnesses.",
        doc_type="research_paper",
        updated_at="2026-01-02",
        metadata={
            "record": {
                "id": "arxiv:2601.00090",
                "url": "https://arxiv.org/abs/2601.00090",
                "pdf_url": "https://arxiv.org/pdf/2601.00090",
                "authors": ["A. Researcher"],
                "categories": ["cs.AI"],
            }
        },
    )

    def _build_connector(_source_type, config):
        assert config["preset"] == "scholarx-papers"
        return SimpleNamespace(load=lambda: [raw_doc])

    monkeypatch.setattr(
        "agent_utilities.protocols.source_connectors.registry.build_connector",
        _build_connector,
    )

    docs = feed_sources.scholarx_feed_documents(categories=["cs.AI"])
    assert len(docs) == 1
    doc = docs[0]
    assert doc.id == "arxiv:2601.00090"
    rec = doc.metadata["record"]
    assert rec["origin"]["streamId"] == "scholarx:arxiv"
    assert rec["pdf_url"] == "https://arxiv.org/pdf/2601.00090"
    assert doc.metadata["source_system"] == "scholarx"


def test_scholarx_feed_documents_noop_when_neither_available(monkeypatch):
    from agent_utilities.automation import feed_sources

    monkeypatch.setattr(feed_sources, "_scholarx_mcp_configured", lambda: False)
    assert feed_sources.scholarx_feed_documents() == []


def test_scholarx_mcp_documents_swallows_unreachable_server(monkeypatch):
    from agent_utilities.automation import feed_sources

    def _boom(_source_type, _config):
        raise ValueError("Unknown mcp_tool preset 'scholarx-papers'.")

    monkeypatch.setattr(
        "agent_utilities.protocols.source_connectors.registry.build_connector", _boom
    )
    assert feed_sources._scholarx_mcp_documents(["cs.AI"], 1) == []
