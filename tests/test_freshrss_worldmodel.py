"""FreshRSS world-model source + relevance gate — CONCEPT:KG-2.115/116/117."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import agent_utilities.knowledge_graph.core.source_sync as ss
from agent_utilities.automation.research_pipeline import (
    RELEVANCE_PROFILES,
    RELEVANCE_TAXONOMY,
    WORLD_MODEL_TAXONOMY,
    ResearchPipelineRunner,
    score_text,
)
from agent_utilities.automation.worldmodel_pipeline import (
    WorldModelConfig,
    WorldModelPipelineRunner,
)
from agent_utilities.protocols.source_connectors.connectors.mcp_tool import (
    MCP_TOOL_PRESETS,
)

# ── preset ────────────────────────────────────────────────────────────────────


def test_freshrss_preset_shape():
    p = MCP_TOOL_PRESETS["freshrss"]
    assert p["server"] == "freshrss-mcp"
    assert p["tool"] == "freshrss_reader" and p["action"] == "stream_contents"
    assert p["params_style"] == "json"
    assert p["records_path"] == "items"
    assert p["text_field"] == "summary.content"
    assert p["updated_field"] == "published"
    assert p["pagination"] == "cursor"
    assert p["cursor_param"] == "continuation" and p["cursor_path"] == "continuation"
    assert p["updated_since_param"] == "newer_than"
    assert p["doc_type"] == "news_article"


# ── relevance scorer strangle (research path stays identical) ───────────────────


def test_score_paper_delegates_to_score_text():
    title, abstract = "Agentic planning", "reasoning and planning with memory and tools"
    runner = ResearchPipelineRunner(engine=None)
    assert runner.score_paper(title, abstract) == score_text(
        title, abstract, RELEVANCE_TAXONOMY
    )


def test_relevance_profiles_registered():
    assert RELEVANCE_PROFILES["research"] is RELEVANCE_TAXONOMY
    assert RELEVANCE_PROFILES["world_model"] is WORLD_MODEL_TAXONOMY


def test_world_model_scoring_finance():
    score, domains = score_text(
        "Nvidia earnings beat",
        "earnings stock equities valuation",
        WORLD_MODEL_TAXONOMY,
    )
    assert score > 3.0
    assert "markets_finance" in domains and "companies" in domains


# ── gate tiering ────────────────────────────────────────────────────────────


def test_tier_matrix():
    r = WorldModelPipelineRunner(engine=None, config=WorldModelConfig())
    # forced always wins
    assert r._tier(0.0, None, True) == "relevant"
    # high score, unknown novelty → relevant
    assert r._tier(5.0, None, False) == "relevant"
    # high score but already-covered (low novelty) → demoted to marginal
    assert r._tier(5.0, 0.1, False) == "marginal"
    # mid score → marginal
    assert r._tier(1.5, None, False) == "marginal"
    # low score, highly novel → marginal
    assert r._tier(0.5, 0.9, False) == "marginal"
    # below everything → skipped
    assert r._tier(0.0, 0.0, False) == "skipped"


def test_is_research_detects_arxiv_feed():
    rec_research = {"categories": [{"label": "Research (ScholarX)"}], "origin": {}}
    rec_news = {"categories": [{"label": "Markets & Finance"}], "origin": {}}
    assert WorldModelPipelineRunner._is_research(rec_research) is True
    assert WorldModelPipelineRunner._is_research(rec_news) is False
    assert (
        WorldModelPipelineRunner._is_research(
            {"origin": {"htmlUrl": "https://arxiv.org/list/cs.AI"}}
        )
        is True
    )


def _doc(rid, title, text, record):
    return SimpleNamespace(
        id=rid,
        title=title,
        text=text,
        metadata={"record": record},
        source_uri=f"mcp-tool://freshrss-mcp/freshrss_reader/{rid}",
        updated_at=record.get("published"),
    )


def test_run_gated_ingest_tiers_without_engine():
    # engine=None → no writes, but tiering/counting still runs (best-effort).
    docs = [
        _doc(
            "i1",
            "Nvidia earnings beat",
            "earnings stock equities valuation",
            {"published": "1700000100", "categories": [{"label": "Markets & Finance"}]},
        ),
        _doc(
            "i2",
            "Fed watch",
            "inflation outlook",
            {"published": "1700000050", "categories": [{"label": "Macro"}]},
        ),
        _doc(
            "i3",
            "Cat picture",
            "a fluffy cat naps",
            {"published": "1700000010", "categories": [{"label": "Misc"}]},
        ),
        _doc(
            "i4",
            "New transformer paper",
            "a novel attention mechanism",
            {
                "published": "1700000200",
                "categories": [{"label": "Research (ScholarX)"}],
            },
        ),
    ]
    report = WorldModelPipelineRunner(engine=None).run_gated_ingest(docs)
    assert report.items_seen == 4
    assert report.relevant == 1
    assert report.marginal == 1
    assert report.skipped == 1
    assert report.research == 1
    assert report.ingested == 3


# ── delta handler ─────────────────────────────────────────────────────────────


def test_sync_freshrss_skips_when_unconfigured(monkeypatch):
    monkeypatch.delenv("FRESHRSS_URL", raising=False)
    # No freshrss-mcp server registered either → genuinely unconfigured.
    monkeypatch.setattr(
        "agent_utilities.protocols.source_connectors.connectors.mcp_tool._load_mcp_config",
        lambda: {},
    )
    res = ss.sync_source(MagicMock(), "freshrss", mode="delta")
    assert res["status"] == "skipped"


def test_sync_freshrss_gates_and_watermarks(monkeypatch):
    monkeypatch.setenv("FRESHRSS_URL", "http://freshrss.arpa")
    monkeypatch.setattr(ss, "_read_watermark", lambda b, s: None)
    written: dict[str, str] = {}
    monkeypatch.setattr(
        ss, "_write_watermark", lambda b, s, w: written.__setitem__(s, w)
    )

    docs = [
        SimpleNamespace(updated_at="1700000100"),
        SimpleNamespace(updated_at="1700000300"),
        SimpleNamespace(updated_at=None),
    ]
    fake_conn = SimpleNamespace(poll_all=lambda: docs)
    monkeypatch.setattr(
        "agent_utilities.protocols.source_connectors.registry.build_connector",
        lambda *a, **k: fake_conn,
    )

    class FakeRunner:
        def __init__(self, engine=None):
            pass

        def run_gated_ingest(self, docs):
            return SimpleNamespace(
                ingested=2, relevant=1, marginal=1, research=0, skipped=1
            )

    monkeypatch.setattr(
        "agent_utilities.automation.worldmodel_pipeline.WorldModelPipelineRunner",
        FakeRunner,
    )

    engine = MagicMock()
    engine.backend = MagicMock()
    res = ss.sync_source(engine, "freshrss", mode="delta")

    assert res["status"] == "ok" and res["source"] == "freshrss"
    assert res["items_seen"] == 3
    assert res["relevant"] == 1 and res["marginal"] == 1
    assert res["skipped_unchanged"] == 1
    assert written["freshrss"] == "1700000300"  # max published epoch
