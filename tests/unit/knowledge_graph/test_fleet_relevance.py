#!/usr/bin/python
"""Fleet-wide relevance grading (CONCEPT:AHE-3.63)."""

import pytest

from agent_utilities.knowledge_graph.research import fleet_relevance as fr

pytestmark = pytest.mark.concept("AHE-3.63")


_PROFILES = {
    "vector-mcp": {"vector", "retrieval", "rag", "embedding", "search"},
    "data-science-mcp": {"training", "model", "evaluation", "reranking"},
    "geniusbot": {"desktop", "cockpit", "frontend"},
}


class _Graph:
    def __init__(self, nodes):
        self._n = nodes

    def nodes(self, data=False):
        return list(self._n.items()) if data else list(self._n)


class _Engine:
    def __init__(self, nodes):
        self.graph = _Graph(nodes)


def test_grade_item_surfaces_matches_above_threshold():
    item = {"sparse", "retrieval", "reranking", "rag", "search"}
    matches = fr.grade_item(item, _PROFILES, threshold_pct=5.0)
    names = [m["target"] for m in matches]
    # overlaps vector-mcp (retrieval/rag/search) and data-science-mcp (reranking);
    # geniusbot shares nothing → excluded
    assert "vector-mcp" in names and "data-science-mcp" in names
    assert "geniusbot" not in names
    # best-first ordering
    assert matches[0]["score"] >= matches[-1]["score"]


def test_low_threshold_is_wider_than_high():
    item = {"reranking", "unrelated", "words", "here", "plus"}
    wide = fr.grade_item(item, _PROFILES, threshold_pct=5.0)
    strict = fr.grade_item(item, _PROFILES, threshold_pct=90.0)
    assert len(wide) >= len(strict)


def test_grade_fleet_over_graph(monkeypatch):
    monkeypatch.setattr(fr, "fleet_target_profiles", lambda yml_path=None: _PROFILES)
    eng = _Engine(
        {
            "paper:rag1": {
                "type": "article",
                "name": "Sparse retrieval reranking for RAG search",
            },
            "repo:dice": {
                "type": "codebase",
                "name": "DICE single-vector embedding retrieval",
            },
            "noise": {"type": "concept", "name": "should be ignored"},
        }
    )
    out = fr.grade_fleet(eng, threshold_pct=5.0)
    assert out["targets"] == 3
    assert out["sources_graded"] == 2  # the concept node is not a source
    by_src = {c["source"]: c for c in out["considerations"]}
    assert "paper:rag1" in by_src
    assert any(m["target"] == "vector-mcp" for m in by_src["paper:rag1"]["matches"])


def test_fleet_target_profiles_parses_manifest(tmp_path):
    yml = tmp_path / "workspace.yml"
    yml.write_text(
        "name: ws\n"
        "path: /tmp\n"
        "subdirectories:\n"
        "  agent-packages:\n"
        "    repositories:\n"
        "      - url: https://x/vector-mcp.git\n"
        "        description: RAG over multiple vector DBs\n"
        "      - url: https://x/geniusbot.git\n"
        "        description: Desktop Cockpit frontend\n"
    )
    profiles = fr.fleet_target_profiles(str(yml))
    assert set(profiles) == {"vector-mcp", "geniusbot"}
    assert "vector" in profiles["vector-mcp"]
    assert "cockpit" in profiles["geniusbot"]
