"""Tests for GraphRAG community summarization (CONCEPT:KG-2.258)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_utilities.knowledge_graph.pipeline.phases import community_reports as cr


class FakeGraph:
    def __init__(self, nodes, edges):
        self._nodes = dict(nodes)  # id -> props
        self._edges = list(edges)  # (u, v, data)
        self.added_edges: list[tuple[str, str, dict]] = []

    def node_count(self):
        return len(self._nodes)

    def nodes(self, data=False):
        return list(self._nodes.items()) if data else list(self._nodes)

    def edges(self, data=False):
        return (
            [(u, v, d) for u, v, d in self._edges]
            if data
            else [(u, v) for u, v, _ in self._edges]
        )

    def add_node(self, nid, props):
        self._nodes[nid] = props

    def add_edge(self, u, v, **kw):
        self.added_edges.append((u, v, kw))


def test_group_by_community():
    nodes = [
        ("a", {"community": 0}),
        ("b", {"community": 0}),
        ("c", {"community": 1}),
        ("d", {}),  # untagged → dropped
    ]
    groups = cr.group_by_community(nodes)
    assert set(groups) == {0, 1}
    assert len(groups[0]) == 2


def test_summarize_community_fallback_without_llm():
    theme, summary = cr.summarize_community(["Acme", "Bob"], [], None)
    assert "Acme" in theme
    assert summary == ""


def test_summarize_community_with_llm():
    def fake_llm(prompt: str) -> str:
        return '```json\n{"theme": "Palm Beach Network", "summary": "A tight cluster."}\n```'

    theme, summary = cr.summarize_community(
        ["Bob", "Acme"], ["Bob knows Acme"], fake_llm
    )
    assert theme == "Palm Beach Network"
    assert summary == "A tight cluster."


def test_parse_theme_summary_bad_json_falls_back():
    theme, summary = cr._parse_theme_summary("not json at all", "fallback")
    assert theme == "fallback"
    assert summary == ""


@pytest.mark.asyncio
async def test_execute_creates_community_report_live_path(monkeypatch):
    # stub the lite LLM so no network call
    def fake_llm(prompt: str) -> str:
        return '{"theme": "Test Theme", "summary": "stub summary"}'

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.enrichment.cards.make_lite_llm_fn",
        lambda: fake_llm,
    )

    # one community of 8 members (>= _MIN_COMMUNITY_SIZE) tagged community=0
    nodes = [(f"n{i}", {"community": 0, "label": f"Entity {i}"}) for i in range(8)]
    edges = [("n0", "n1", {"rel_type": "knows"}), ("n1", "n2", {"rel_type": "knows"})]
    graph = FakeGraph(nodes, edges)
    ctx = SimpleNamespace(graph=graph)

    result = await cr.execute_community_reports(ctx, {})
    assert result["community_reports"] >= 1

    report = graph._nodes.get("community_report:0")
    assert report is not None
    assert report["type"] == "CommunityReport"
    assert report["theme"] == "Test Theme"
    assert report["member_count"] == 8
    assert report["level"] == 0
    # every member linked PART_OF_COMMUNITY → report
    part_of = [e for e in graph.added_edges if e[2].get("type") == "PART_OF_COMMUNITY"]
    assert len(part_of) >= 8


@pytest.mark.asyncio
async def test_small_communities_skipped(monkeypatch):
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.enrichment.cards.make_lite_llm_fn",
        lambda: lambda p: "{}",
    )
    nodes = [(f"n{i}", {"community": 0}) for i in range(3)]  # below min size
    graph = FakeGraph(nodes, [])
    result = await cr.execute_community_reports(SimpleNamespace(graph=graph), {})
    assert result["community_reports"] == 0


@pytest.mark.asyncio
async def test_global_report_when_multiple_communities(monkeypatch):
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.enrichment.cards.make_lite_llm_fn",
        lambda: lambda p: '{"theme": "T", "summary": "s"}',
    )
    nodes = [(f"a{i}", {"community": 0, "label": f"A{i}"}) for i in range(8)]
    nodes += [(f"b{i}", {"community": 1, "label": f"B{i}"}) for i in range(8)]
    graph = FakeGraph(nodes, [])
    await cr.execute_community_reports(SimpleNamespace(graph=graph), {})
    # a level-1 global report links the two level-0 reports
    glob = graph._nodes.get("community_report:global")
    assert glob is not None and glob["level"] == 1
