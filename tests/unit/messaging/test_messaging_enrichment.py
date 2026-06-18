"""Tests for post-conversation KG enrichment (CONCEPT:ECO-4.65)."""

from __future__ import annotations

import pytest

from agent_utilities.messaging import enrichment


class _Concept:
    def __init__(self, cid: str, name: str) -> None:
        self.id, self.name, self.kind, self.summary, self.source_ids = (
            cid,
            name,
            "topic",
            "s",
            [],
        )


class _Eng:
    def __init__(self) -> None:
        self.nodes: list[str] = []
        self.links: list[tuple[str, str, str]] = []

    def add_node(self, node_id: str, node_type: str, properties: dict) -> None:
        self.nodes.append(f"{node_type}:{node_id}")

    def link_nodes(
        self, source_id: str, target_id: str, rel_type: str, properties: dict
    ) -> None:
        self.links.append((source_id, target_id, rel_type))


def test_enrich_writes_concepts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MESSAGING_ENRICH", "1")
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.enrichment.cards.make_lite_llm_fn",
        lambda: lambda *a, **k: "{}",
    )
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.enrichment.extractors.text.extract_text_concepts",
        lambda *a, **k: ([_Concept("concept:kg", "knowledge graph")], []),
    )
    eng = _Eng()
    n = enrichment.enrich_conversation(
        eng,
        "let's talk about the knowledge graph",
        platform="telegram",
        channel_id="42",
    )
    assert n == 1
    assert any(x.startswith("Concept:") for x in eng.nodes)
    assert any(x.startswith("Thread:") for x in eng.nodes)  # the chat-turn anchor
    assert eng.links and eng.links[0][2] == "MENTIONS"


def test_enrich_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MESSAGING_ENRICH", "0")
    eng = _Eng()
    assert enrichment.enrich_conversation(eng, "x", platform="t", channel_id="1") == 0
    assert eng.nodes == []


def test_enrich_no_engine() -> None:
    assert enrichment.enrich_conversation(None, "x", platform="t", channel_id="1") == 0
