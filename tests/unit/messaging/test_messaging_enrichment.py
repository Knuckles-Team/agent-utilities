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


def test_surface_intents_creates_goal_and_spec() -> None:
    from agent_utilities.messaging import enrichment

    class _Eng:
        def __init__(self):
            self.nodes = []
            self.links = []

        def add_node(self, node_id, node_type, properties):
            self.nodes.append((node_type, node_id))

        def link_nodes(self, source_id, target_id, rel_type, properties):
            self.links.append(rel_type)

    eng = _Eng()

    def llm(p):
        return '{"goal": "deploy the tunnel", "spec": "webhook receiver spec"}'

    enrichment._surface_intents(
        eng, "let's deploy the tunnel and spec the webhook", "chatturn:x", llm
    )
    types = {t for t, _ in eng.nodes}
    assert "Goal" in types and "Spec" in types
    assert "HAS_GOAL" in eng.links and "PROPOSES_SPEC" in eng.links


def test_surface_intents_disabled(monkeypatch) -> None:
    from agent_utilities.messaging import enrichment

    monkeypatch.setenv("MESSAGING_GOALS", "0")

    class _Eng:
        def add_node(self, **k):
            raise AssertionError("should not write")

        def link_nodes(self, **k):
            raise AssertionError("should not link")

    enrichment._surface_intents(_Eng(), "deploy it", "s", lambda p: '{"goal":"x"}')
