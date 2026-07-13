"""Topic classification → topology (CONCEPT:AU-KG.enrichment.topic-classification-topology)."""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.enrichment import topic_classifier as tc


class _FakeBackend:
    """Records add_node/add_edge calls like the real graph backend."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.edges: list[tuple[str, str, dict]] = []

    def add_node(self, node_id: str, **props) -> None:
        self.nodes[node_id] = props

    def add_edge(self, source: str, target: str, **props) -> None:
        self.edges.append((source, target, props))


def test_worldview_taxonomy_shape() -> None:
    """Every top-level domain has a non-empty first layer of sub-domains."""
    assert len(tc.WORLDVIEW_TAXONOMY) >= 10
    for top, subs in tc.WORLDVIEW_TAXONOMY.items():
        assert isinstance(top, str) and top
        assert subs, f"{top} has no seed sub-topics"


def test_topic_node_id_deterministic_and_slugified() -> None:
    a = tc.topic_node_id("Technology & Computing", "Artificial Intelligence & Machine Learning")
    b = tc.topic_node_id("Technology & Computing", "Artificial Intelligence & Machine Learning")
    assert a == b
    assert a == "topic:technology-and-computing/artificial-intelligence-and-machine-learning"
    assert tc.topic_node_id("Science") == "topic:science"


def test_heuristic_classify_picks_valid_taxonomy_entry() -> None:
    assignment = tc._heuristic_classify(
        "A new transformer-based large language model architecture for code generation "
        "using attention and neural networks.",
        title="LLM architecture paper",
    )
    assert assignment.top_level in tc.WORLDVIEW_TAXONOMY
    assert 0.0 <= assignment.confidence <= 1.0


def test_heuristic_classify_empty_text_degrades_safely() -> None:
    assignment = tc._heuristic_classify("", title="")
    assert assignment.top_level in tc.WORLDVIEW_TAXONOMY


async def test_classify_and_link_topics_mints_hierarchy_and_edges(monkeypatch) -> None:
    async def _fake_classify(text, *, title="", source_type=""):
        return tc.TopicAssignment(
            top_level="Technology & Computing",
            sub_topic="Artificial Intelligence & Machine Learning",
            confidence=0.87,
            reasoning="mentions AI/ML explicitly",
        )

    monkeypatch.setattr(tc, "classify_topic", _fake_classify)

    backend = _FakeBackend()
    result = await tc.classify_and_link_topics(
        backend, "doc:test:1", "some AI content", title="Test Doc", source_type="document"
    )

    assert result["status"] == "classified"
    assert result["top_level"] == "Technology & Computing"
    assert result["sub_topic"] == "Artificial Intelligence & Machine Learning"
    assert result["confidence"] == 0.87

    top_id = tc.topic_node_id("Technology & Computing")
    sub_id = tc.topic_node_id("Technology & Computing", "Artificial Intelligence & Machine Learning")
    assert result["topic_ids"] == [top_id, sub_id]
    assert result["primary_topic_id"] == sub_id

    # :Topic nodes minted for both levels, idempotently addressable by id.
    assert backend.nodes[top_id]["type"] == "Topic"
    assert backend.nodes[top_id]["is_worldview_domain"] is True
    assert backend.nodes[sub_id]["type"] == "Topic"
    assert backend.nodes[sub_id]["is_worldview_domain"] is False

    # BROADER/NARROWER hierarchy edges between the two Topic nodes.
    rels = {(s, t, p["rel_type"]) for s, t, p in backend.edges}
    assert (sub_id, top_id, "BROADER") in rels
    assert (top_id, sub_id, "NARROWER") in rels

    # HAS_TOPIC on both levels; CLASSIFIED_AS only on the primary (most specific).
    has_topic = [(s, t) for s, t, p in backend.edges if p.get("rel_type") == "HAS_TOPIC"]
    assert ("doc:test:1", top_id) in has_topic
    assert ("doc:test:1", sub_id) in has_topic
    classified_as = [(s, t) for s, t, p in backend.edges if p.get("rel_type") == "CLASSIFIED_AS"]
    assert classified_as == [("doc:test:1", sub_id)]

    # Confidence is carried on the edges.
    for s, t, p in backend.edges:
        if s == "doc:test:1":
            assert p.get("confidence") == 0.87


async def test_classify_and_link_topics_no_subtopic_uses_top_level_as_primary(monkeypatch) -> None:
    async def _fake_classify(text, *, title="", source_type=""):
        return tc.TopicAssignment(top_level="Science", sub_topic="", confidence=0.4)

    monkeypatch.setattr(tc, "classify_topic", _fake_classify)

    backend = _FakeBackend()
    result = await tc.classify_and_link_topics(backend, "doc:test:2", "general science content")

    top_id = tc.topic_node_id("Science")
    assert result["primary_topic_id"] == top_id
    assert result["topic_ids"] == [top_id]
    classified_as = [(s, t) for s, t, p in backend.edges if p.get("rel_type") == "CLASSIFIED_AS"]
    assert classified_as == [("doc:test:2", top_id)]


async def test_classify_and_link_topics_skips_empty_text() -> None:
    backend = _FakeBackend()
    result = await tc.classify_and_link_topics(backend, "doc:test:3", "   ")
    assert result["status"] == "skipped"
    assert backend.nodes == {}
    assert backend.edges == []


async def test_classify_and_link_topics_skips_backend_without_graph_methods() -> None:
    result = await tc.classify_and_link_topics(object(), "doc:test:4", "some text")
    assert result["status"] == "skipped"


@pytest.mark.parametrize("label", list(tc.WORLDVIEW_TAXONOMY.keys()))
def test_every_seed_subtopic_is_known(label: str) -> None:
    assert set(tc.WORLDVIEW_TAXONOMY[label]) <= tc._KNOWN_SUBTOPICS[label]
