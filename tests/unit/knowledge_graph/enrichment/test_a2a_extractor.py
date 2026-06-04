"""Tests for the A2A agent-card source extractor (CONCEPT:KG-2.10)."""

from __future__ import annotations

import json

from agent_utilities.knowledge_graph.enrichment.extractors.a2a import extract
from agent_utilities.knowledge_graph.enrichment.registry import (
    get_source,
    write_batch,
)


class FakeBackend:
    def __init__(self):
        self.nodes = {}
        self.edges = []

    def add_node(self, node_id, **props):
        self.nodes[node_id] = props

    def add_edge(self, s, t, **props):
        self.edges.append((s, t, props.get("rel_type")))


def _sample_cards():
    return [
        {
            "name": "Weather Agent",
            "description": "Provides weather forecasts.",
            "url": "https://weather.example.com/a2a",
            "version": "1.2.0",
            "provider": {"organization": "Acme Weather"},
            "capabilities": {"streaming": True},
            "skills": [
                {
                    "id": "forecast",
                    "name": "Forecast",
                    "description": "Multi-day forecast.",
                    "tags": ["weather", "forecast"],
                },
                {
                    "name": "Current Conditions",
                    "description": "Now-cast.",
                    "tags": ["weather"],
                },
            ],
        },
        {
            "name": "Translator Agent",
            "description": "Translates text.",
            "endpoint": "https://translate.example.com/a2a",
            "version": "0.9.1",
            "provider": "Lingua Inc",
            "skills": [
                {
                    "id": "translate",
                    "description": "Translate between languages.",
                    "tags": ["nlp", "translation"],
                }
            ],
        },
    ]


def test_extract_cards_and_skills():
    batch = extract({"cards": _sample_cards()})
    assert batch.category == "a2a"

    by_id = {n.id: n for n in batch.nodes}

    # AgentCard nodes
    weather = by_id["a2a:weather-agent"]
    assert weather.type == "A2AAgentCard"
    assert weather.props["name"] == "Weather Agent"
    assert weather.props["url"] == "https://weather.example.com/a2a"
    assert weather.props["version"] == "1.2.0"
    assert weather.props["provider"] == "Acme Weather"

    translator = by_id["a2a:translator-agent"]
    assert translator.type == "A2AAgentCard"
    # endpoint used when url absent
    assert translator.props["url"] == "https://translate.example.com/a2a"
    assert translator.props["provider"] == "Lingua Inc"

    # Skill nodes
    forecast = by_id["skill:a2a:weather-agent:forecast"]
    assert forecast.type == "Skill"
    assert forecast.props["description"] == "Multi-day forecast."
    assert forecast.props["tags"] == ["weather", "forecast"]

    # skill keyed off id when no name
    assert "skill:a2a:translator-agent:translate" in by_id

    # EXPOSES_SKILL edges
    rels = {(e.source, e.target, e.rel_type) for e in batch.edges}
    assert (
        "a2a:weather-agent",
        "skill:a2a:weather-agent:forecast",
        "EXPOSES_SKILL",
    ) in rels
    assert (
        "a2a:translator-agent",
        "skill:a2a:translator-agent:translate",
        "EXPOSES_SKILL",
    ) in rels

    # 2 cards + 3 skills
    assert len(batch.nodes) == 5
    assert len(batch.edges) == 3


def test_extract_from_json_files(tmp_path):
    cards = _sample_cards()
    p1 = tmp_path / "weather.json"
    p2 = tmp_path / "translator.json"
    p1.write_text(json.dumps(cards[0]), encoding="utf-8")
    p2.write_text(json.dumps(cards[1]), encoding="utf-8")

    # config as a list of path strings
    batch = extract({"cards": [str(p1), str(p2)]})
    ids = {n.id for n in batch.nodes}
    assert "a2a:weather-agent" in ids
    assert "a2a:translator-agent" in ids

    # config as a bare path string
    batch2 = extract(str(p1))
    ids2 = {n.id for n in batch2.nodes}
    assert "a2a:weather-agent" in ids2


def test_tolerant_of_missing_keys():
    batch = extract({"cards": [{}, {"name": "Bare Agent"}]})
    ids = {n.id for n in batch.nodes}
    assert "a2a:bare-agent" in ids  # nameless card skipped, bare card kept
    bare = next(n for n in batch.nodes if n.id == "a2a:bare-agent")
    assert bare.props["url"] is None


def test_source_registered():
    src = get_source("a2a")
    assert src is not None
    assert src.category == "a2a"
    assert "A2A" in src.description


def test_write_batch_persists():
    batch = extract({"cards": _sample_cards()})
    backend = FakeBackend()
    n, e = write_batch(backend, batch)
    assert n == 5 and e == 3
    assert backend.nodes["a2a:weather-agent"]["type"] == "A2AAgentCard"
    assert backend.nodes["skill:a2a:weather-agent:forecast"]["type"] == "Skill"
    assert (
        "a2a:weather-agent",
        "skill:a2a:weather-agent:forecast",
        "EXPOSES_SKILL",
    ) in backend.edges
