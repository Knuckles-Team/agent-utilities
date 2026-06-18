"""Native multi-perspective (STORM) inquiry (KG-2.127/2.128/2.129).

Live-path: the loop's ``acquire_for_topic_perspectival`` fans the probe across lenses,
materializes the inquiry as typed KG nodes, and returns the union of sources; the engine
derives agreement/divergence/blind-spot + a peer-review frontier question.
"""

from __future__ import annotations

from typing import Any

from agent_utilities.knowledge_graph.research import search
from agent_utilities.knowledge_graph.research.perspective import PerspectiveEngine


class _FakeEngine:
    def __init__(self) -> None:
        self.batches: list[tuple] = []
        self.loops: list[str] = []

    def query_cypher(self, query: str, params: dict[str, Any] | None = None):
        if "--(n)" in query:
            return [{"labels": ["Service"]}]
        return []

    def ingest_external_batch(self, domain, entities, rels=None):
        self.batches.append((domain, entities, rels))
        return {"status": "ok"}

    def add_node(self, *a, **k):  # used by submit_loop
        self.loops.append(a[0] if a else "")


def test_engine_builds_map_and_peer_review():
    eng = PerspectiveEngine(_FakeEngine())

    def acquire(q: str) -> list[str]:
        if "practitioner" in q:
            return ["s1", "s2"]
        if "academic" in q:
            return ["s1"]
        return []

    inq = eng.inquire({"id": "t1", "name": "agentic RAG"}, acquire)
    assert len(inq.perspectives) == 5
    assert "s1" in inq.contradiction_map.agreements  # corroborated by >=2 lenses
    assert inq.contradiction_map.blind_spot == ["Service"]
    assert inq.peer_review.dominant_lens == "practitioner"
    assert inq.peer_review.missing_perspective is not None
    assert inq.peer_review.frontier_question
    # typed KG structures
    ents, rels = inq.to_entities()
    assert {"research_inquiry", "perspective", "agreement", "peer_review"} <= {
        e["type"] for e in ents
    }
    assert {"asks_from", "reviews"} <= {r["type"] for r in rels}


def test_acquire_for_topic_perspectival_is_native_and_materializes(monkeypatch):
    eng = _FakeEngine()

    # Each per-question probe returns a deterministic source.
    monkeypatch.setattr(
        search,
        "acquire_for_topic",
        lambda engine, topic, **k: ["src-" + topic["name"].split()[0].lower()],
    )
    srcs = search.acquire_for_topic_perspectival(
        eng, {"id": "t1", "name": "agentic RAG"}
    )
    assert srcs  # union of per-lens sources
    # materialized the inquiry as a KG batch
    assert eng.batches and eng.batches[0][0] == "research"


def test_perspectival_falls_back_when_fanout_empty(monkeypatch):
    eng = _FakeEngine()
    calls = {"n": 0}

    def fake_acquire(engine, topic, **k):
        calls["n"] += 1
        # perspective questions find nothing; the direct topic probe finds one
        return ["direct"] if topic["name"] == "agentic RAG" else []

    monkeypatch.setattr(search, "acquire_for_topic", fake_acquire)
    srcs = search.acquire_for_topic_perspectival(
        eng, {"id": "t1", "name": "agentic RAG"}
    )
    assert srcs == ["direct"]
