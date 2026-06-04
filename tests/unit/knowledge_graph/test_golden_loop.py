"""Unit tests for the propose-only self-evolution golden loop (CONCEPT:KG-2.7)."""

from __future__ import annotations

from typing import Any

from agent_utilities.knowledge_graph.adaptation.topic_resolver import (
    mark_addressed,
    unresolved_topics,
)
from agent_utilities.knowledge_graph.research.golden_loop import GoldenLoopController


class _StubEngine:
    """Minimal engine: canned cypher results + records link_nodes calls."""

    def __init__(self, concepts, addressed):
        self._concepts = concepts  # list[(id, name)]
        self._addressed = set(addressed)  # ids with ADDRESSED_BY
        self.links: list[tuple[str, str, str]] = []
        self.backend = object()  # no semantic_search → acquire returns []

    def query_cypher(self, q: str, params: dict | None = None) -> list[dict[str, Any]]:
        if "ADDRESSED_BY" in q and "RETURN c.id AS id" in q and "name" not in q:
            return [{"id": i} for i in self._addressed]
        if "MATCH (c:Concept) RETURN c.id AS id, c.name AS name" in q:
            return [{"id": i, "name": n} for i, n in self._concepts]
        return []

    def link_nodes(self, source_id, target_id, rel_type, properties=None):
        self.links.append((source_id, target_id, rel_type))


def test_unresolved_topics_subtracts_addressed():
    eng = _StubEngine(
        concepts=[("c:1", "A"), ("c:2", "B"), ("c:3", "C")],
        addressed=["c:2"],
    )
    topics = unresolved_topics(eng, limit=10)
    ids = {t["id"] for t in topics}
    assert ids == {"c:1", "c:3"}  # c:2 is already addressed → excluded


def test_mark_addressed_writes_both_directions():
    eng = _StubEngine([], [])
    n = mark_addressed(eng, "c:1", ["src:a", "src:b", "c:1"], source="t")
    assert n == 2  # self-link (c:1) skipped
    rels = {(s, t, r) for s, t, r in eng.links}
    assert ("src:a", "c:1", "ADDRESSES") in rels
    assert ("c:1", "src:a", "ADDRESSED_BY") in rels


def test_run_one_cycle_intake_only_propose_only():
    eng = _StubEngine(concepts=[("c:1", "A"), ("c:2", "B")], addressed=[])
    # acquire returns [] (no semantic_search) → resolve does nothing, but the
    # cycle must complete cleanly and stay propose-only.
    rep = GoldenLoopController(eng).run_one_cycle(synthesize=False, distill=False)
    assert rep["propose_only"] is True
    assert rep["topics_intake"] == 2
    assert rep["topics_resolved"] == 0
    assert rep["errors"] == []
