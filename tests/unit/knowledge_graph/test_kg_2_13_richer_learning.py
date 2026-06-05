"""CONCEPT:KG-2.13 (Richer Learning enhancement) — typed, outcome-grounded extraction.

Covers the new MemoryEdit fields (entry_type / training_value / outcome_gate / evidence_ids), the
outcome-grounding gate (un-grounded decisions are dropped), node tagging, and GROUNDED_BY edges.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.memory.learning_engine import (
    BackgroundLearner,
    MemoryEdit,
    parse_memory_edits,
)


class _FakeEngine:
    def __init__(self):
        self.nodes = {}
        self.edges = []

    def add_memory_node(self, node):
        self.nodes[node.id] = node

    def get_memory_node(self, mid):
        return self.nodes.get(mid)

    def update_memory_node(self, mid, node):
        self.nodes[mid] = node

    def link_nodes(self, src, dst, rel, properties=None):
        self.edges.append((src, dst, rel))


@pytest.mark.concept(id="KG-2.13")
def test_memory_edit_new_fields_defaults():
    e = MemoryEdit(action="ADD", content="x")
    assert e.entry_type == "fact" and e.training_value == "normal"
    assert e.outcome_gate is True and e.evidence_ids == []


@pytest.mark.concept(id="KG-2.13")
def test_ungrounded_decision_is_gated_out():
    eng = _FakeEngine()
    learner = BackgroundLearner(eng)
    counts = learner.apply_edits([
        MemoryEdit(action="ADD", id="d1", content="We will use Postgres", entry_type="decision"),
    ])
    assert counts["gated"] == 1 and counts["added"] == 0
    assert "d1" not in eng.nodes  # not persisted


@pytest.mark.concept(id="KG-2.13")
def test_grounded_decision_persists_with_tags_and_edges():
    eng = _FakeEngine()
    learner = BackgroundLearner(eng)
    counts = learner.apply_edits([
        MemoryEdit(
            action="ADD", id="d1", content="We will use Postgres", entry_type="decision",
            training_value="high", evidence_ids=["msg7", "msg9"],
        )
    ])
    assert counts["added"] == 1
    node = eng.nodes["d1"]
    assert "type:decision" in node.tags and "train:high" in node.tags
    # GROUNDED_BY edges written to each evidence id.
    assert ("d1", "msg7", "GROUNDED_BY") in eng.edges
    assert ("d1", "msg9", "GROUNDED_BY") in eng.edges


@pytest.mark.concept(id="KG-2.13")
def test_plain_fact_not_gated():
    eng = _FakeEngine()
    learner = BackgroundLearner(eng)
    counts = learner.apply_edits([MemoryEdit(action="ADD", id="f1", content="User name is Sam")])
    assert counts["added"] == 1 and counts["gated"] == 0


@pytest.mark.concept(id="KG-2.13")
def test_parse_memory_edits_reads_new_fields():
    raw = '{"actions": [{"action": "ADD", "content": "c", "entry_type": "note", "training_value": "low"}]}'
    edits = parse_memory_edits(raw)
    assert edits[0].entry_type == "note" and edits[0].training_value == "low"
