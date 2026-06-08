#!/usr/bin/python
"""Plan synthesis from KG neighborhood (VU-8).

CONCEPT:KG-2.7
"""

import pytest

from agent_utilities.knowledge_graph.assimilation import (
    hydrate_feature,
    synthesize_plan_for_feature,
    synthesize_plans,
)
from agent_utilities.knowledge_graph.assimilation.plan_synthesis import _default_synth

pytestmark = pytest.mark.concept("KG-2.7")


class _Graph:
    def __init__(self, nodes):
        self._n = dict(nodes)
        self._out: dict = {}
        self._in: dict = {}

    def nodes(self, data=False):
        return list(self._n.items()) if data else list(self._n)

    def add_node(self, nid, attrs):
        self._n[nid] = attrs

    def add_edge(self, src, dst, props):
        self._out.setdefault(src, []).append((src, dst, props))
        self._in.setdefault(dst, []).append((src, dst, props))

    def out_edges(self, nid, data=False):
        e = self._out.get(nid, [])
        return e if data else [(s, t) for s, t, _ in e]

    def in_edges(self, nid, data=False):
        e = self._in.get(nid, [])
        return e if data else [(s, t) for s, t, _ in e]


class _Engine:
    def __init__(self, nodes):
        self.graph = _Graph(nodes)

    def add_node(self, nid, node_type, properties=None, ephemeral=False):
        self.graph.add_node(nid, {**(properties or {}), "type": node_type})

    def link_nodes(self, src, dst, rel_type, properties=None, ephemeral=False):
        self.graph.add_edge(src, dst, properties or {})


def _nodes():
    return {
        "f1": {
            "type": "capability",
            "name": "exec-rag planner",
            "concept_ids": ["KG-2.12"],
            "research_sources": ["arxiv:pyrag"],
            "status": "open",
        },
        "f2": {
            "type": "capability",
            "name": "social swarm",
            "concept_ids": ["ORCH-1.32"],
            "research_sources": ["arxiv:mass"],
            "status": "open",
        },
    }


def test_hydrate_feature_pulls_neighborhood():
    engine = _Engine(_nodes())
    engine.link_nodes(
        "f1", "f2", "HAS_SYNERGY_WITH", properties={"_rel": "HAS_SYNERGY_WITH"}
    )
    nb = hydrate_feature(engine, "f1")
    assert nb["name"] == "exec-rag planner"
    assert nb["pillar"] == "KG"
    assert nb["sources"] == ["arxiv:pyrag"]
    assert nb["synergies"] == ["f2"]


def test_default_template_is_grounded():
    # The deterministic fallback (no LLM) is grounded in the feature's neighborhood.
    engine = _Engine(_nodes())
    plan = _default_synth(hydrate_feature(engine, "f1"))
    assert "exec-rag planner" in plan["title"]
    assert "arxiv:pyrag" in plan["body"]  # grounded in the source
    assert "KG-2.12" in plan["body"]


def test_synthesize_persists_proposal():
    engine = _Engine(_nodes())
    # inject a synth_fn so the test does not depend on LLM availability
    proposal = synthesize_plan_for_feature(
        engine, "f1", synth_fn=lambda nb: {"title": "T", "body": "B"}
    )
    assert proposal.plan_id == "plan:f1"
    # persisted: plan node + ADDRESSED_BY edge + feature flipped to proposed
    data = dict(engine.graph.nodes(data=True))
    assert data["plan:f1"]["type"] == "sdd_plan"
    assert data["f1"]["status"] == "proposed"
    addressed = [
        e for e in engine.graph._out["f1"] if e[2].get("_rel") == "ADDRESSED_BY"
    ]
    assert addressed and addressed[0][1] == "plan:f1"


def test_injected_synth_fn_is_used():
    engine = _Engine(_nodes())
    proposal = synthesize_plan_for_feature(
        engine, "f1", synth_fn=lambda nb: {"title": "Custom", "body": "custom body"}
    )
    assert proposal.title == "Custom" and proposal.body == "custom body"


def test_synthesize_plans_top_n_and_idempotent():
    engine = _Engine(_nodes())
    first = synthesize_plans(engine, top_n=5)
    assert {p.feature_id for p in first} == {"f1", "f2"}
    # second pass: both are now 'proposed' → skipped (idempotent)
    second = synthesize_plans(engine, top_n=5)
    assert second == []


def test_synthesize_plans_respects_top_n():
    engine = _Engine(_nodes())
    out = synthesize_plans(engine, top_n=1)
    assert len(out) == 1
