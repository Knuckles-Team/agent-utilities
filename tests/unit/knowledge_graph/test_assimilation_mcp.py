#!/usr/bin/python
"""Assimilation MCP action + public pass entrypoint (VU-9).

CONCEPT:KG-2.7
"""

from pathlib import Path

import pytest

from agent_utilities.knowledge_graph.research.loop_controller import run_assimilation_pass

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
        self.backend = None

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


def test_run_assimilation_pass_without_synthesis():
    rep = run_assimilation_pass(_Engine(_nodes()))
    assert rep["skipped"] is False
    assert rep["open_gaps"] == 2
    assert "proposed_plans" not in rep


def test_run_assimilation_pass_with_synthesis():
    engine = _Engine(_nodes())
    rep = run_assimilation_pass(engine, synthesize=True, top_n=5)
    assert {p["feature_id"] for p in rep["proposed_plans"]} == {"f1", "f2"}
    # features flipped to proposed (idempotent next pass)
    assert dict(engine.graph.nodes(data=True))["f1"]["status"] == "proposed"


def test_run_assimilation_pass_idempotent_and_force():
    engine = _Engine(_nodes())
    run_assimilation_pass(engine)
    assert run_assimilation_pass(engine)["skipped"] is True
    assert run_assimilation_pass(engine, force=True)["skipped"] is False


def test_mcp_assimilate_action_is_wired():
    # Regression guard: the graph_orchestrate tool routes an 'assimilate' action.
    src = Path("agent_utilities/mcp/tools/analysis_tools.py").read_text(encoding="utf-8")
    assert 'action == "assimilate"' in src
    assert "run_assimilation_pass" in src
