"""Live-path tests for the round-4 assimilation: PauseRec + LLM-coder + FST trainer + neural reranker."""

import json
import time

import pytest

from agent_utilities.harness.agentic_evolution_engine import AgenticEvolutionEngine
from agent_utilities.harness.substrate_trainer import SubstrateTrainer
from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
from agent_utilities.knowledge_graph.retrieval.reasoning_reranker import (
    ReasoningAwareReranker,
)
from agent_utilities.mcp import kg_server
from unittest.mock import MagicMock


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.engine.get_active_backend",
        lambda: None,
    )
    g = GraphComputeEngine(backend_type="rust")
    for node in g.node_ids():
        g.remove_node(node)
    eng = IntelligenceGraphEngine(db_path=":memory:")
    now = time.time()
    for i, (cid, vec) in enumerate(
        [
            ("py1", [1.0, 0.1, 0.0, 0.0]),
            ("py2", [0.9, 0.2, 0.0, 0.0]),
            ("rs1", [0.0, 0.0, 1.0, 0.1]),
        ]
    ):
        eng.graph.add_node(
            cid, name=cid, description=f"item {cid} python tooling",
            embedding=vec, event_time=now,
        )
    return eng


def test_reasoning_reranker_default_scorer_functional():
    """KG-2.85 — the default reranker scorer scores without breaking (neural or lexical fallback)."""
    r = ReasoningAwareReranker()
    s = r.scorer.score("python packaging", "helps with python packaging")
    assert 0.0 <= s <= 1.0


def test_fst_substrate_trainer_wired_and_records_jobs():
    """ORCH-1.56/1.57 — repeated cycles for one base trigger a recorded GRPO training job."""
    eng = AgenticEvolutionEngine(engine=MagicMock())
    eng._lazy_init()
    assert isinstance(eng._substrate_trainer, SubstrateTrainer)
    vp = MagicMock()
    vp.tournament_select.return_value = ["v1"]
    vp.prune_losers.return_value = 0
    vp.population_health.return_value = {"spread": 0.5, "collapsed": False}
    eng._variant_pool = vp
    eng._skill_detector = None
    for _ in range(6):  # exceed the recurrence threshold (5) for the same base
        eng.run_evolution_cycle("baseX", top_k=1)
    jobs = eng._substrate_trainer.jobs()
    assert jobs, "expected a recorded GRPO training-job spec after recurring cycles"
    assert jobs[0].task_key == "baseX"
    assert jobs[0].method == "grpo"


@pytest.mark.asyncio
async def test_recommend_action_pauserec(engine, monkeypatch):
    """KG-2.93 — graph_analyze action='recommend' returns ranked recommendations."""
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)
    kg_server.ensure_tools_registered()
    res = await kg_server._execute_tool(
        "graph_analyze", action="recommend", query="python", top_k=3
    )
    recs = json.loads(res)
    assert isinstance(recs, list) and recs
    for r in recs:
        assert "item_id" in r and "semantic_id" in r and "score" in r


@pytest.mark.asyncio
async def test_evolve_code_action_offline_fallback(engine, monkeypatch):
    """KG-2.92 — evolve_code runs end-to-end (LLM coder falls back deterministically offline)."""
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)
    kg_server.ensure_tools_registered()
    res = await kg_server._execute_tool(
        "graph_analyze", action="evolve_code", query="sort a list efficiently", top_k=4
    )
    out = json.loads(res)
    assert "best_metric" in out and "branch_id" in out
