"""Live-path (Wire-First) tests for the final assimilation phases.

- KG-2.92 MLEvolve graph-search code evolution — reachable via graph_analyze
  action='evolve_code' (MCP + REST twin /graph/analyze) and the engine method.
- ORCH-1.56 Fast-Slow controller — run_evolution_cycle observes a trace and runs
  fast/slow steps.
- ORCH-1.55 eval-set optimizer — TraceDistiller.distill grows the eval set from
  each round's failures (compounding IP).
"""

from unittest.mock import MagicMock

import pytest

from agent_utilities.harness.agentic_evolution_engine import AgenticEvolutionEngine
from agent_utilities.harness.continuous_evaluation_engine import (
    DistillationConfig,
    TraceDistiller,
)


def test_graph_search_evolution_engine_method():
    """KG-2.92 — the evolution engine evolves a solution by graph search."""
    eng = AgenticEvolutionEngine(engine=MagicMock())
    out = eng.evolve_via_graph_search("optimize a matrix multiply", num_steps=8)
    assert "best_metric" in out
    assert "branch_id" in out
    assert "stage" in out


def test_fast_slow_controller_runs_in_cycle():
    """ORCH-1.56 — the cycle observes a trace + runs the fast loop."""
    eng = AgenticEvolutionEngine(engine=MagicMock())
    eng._lazy_init()
    assert eng._fast_slow is not None
    vp = MagicMock()
    vp.tournament_select.return_value = ["v1"]
    vp.prune_losers.return_value = 0
    vp.population_health.return_value = {"spread": 0.4, "collapsed": False}
    eng._variant_pool = vp
    eng._skill_detector = None

    report = eng.run_evolution_cycle("base1", top_k=1)
    assert "fast_harness_id" in report


class _StubBackend:
    def __init__(self, traces):
        self._t = traces

    async def get_traces(self, round_id):
        return self._t

    async def store_evidence(self, *a, **k):
        return None


@pytest.mark.asyncio
async def test_distill_grows_eval_set_from_failures():
    """ORCH-1.55 — every distilled round's failures grow the compounding eval set."""
    traces = [
        {"id": "t0", "name": "task0", "score": 0.9, "error": ""},
        {"id": "t1", "name": "task1", "score": 0.1, "error": "boom"},
        {"id": "t2", "name": "task2", "score": 0.05, "error": "kaboom"},
    ]
    d = TraceDistiller(_StubBackend(traces), config=DistillationConfig())
    assert len(d.eval_set) == 0
    await d.distill("r1")
    # The two failures became eval cases (the eval set grows = compounding IP).
    assert len(d.eval_set) == 2
    assert all(c.source == "production_failure" for c in d.eval_set.cases())
