"""Live-path: the evolution cycle records into decentralized memory (KG-2.82/AHE-3.33).

Wire-First proof — `AgenticEvolutionEngine` constructs a `DecentralizedMemory`
on init and `run_evolution_cycle` records each base's winners into ITS OWN
exploitation pool and feeds the cycle outcome back to its bandit.
"""

from unittest.mock import MagicMock

from agent_utilities.harness.agentic_evolution_engine import AgenticEvolutionEngine
from agent_utilities.harness.decentralized_memory import (
    DecentralizedMemory,
    MemoryPool,
)


def test_evolution_engine_wires_decentralized_memory():
    eng = AgenticEvolutionEngine(engine=MagicMock())
    eng._lazy_init()
    assert isinstance(eng._decentralized_memory, DecentralizedMemory)


def test_cycle_records_winners_and_router_into_decentralized_memory():
    eng = AgenticEvolutionEngine(engine=MagicMock())
    eng._lazy_init()

    vp = MagicMock()
    vp.tournament_select.return_value = ["v1", "v2"]
    vp.prune_losers.return_value = 0
    vp.population_health.return_value = {"spread": 0.5, "collapsed": False}
    eng._variant_pool = vp

    report = eng.run_evolution_cycle("base1", top_k=2)

    # The new behaviour fired on the live cycle path.
    assert report["winners"] == ["v1", "v2"]
    assert "decentralized_router" in report
    assert report["decentralized_router"]["arms"]["exploit"]["count"] == 1

    # base1's own exploitation pool now holds reusable trajectories.
    recs = eng._decentralized_memory.recall(
        "base1", "variant", pool=MemoryPool.EXPLOIT, top_k=5
    )
    assert recs
    assert any("v1" in r.content or "v2" in r.content for r in recs)


def test_collapse_rewards_exploration():
    eng = AgenticEvolutionEngine(engine=MagicMock())
    eng._lazy_init()
    vp = MagicMock()
    vp.tournament_select.return_value = ["v1"]
    vp.prune_losers.return_value = 0
    vp.population_health.return_value = {"spread": 0.0, "collapsed": True}
    eng._variant_pool = vp

    eng.run_evolution_cycle("base2", top_k=1)
    stats = eng._decentralized_memory.router_stats("base2")
    # A diversity collapse rewards the explore arm to diversify next cycle.
    assert stats["arms"]["explore"]["count"] == 1
    assert stats["arms"]["explore"]["mean"] == 1.0
