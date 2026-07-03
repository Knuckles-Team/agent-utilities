"""Live-path: the evolution engine runs self-guided self-play (AHE-3.37).

Wire-First proof — AgenticEvolutionEngine builds a SelfGuidedSelfPlay on init and
run_evolution_cycle(task_text=...) runs a guided curriculum, surfacing its rates.
"""

from unittest.mock import MagicMock

from agent_utilities.harness.agentic_evolution_engine import AgenticEvolutionEngine
from agent_utilities.harness.self_guided_play import SelfGuidedSelfPlay


def test_engine_wires_self_play():
    eng = AgenticEvolutionEngine(engine=MagicMock())
    eng._lazy_init()
    assert isinstance(eng._self_play, SelfGuidedSelfPlay)


def test_cycle_with_task_runs_self_play():
    eng = AgenticEvolutionEngine(engine=MagicMock())
    eng._lazy_init()
    vp = MagicMock()
    vp.tournament_select.return_value = ["v1"]
    vp.prune_losers.return_value = 0
    vp.population_health.return_value = {"spread": 0.5, "collapsed": False}
    eng._variant_pool = vp
    # No skill detector so only the self-play branch exercises task_text.
    eng._skill_detector = None

    report = eng.run_evolution_cycle(
        "base1", task_text="optimize a sorting routine", top_k=1
    )
    assert "self_play" in report
    sp = report["self_play"]
    assert 0.0 <= sp["accept_rate"] <= 1.0
    assert 0.0 <= sp["solve_rate"] <= 1.0
    assert isinstance(sp["plateaued"], bool)
    assert sp["rounds"] >= 1
