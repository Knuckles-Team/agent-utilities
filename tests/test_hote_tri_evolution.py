"""HOTE tri-evolution tests — CONCEPT:AHE-3.50 (arXiv:2606.13710).

Verifies the joint co-evolution loop and the paper's central, falsifiable claim:
co-evolving proposer+solver+judge together is *indispensable* — freezing any one
module stalls the others, so joint strictly beats every solo ablation.
"""

import pytest

from agent_utilities.harness.hote_tri_evolution import (
    HybridTriEvolutionController,
    TriModulePolicies,
)


def test_joint_co_evolution_grows_solver_skill():
    ctrl = HybridTriEvolutionController()
    res = ctrl.evolve(rounds=20, mode="joint")
    assert res.mode == "joint"
    assert res.final_skill > 0.0
    # A co-evolved judge calibrates (reward → 1) and the proposer holds the band.
    assert res.rewards["judge"] > 0.95
    assert res.curve_metrics["learning_auc"] > 0.0
    assert len(res.records) == 20


def test_frozen_solver_modes_do_not_grow_skill():
    ctrl = HybridTriEvolutionController()
    # Evolving only the proposer or only the judge leaves the solver frozen.
    assert ctrl.evolve(rounds=20, mode="proposer").final_skill == 0.0
    assert ctrl.evolve(rounds=20, mode="judge").final_skill == 0.0


def test_solo_solver_underperforms_joint():
    ctrl = HybridTriEvolutionController()
    joint = ctrl.evolve(rounds=20, mode="solver")
    full = ctrl.evolve(rounds=20, mode="joint")
    # A solver evolving against a FROZEN proposer stalls as tasks become trivial.
    assert full.final_skill > joint.final_skill


def test_ablation_proves_co_evolution_indispensable():
    ctrl = HybridTriEvolutionController()
    ab = ctrl.run_ablation(rounds=20)
    assert ab["indispensable"] is True
    assert ab["joint"] > ab["best_solo"]
    assert ab["marginal_speed_gain"] > 0.0
    assert set(ab["final_skill"]) == {"joint", "proposer", "solver", "judge"}


def test_injected_modules_override_defaults():
    # A custom always-succeed solver ⇒ p=1 ⇒ zero informativeness ⇒ no learning,
    # which proves the frontier coupling drives skill (not raw success).
    ctrl = HybridTriEvolutionController(success_fn=lambda d, s: 1.0)
    res = ctrl.evolve(rounds=10, mode="joint")
    assert res.final_skill == 0.0


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        HybridTriEvolutionController().evolve(rounds=3, mode="bogus")


def test_policies_copy_is_isolated():
    p = TriModulePolicies(solver_skill=1.0)
    q = p.copy()
    q.solver_skill = 9.0
    assert p.solver_skill == 1.0
