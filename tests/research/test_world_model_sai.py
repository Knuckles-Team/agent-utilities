#!/usr/bin/python
"""World-model SAI specialization track (KG-2.73) — learned backend + factory.

CPU-only: a deterministic toy environment exercises the learned latent-dynamics
backend (generalization), the WorldModelVerifier (held-out accuracy reward), and
the SaiFactoryController specializing a world-model config end-to-end.
"""

from __future__ import annotations

import json

from agent_utilities.harness.world_model_task import (
    WorldModelVerifier,
    build_world_model_task,
)
from agent_utilities.knowledge_graph.core.world_model import (
    LatentDynamicsModel,
    WorldModel,
)
from agent_utilities.knowledge_graph.research.sai_factory import SaiFactoryController

# Deterministic ring environment: 5 states, inc/dec move around the ring.
_STATES = [f"s{i}" for i in range(5)]


def _ring_transitions() -> list[tuple[str, str, str]]:
    t = []
    for i in range(5):
        t.append((f"s{i}", "inc", f"s{(i + 1) % 5}"))
        t.append((f"s{i}", "dec", f"s{(i - 1) % 5}"))
    return t


def test_latent_backend_learns_deterministic_dynamics():
    model = LatentDynamicsModel(dim=128, alpha=0.01, neighbors=1)
    for s, a, ns in _ring_transitions():
        model.observe(s, a, ns)
    model.fit()
    # predicts the correct next state for a trained pair
    preds = model.predict("s0", "inc", k=1)
    assert preds and preds[0][0] == "s1"


def test_world_model_latent_backend_routes_predict():
    wm = WorldModel(backend="latent", latent_dim=128, ridge_alpha=0.01)
    for s, a, ns in _ring_transitions():
        wm.observe(s, a, ns)
    step = wm.step("s2", "dec")
    assert step.next_state == "s1"


def test_symbolic_backend_still_default():
    wm = WorldModel()  # no backend kwarg → symbolic
    assert wm._latent is None
    for s, a, ns in _ring_transitions():
        wm.observe(s, a, ns)
    assert wm.step("s0", "inc").next_state == "s1"


def test_verifier_rewards_accuracy_above_majority_baseline():
    trans = _ring_transitions()
    verifier = WorldModelVerifier(train=trans, holdout=trans)
    good = verifier.verify(json.dumps({"dim": 128, "alpha": 0.01, "neighbors": 1}))
    assert good.passed is True
    assert good.reward > good.detail["majority_baseline"]
    # a degenerate tiny embedding cannot separate the states → low accuracy
    bad = verifier.verify(json.dumps({"dim": 1, "alpha": 100.0, "neighbors": 1}))
    assert bad.reward <= good.reward


def test_verifier_handles_bad_config():
    verifier = WorldModelVerifier(
        train=_ring_transitions(), holdout=_ring_transitions()
    )
    assert verifier.verify("not json").passed is False
    assert verifier.verify("[1,2,3]").passed is False


def test_factory_specializes_world_model_end_to_end():
    trans = _ring_transitions()
    task = build_world_model_task(train=trans, holdout=trans, target_tau=0.9)
    # the candidate IS the config scaffold (no LLM needed for this track)
    controller = SaiFactoryController(task, generate_fn=lambda scaffold: scaffold)
    result = controller.run(rounds=1)
    assert result.specialist.reward > 0.5  # learned a useful forward model
    assert result.curve.reached(0.5) is True
    # the winning scaffold is one of the offered configs
    assert json.loads(result.specialist.scaffold)["dim"] in (16, 64, 128)
