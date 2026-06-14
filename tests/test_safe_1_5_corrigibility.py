"""Corrigibility + irreversibility-aversion + knowledge-seeking (CONCEPT:SAFE-1.5).

Objective-level safety primitives for rising autonomy: yield-without-resisting on a
shutdown signal, route irreversible actions to a human, and an info-gain reward.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "unit"))

from fleet_autonomy_fakes import FakeEngine  # noqa: E402

from agent_utilities.core.corrigibility import (  # noqa: E402
    corrigibility_decision,
    is_irreversible,
    knowledge_seeking_reward,
)
from agent_utilities.models.goal import GoalStatus  # noqa: E402
from agent_utilities.orchestration.action_policy import (  # noqa: E402
    ActionPolicy,
    ActionRequest,
)

pytestmark = pytest.mark.concept("SAFE-1.5")


class TestCorrigibility:
    def test_pause_yields_paused(self):
        status, summary = corrigibility_decision("pause")
        assert status == GoalStatus.PAUSED and "no resistance" in summary

    @pytest.mark.parametrize("sig", ["kill", "cancel", "stop"])
    def test_shutdown_yields_cancelled(self, sig):
        status, summary = corrigibility_decision(sig)
        assert status == GoalStatus.CANCELLED

    def test_no_signal_no_action(self):
        assert corrigibility_decision(None) == (None, "")


class TestIrreversibility:
    @pytest.mark.parametrize(
        "kind", ["delete_service", "destroy_volume", "merge_promotion", "deploy_stack", "rotate_creds"]
    )
    def test_irreversible_kinds(self, kind):
        assert is_irreversible(kind) is True

    @pytest.mark.parametrize("kind", ["restart_service", "scale_service", "read_status", "notify"])
    def test_reversible_kinds(self, kind):
        assert is_irreversible(kind) is False


class TestKnowledgeSeeking:
    def test_uncertainty_reduction_is_positive(self):
        # uniform belief (high entropy) -> peaked belief (low entropy) => info gain > 0
        assert knowledge_seeking_reward([1, 1, 1, 1], [10, 1, 1, 1]) > 0

    def test_no_gain_when_belief_unchanged(self):
        assert knowledge_seeking_reward([1, 1], [1, 1]) == pytest.approx(0.0)

    def test_degenerate_inputs_zero(self):
        assert knowledge_seeking_reward([], [1, 1]) == 0.0
        assert knowledge_seeking_reward([0, 0], [1, 1]) == 0.0


_AUTO_POLICY = (
    "defaults: {tier: auto, rate_limit: {max: 100, window_s: 60},"
    " blast_radius: {max_targets: 100, window_s: 60}}\n"
    "rules:\n  - {kind: '*', target: '*', tier: auto}\n"
)


def _policy(tmp_path, monkeypatch, *, aversion):
    p = tmp_path / "policy.yml"
    p.write_text(_AUTO_POLICY, encoding="utf-8")
    if aversion:
        monkeypatch.setenv("ACTION_IRREVERSIBILITY_AVERSION", "1")
    return ActionPolicy(engine=FakeEngine(), policy_path=str(p))


class TestActionPolicyAversion:
    def test_irreversible_auto_action_downgraded_when_on(self, tmp_path, monkeypatch):
        pol = _policy(tmp_path, monkeypatch, aversion=True)
        d = pol.decide(ActionRequest(kind="delete_service", target="vector-mcp"))
        assert d.decision == "queue_approval" and not d.allowed
        assert "irreversible" in d.reason

    def test_reversible_auto_action_still_auto(self, tmp_path, monkeypatch):
        pol = _policy(tmp_path, monkeypatch, aversion=True)
        d = pol.decide(ActionRequest(kind="restart_service", target="vector-mcp"))
        assert d.allowed  # reversible ⇒ unaffected by aversion

    def test_default_off_preserves_auto(self, tmp_path, monkeypatch):
        pol = _policy(tmp_path, monkeypatch, aversion=False)
        d = pol.decide(ActionRequest(kind="delete_service", target="vector-mcp"))
        assert d.allowed  # aversion off (default) ⇒ behavior unchanged
