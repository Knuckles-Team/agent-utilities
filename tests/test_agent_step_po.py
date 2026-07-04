#!/usr/bin/python
"""Agent-Step Policy Optimization (ARPO) — CONCEPT:AHE-3.15.

Covers the entropy signal, the branch gate, the capability reward-EMA step-credit
write-back, and the live router path (SubagentLifecyclePolicy branches on a
high-entropy step).
"""

import pytest

from agent_utilities.graph.agent_step_po import (
    should_branch,
    step_entropy,
    write_back_step_credit,
)
from agent_utilities.graph.reward_decomposition import RewardDecomposer
from agent_utilities.graph.routing.strategies.policy import SubagentLifecyclePolicy
from agent_utilities.knowledge_graph.retrieval.capability_index import CapabilityIndex
from agent_utilities.numeric import xp as np

pytestmark = pytest.mark.concept("AHE-3.15")


def test_step_entropy_low_when_decisive_high_when_uncertain():
    assert step_entropy([5.0]) == 0.0  # single candidate → no uncertainty
    decisive = step_entropy([10.0, 0.0, 0.0])  # one clear winner
    uncertain = step_entropy([1.0, 1.0, 1.0])  # flat → max entropy
    assert uncertain == pytest.approx(1.0, abs=1e-6)
    assert decisive < 0.5


def test_branch_gate_respects_threshold_and_bound():
    assert should_branch(0.8, threshold=0.6) is True
    assert should_branch(0.5, threshold=0.6) is False
    # bounded: at/over max_branches it stops branching even when uncertain
    assert should_branch(0.9, threshold=0.6, branch_count=3, max_branches=3) is False


def test_step_credit_writes_back_into_capability_ema():
    idx = CapabilityIndex(dim=3)
    idx.add("tool_a", np.array([1.0, 0.0, 0.0], dtype=np.float32), ["x"])
    idx.add("tool_b", np.array([0.0, 1.0, 0.0], dtype=np.float32), ["x"])
    before_a = idx.reward_of("tool_a")
    # tool_a got a strongly positive step advantage, tool_b a negative one
    n = write_back_step_credit(idx, ["tool_a", "tool_b"], [3.0, -3.0])
    assert n == 2
    assert idx.reward_of("tool_a") > before_a  # positive advantage raised its EMA
    assert idx.reward_of("tool_b") < before_a  # negative advantage lowered it


def test_step_advantages_feed_arpo_writeback_live():
    # The RewardDecomposer.step_advantages output is exactly what ARPO writes back.
    rd = RewardDecomposer()
    rd.decompose(
        "t1",
        steps=[
            {"action": "good", "success": True},
            {"action": "bad", "success": False},
        ],
        goal_achieved=False,
    )
    adv = rd.step_advantages()
    assert len(adv) == 2
    assert adv[0] > adv[1]  # the correct step out-advantages the incorrect one
    # entropy-progress reweighting (EP-GRPO consumer) accepted on the same call
    adv_w = rd.step_advantages(step_entropies=[1.0, 0.2])
    assert len(adv_w) == 2


def test_router_branches_on_high_entropy_step_live_path():
    policy = SubagentLifecyclePolicy()
    # moderate complexity that would normally stay inline...
    assert policy.determine_route({"task_complexity": 3}) == "inline_tool"
    # ...but a high-entropy decision step escalates it to a branched fan-out (ARPO).
    assert (
        policy.determine_route({"task_complexity": 3, "step_entropy": 0.85})
        == "fan_out"
    )
    # bound respected: once branch_count hits the cap, it stops branching
    assert (
        policy.determine_route(
            {"task_complexity": 3, "step_entropy": 0.85, "branch_count": 3}
        )
        == "inline_tool"
    )
