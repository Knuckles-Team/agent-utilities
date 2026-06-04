"""Plan 03 Step 3: policy-driven routing migrated from the deleted
graph/policy_driven_router.py into the routing strategy package.
"""

from __future__ import annotations

import importlib
import types

import pytest

from agent_utilities.graph.routing.strategies.policy import (
    LearnedAgentPolicy,
    PolicyDrivenRouter,
    PolicyStrategy,
    SubagentLifecyclePolicy,
    SwarmPresetPolicy,
)


def test_old_module_is_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("agent_utilities.graph.policy_driven_router")


def test_swarm_preset_policy():
    p = SwarmPresetPolicy({"transitions": {"START": "design", "design": "build"}})
    assert p.determine_route({"current_node": "START"}) == "design"
    assert p.determine_route({"current_node": "build"}) == "END"  # no transition


def test_learned_agent_policy():
    class Backend:
        def predict_best_route(self, emb):
            return "specialist_x"

    p = LearnedAgentPolicy(Backend())
    assert p.determine_route({"task_embedding": [0.1]}) == "specialist_x"
    assert p.determine_route({}) == "fallback_specialist"  # no embedding


def test_subagent_lifecycle_policy():
    p = SubagentLifecyclePolicy()
    assert p.determine_route({"task_complexity": 9}) == "spawn_team"
    assert p.determine_route({"task_complexity": 5}) == "fan_out"
    assert p.determine_route({"task_complexity": 1}) == "inline_tool"


def test_policy_driven_router_hot_swap():
    router = PolicyDrivenRouter(SubagentLifecyclePolicy())
    assert router.route({"task_complexity": 9}) == "spawn_team"
    router.set_policy(SwarmPresetPolicy({"transitions": {"START": "A"}}))
    assert router.route({"current_node": "START"}) == "A"


@pytest.mark.asyncio
async def test_policy_strategy_adapts_to_async():
    strat = PolicyStrategy(SubagentLifecyclePolicy())
    ctx = types.SimpleNamespace(execution_context={"task_complexity": 9})
    assert await strat.decide(ctx) == "spawn_team"
