"""Action-conditioned world model (CONCEPT:AU-KG.compute.first-class-action-conditioned).

Wraps the graph's Markov transition kernel with a composite state|action key + a
reward table so an agent can roll a policy forward over predicted next-states and
rewards, and persist the imagined trajectory back to the graph.
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.knowledge_graph.core.world_model import Transition, WorldModel

pytestmark = pytest.mark.concept("AU-KG.compute.first-class-action-conditioned")


class WMEngine:
    def __init__(self, transitions=()):
        self.nodes: dict[str, dict] = {}
        self._transitions = list(transitions)

    def add_node(self, node_id, node_type, properties=None):
        self.nodes[node_id] = {"id": node_id, "type": node_type, **(properties or {})}

    def by_type(self, t):
        return [n for n in self.nodes.values() if n["type"] == t]

    def query_cypher(self, query, params=None):
        if "WorldModelTransition" in query:
            return [
                {"state": s, "action": a, "next_state": n, "reward": r}
                for (s, a, n, r) in self._transitions
            ]
        return []


class TestDynamics:
    def test_predict_and_expected_reward(self):
        wm = WorldModel()
        wm.observe("s1", "go", "s2", reward=1.0)
        wm.observe("s1", "go", "s2", reward=1.0)
        preds = wm.predict("s1", "go", k=2)
        assert preds[0][0] == "s2" and preds[0][1] == pytest.approx(1.0)
        assert wm.expected_reward("s1", "go") == 1.0

    def test_action_conditioning(self):
        # same state, different actions lead to different next-states
        wm = WorldModel()
        wm.observe("s", "left", "L")
        wm.observe("s", "right", "R")
        assert wm.step("s", "left").next_state == "L"
        assert wm.step("s", "right").next_state == "R"

    def test_unknown_transition_is_absorbing(self):
        t = WorldModel().step("ghost", "noop")
        assert t.next_state == "ghost" and t.probability == 0.0

    def test_rollout_and_expected_return(self):
        wm = WorldModel()
        wm.observe("a", "step", "b", reward=1.0)
        wm.observe("b", "step", "c", reward=2.0)
        traj = wm.rollout("a", policy=lambda _s: "step", horizon=2)
        assert [t.next_state for t in traj] == ["b", "c"]
        # discounted: 1.0 + 0.95*2.0
        assert wm.expected_return(traj) == pytest.approx(1.0 + 0.95 * 2.0)


class TestGroundingAndPersistence:
    def test_from_engine_grounds_from_history(self):
        eng = WMEngine(transitions=[("s1", "go", "s2", 1.0), ("s1", "go", "s2", 1.0)])
        wm = WorldModel.from_engine(eng)
        assert wm.step("s1", "go").next_state == "s2"
        assert wm.expected_reward("s1", "go") == 1.0

    def test_from_engine_empty_when_no_history(self):
        wm = WorldModel.from_engine(WMEngine())
        assert wm.step("x", "y").probability == 0.0

    def test_record_observation_persists_node(self):
        eng = WMEngine()
        wm = WorldModel(eng)
        wm.record_observation("s1", "go", "s2", reward=1.0)
        nodes = eng.by_type("WorldModelTransition")
        assert nodes and nodes[0]["next_state"] == "s2"
        # and it is learned in-memory too
        assert wm.step("s1", "go").next_state == "s2"

    def test_persist_rollout_writes_graph_node(self):
        eng = WMEngine()
        wm = WorldModel(eng)
        traj = [Transition("a", "go", "b", 1.0, 2.0)]
        rid = wm.persist_rollout(traj)
        assert rid is not None
        node = eng.by_type("WorldModelRollout")[0]
        assert node["horizon"] == 1
        assert json.loads(node["steps_json"])[0]["next_state"] == "b"

    def test_persist_without_engine_is_noop(self):
        assert WorldModel().persist_rollout([Transition("a", "go", "b")]) is None
