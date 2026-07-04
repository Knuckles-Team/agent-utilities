"""Persistent latent rollout memory (CONCEPT:AU-KG.compute.reuse-model-latent).

A learned world-model rollout carries the predicted next-state latent forward and
EMA-blends it each step (the persistent-latent-memory analogue of arXiv:2606.09828)
instead of discarding it and re-deriving from the bare next-state string. This keeps
an imagined trajectory on-manifold — measurably lower step-to-step latent drift —
while reproducing the legacy memoryless rollout exactly when memory is off.
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.knowledge_graph.core.world_model import (
    LatentDynamicsModel,
    WorldModel,
)
from agent_utilities.mcp import kg_server
from agent_utilities.numeric import xp as np

pytestmark = pytest.mark.concept("AU-KG.compute.reuse-model-latent")

CYCLE = ["room_alpha", "room_beta", "room_gamma", "room_delta"]


def _learned_model() -> WorldModel:
    wm = WorldModel(backend="latent")
    for _ in range(4):
        for i, s in enumerate(CYCLE):
            wm.observe(s, "go", CYCLE[(i + 1) % len(CYCLE)])
    return wm


class TestMechanism:
    def test_predict_latent_returns_carryable_latent(self):
        m = LatentDynamicsModel()
        m.observe("a", "go", "b")
        m.observe("b", "go", "c")
        results, latent = m.predict_latent("a", "go")
        assert results and latent is not None
        # predict() is now a thin delegate — identical ranking, latent dropped.
        assert [r[0] for r in results] == [r[0] for r in m.predict("a", "go")]

    def test_blending_pulls_latent_toward_prior(self):
        """With a prior + weight>0 the carried latent is closer to the prior than the
        raw prediction — the EMA smoothing that lowers step-to-step drift."""
        m = _learned_model()._latent
        assert m is not None
        _, y1 = m.predict_latent("room_alpha", "go")
        _, raw = m.predict_latent("room_beta", "go")
        _, blended = m.predict_latent("room_beta", "go", prior=y1, memory_weight=0.25)
        d_raw = float(np.linalg.norm(np.asarray(raw) - np.asarray(y1)))
        d_blended = float(np.linalg.norm(np.asarray(blended) - np.asarray(y1)))
        assert d_blended < d_raw


class TestRollout:
    def test_memory_reduces_drift(self):
        wm = _learned_model()
        base = wm.rollout("room_alpha", lambda _s: "go", 8, memory_weight=0.0)
        ours = wm.rollout("room_alpha", lambda _s: "go", 8, memory_weight=0.25)
        # Baseline (memory_weight=0) still records drift but does not blend.
        assert sum(t.drift for t in ours) < sum(t.drift for t in base)
        # Memory must not degrade which states the trajectory visits.
        assert [t.next_state for t in ours] == [t.next_state for t in base]

    def test_memory_off_reproduces_memoryless_path(self):
        """latent_memory=False is bit-identical to the legacy memoryless rollout."""
        wm = _learned_model()
        off = wm.rollout("room_alpha", lambda _s: "go", 6, latent_memory=False)
        w0 = wm.rollout("room_alpha", lambda _s: "go", 6, memory_weight=0.0)
        assert [t.next_state for t in off] == [t.next_state for t in w0]
        # No cache carried ⇒ no drift telemetry on the memory-off path.
        assert all(t.drift == 0.0 and t.latent_norm == 0.0 for t in off)

    def test_symbolic_backend_rollout_unchanged(self):
        """The symbolic backend has no latents — rollout stays memoryless + correct."""
        wm = WorldModel()  # default symbolic
        wm.observe("a", "step", "b")
        wm.observe("b", "step", "c")
        traj = wm.rollout("a", lambda _s: "step", 2)
        assert [t.next_state for t in traj] == ["b", "c"]
        assert all(t.drift == 0.0 for t in traj)


class _WMEngine:
    """Minimal engine exposing the transition history + add_node sink."""

    def __init__(self, transitions):
        self.nodes: dict[str, dict] = {}
        self._transitions = list(transitions)

    def add_node(self, node_id, node_type, properties=None):
        self.nodes[node_id] = {"id": node_id, "type": node_type, **(properties or {})}

    def by_type(self, t):
        return [n for n in self.nodes.values() if n["type"] == t]

    def query_cypher(self, query, params=None):
        if "WorldModelTransition" in query:
            return [
                {"state": s, "action": a, "next_state": n, "reward": None}
                for (s, a, n) in self._transitions
            ]
        return []


@pytest.mark.asyncio
async def test_world_model_rollout_live_path(monkeypatch):
    """graph_analyze action='world_model_rollout' rolls out + persists with drift."""
    transitions = []
    for _ in range(4):
        for i, s in enumerate(CYCLE):
            transitions.append((s, "advance", CYCLE[(i + 1) % len(CYCLE)]))
    engine = _WMEngine(transitions)
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)
    kg_server.ensure_tools_registered()

    res = await kg_server._execute_tool(
        "graph_analyze",
        action="world_model_rollout",
        query="room_alpha",
        top_k=6,
    )
    report = json.loads(res)
    assert report["status"] == "ok"
    assert report["horizon"] == 6
    assert len(report["steps"]) == 6
    assert all("drift" in s and "latent_norm" in s for s in report["steps"])
    assert "total_drift" in report
    # The imagined trajectory was persisted as a graph-native rollout node.
    assert report["rollout_id"] is not None
    assert engine.by_type("WorldModelRollout")
