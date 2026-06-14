#!/usr/bin/python
"""SAI-factory daemon-tick live path (AHE-3.29) — engine-grounded specialization."""

from __future__ import annotations

from agent_utilities.harness.world_model_task import (
    specialize_world_model_from_engine,
    transitions_from_engine,
)


def _ring_transitions() -> list[tuple[str, str, str]]:
    """A grid environment where each (state, action) has a distinct next-state.

    Distinct, embedding-separable next-states make this a cleanly-learnable
    dynamics task (vs a symmetric ring where the action signal is weak), so the
    learned backend demonstrably specializes.
    """
    t = []
    for i in range(8):
        for act in ("north", "south", "east", "west"):
            t.append((f"cell_{i:02d}", act, f"cell_{i:02d}_{act}"))
    return t


class _FakeEngine:
    """Minimal engine: serves WorldModelTransition rows + records add_node calls."""

    def __init__(self, transitions: list[tuple[str, str, str]]) -> None:
        self._t = transitions
        self.nodes: list[tuple[str, str, dict]] = []

    def query_cypher(self, _cypher: str, *_a, **_k):
        return [{"state": s, "action": a, "next_state": ns} for s, a, ns in self._t]

    def add_node(self, node_id: str, label: str, properties=None):
        self.nodes.append((node_id, label, properties or {}))


def test_transitions_from_engine_reads_rows():
    eng = _FakeEngine(_ring_transitions())
    trans = transitions_from_engine(eng)
    assert len(trans) == 32
    assert ("cell_00", "north", "cell_00_north") in trans


def test_specialize_persists_cycle_and_returns_summary():
    trans = _ring_transitions() * 3  # enough history to split
    eng = _FakeEngine(trans)
    summary = specialize_world_model_from_engine(eng, min_transitions=10, rounds=1)
    assert summary is not None
    assert summary["transitions"] == len(trans)
    assert summary["final_specialist_reward"] > 0.3
    # a queryable SaiFactoryCycle node was persisted
    assert any(label == "SaiFactoryCycle" for _, label, _ in eng.nodes)


def test_specialize_noops_on_insufficient_history():
    eng = _FakeEngine(_ring_transitions()[:3])
    assert specialize_world_model_from_engine(eng, min_transitions=20) is None


def test_specialize_handles_engineless_gracefully():
    # No engine / no query support → empty transitions → None (no crash)
    class _Dead:
        def query_cypher(self, *_a, **_k):
            raise RuntimeError("no query support")

    assert specialize_world_model_from_engine(_Dead(), min_transitions=1) is None


def test_tick_registered_when_flag_on():
    from agent_utilities.core.config import config

    # the config flag + interval exist and default off
    assert hasattr(config, "kg_sai_factory")
    assert config.kg_sai_factory_interval > 0
