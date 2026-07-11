"""Durable contextual-bandit outcome persistence (CONCEPT:AU-P1-3).

Verifies routing outcomes are written onto the engine's own node properties (a
Cypher ``SET`` via ``backend.execute``) instead of staying a local, process-only
EMA — so the learned preference survives a restart. The engine backend is a fake
in-memory Cypher executor; no real database involved.
"""

from __future__ import annotations

import types
from typing import Any

import pytest

from agent_utilities.knowledge_graph.retrieval.durable_outcome_store import (
    persist_capability_reward,
    read_capability_reward,
)


class _FakeCypherBackend:
    """A minimal in-memory stand-in for ``EpistemicGraphBackend.execute``.

    Understands exactly the two Cypher shapes ``durable_outcome_store`` issues:
    a ``RETURN`` read and a ``SET`` write, keyed by ``n.id = $id``.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def execute(self, query: str, params: dict[str, Any] | None = None):
        params = params or {}
        self.calls.append((query, params))
        nid = str(params.get("id"))
        if "SET" in query:
            node = self.nodes.setdefault(nid, {})
            if "$r" in query or "n.capability_reward = $r" in query:
                node["capability_reward"] = params.get("r")
            node["capability_reward_count"] = params.get("c")
            node["capability_reward_updated_at"] = params.get("ts")
            return []
        # RETURN query
        node = self.nodes.get(nid, {})
        return [
            {
                "reward": node.get("capability_reward"),
                "count": node.get("capability_reward_count"),
            }
        ]


def _engine(backend: Any) -> Any:
    return types.SimpleNamespace(backend=backend)


def test_first_outcome_persists_from_neutral_prior():
    backend = _FakeCypherBackend()
    engine = _engine(backend)

    updated = persist_capability_reward(engine, "tool:search", success=True)

    # EMA from neutral 0.5 toward 1.0 at alpha=0.3: 0.7*0.5 + 0.3*1.0 = 0.65
    assert updated == pytest.approx(0.65)
    assert backend.nodes["tool:search"]["capability_reward"] == pytest.approx(0.65)
    assert backend.nodes["tool:search"]["capability_reward_count"] == 1


def test_reward_persists_across_a_fresh_process_read():
    """The defining AU-P1-3 assertion: the reward is NOT just a local EMA — a
    brand-new read (simulating a restarted process) sees the durably written value."""
    backend = _FakeCypherBackend()
    engine = _engine(backend)

    persist_capability_reward(engine, "tool:search", success=True)
    persist_capability_reward(engine, "tool:search", success=True)

    # A fresh call to read_capability_reward (no in-process state at all) recovers it.
    recovered = read_capability_reward(engine, "tool:search")
    assert recovered is not None
    assert recovered > 0.5


def test_repeated_outcomes_compound_the_durable_ema():
    backend = _FakeCypherBackend()
    engine = _engine(backend)

    r1 = persist_capability_reward(engine, "tool:x", success=True, alpha=0.5)
    r2 = persist_capability_reward(engine, "tool:x", success=True, alpha=0.5)
    r3 = persist_capability_reward(engine, "tool:x", success=False, alpha=0.5)
    assert r1 is not None and r2 is not None and r3 is not None

    assert r1 == pytest.approx(0.75)
    assert r2 == pytest.approx(0.875)
    assert r3 < r2  # a failure pulls the durable EMA back down
    assert backend.nodes["tool:x"]["capability_reward_count"] == 3


def test_explicit_reward_value_is_honoured_and_clamped():
    backend = _FakeCypherBackend()
    engine = _engine(backend)

    updated = persist_capability_reward(engine, "tool:y", reward=5.0)  # out-of-range
    assert updated is not None
    assert updated <= 1.0

    updated2 = persist_capability_reward(engine, "tool:y", reward=-5.0)
    assert updated2 is not None
    assert updated2 >= 0.0


def test_no_backend_reachable_returns_none_not_raise():
    engine = _engine(None)
    assert persist_capability_reward(engine, "tool:z", success=True) is None
    assert read_capability_reward(engine, "tool:z") is None


def test_missing_success_and_reward_raises_value_error():
    backend = _FakeCypherBackend()
    engine = _engine(backend)
    with pytest.raises(ValueError):
        persist_capability_reward(engine, "tool:missing-args")


def test_backend_execute_failure_degrades_gracefully():
    class _BrokenBackend:
        def execute(self, *a, **k):
            raise RuntimeError("db unreachable")

    engine = _engine(_BrokenBackend())
    # Read failure -> None, not a raise.
    assert read_capability_reward(engine, "tool:a") is None
    # Write failure -> still returns the computed value (best-effort durability).
    updated = persist_capability_reward(engine, "tool:a", success=True)
    assert updated == pytest.approx(0.65)
