"""Unit tests for the agent-native memory lifecycle loop (CONCEPT:KG-2.307).

Drives the AU-side lifecycle policy with a MOCK engine client (records calls to the
engine memory primitives ``create_summary_node`` / ``consolidate`` / ``maintain``)
and a MOCK LLM (returns fixed summary text). No live engine or model is required.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from agent_utilities.knowledge_graph.memory.lifecycle import (
    MemoryLifecycle,
    MemoryLifecycleConfig,
    run_memory_lifecycle,
)


# ── Mock engine + backend ─────────────────────────────────────────────────────
class _MockBackend:
    """Minimal backend exposing the Cypher-subset ``execute`` the loop reads with."""

    def __init__(self, nodes: list[dict[str, Any]]) -> None:
        self._nodes = nodes
        self.execute_calls = 0

    def execute(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        self.execute_calls += 1
        return [{"id": n["id"], "data": dict(n)} for n in self._nodes]


class _MockEngine:
    """Mock engine exposing the EG-220/221/222 primitives as typed methods."""

    def __init__(self, nodes: list[dict[str, Any]]) -> None:
        self.backend = _MockBackend(nodes)
        self.create_summary_node_calls: list[dict[str, Any]] = []
        self.consolidate_calls: list[dict[str, Any]] = []
        self.maintain_calls: list[dict[str, Any]] = []

    def create_summary_node(self, **kwargs: Any) -> dict[str, Any]:
        self.create_summary_node_calls.append(kwargs)
        return {"id": "summary-1"}

    def consolidate(self, **kwargs: Any) -> dict[str, Any]:
        self.consolidate_calls.append(kwargs)
        return {"consolidated": len(kwargs.get("node_ids", []))}

    def maintain(self, **kwargs: Any) -> dict[str, Any]:
        self.maintain_calls.append(kwargs)
        return {"decayed": len(kwargs.get("node_ids", [])), "evicted": 0}


def _mock_llm(system_prompt: str, user_content: str) -> str:
    return "SEMANTIC SUMMARY: durable facts distilled from the episodes."


def _iso(hours_ago: float, now: datetime) -> str:
    return (now - timedelta(hours=hours_ago)).isoformat()


def _episodes_ripe_and_unripe(now: datetime) -> list[dict[str, Any]]:
    """A ripe cluster (3 old episodes on entity 'alpha') + noise the loop must skip.

    Noise: a too-small/too-new cluster ('beta'), a semantic memory, and an already
    consolidated episode — none should be chosen.
    """
    ripe = [
        {
            "id": f"ep-alpha-{i}",
            "memory_type": "episodic",
            "status": "ACTIVE",
            "target_entity": "alpha",
            "content": f"alpha episode {i}",
            "created_at": _iso(48 + i, now),
        }
        for i in range(3)
    ]
    unripe_new = [
        {
            "id": "ep-beta-0",
            "memory_type": "episodic",
            "status": "ACTIVE",
            "target_entity": "beta",
            "content": "beta episode fresh",
            "created_at": _iso(0.5, now),  # too new
        }
    ]
    semantic = [
        {
            "id": "sem-0",
            "memory_type": "semantic",
            "status": "ACTIVE",
            "target_entity": "alpha",
            "content": "already-semantic fact",
            "created_at": _iso(72, now),
        }
    ]
    already = [
        {
            "id": "ep-old-consolidated",
            "memory_type": "episodic",
            "status": "CONSOLIDATED",
            "target_entity": "gamma",
            "content": "old consolidated",
            "created_at": _iso(200, now),
        }
    ]
    return ripe + unripe_new + semantic + already


def _lifecycle(engine: _MockEngine, **cfg: Any) -> MemoryLifecycle:
    config = MemoryLifecycleConfig(
        enabled=True, min_cluster_size=3, min_cluster_age_hours=6.0, **cfg
    )
    return MemoryLifecycle(engine, config=config, llm=_mock_llm)


# ── select_consolidation_candidates ───────────────────────────────────────────
def test_kg_2_307_candidate_selection_picks_the_ripe_episodic_cluster() -> None:
    now = datetime.now(UTC)
    nodes = _episodes_ripe_and_unripe(now)
    engine = _MockEngine(nodes)
    life = _lifecycle(engine)

    cluster = life.select_consolidation_candidates(nodes, now)

    ids = {n["id"] for n in cluster}
    assert ids == {"ep-alpha-0", "ep-alpha-1", "ep-alpha-2"}
    # Must exclude semantic, too-new, and already-consolidated nodes.
    assert "sem-0" not in ids
    assert "ep-beta-0" not in ids
    assert "ep-old-consolidated" not in ids


def test_kg_2_307_candidate_selection_empty_when_no_cluster_ripe() -> None:
    now = datetime.now(UTC)
    # Only two episodes → below min_cluster_size of 3.
    nodes = [
        {
            "id": f"ep-{i}",
            "memory_type": "episodic",
            "status": "ACTIVE",
            "target_entity": "alpha",
            "content": f"e{i}",
            "created_at": _iso(48, now),
        }
        for i in range(2)
    ]
    life = _lifecycle(_MockEngine(nodes))
    assert life.select_consolidation_candidates(nodes, now) == []


# ── run_summarization → create_summary_node ───────────────────────────────────
def test_kg_2_307_summarization_calls_create_summary_node_with_generated_text() -> None:
    now = datetime.now(UTC)
    nodes = _episodes_ripe_and_unripe(now)
    engine = _MockEngine(nodes)
    life = _lifecycle(engine)
    cluster = life.select_consolidation_candidates(nodes, now)

    res = life.run_summarization(cluster)

    assert res["status"] == "ok"
    assert res["summary_id"] == "summary-1"
    assert len(engine.create_summary_node_calls) == 1
    call = engine.create_summary_node_calls[0]
    # AU produced the summary TEXT; the engine stores it.
    assert call["summary_text"] == _mock_llm("", "")
    assert set(call["child_ids"]) == {"ep-alpha-0", "ep-alpha-1", "ep-alpha-2"}
    assert call["metadata"]["concept"] == "KG-2.307"


def test_kg_2_307_summarization_skips_when_llm_returns_no_text() -> None:
    now = datetime.now(UTC)
    nodes = _episodes_ripe_and_unripe(now)
    engine = _MockEngine(nodes)
    life = MemoryLifecycle(
        engine,
        config=MemoryLifecycleConfig(enabled=True),
        llm=lambda s, u: "",  # LLM yields nothing → no empty summary node created
    )
    cluster = life.select_consolidation_candidates(nodes, now)

    res = life.run_summarization(cluster)

    assert res["status"] == "skipped"
    assert res["reason"] == "no_summary_text"
    assert engine.create_summary_node_calls == []


# ── run_consolidation → consolidate ───────────────────────────────────────────
def test_kg_2_307_consolidation_calls_engine_consolidate() -> None:
    now = datetime.now(UTC)
    nodes = _episodes_ripe_and_unripe(now)
    engine = _MockEngine(nodes)
    life = _lifecycle(engine)
    cluster = life.select_consolidation_candidates(nodes, now)

    res = life.run_consolidation(cluster, summary_id="summary-1")

    assert res["status"] == "ok"
    assert len(engine.consolidate_calls) == 1
    call = engine.consolidate_calls[0]
    assert set(call["node_ids"]) == {"ep-alpha-0", "ep-alpha-1", "ep-alpha-2"}
    assert call["summary_id"] == "summary-1"


# ── run_maintenance → maintain (localized: only the working set) ───────────────
def test_kg_2_307_maintenance_calls_maintain_on_the_working_set() -> None:
    now = datetime.now(UTC)
    nodes = _episodes_ripe_and_unripe(now)
    engine = _MockEngine(nodes)
    life = _lifecycle(engine)

    res = life.run_maintenance(nodes, now)

    assert res["status"] == "ok"
    assert len(engine.maintain_calls) == 1
    call = engine.maintain_calls[0]
    # Localized maintenance: scoped to exactly the working-set node ids.
    assert set(call["node_ids"]) == {n["id"] for n in nodes}
    assert call["half_life_secs"] == life.config.decay_half_life_secs
    assert "now" in call


# ── tick: end-to-end wiring + idempotency + safety ────────────────────────────
def test_kg_2_307_tick_runs_full_cycle_and_is_idempotent() -> None:
    now = datetime.now(UTC)
    nodes = _episodes_ripe_and_unripe(now)
    engine = _MockEngine(nodes)
    life = _lifecycle(engine)

    first = life.tick(now)
    assert first["status"] == "ok"
    assert first["summarized"] == 1
    assert first["consolidated"] == 1
    assert first["maintained"] == len(nodes)

    # Second tick over the SAME working set must NOT re-summarize/re-consolidate
    # the already-processed cluster (idempotent), though maintenance still runs.
    second = life.tick(now)
    assert second["status"] == "ok"
    assert second.get("skipped") == "already_processed"
    assert second["summarized"] == 0

    assert len(engine.create_summary_node_calls) == 1  # exactly once across 2 ticks
    assert len(engine.consolidate_calls) == 1
    assert len(engine.maintain_calls) == 2  # maintenance runs each tick


def test_kg_2_307_tick_disabled_by_default_is_a_noop() -> None:
    engine = _MockEngine(_episodes_ripe_and_unripe(datetime.now(UTC)))
    # Default config: enabled=False.
    life = MemoryLifecycle(engine, config=MemoryLifecycleConfig(), llm=_mock_llm)

    res = life.tick()

    assert res == {"status": "disabled"}
    assert engine.create_summary_node_calls == []
    assert engine.consolidate_calls == []
    assert engine.maintain_calls == []


def test_kg_2_307_tick_is_safe_when_primitives_unavailable() -> None:
    """A build where the engine has not wrapped the primitives degrades, never raises."""
    now = datetime.now(UTC)
    nodes = _episodes_ripe_and_unripe(now)

    class _BareEngine:
        def __init__(self, ns: list[dict[str, Any]]) -> None:
            self.backend = _MockBackend(ns)

    life = MemoryLifecycle(
        _BareEngine(nodes),
        config=MemoryLifecycleConfig(enabled=True),
        llm=_mock_llm,
    )
    res = life.tick(now)

    assert res["status"] == "ok"
    # Summary text was generated but the primitive was unavailable → skipped, safely.
    assert res["summarized"] == 0
    assert res["summarization"]["reason"].startswith("primitive_unavailable")


def test_kg_2_307_run_memory_lifecycle_entrypoint_respects_disabled_default() -> None:
    engine = _MockEngine(_episodes_ripe_and_unripe(datetime.now(UTC)))
    # No config → from_env(), which is disabled unless the env var is set.
    assert run_memory_lifecycle(engine) == {"status": "disabled"}
