"""CONCEPT:AU-KG.memory.background-learning-engine — Background Learning Engine.

Covers relative-date resolution, defensive edit parsing, targeted ADD/UPDATE/DELETE application
as bi-temporal mutations (with a fake engine), bounded backoff, and the async sync barrier.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.memory.learning_engine import (
    BackgroundLearner,
    MemoryEdit,
    parse_memory_edits,
    resolve_relative_dates,
    with_backoff,
)

# ── pure helpers ──────────────────────────────────────────────────────────────


@pytest.mark.concept(id="AU-KG.memory.background-learning-engine")
def test_resolve_relative_dates_absolute():
    now = "2026-06-04T12:00:00+00:00"
    assert "2026-06-03" in resolve_relative_dates("I did it yesterday", now=now)
    assert "2026-06-04" in resolve_relative_dates("happening today", now=now)
    assert "2026-05-21" in resolve_relative_dates(
        "2 weeks ago", now=now
    )  # 14 days back
    # Vague recency is intentionally left as report-time context.
    assert resolve_relative_dates("recently moved", now=now) == "recently moved"


@pytest.mark.concept(id="AU-KG.memory.background-learning-engine")
def test_parse_memory_edits_actions_envelope_and_skips_bad_rows():
    raw = (
        'noise {"actions": [{"action": "ADD", "content": "User name is Sam"}, '
        '{"action": "BOGUS"}, {"action": "DELETE", "id": "x"}]} trailing'
    )
    edits = parse_memory_edits(raw)
    assert [e.action for e in edits] == ["ADD", "DELETE"]


@pytest.mark.concept(id="AU-KG.memory.background-learning-engine")
def test_parse_memory_edits_garbage_returns_empty():
    assert parse_memory_edits("no json here") == []


# ── apply_edits against a fake engine ───────────────────────────────────────────


class _FakeEngine:
    def __init__(self):
        self.nodes: dict[str, object] = {}

    def add_memory_node(self, node):
        self.nodes[node.id] = node

    def get_memory_node(self, mid):
        return self.nodes.get(mid)

    def update_memory_node(self, mid, node):
        self.nodes[mid] = node


@pytest.mark.concept(id="AU-KG.memory.background-learning-engine")
def test_apply_add_stamps_bitemporal_and_type():
    eng = _FakeEngine()
    learner = BackgroundLearner(eng)  # type: ignore[arg-type]
    counts = learner.apply_edits(
        [
            MemoryEdit(
                action="ADD",
                id="m1",
                content="User prefers a formal tone",
                memory_type="procedural",
                target_entity="global",
                event_time="2026-06-04T00:00:00+00:00",
            )
        ],
        now="2026-06-04T12:00:00+00:00",
    )
    assert counts["added"] == 1
    node = eng.nodes["m1"]
    assert node.memory_type == "procedural"  # type: ignore[attr-defined]
    assert node.target_entity == "global"  # type: ignore[attr-defined]
    assert node.event_time == "2026-06-04T00:00:00+00:00"  # type: ignore[attr-defined]
    assert node.storage_time == "2026-06-04T12:00:00+00:00"  # type: ignore[attr-defined]


@pytest.mark.concept(id="AU-KG.memory.background-learning-engine")
def test_apply_update_supersedes_content_and_stamps():
    eng = _FakeEngine()
    learner = BackgroundLearner(eng)  # type: ignore[arg-type]
    learner.apply_edits([MemoryEdit(action="ADD", id="m1", content="lives in Boston")])
    learner.apply_edits(
        [
            MemoryEdit(
                action="UPDATE",
                id="m1",
                content="lives in Denver",
                event_time="2026-06-04T00:00:00+00:00",
            )
        ],
        now="2026-06-04T12:00:00+00:00",
    )
    assert eng.nodes["m1"].content == "lives in Denver"  # type: ignore[attr-defined]
    assert eng.nodes["m1"].event_time == "2026-06-04T00:00:00+00:00"  # type: ignore[attr-defined]


@pytest.mark.concept(id="AU-KG.memory.background-learning-engine")
def test_apply_delete_is_soft_and_preserves_node():
    eng = _FakeEngine()
    learner = BackgroundLearner(eng)  # type: ignore[arg-type]
    learner.apply_edits([MemoryEdit(action="ADD", id="m1", content="ephemeral")])
    counts = learner.apply_edits(
        [MemoryEdit(action="DELETE", id="m1")], now="2026-06-04T12:00:00+00:00"
    )
    assert counts["deleted"] == 1
    # Soft delete: node still exists, marked REMOVED with a closed validity interval.
    assert eng.nodes["m1"].status == "REMOVED"  # type: ignore[attr-defined]
    assert eng.nodes["m1"].valid_to == "2026-06-04T12:00:00+00:00"  # type: ignore[attr-defined]


@pytest.mark.concept(id="AU-KG.memory.background-learning-engine")
def test_apply_update_missing_node_is_skipped():
    eng = _FakeEngine()
    learner = BackgroundLearner(eng)  # type: ignore[arg-type]
    counts = learner.apply_edits([MemoryEdit(action="UPDATE", id="ghost", content="x")])
    assert counts == {"added": 0, "updated": 0, "deleted": 0, "skipped": 1, "gated": 0}


# ── async: backoff + sync barrier ───────────────────────────────────────────────


@pytest.mark.concept(id="AU-KG.memory.background-learning-engine")
@pytest.mark.asyncio
async def test_with_backoff_retries_then_succeeds():
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient")
        return "ok"

    result = await with_backoff(flaky, initial=0.001, max_delay=0.002, max_attempts=5)
    assert result == "ok"
    assert calls["n"] == 3


@pytest.mark.concept(id="AU-KG.memory.background-learning-engine")
@pytest.mark.asyncio
async def test_schedule_and_await_pending_drains_tasks():
    eng = _FakeEngine()
    learner = BackgroundLearner(eng, concurrency=2)  # type: ignore[arg-type]
    learner.schedule([MemoryEdit(action="ADD", id="a", content="fact A")])
    learner.schedule([MemoryEdit(action="ADD", id="b", content="fact B")])
    await learner.await_pending()
    assert set(eng.nodes) == {"a", "b"}
