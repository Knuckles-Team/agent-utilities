"""Durable inbound inbox + retry — nothing goes unanswered (CONCEPT:ECO-4.83)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from agent_utilities.messaging.inbox import (
    mark_answered,
    pending_unanswered,
    record_inbound,
    retry_unanswered,
)


class _FakeEngine:
    def __init__(self, pending: list[dict] | None = None) -> None:
        self.nodes: dict[str, dict] = {}
        self._pending = pending or []

    def add_node(self, node_id, node_type, properties=None):  # MERGE/upsert semantics
        self.nodes.setdefault(node_id, {}).update(properties or {})

    def query_cypher(self, cypher, params=None):
        return self._pending if "InboundMessage" in cypher else []


def _old_ts(seconds_ago: int) -> str:
    return (datetime.now(UTC) - timedelta(seconds=seconds_ago)).isoformat()


def test_record_inbound_persists_pending_idempotently():
    eng = _FakeEngine()
    iid = record_inbound(
        eng,
        platform="telegram",
        channel_id="42",
        message_id="1",
        text="hi",
        session="s",
    )
    assert iid and iid.startswith("inbound:telegram:42:")
    assert eng.nodes[iid]["status"] == "pending" and eng.nodes[iid]["text"] == "hi"
    # same content → same id (idempotent)
    iid2 = record_inbound(
        eng,
        platform="telegram",
        channel_id="42",
        message_id="1",
        text="hi",
        session="s",
    )
    assert iid2 == iid


def test_mark_answered_sets_status():
    eng = _FakeEngine()
    iid = record_inbound(
        eng, platform="t", channel_id="c", message_id="1", text="hi", session="s"
    )
    mark_answered(eng, iid)
    assert eng.nodes[iid]["status"] == "answered"


def test_pending_unanswered_honors_grace_window():
    eng = _FakeEngine(
        pending=[
            {"id": "a", "received_at": _old_ts(1000), "attempts": 0},  # old → eligible
            {
                "id": "b",
                "received_at": _old_ts(5),
                "attempts": 0,
            },  # fresh → skip (grace)
        ]
    )
    ids = {m["id"] for m in pending_unanswered(eng)}
    assert ids == {"a"}


@pytest.mark.asyncio
async def test_retry_unanswered_answers_then_bumps_then_dead_letters():
    eng = _FakeEngine(
        pending=[{"id": "x", "received_at": _old_ts(1000), "attempts": 0}]
    )
    eng.nodes["x"] = {"status": "pending", "attempts": 0}
    sent: list[dict] = []

    async def ok_send(m):
        sent.append(m)
        return True

    assert await retry_unanswered(eng, ok_send) == 1
    assert len(sent) == 1 and eng.nodes["x"]["status"] == "answered"

    # failure → bumps attempts, stays pending
    eng2 = _FakeEngine(
        pending=[{"id": "y", "received_at": _old_ts(1000), "attempts": 2}]
    )
    eng2.nodes["y"] = {"status": "pending", "attempts": 2}

    async def fail_send(m):
        return False

    assert await retry_unanswered(eng2, fail_send) == 0
    assert eng2.nodes["y"]["attempts"] == 3

    # at the cap → dead_letter, no send
    eng3 = _FakeEngine(
        pending=[{"id": "z", "received_at": _old_ts(1000), "attempts": 4}]
    )
    eng3.nodes["z"] = {"status": "pending", "attempts": 4}
    calls = []

    async def track(m):
        calls.append(m)
        return True

    await retry_unanswered(eng3, track)
    assert eng3.nodes["z"]["status"] == "dead_letter" and calls == []
