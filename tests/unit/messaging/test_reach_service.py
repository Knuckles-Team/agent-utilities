"""Unit tests for the messaging reach service + planner handler (CONCEPT:AU-ECO.messaging.messaging-reach-service-governed–4.52).

Covers governed sends, last-active channel routing + default fallback, the awaited-reply
bridge, and that the inbound planner handler is no longer the canned-acknowledgment stub.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agent_utilities.messaging.models import (
    EventType,
    InboundEvent,
    SendResult,
)
from agent_utilities.messaging.service import MessagingService


class _FakeBackend:
    """Minimal connected backend that records sends and supports reply_to."""

    def __init__(self) -> None:
        self.id = "telegram"
        self._connected = True
        self.sent: list[tuple[str, str]] = []
        self.replies: list[tuple[str, str, str]] = []

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def send_message(self, channel_id: str, text: str, **_: Any) -> SendResult:
        self.sent.append((channel_id, text))
        return SendResult(
            success=True, message_id="m1", platform="telegram", channel_id=channel_id
        )

    async def reply_to(self, channel_id: str, message_id: str, text: str, **_: Any):
        self.replies.append((channel_id, message_id, text))
        return SendResult(success=True, platform="telegram", channel_id=channel_id)


class _FakeEngine:
    """In-memory engine stub for add_node / query_cypher / recall_memory."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}

    def add_node(self, node_id: str, _label: str, properties: dict[str, Any]) -> None:
        self.nodes[node_id] = dict(properties)

    def query_cypher(self, _query: str, params: dict[str, Any]):
        node = self.nodes.get(params.get("id", ""))
        return [{"p": {"properties": node}}] if node else []

    def recall_memory(self, **_: Any):
        return []

    def store_memory(self, **_: Any):
        return "mem-1"


@pytest.fixture()
def svc(monkeypatch: pytest.MonkeyPatch) -> MessagingService:
    """Fresh service bound to a fake engine + backend, with the policy gate allowing."""
    MessagingService._instance = None
    service = MessagingService.instance(_FakeEngine())
    backend = _FakeBackend()
    service.register_connected(backend)

    async def _get_backend(_platform: str):
        return backend

    monkeypatch.setattr(service, "get_backend", _get_backend)
    # Isolate routing from the policy engine — gate allows.
    monkeypatch.setattr(
        service, "_gate", lambda *a, **k: type("D", (), {"allowed": True})()
    )
    return service


@pytest.mark.asyncio
async def test_reach_user_uses_last_active_channel(svc: MessagingService) -> None:
    # No pref yet, no default → cannot route.
    res = await svc.reach_user("hi", user_id="u1")
    assert not res.success

    # An inbound event records the last-active channel (ECO-4.49)…
    svc.record_inbound(
        InboundEvent(
            event_type=EventType.MESSAGE,
            platform="telegram",
            channel_id="555",
            user_id="u1",
        )
    )
    # …so reach_user now routes there.
    res = await svc.reach_user("hello", user_id="u1")
    assert res.success
    backend = await svc.get_backend("telegram")
    assert backend.sent[-1] == ("555", "hello")


@pytest.mark.asyncio
async def test_reach_user_falls_back_to_default_channel(
    svc: MessagingService, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MESSAGING_DEFAULT_PLATFORM", "telegram")
    monkeypatch.setenv("MESSAGING_DEFAULT_CHANNEL", "999")
    res = await svc.reach_user("yo", user_id="unknown-user")
    assert res.success
    backend = await svc.get_backend("telegram")
    assert backend.sent[-1] == ("999", "yo")


@pytest.mark.asyncio
async def test_send_blocked_when_policy_denies(
    svc: MessagingService, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        svc,
        "_gate",
        lambda *a, **k: type(
            "D", (), {"allowed": False, "decision": "deny", "reason": "x"}
        )(),
    )
    res = await svc.send("telegram", "1", "blocked?")
    assert not res.success
    assert "policy" in res.error


@pytest.mark.asyncio
async def test_deliver_reply_resolves_awaited_reply(svc: MessagingService) -> None:
    svc.record_inbound(
        InboundEvent(
            event_type=EventType.MESSAGE,
            platform="telegram",
            channel_id="42",
            user_id="u1",
        )
    )

    async def _answer_later() -> None:
        await asyncio.sleep(0.05)
        assert svc.deliver_reply("telegram", "42", "the answer")

    asyncio.create_task(_answer_later())
    reply = await svc.reach_user_and_wait("question?", user_id="u1", timeout=5.0)
    assert reply == "the answer"


@pytest.mark.asyncio
async def test_planner_handler_is_not_the_stub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The inbound handler must run a real path, not emit the old canned string."""
    from agent_utilities.messaging.router import create_planner_handler

    MessagingService._instance = None
    service = MessagingService.instance(_FakeEngine())
    backend = _FakeBackend()
    service.register_connected(backend)
    monkeypatch.setattr(
        service, "_gate", lambda *a, **k: type("D", (), {"allowed": True})()
    )

    # A message that is the answer to an awaited question must be delivered to the
    # waiting loop and NOT trigger a planner reply on the backend.
    loop = asyncio.get_running_loop()
    fut: asyncio.Future[str] = loop.create_future()
    service._pending["telegram:77"] = fut

    handler = await create_planner_handler(service._engine)
    await handler(
        InboundEvent(
            event_type=EventType.MESSAGE,
            platform="telegram",
            channel_id="77",
            user_id="u1",
            content="answer text",
        ),
        backend,
    )
    # deliver_reply resolves via call_soon_threadsafe (cross-loop safe) — yield a tick.
    await asyncio.sleep(0.01)
    assert fut.result() == "answer text"
    # Delivered to the loop → no canned acknowledgment was sent back.
    assert backend.sent == []
    assert backend.replies == []
