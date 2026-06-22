"""Mattermost backend live-path tests (CONCEPT:ECO-4.90).

Mirrors the Telegram integration: prove a bidirectional Mattermost backend so an inbound
Mattermost post reaches the universal ``InboundRouter`` and a reply renders back through the
Mattermost outbound API — with the ``mattermostdriver`` transport mocked. This is a
LIVE-PATH test (it drives the real ``InboundRouter._dispatch`` → default handler), not just
an API unit test.

CONCEPT:ECO-4.90 — Mattermost as a first-class bidirectional messaging platform
"""

from __future__ import annotations

import asyncio
import json
import sys
import types
from typing import Any

import pytest

from agent_utilities.messaging.models import EventType, PlatformId

# ── A fake mattermostdriver transport ────────────────────────────────


class _FakePosts:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []

    def create_post(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.created.append(payload)
        return {"id": f"post-{len(self.created)}"}


class _FakeReactions:
    def __init__(self) -> None:
        self.added: list[dict[str, Any]] = []

    def create_reaction(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.added.append(payload)
        return payload


class _FakeUsers:
    def get_user(self, _uid: str) -> dict[str, Any]:
        return {"id": "BOT123", "username": "agent-bot"}


class _FakeDriver:
    """Stand-in for ``mattermostdriver.Driver`` — records calls, no network."""

    def __init__(self, options: dict[str, Any]) -> None:
        self.options = options
        self.posts = _FakePosts()
        self.reactions = _FakeReactions()
        self.users = _FakeUsers()
        self.logged_in = False
        self._ws_handler: Any = None

    def login(self) -> dict[str, Any]:
        self.logged_in = True
        return {"id": "BOT123"}

    def logout(self) -> None:
        self.logged_in = False

    def init_websocket(self, handler: Any) -> None:
        # The real driver blocks here on its own loop; the test feeds frames directly.
        self._ws_handler = handler

    def disconnect(self) -> None:
        self._ws_handler = None


@pytest.fixture()
def _fake_mattermost(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a fake ``mattermostdriver`` module so the backend imports without the dep."""
    mod = types.ModuleType("mattermostdriver")
    mod.Driver = _FakeDriver  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mattermostdriver", mod)
    monkeypatch.setenv("MATTERMOST_URL", "https://mm.example.com")


def _backend() -> Any:
    from agent_utilities.messaging.backends.mattermost import MattermostBackend
    from agent_utilities.messaging.models import MessagingConfig

    return MattermostBackend(MessagingConfig(platform="mattermost", token="bot-token"))


# ── Identity / capabilities ───────────────────────────────────────────


def test_id_and_capabilities() -> None:
    b = _backend()
    assert b.id == "mattermost"
    assert b.capabilities.inbound_listen is True  # declared AND implemented
    assert b.capabilities.send_text is True


def test_registered_as_first_class_platform() -> None:
    # Same enum + entry-point + capability matrix membership as Telegram.
    from agent_utilities.messaging.capabilities import CAPABILITY_MATRIX

    assert PlatformId.MATTERMOST == "mattermost"
    assert "mattermost" in CAPABILITY_MATRIX


# ── Connect (send-ready) resolves the bot id, no websocket yet ────────


@pytest.mark.asyncio
async def test_connect_is_send_ready_without_websocket(_fake_mattermost: None) -> None:
    b = _backend()
    await b.connect()
    assert b.is_connected
    assert b._bot_user_id == "BOT123"  # resolved from the token
    assert (
        b._listening is False
    )  # websocket NOT started by connect (lazy, like Telegram)


# ── Inbound: a posted frame normalizes into the shared InboundEvent ───


def test_normalize_posted_frame() -> None:
    b = _backend()
    b._bot_user_id = "BOT123"
    frame = {
        "event": "posted",
        "data": {
            "post": json.dumps(
                {
                    "id": "p1",
                    "user_id": "USER9",
                    "channel_id": "chan1",
                    "root_id": "",
                    "message": "hello agent",
                }
            ),
            "sender_name": "@alice",
            "channel_type": "D",
        },
    }
    event = b._normalize_post_event(frame)
    assert event is not None
    assert event.platform == PlatformId.MATTERMOST
    assert event.channel_id == "chan1"
    assert event.user_id == "USER9"
    assert event.user_name == "alice"
    assert event.content == "hello agent"
    assert event.message and event.message.id == "p1"


def test_drops_bot_own_post_no_echo_loop() -> None:
    b = _backend()
    b._bot_user_id = "BOT123"
    frame = {
        "event": "posted",
        "data": {
            "post": json.dumps(
                {
                    "id": "p2",
                    "user_id": "BOT123",
                    "channel_id": "c",
                    "message": "my reply",
                }
            ),
            "sender_name": "@agent-bot",
        },
    }
    assert b._normalize_post_event(frame) is None  # don't route our own posts


def test_ignores_non_posted_events() -> None:
    b = _backend()
    assert b._normalize_post_event({"event": "typing", "data": {}}) is None


# ── Outbound: a reply renders via the Mattermost posts API ────────────


@pytest.mark.asyncio
async def test_send_message_posts_via_driver(_fake_mattermost: None) -> None:
    b = _backend()
    await b.connect()
    result = await b.send_message("chan1", "the answer", thread_id="root7")
    assert result.success
    assert result.platform == PlatformId.MATTERMOST
    posted = b._driver.posts.created[-1]
    assert posted == {
        "channel_id": "chan1",
        "message": "the answer",
        "root_id": "root7",
    }


@pytest.mark.asyncio
async def test_reply_to_roots_under_message(_fake_mattermost: None) -> None:
    b = _backend()
    await b.connect()
    await b.reply_to("chan1", "msg-root", "threaded reply")
    posted = b._driver.posts.created[-1]
    assert posted["root_id"] == "msg-root"


@pytest.mark.asyncio
async def test_send_reaction_uses_bot_user(_fake_mattermost: None) -> None:
    b = _backend()
    await b.connect()
    await b.send_reaction("chan1", "p1", ":tada:")
    added = b._driver.reactions.added[-1]
    assert added == {"user_id": "BOT123", "post_id": "p1", "emoji_name": "tada"}


# ── LIVE PATH: inbound frame → InboundRouter → default handler reply ──


@pytest.mark.asyncio
async def test_inbound_frame_reaches_router_and_reply_renders(
    _fake_mattermost: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end thin-transport proof: feed a Mattermost ``posted`` WebSocket frame, let the
    universal ``InboundRouter`` consume it via ``backend.listen()``, and assert the default
    handler's reply is rendered back through the Mattermost outbound API (the bot posts it).
    The orchestrator itself is stubbed — we are proving the transport wiring, not the LLM.
    """
    from agent_utilities.messaging.router import InboundRouter

    b = _backend()
    await b.connect()

    # Stub the default handler to stand in for the universal orchestrator: it renders a reply
    # back through the SAME backend (what create_planner_handler does after a graph run).
    replies: list[tuple[str, str]] = []

    async def _handler(event: Any, backend: Any) -> None:
        assert event.platform == PlatformId.MATTERMOST
        out = await backend.send_message(event.channel_id, f"echo: {event.content}")
        replies.append((event.channel_id, out.message_id))

    router = InboundRouter()
    router.register_backend(b)
    router.set_default_handler(_handler)

    # Feed an inbound post by enqueuing what the websocket handler would produce.
    frame = {
        "event": "posted",
        "data": {
            "post": json.dumps(
                {
                    "id": "p1",
                    "user_id": "USER9",
                    "channel_id": "chanX",
                    "message": "ping",
                }
            ),
            "sender_name": "@alice",
        },
    }
    event = b._normalize_post_event(frame)
    assert event is not None
    b._event_queue.put_nowait(event)

    # Run the router briefly, then stop it (listen() drains the queue → dispatch → handler).
    task = asyncio.create_task(router.start())
    await asyncio.sleep(0.3)
    await router.stop()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    # The inbound message reached the router AND a reply was rendered to Mattermost.
    assert ("chanX", "post-1") in replies
    assert b._driver.posts.created[-1]["message"] == "echo: ping"
    assert b._driver.posts.created[-1]["channel_id"] == "chanX"


# ── Mattermost participates in last-active routing like Telegram ──────


@pytest.mark.asyncio
async def test_reach_user_routes_to_mattermost_last_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``reach_user`` follows the user to a Mattermost channel they were last active on —
    proving Mattermost is a first-class target of the platform-agnostic reach service."""
    from agent_utilities.messaging.models import InboundEvent, SendResult
    from agent_utilities.messaging.service import MessagingService

    class _Eng:
        def __init__(self) -> None:
            self.nodes: dict[str, dict[str, Any]] = {}

        def add_node(self, nid: str, _l: str, properties: dict[str, Any]) -> None:
            self.nodes[nid] = dict(properties)

        def query_cypher(self, _q: str, p: dict[str, Any]) -> list[Any]:
            n = self.nodes.get(p.get("id", ""))
            return [{"p": {"properties": n}}] if n else []

    class _FakeBackend:
        def __init__(self) -> None:
            self.id = "mattermost"
            self._connected = True
            self.sent: list[tuple[str, str]] = []

        @property
        def is_connected(self) -> bool:
            return self._connected

        async def send_message(
            self, channel_id: str, text: str, **_: Any
        ) -> SendResult:
            self.sent.append((channel_id, text))
            return SendResult(
                success=True, platform="mattermost", channel_id=channel_id
            )

    MessagingService._instance = None
    svc = MessagingService.instance(_Eng())
    backend = _FakeBackend()
    svc.register_connected(backend)
    monkeypatch.setattr(svc, "get_backend", lambda _p: _wrap(backend))
    monkeypatch.setattr(
        svc, "_gate", lambda *a, **k: type("D", (), {"allowed": True})()
    )

    svc.record_inbound(
        InboundEvent(
            event_type=EventType.MESSAGE,
            platform="mattermost",
            channel_id="chanX",
            user_id="u1",
        )
    )
    assert svc.resolve_channel("u1") == ("mattermost", "chanX")
    await svc.reach_user("proactive ping", user_id="u1")
    assert backend.sent[-1] == ("chanX", "proactive ping")


async def _wrap(backend: Any) -> Any:
    return backend
