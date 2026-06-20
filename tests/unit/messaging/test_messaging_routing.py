"""Tests for the universal-path messaging reply flow (CONCEPT:ECO-4.78).

Messaging is thin transport: an inbound chat turn runs the ONE universal graph agent
(``Orchestrator.execute_agent`` → ``run_agent``), session-scoped per channel. These tests
prove the reply routes through that universal path (not a bespoke messaging-only path), that
continuity + dynamic delegation come from the core, and that a slow/hung graph run still
yields a reply via the plain-chat fallback. They also cover the preserved local-default /
Claude-address responder selection used by that fallback, and several concurrent backends.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agent_utilities.messaging.models import EventType, InboundEvent, SendResult
from agent_utilities.messaging.router import (
    _channel_session,
    _graph_agent_reply,
    _plain_chat_reply,
    _select_responder,
)
from agent_utilities.messaging.service import MessagingService

# ── Responder selection (local default / Claude address) ─────────────


def test_default_responder_is_local() -> None:
    label, provider, _model, task = _select_responder("what's the weather?")
    assert label == "local"
    assert provider == ""
    assert task == "what's the weather?"


def test_claude_address_routes_to_claude_when_key_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_utilities.core.config import config

    monkeypatch.setattr(config, "anthropic_api_key", "sk-test", raising=False)
    label, provider, model_id, task = _select_responder("/claude summarize this")
    assert label == "claude"
    assert provider == "anthropic"
    assert model_id  # a claude model id
    assert task == "summarize this"  # trigger stripped


def test_claude_address_falls_back_to_local_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_utilities.core.config import config

    monkeypatch.setattr(config, "anthropic_api_key", None, raising=False)
    label, provider, _model, _task = _select_responder("/claude hi")
    assert "no Anthropic key" in label
    assert provider == ""  # local fallback


# ── The reply IS the universal graph agent (CONCEPT:ECO-4.78) ─────────


def test_channel_session_is_stable_per_channel() -> None:
    # The session key is one stable id per (platform, channel) so successive turns share it.
    assert _channel_session("telegram", "42") == "messaging:telegram:42"
    assert _channel_session("telegram", "42") == _channel_session("telegram", "42")
    assert _channel_session("slack", "C1") != _channel_session("telegram", "42")


@pytest.mark.asyncio
async def test_reply_routes_through_universal_execute_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A chat turn runs ``Orchestrator.execute_agent`` with the per-channel session — NOT a
    bespoke messaging-only path. We capture the call to prove the universal path is taken and
    that the session/memento source is wired so continuity comes from the core memory."""
    from agent_utilities.orchestration import manager as mgr

    captured: dict[str, Any] = {}

    class _Orch:
        def __init__(self, _engine: Any) -> None:
            pass

        async def execute_agent(self, **kwargs: Any) -> str:
            captured.update(kwargs)
            return "answer from the universal graph agent"

    monkeypatch.setattr(mgr, "Orchestrator", _Orch)

    reply = await _graph_agent_reply(
        object(), "what's the github status?", session="messaging:telegram:42"
    )
    assert reply == "answer from the universal graph agent"
    # Routed through execute_agent, session-scoped, with the memento source = the session so
    # the next turn recalls this conversation via the core memory (no messaging recall query).
    assert captured["session_id"] == "messaging:telegram:42"
    assert captured["memento_source"] == "messaging:telegram:42"
    assert captured["task"] == "what's the github status?"


@pytest.mark.asyncio
async def test_reply_unwraps_channel_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    """CONCEPT:ORCH-1.40 — when the run opened a native message channel, run_agent returns a
    JSON envelope {"output", "channel_id"} (not the bare reply). The messaging layer must
    deliver the ``output`` text, not the raw JSON (which rendered as literal JSON in Telegram)."""
    import json

    from agent_utilities.orchestration import manager as mgr

    envelope = json.dumps(
        {
            "output": "Here are your portainer stacks:\n- **web** (running)",
            "channel_id": "orch:messaging:telegram:42:run:abc",
        }
    )

    class _Orch:
        def __init__(self, _engine: Any) -> None:
            pass

        async def execute_agent(self, **_kwargs: Any) -> str:
            return envelope

    monkeypatch.setattr(mgr, "Orchestrator", _Orch)

    reply = await _graph_agent_reply(
        object(), "list my portainer stacks", session="messaging:telegram:42"
    )
    assert reply == "Here are your portainer stacks:\n- **web** (running)"
    assert "channel_id" not in reply and not reply.startswith("{")


@pytest.mark.asyncio
async def test_reply_does_not_unwrap_a_genuine_json_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real JSON reply from the agent (keys beyond the envelope's) is delivered verbatim —
    the unwrap is exact-key so it never mis-extracts a legitimate JSON payload."""
    from agent_utilities.orchestration import manager as mgr

    genuine = '{"output": "x", "status": "ok", "items": [1, 2]}'

    class _Orch:
        def __init__(self, _engine: Any) -> None:
            pass

        async def execute_agent(self, **_kwargs: Any) -> str:
            return genuine

    monkeypatch.setattr(mgr, "Orchestrator", _Orch)

    reply = await _graph_agent_reply(object(), "give me json", session="s:1")
    assert reply == genuine  # untouched — has non-envelope keys


@pytest.mark.asyncio
async def test_reply_timeout_does_not_double_call_the_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A backend TIMEOUT must NOT trigger a second full LLM call (CONCEPT:ORCH-1.62).

    The measured >90 s came from a stalled first round + a 45 s wall + ANOTHER slow plain-chat
    call to the same degraded endpoint. When the universal run hits the reply-timeout wall we
    now surface a graceful message and do NOT call ``_plain_chat_reply`` (no double-LLM tax)."""
    import time

    from agent_utilities.messaging import router as router_mod
    from agent_utilities.orchestration import manager as mgr

    class _SlowOrch:
        def __init__(self, _engine: Any) -> None:
            pass

        async def execute_agent(self, **kwargs: Any) -> str:
            await asyncio.sleep(10)  # simulate a hung graph run on a degraded backend
            return "never reached"

    plain_calls: list[str] = []

    async def _spy_plain(content: str, **_: Any) -> str:
        plain_calls.append(content)
        return "[local] SHOULD NOT BE CALLED ON TIMEOUT"

    monkeypatch.setattr(mgr, "Orchestrator", _SlowOrch)
    monkeypatch.setattr(router_mod, "_plain_chat_reply", _spy_plain)
    monkeypatch.setenv("MESSAGING_REPLY_TIMEOUT", "0.3")
    monkeypatch.setenv("AGENT_UTILITIES_TESTING", "true")

    start = time.monotonic()
    reply = await _graph_agent_reply(
        object(), "hello there", session="messaging:telegram:42"
    )
    elapsed = time.monotonic() - start
    assert elapsed < 5, f"timeout did not fire promptly ({elapsed:.2f}s)"
    # The double-LLM tax is removed: no second call to the (degraded) endpoint on timeout.
    assert plain_calls == [], "timeout must NOT trigger a second plain-chat LLM call"
    assert "slowly" in reply.lower() or "try again" in reply.lower()


@pytest.mark.asyncio
async def test_reply_error_falls_back_to_plain_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the universal run errors (e.g. a delegation failure), the reply degrades to plain
    chat so the user always gets an answer."""
    from agent_utilities.orchestration import manager as mgr

    class _BoomOrch:
        def __init__(self, _engine: Any) -> None:
            pass

        async def execute_agent(self, **kwargs: Any) -> str:
            raise RuntimeError("delegation exploded")

    monkeypatch.setattr(mgr, "Orchestrator", _BoomOrch)
    monkeypatch.setenv("AGENT_UTILITIES_TESTING", "true")

    reply = await _graph_agent_reply(
        object(), "hello there", session="messaging:telegram:42"
    )
    assert reply.startswith("[local] ")
    assert "couldn't draft a reply" not in reply


@pytest.mark.asyncio
async def test_plain_chat_reply_tags_responder(monkeypatch: pytest.MonkeyPatch) -> None:
    # The plain-chat fallback tags the reply with who answered (CONCEPT:ECO-4.55).
    monkeypatch.setenv("AGENT_UTILITIES_TESTING", "true")
    reply = await _plain_chat_reply("hello there")
    assert reply.startswith("[local] ")
    assert "couldn't draft a reply" not in reply


# ── Image / multimodal input (ECO-4.67) ──────────────────────────────


def test_agent_input_plain_vs_multimodal() -> None:
    from agent_utilities.messaging.router import _agent_input

    assert _agent_input("hi", None) == "hi"
    assert _agent_input("hi", []) == "hi"
    parts = ["<img1>", "<img2>"]
    assert _agent_input("describe", parts) == ["describe", "<img1>", "<img2>"]


# ── Multiple concurrent backends ─────────────────────────────────────


class _FakeBackend:
    def __init__(self, platform: str) -> None:
        self.id = platform
        self._connected = True
        self.sent: list[tuple[str, str]] = []

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def send_message(self, channel_id: str, text: str, **_: Any) -> SendResult:
        self.sent.append((channel_id, text))
        return SendResult(success=True, platform=self.id, channel_id=channel_id)


class _Eng:
    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}

    def add_node(self, nid: str, _l: str, properties: dict[str, Any]) -> None:
        self.nodes[nid] = dict(properties)

    def query_cypher(self, _q: str, p: dict[str, Any]):
        n = self.nodes.get(p.get("id", ""))
        return [{"p": {"properties": n}}] if n else []

    def store_memory(self, **_: Any):
        return "m"


@pytest.fixture()
def multi(monkeypatch: pytest.MonkeyPatch) -> MessagingService:
    MessagingService._instance = None
    svc = MessagingService.instance(_Eng())
    backends = {"telegram": _FakeBackend("telegram"), "slack": _FakeBackend("slack")}
    for b in backends.values():
        svc.register_connected(b)

    async def _get_backend(platform: str):
        return backends.get(platform)

    monkeypatch.setattr(svc, "get_backend", _get_backend)
    monkeypatch.setattr(
        svc, "_gate", lambda *a, **k: type("D", (), {"allowed": True})()
    )
    return svc


@pytest.mark.asyncio
async def test_send_targets_the_right_service(multi: MessagingService) -> None:
    await multi.send("telegram", "100", "to tg")
    await multi.send("slack", "C200", "to slack")
    tg = await multi.get_backend("telegram")
    sl = await multi.get_backend("slack")
    assert tg.sent == [("100", "to tg")]
    assert sl.sent == [("C200", "to slack")]


@pytest.mark.asyncio
async def test_reach_user_follows_last_active_service(multi: MessagingService) -> None:
    # User talks on telegram, then on slack — reach_user must follow to slack.
    for plat, chan in (("telegram", "100"), ("slack", "C200")):
        multi.record_inbound(
            InboundEvent(
                event_type=EventType.MESSAGE,
                platform=plat,
                channel_id=chan,
                user_id="u1",
            )
        )
    assert multi.resolve_channel("u1") == ("slack", "C200")
    await multi.reach_user("hi", user_id="u1")
    assert (await multi.get_backend("slack")).sent[-1] == ("C200", "hi")

    # They reply on telegram again — routing follows back.
    multi.record_inbound(
        InboundEvent(
            event_type=EventType.MESSAGE,
            platform="telegram",
            channel_id="100",
            user_id="u1",
        )
    )
    assert multi.resolve_channel("u1") == ("telegram", "100")


# ── Reply path must not block on slow KG writes (ECO-4.72/4.74) ───────


@pytest.mark.asyncio
async def test_inbound_reply_path_not_blocked_by_slow_kg() -> None:
    """planner_handler must NOT await blocking KG writes (last-active + ingest).

    Regression for the 'message ingested but no reply' stall: record_inbound (add_node)
    and ingest (store_memory + embed) are blocking; awaiting them inline starved the burst
    reply. They now run in a background thread, so the handler returns immediately.
    """
    import time

    from agent_utilities.messaging.router import create_planner_handler

    MessagingService._instance = None

    class _SlowEng:
        def add_node(self, *a: Any, **k: Any) -> None:
            time.sleep(2)  # blocking last-active write

        def store_memory(self, **k: Any) -> str:
            time.sleep(2)  # blocking ingest + embedding
            return "m"

        def recall_memory(self, **k: Any) -> list[Any]:
            return []

        def query_cypher(self, *a: Any, **k: Any) -> list[Any]:
            return []

    handler = await create_planner_handler(knowledge_engine=_SlowEng())
    backend = _FakeBackend("telegram")
    ev = InboundEvent(
        event_type=EventType.MESSAGE,
        platform="telegram",
        channel_id="42",
        user_id="u1",
        content="hello there",
    )

    start = time.monotonic()
    await handler(ev, backend)
    elapsed = time.monotonic() - start
    assert elapsed < 1.0, f"reply path blocked on KG writes ({elapsed:.2f}s)"


# ── Continuity via the CORE memory — two turns share a session (ECO-4.78) ──


@pytest.mark.asyncio
async def test_two_turns_share_one_session_for_continuity(monkeypatch) -> None:
    """Two messages in the same channel → both runs use the SAME per-channel session, so the
    core memory (mementos under that session source) carries continuity from turn 1 to turn 2
    — WITHOUT any messaging-specific recall query. We capture the session passed to the
    universal path on each turn to prove it is stable and channel-scoped.
    """
    from agent_utilities.messaging import router
    from agent_utilities.messaging.router import create_planner_handler

    MessagingService._instance = None

    sessions: list[str] = []

    async def _fake_reply(_engine, _content, *, session, image_parts=None, budget=None):
        sessions.append(session)
        return "ok"

    monkeypatch.setattr(router, "_graph_agent_reply", _fake_reply)
    # Tight burst window so the coalescer flushes promptly in-test.
    monkeypatch.setenv("MESSAGING_BURST_WINDOW_S", "0.2")
    monkeypatch.setenv("MESSAGING_BURST_MAX_S", "1")

    handler = await create_planner_handler(knowledge_engine=_Eng())
    backend = _FakeBackend("telegram")

    def _ev(text: str) -> InboundEvent:
        return InboundEvent(
            event_type=EventType.MESSAGE,
            platform="telegram",
            channel_id="42",
            user_id="u1",
            content=text,
        )

    await handler(_ev("what is the capital of France?"), backend)
    await asyncio.sleep(0.6)
    await handler(_ev("and its population?"), backend)
    await asyncio.sleep(0.6)

    assert len(sessions) == 2, sessions
    # Both turns of the SAME channel share one stable session → continuity via the core.
    assert sessions[0] == sessions[1] == "messaging:telegram:42"


def test_load_fleet_auth_noop_when_already_set(monkeypatch) -> None:
    # CONCEPT:ECO-4.75 — if MCP_CLIENT_AUTH is already in the env (deploy/OpenBao path),
    # the bootstrap is a no-op (doesn't overwrite or read other sources).
    import os

    from agent_utilities.messaging import daemon

    monkeypatch.setenv("MCP_CLIENT_AUTH", "oidc-client-credentials")
    monkeypatch.setenv("OIDC_CLIENT_ID", "preset")
    daemon._load_fleet_auth()
    assert os.environ["OIDC_CLIENT_ID"] == "preset"


@pytest.mark.asyncio
async def test_image_turn_routes_to_vision_responder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CONCEPT:ECO-4.67 — a turn with image attachments goes straight to the vision-capable
    responder, NOT the universal graph (which drops images and would answer text-only)."""
    from agent_utilities.messaging import router as rt
    from agent_utilities.orchestration import manager as mgr

    called = {"execute": 0, "vision": 0}

    class _Orch:
        def __init__(self, _engine: Any) -> None: ...
        async def execute_agent(self, **_k: Any) -> str:
            called["execute"] += 1
            return "graph (should not run for an image turn)"

    async def _fake_vision(content: str, *, image_parts: Any = None) -> str:
        called["vision"] += 1
        return f"[local] I can see {len(image_parts)} image(s)"

    monkeypatch.setattr(mgr, "Orchestrator", _Orch)
    monkeypatch.setattr(rt, "_plain_chat_reply", _fake_vision)

    reply = await _graph_agent_reply(
        object(),
        "what is this photo of?",
        session="messaging:telegram:1",
        image_parts=["img"],
    )
    assert called["vision"] == 1 and called["execute"] == 0
    assert "1 image" in reply
