"""Tests for model-routed replies + multiple concurrent backends (CONCEPT:ECO-4.55).

Verifies the local-default / Claude-address responder selection and that several
messaging services work side by side (last-active routing + per-platform send).
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.messaging.models import EventType, InboundEvent, SendResult
from agent_utilities.messaging.router import _model_routed_reply, _select_responder
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


@pytest.mark.asyncio
async def test_model_routed_reply_uses_dedicated_agent_and_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Stub the cached dedicated agent (ECO-4.56) so the test stays hermetic (no MCP/skills
    # build, no live LLM) while exercising the routing + tag + agent.run path.
    from agent_utilities.messaging import router

    class _Result:
        output = "hi from the dedicated agent"

    class _StubAgent:
        async def run(self, _prompt: str):
            return _Result()

    monkeypatch.setattr(
        router, "_get_messaging_agent", lambda provider, model_id: _StubAgent()
    )
    reply = await _model_routed_reply("hello there", "")
    assert reply == "[local] hi from the dedicated agent"


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
