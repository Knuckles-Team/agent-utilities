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


@pytest.mark.asyncio
async def test_model_routed_reply_falls_back_to_plain_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # When the dedicated agent fails (e.g. a local model without tool support), the reply
    # must degrade to a plain chat completion, not error out (CONCEPT:ECO-4.56).
    from agent_utilities.messaging import router

    def _boom(provider, model_id):
        raise RuntimeError("System message must be at the beginning.")

    monkeypatch.setattr(router, "_get_messaging_agent", _boom)
    # AGENT_UTILITIES_TESTING=true → create_model returns a TestModel for the plain path.
    monkeypatch.setenv("AGENT_UTILITIES_TESTING", "true")
    reply = await _model_routed_reply("hello there", "")
    assert reply.startswith("[local] ")
    assert "couldn't draft a reply" not in reply


@pytest.mark.asyncio
async def test_run_until_text_plain_output() -> None:
    # _run_until_text returns the agent's text when it doesn't defer tools (ECO-4.62).
    from agent_utilities.messaging import router

    class _Result:
        output = "the answer"

    class _Agent:
        async def run(self, *a: Any, **k: Any):
            return _Result()

    out = await router._run_until_text(_Agent(), "q")
    assert out == "the answer"


def test_auto_approvable_delegation_and_reads_yes_mutations_no() -> None:
    # CONCEPT:ECO-4.75 — KG reads + the graph-os delegation/discovery surface auto-run;
    # read-only fleet tools auto-run; mutating tools stay gated.
    from agent_utilities.messaging.router import _auto_approvable

    # delegation + discovery + KG reads
    for t in (
        "kg_search",
        "graph_orchestrate",
        "graph_search",
        "find_tools",
        "load_tools",
    ):
        assert _auto_approvable(t), t
    # read-only fleet tools (incl. multiplexer-prefixed) auto-run
    assert _auto_approvable("go__github_list_issues")
    assert _auto_approvable("gith__repos_get")
    # mutations stay denied
    for t in (
        "github_create_issue",
        "go__graph_write",
        "delete_node",
        "save_chat_message",
        "kafka_send",
    ):
        assert not _auto_approvable(t), t


# ── Lean / lazy loadout (ECO-4.58) ───────────────────────────────────


@pytest.mark.asyncio
async def test_messaging_agent_is_lean_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_utilities.agent.factory as factory
    from agent_utilities.messaging import router

    captured: dict[str, Any] = {}

    def _fake_create_agent(**kwargs: Any):
        captured.clear()
        captured.update(kwargs)
        return object(), []

    monkeypatch.setattr(factory, "create_agent", _fake_create_agent)

    router._MESSAGING_AGENTS.clear()
    router._get_messaging_agent("", None)
    # Lean: skills NOT pre-loaded; universal tools on (fleet tools load on demand).
    assert captured["enable_skills"] is False
    assert captured["enable_universal_tools"] is True

    # Opt into skills + scoped tool tags.
    monkeypatch.setenv("MESSAGING_ENABLE_SKILLS", "1")
    monkeypatch.setenv("MESSAGING_TOOL_TAGS", "kg,reach")
    router._MESSAGING_AGENTS.clear()
    router._get_messaging_agent("", None)
    assert captured["enable_skills"] is True
    assert captured["tool_tags"] == ["kg", "reach"]


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


# ── Image / multimodal input (ECO-4.67) ──────────────────────────────


def test_agent_input_plain_vs_multimodal() -> None:
    from agent_utilities.messaging.router import _agent_input

    assert _agent_input("hi", None) == "hi"
    assert _agent_input("hi", []) == "hi"
    parts = ["<img1>", "<img2>"]
    assert _agent_input("describe", parts) == ["describe", "<img1>", "<img2>"]


@pytest.mark.asyncio
async def test_run_until_text_passes_images_to_agent() -> None:
    from agent_utilities.messaging import router

    seen: dict[str, Any] = {}

    class _Result:
        output = "a white square"

    class _Agent:
        async def run(self, inp: Any = None, **k: Any):
            seen["input"] = inp
            return _Result()

    out = await router._run_until_text(_Agent(), "describe", image_parts=["<imgbytes>"])
    assert out == "a white square"
    assert seen["input"] == ["describe", "<imgbytes>"]


# ── Non-blocking recall (ECO-4.72) ───────────────────────────────────


@pytest.mark.asyncio
async def test_recall_context_times_out_without_freezing(monkeypatch) -> None:
    import time as _t

    from agent_utilities.messaging import router

    monkeypatch.setenv("MESSAGING_RECALL_TIMEOUT", "1")

    class _Eng:
        def recall_memory(self, **kw):
            _t.sleep(10)  # simulate a hung blocking retrieval
            return []

    # Must return "" within the timeout, NOT hang for 10s.
    start = _t.monotonic()
    out = await router._recall_context(_Eng(), "hello", "telegram")
    assert out == "" and (_t.monotonic() - start) < 5


@pytest.mark.asyncio
async def test_recall_context_returns_memories() -> None:
    from agent_utilities.messaging import router

    class _Eng:
        def recall_memory(self, **kw):
            return [{"description": "prior chat about webhooks"}]

    out = await router._recall_context(_Eng(), "hi", "telegram")
    assert "prior chat about webhooks" in out


# ── Reply path must not block on slow KG writes (ECO-4.72) ───────────


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


def test_load_fleet_auth_noop_when_already_set(monkeypatch) -> None:
    # CONCEPT:ECO-4.75 — if MCP_CLIENT_AUTH is already in the env (deploy/OpenBao path),
    # the bootstrap is a no-op (doesn't overwrite or read other sources).
    import os
    from agent_utilities.messaging import daemon

    monkeypatch.setenv("MCP_CLIENT_AUTH", "oidc-client-credentials")
    monkeypatch.setenv("OIDC_CLIENT_ID", "preset")
    daemon._load_fleet_auth()
    assert os.environ["OIDC_CLIENT_ID"] == "preset"
