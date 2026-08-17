"""Tests for the universal-path messaging reply flow (CONCEPT:AU-ECO.messaging.universal-graph-agent).

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


class _EmptyEvidenceEngine:
    """Bare ``ContextCompiler`` source (CONCEPT:AU-KG.retrieval.context-compiler)
    standing in for the real epistemic-graph engine this sandbox has no
    native binary for — an explicit empty-evidence retrieval surface, not a
    ContextCompiler bypass."""

    def search_hybrid(
        self, query: str, *, top_k: int = 8, as_of: str | None = None
    ) -> list[dict[str, Any]]:
        del query, top_k, as_of
        return []

    def retrieve_epistemic_view(self, query: str, *, top_k: int = 8) -> dict[str, Any]:
        del query, top_k
        return {}


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


# ── The reply IS the universal graph agent (CONCEPT:AU-ECO.messaging.universal-graph-agent) ─────────


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
    """CONCEPT:AU-ORCH.session.session-anchored-collections-native — when the run opened a native message channel, run_agent returns a
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
    """A backend TIMEOUT must NOT trigger a second full LLM call (CONCEPT:AU-ORCH.execution.chat-profile-timeouts).

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

    from agent_utilities.core.contextual_model import use_context_compiler_engine

    # The plain-chat fallback's model call is mandatorily context-compiled
    # (CONCEPT:AU-KG.retrieval.context-compiler) — an explicit (bare, empty-
    # evidence) engine stands in for the real epistemic-graph engine this
    # sandbox has no native binary for.
    with use_context_compiler_engine(_EmptyEvidenceEngine()):
        reply = await _graph_agent_reply(
            object(), "hello there", session="messaging:telegram:42"
        )
    assert reply.startswith("[local] ")
    assert "couldn't draft a reply" not in reply


@pytest.mark.asyncio
async def test_plain_chat_reply_tags_responder(monkeypatch: pytest.MonkeyPatch) -> None:
    # The plain-chat fallback tags the reply with who answered (CONCEPT:AU-ECO.messaging.model-routed-inbound-responder).
    monkeypatch.setenv("AGENT_UTILITIES_TESTING", "true")

    from agent_utilities.core.contextual_model import use_context_compiler_engine

    with use_context_compiler_engine(_EmptyEvidenceEngine()):
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


# ── Untranscribable voice attachment must fail visibly, not silently drop (ECO) ──


@pytest.mark.asyncio
async def test_untranscribable_voice_attachment_gets_explicit_failure_notice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CONCEPT:AU-ECO.messaging.voice-attachment-fallback — an audio/voice attachment that
    fails to transcribe (disabled, download error, empty ASR result) must produce an
    explicit, visible reply — never a silently dropped message. Mirrors this repo's
    fail-closed rule that a degraded read must never be read by its caller as success.
    """
    from agent_utilities.messaging import voice
    from agent_utilities.messaging.models import MediaAttachment, MediaType, Message
    from agent_utilities.messaging.router import create_planner_handler

    MessagingService._instance = None

    async def _fails(
        url: str, *, headers: dict[str, str] | None = None, mime_type: str = ""
    ) -> str:
        return ""

    monkeypatch.setattr(voice, "transcribe_voice", _fails)

    handler = await create_planner_handler(knowledge_engine=_Eng())
    backend = _FakeBackend("telegram")
    ev = InboundEvent(
        event_type=EventType.MESSAGE,
        platform="telegram",
        channel_id="42",
        user_id="u1",
        message=Message(
            attachments=[MediaAttachment(media_type=MediaType.VOICE_NOTE, url="u")]
        ),
    )

    await handler(ev, backend)

    assert len(backend.sent) == 1, backend.sent
    channel_id, text = backend.sent[0]
    assert channel_id == "42"
    assert "couldn't transcribe" in text.lower()


@pytest.mark.asyncio
async def test_no_attachment_and_no_text_is_a_silent_noop() -> None:
    """A message with neither text nor an audio attachment (e.g. a bare reaction/sticker
    the model layer doesn't carry as an attachment) is correctly a no-op — this must NOT
    regress into sending a spurious failure notice for every non-text event."""
    from agent_utilities.messaging.router import create_planner_handler

    MessagingService._instance = None
    handler = await create_planner_handler(knowledge_engine=_Eng())
    backend = _FakeBackend("telegram")
    ev = InboundEvent(
        event_type=EventType.MESSAGE, platform="telegram", channel_id="42", user_id="u1"
    )

    await handler(ev, backend)

    assert backend.sent == []


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

    async def _fake_reply(
        _engine, _content, *, session, image_parts=None, budget=None, shape=None
    ):
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


def test_validate_fleet_auth_consumes_xdg_config_contract(monkeypatch) -> None:
    import agent_utilities.core.config as config_module
    import agent_utilities.mcp.client_credentials as client_credentials
    from agent_utilities.messaging import daemon

    calls: list[str] = []
    monkeypatch.setattr(config_module, "load_config", lambda: calls.append("load"))
    monkeypatch.setattr(
        client_credentials,
        "outbound_auth_configuration_status",
        lambda: {
            "mode": "oidc-client-credentials",
            "ready": True,
            "missing": (),
            "invalid": (),
            "redacted": True,
        },
    )
    monkeypatch.setattr(
        client_credentials,
        "validate_outbound_auth_configuration",
        lambda: calls.append("validate"),
    )

    daemon._validate_fleet_auth()

    assert calls == ["load", "validate"]


def test_validate_fleet_auth_fails_closed_when_configuration_is_incomplete(
    monkeypatch,
) -> None:
    import agent_utilities.mcp.client_credentials as client_credentials
    from agent_utilities.messaging import daemon

    monkeypatch.setattr(
        client_credentials,
        "outbound_auth_configuration_status",
        lambda: {
            "mode": "oidc-client-credentials",
            "ready": False,
            "missing": ("OIDC_AUDIENCE",),
            "invalid": (),
            "redacted": True,
        },
    )
    monkeypatch.setattr(
        client_credentials,
        "validate_outbound_auth_configuration",
        lambda: (_ for _ in ()).throw(RuntimeError("incomplete")),
    )

    with pytest.raises(RuntimeError, match="incomplete"):
        daemon._validate_fleet_auth()


@pytest.mark.asyncio
async def test_image_turn_routes_to_vision_responder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CONCEPT:AU-ECO.messaging.image-attachment-fallback — a turn with image attachments goes straight to the vision-capable
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


@pytest.mark.asyncio
async def test_varied_ack_lite_llm_with_static_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CONCEPT:AU-ECO.messaging.image-attachment-fallback — the deferred-turn ack is LLM-varied (not a fixed template), with a
    varied static fallback when the lite model is unavailable."""
    from types import SimpleNamespace

    from agent_utilities.knowledge_graph.enrichment import cards
    from agent_utilities.messaging import router as rt

    shape = SimpleNamespace(tool_servers=("github-mcp",))
    # lite model available → its varied line is used
    monkeypatch.setattr(
        cards, "make_lite_llm_fn", lambda: lambda p: "Grabbing that now ⏳"
    )
    assert await rt._varied_ack("list my issues", shape) == "Grabbing that now ⏳"
    # lite model unavailable → varied static fallback (references the server, has ⏳)
    monkeypatch.setattr(cards, "make_lite_llm_fn", lambda: None)
    fb = await rt._varied_ack("list my issues", shape)
    assert "github" in fb and "⏳" in fb


# ── Messaging-orchestration transparency (CONCEPT:AU-ORCH.execution.messaging-orchestration-transparency) ──
# A failure must be a troubleshooting ENTRY POINT, never a black box: every non-ok run_summary
# gets a concise, translated footer appended to the reply — including the plain-chat and
# reply-budget-timeout fallbacks, which previously gave zero indication anything failed.


def _degraded_summary(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "route": {"agents": [], "servers": ["github-mcp"], "why": "lexical gate"},
        "outcome": "degraded",
        "stage_reached": "tool-call: github-mcp",
        "trace_ref": "trace:pref_run_deadbeef",
        "failure": {
            "raw": "RuntimeError: fleet MCP endpoint requires HTTPS outside loopback",
            "translated": (
                "The fleet gateway rejected a plain-HTTP connection outside loopback "
                "(TLS is required for any non-local endpoint)."
            ),
            "category": "fleet_https_gate",
            "hint": "Ask an operator to put that server behind HTTPS, or run it on loopback.",
        },
    }
    base.update(overrides)
    return base


class TestTransparencyFooter:
    def test_ok_outcome_produces_no_footer(self) -> None:
        from agent_utilities.messaging.router import _transparency_footer

        summary = {
            "outcome": "ok",
            "route": {},
            "stage_reached": "x",
            "trace_ref": "trace:y",
        }
        assert _transparency_footer(summary) == ""

    def test_none_or_non_dict_summary_produces_no_footer(self) -> None:
        from agent_utilities.messaging.router import _transparency_footer

        assert _transparency_footer(None) == ""
        assert _transparency_footer("not a dict") == ""  # type: ignore[arg-type]
        assert _transparency_footer(123) == ""  # type: ignore[arg-type]

    def test_degraded_outcome_renders_translated_hint_and_trace_ref(self) -> None:
        from agent_utilities.messaging.router import _transparency_footer

        footer = _transparency_footer(_degraded_summary())
        assert "⚠️" in footer
        assert "TLS" in footer or "HTTPS" in footer
        assert "operator" in footer.lower()
        assert "trace:pref_run_deadbeef" in footer

    def test_failed_and_timeout_outcomes_also_render(self) -> None:
        from agent_utilities.messaging.router import _transparency_footer

        for outcome in ("failed", "timeout"):
            footer = _transparency_footer(_degraded_summary(outcome=outcome))
            assert footer  # non-empty for every non-ok outcome

    def test_missing_failure_detail_still_names_stage_and_outcome(self) -> None:
        """A degraded/failed run_summary with NO failure sub-dict (e.g. a bare empty
        output, no captured cause) must still say SOMETHING — never a silent no-op."""
        from agent_utilities.messaging.router import _transparency_footer

        summary = {
            "outcome": "degraded",
            "route": {},
            "stage_reached": "tool-call: portainer-mcp",
            "trace_ref": "trace:pref_run_x",
        }
        footer = _transparency_footer(summary)
        assert footer
        assert "degraded" in footer
        assert "portainer-mcp" in footer

    def test_footer_respects_the_off_setting(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agent_utilities.messaging.router import _transparency_footer

        monkeypatch.setenv("MESSAGING_TRANSPARENCY_FOOTER", "false")
        assert _transparency_footer(_degraded_summary()) == ""

    def test_footer_never_raises_on_malformed_summary(self) -> None:
        from agent_utilities.messaging.router import _transparency_footer

        for junk in (
            {"outcome": "degraded", "failure": "not a dict"},
            {"outcome": object()},
        ):
            assert isinstance(_transparency_footer(junk), str)  # never raises

    def test_with_transparency_appends_only_when_non_empty(self) -> None:
        from agent_utilities.messaging.router import _with_transparency

        assert _with_transparency("hello", None) == "hello"
        assert _with_transparency("hello", {"outcome": "ok"}) == "hello"
        combined = _with_transparency("hello", _degraded_summary())
        assert combined.startswith("hello\n\n⚠️")


class TestSyntheticRunSummaries:
    """The router-side synthesized summaries for the two exits run_agent cannot describe
    itself: a caller-side reply-budget cancellation, and an exception above run_agent."""

    def test_timeout_run_summary_uses_the_planned_route_when_known(self) -> None:
        from types import SimpleNamespace

        from agent_utilities.messaging.router import _timeout_run_summary

        shape = SimpleNamespace(tool_servers=("github-mcp", "portainer-mcp"))
        summary = _timeout_run_summary("run:" + "a" * 32, shape, 45.0)
        assert summary["outcome"] == "timeout"
        assert summary["route"]["servers"] == ["github-mcp", "portainer-mcp"]
        assert summary["stage_reached"] == "tool-call: github-mcp,portainer-mcp"
        assert summary["failure"]["category"] == "reply_budget_timeout"
        assert summary["trace_ref"].startswith("trace:")

    def test_timeout_run_summary_falls_back_generically_with_no_shape(self) -> None:
        from agent_utilities.messaging.router import _timeout_run_summary

        summary = _timeout_run_summary("run:" + "b" * 32, None, 45.0)
        assert summary["outcome"] == "timeout"
        assert summary["route"]["servers"] == []
        assert summary["stage_reached"] == "reply-budget"
        assert summary["failure"]["category"] == "reply_budget_timeout"

    def test_exception_run_summary_translates_the_real_exception(self) -> None:
        from agent_utilities.messaging.router import _exception_run_summary

        summary = _exception_run_summary(
            "run:" + "c" * 32, RuntimeError("delegation exploded")
        )
        assert summary["outcome"] == "failed"
        assert summary["stage_reached"] == "messaging-dispatch"
        assert "delegation exploded" in summary["failure"]["raw"]
        assert summary["trace_ref"].startswith("trace:")


class TestGraphAgentReplyTransparency:
    """``_graph_agent_reply`` end to end: every non-ok outcome carries the footer through to
    the final reply string; an ok outcome is untouched."""

    @pytest.mark.asyncio
    async def test_ok_result_has_no_footer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agent_utilities.orchestration import manager as mgr

        class _Orch:
            def __init__(self, _engine: Any) -> None: ...

            async def execute_agent(self, **kwargs: Any) -> str:
                assert kwargs["include_run_summary"] is True
                assert kwargs["run_id"]  # pre-generated by the router
                import json as _json

                return _json.dumps(
                    {
                        "output": "Found 3 running containers.",
                        "run_id": kwargs["run_id"],
                        "run_summary": {
                            "route": {
                                "agents": [],
                                "servers": ["portainer-mcp"],
                                "why": "x",
                            },
                            "outcome": "ok",
                            "stage_reached": "tool-call: portainer-mcp",
                            "trace_ref": "trace:pref_run_ok",
                        },
                    }
                )

        monkeypatch.setattr(mgr, "Orchestrator", _Orch)
        reply = await _graph_agent_reply(
            object(), "list my containers", session="messaging:telegram:1"
        )
        assert reply == "Found 3 running containers."
        assert "⚠️" not in reply

    @pytest.mark.asyncio
    async def test_degraded_fleet_gate_result_reply_is_transparent_end_to_end(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reproduces the reported bug end to end at the ROUTER level: run_agent already
        composed a truthful failure string AND a degraded run_summary (github-mcp's fleet
        HTTPS gate); the reply must contain the TRANSLATED cause, the hint, and a trace_ref —
        not a generic 'some sort of failure'."""
        from agent_utilities.orchestration import manager as mgr

        class _Orch:
            def __init__(self, _engine: Any) -> None: ...

            async def execute_agent(self, **kwargs: Any) -> str:
                import json as _json

                return _json.dumps(
                    {
                        "output": (
                            "Delegation to fleet server 'github-mcp' could not produce a "
                            "tool-grounded result (RuntimeError: fleet MCP endpoint "
                            "requires HTTPS outside loopback). Refusing to fall back to a "
                            "general answer, which would fabricate tool output."
                        ),
                        "run_id": kwargs["run_id"],
                        "run_summary": _degraded_summary(),
                    }
                )

        monkeypatch.setattr(mgr, "Orchestrator", _Orch)
        reply = await _graph_agent_reply(
            object(),
            "does my github org have issues/PRs",
            session="messaging:telegram:1",
        )
        # The already-composed truthful output text is preserved...
        assert "could not produce a tool-grounded result" in reply
        # ...AND the transparency footer adds the TRANSLATED cause + hint + trace_ref.
        assert "⚠️" in reply
        assert "TLS" in reply or "HTTPS" in reply
        assert "operator" in reply.lower()
        assert "trace:pref_run_deadbeef" in reply

    @pytest.mark.asyncio
    async def test_reply_budget_timeout_names_the_planned_route_not_a_bare_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CRITICAL path (the originally-reported bug): the run is cancelled by the
        reply-budget wall before it can return anything. The reply must still name the
        planned route/stage + a translated cause — never a bare generic message."""
        from types import SimpleNamespace

        from agent_utilities.orchestration import manager as mgr

        class _SlowOrch:
            def __init__(self, _engine: Any) -> None: ...

            async def execute_agent(self, **kwargs: Any) -> str:
                await asyncio.sleep(10)
                return "never reached"

        monkeypatch.setattr(mgr, "Orchestrator", _SlowOrch)
        monkeypatch.setenv("AGENT_UTILITIES_TESTING", "true")

        shape = SimpleNamespace(tool_servers=("github-mcp",), reply_budget_s=0.2)
        reply = await _graph_agent_reply(
            object(),
            "does my github org have issues/PRs",
            session="messaging:telegram:1",
            budget=0.2,
            shape=shape,
        )
        # The graceful no-double-LLM-tax message is preserved...
        assert "slowly" in reply.lower() or "try again" in reply.lower()
        # ...AND now names the planned route/stage instead of staying silent about it.
        assert "⚠️" in reply
        assert "github-mcp" in reply
        assert "reply time budget" in reply.lower() or "reply budget" in reply.lower()

    @pytest.mark.asyncio
    async def test_exception_fallback_reply_carries_the_translated_cause(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A genuine exception above run_agent's own structured handling still threads the
        cause into the plain-chat fallback's reply instead of silently discarding it."""
        from agent_utilities.core.contextual_model import use_context_compiler_engine
        from agent_utilities.orchestration import manager as mgr

        class _BoomOrch:
            def __init__(self, _engine: Any) -> None: ...

            async def execute_agent(self, **kwargs: Any) -> str:
                raise RuntimeError("fleet MCP endpoint requires HTTPS outside loopback")

        monkeypatch.setattr(mgr, "Orchestrator", _BoomOrch)
        monkeypatch.setenv("AGENT_UTILITIES_TESTING", "true")

        with use_context_compiler_engine(_EmptyEvidenceEngine()):
            reply = await _graph_agent_reply(
                object(), "hello there", session="messaging:telegram:42"
            )
        assert "⚠️" in reply
        assert "TLS" in reply or "HTTPS" in reply
        assert "trace:" in reply
