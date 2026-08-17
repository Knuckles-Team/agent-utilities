"""BUG-041 messaging conversation-history backfill (CONCEPT:AU-ECO.messaging.conversational-history-backfill).

Known-bad proof: a message deleted from the KG is recovered byte-for-byte (same content,
same content-addressed id) by re-running the backfill against the platform's history API.
Known-good proof: a genuinely unrecoverable platform (Telegram) refuses loudly instead of
silently doing nothing, and never contacts a network transport for it.
"""

from __future__ import annotations

import httpx
import pytest

from agent_utilities.messaging.backfill import (
    NOT_YET_IMPLEMENTED_RECOVERABLE_PLATFORMS,
    RECOVERABLE_PLATFORM_PRESETS,
    UNRECOVERABLE_PLATFORMS,
    MessagingBackfillError,
    PlatformNotRecoverableError,
    backfill_platform_history,
)
from agent_utilities.messaging.inbox import _inbox_id, record_inbound


class _FakeEngine:
    """Mirrors ``tests/unit/messaging/test_inbox.py``'s fake, plus node deletion."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}

    def add_node(self, node_id, node_type, properties=None):  # MERGE/upsert semantics
        self.nodes.setdefault(node_id, {}).update(properties or {})

    def delete_node(self, node_id: str) -> None:
        self.nodes.pop(node_id, None)


def test_unrecoverable_platform_refuses_loudly_never_silent():
    """Known-bad proof: Telegram (and every other confirmed-unrecoverable platform) raises
    instead of returning an empty "success" — a degraded/refused read must never look like
    a clean zero-row backfill (AGENTS.md 'Fail closed')."""
    eng = _FakeEngine()
    for platform in UNRECOVERABLE_PLATFORMS:
        with pytest.raises(PlatformNotRecoverableError, match="BUG-041"):
            backfill_platform_history(eng, platform=platform, channel_id="c1")
    assert eng.nodes == {}  # nothing was silently written


@pytest.mark.parametrize("platform", sorted(UNRECOVERABLE_PLATFORMS))
def test_unrecoverable_platform_never_touches_the_network(platform):
    calls: list[str] = []

    def _never(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, json={})

    eng = _FakeEngine()
    with pytest.raises(PlatformNotRecoverableError):
        backfill_platform_history(
            eng,
            platform=platform,
            channel_id="c1",
            transport=httpx.MockTransport(_never),
        )
    assert calls == []


@pytest.mark.parametrize("platform", sorted(NOT_YET_IMPLEMENTED_RECOVERABLE_PLATFORMS))
def test_not_yet_implemented_platform_names_the_reason(platform):
    eng = _FakeEngine()
    with pytest.raises(MessagingBackfillError, match=platform):
        backfill_platform_history(eng, platform=platform, channel_id="c1")


def test_unknown_platform_raises():
    eng = _FakeEngine()
    with pytest.raises(MessagingBackfillError):
        backfill_platform_history(eng, platform="not-a-real-platform", channel_id="c1")


def test_every_platform_is_classified_exactly_once():
    """No platform can be simultaneously 'refuse' and 'attempt' — the three buckets used by
    backfill_platform_history's dispatch must partition the fleet with zero overlap."""
    recoverable = set(RECOVERABLE_PLATFORM_PRESETS)
    unrecoverable = set(UNRECOVERABLE_PLATFORMS)
    not_yet = set(NOT_YET_IMPLEMENTED_RECOVERABLE_PLATFORMS)
    assert not (recoverable & unrecoverable)
    assert not (recoverable & not_yet)
    assert not (unrecoverable & not_yet)
    # The full known 17-platform fleet (see messaging/capabilities.py CAPABILITY_MATRIX).
    all_classified = recoverable | unrecoverable | not_yet
    assert all_classified == {
        "discord",
        "slack",
        "telegram",
        "whatsapp",
        "teams",
        "googlechat",
        "googlemeet",
        "mattermost",
        "matrix",
        "irc",
        "signal",
        "imessage",
        "line",
        "twitch",
        "synology",
        "voicecall",
        "nextcloud",
    }


def test_discord_backfill_recovers_a_message_deleted_from_the_kg(monkeypatch):
    """The full known-bad proof: record a live message, delete its node, run backfill
    against a mocked Discord history API returning that same message, and assert the
    IDENTICAL InboundMessage node (same content-addressed id) reappears with its text."""
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "fake-bot-token")
    eng = _FakeEngine()

    original_iid = record_inbound(
        eng,
        platform="discord",
        channel_id="42",
        message_id="999",
        text="the original message",
        session="s1",
    )
    assert original_iid in eng.nodes

    # Simulate the loss the ledger is about: the node is gone from the KG.
    eng.delete_node(original_iid)
    assert original_iid not in eng.nodes

    seen_auth: list[str] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        seen_auth.append(request.headers.get("authorization", ""))
        assert "channels/42/messages" in str(request.url)
        before = httpx.QueryParams(request.url.query).get("before")
        if before == "999":
            return httpx.Response(200, json=[])  # end of history
        return httpx.Response(
            200,
            json=[
                {
                    "id": "999",
                    "content": "the original message",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                }
            ],
        )

    result = backfill_platform_history(
        eng,
        platform="discord",
        channel_id="42",
        session="s1",
        transport=httpx.MockTransport(_handler),
    )

    assert result["recovered"] == 1
    assert result["errors"] == 0
    # One page returning the message, one terminating page — same credential every call.
    assert seen_auth == ["Bot fake-bot-token", "Bot fake-bot-token"]

    # The reconstructed node has the SAME content-addressed id as the one that was deleted.
    recovered_id = _inbox_id("discord", "42", "999", "the original message")
    assert recovered_id == original_iid
    assert eng.nodes[recovered_id]["text"] == "the original message"
    assert eng.nodes[recovered_id]["status"] == "backfilled"
    assert eng.nodes[recovered_id]["received_at"] == "2026-01-01T00:00:00+00:00"


def test_discord_missing_credential_raises_before_any_network_call(monkeypatch):
    monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
    monkeypatch.delenv("MESSAGING_DISCORD_TOKEN", raising=False)
    calls: list[str] = []

    def _never(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, json=[])

    eng = _FakeEngine()
    with pytest.raises(MessagingBackfillError, match="DISCORD_BOT_TOKEN"):
        backfill_platform_history(
            eng,
            platform="discord",
            channel_id="42",
            transport=httpx.MockTransport(_never),
        )
    assert calls == []


def test_slack_dotted_cursor_paginates_across_multiple_pages(monkeypatch):
    """Proves the framework generalizes past Discord's shape: Slack's cursor lives at a
    NESTED path (``response_metadata.next_cursor``), not the top level or last-record id."""
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-fake")
    eng = _FakeEngine()

    pages = {
        None: {
            "ok": True,
            "messages": [{"ts": "1.0", "text": "first"}],
            "response_metadata": {"next_cursor": "page2"},
        },
        "page2": {
            "ok": True,
            "messages": [{"ts": "2.0", "text": "second"}],
            "response_metadata": {"next_cursor": ""},
        },
    }

    def _handler(request: httpx.Request) -> httpx.Response:
        cursor = httpx.QueryParams(request.url.query).get("cursor")
        return httpx.Response(200, json=pages[cursor])

    result = backfill_platform_history(
        eng,
        platform="slack",
        channel_id="C123",
        transport=httpx.MockTransport(_handler),
    )

    assert result["recovered"] == 2
    texts = sorted(v["text"] for v in eng.nodes.values())
    assert texts == ["first", "second"]


def test_matrix_nested_content_body_field_is_extracted(monkeypatch):
    """Proves dotted-field extraction reaches into Matrix's nested ``content.body``."""
    monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "fake-matrix-token")
    monkeypatch.setenv("MATRIX_HOMESERVER", "https://matrix.example.org")
    eng = _FakeEngine()

    def _handler(request: httpx.Request) -> httpx.Response:
        assert "matrix.example.org" in str(request.url)
        return httpx.Response(
            200,
            json={
                "chunk": [
                    {
                        "event_id": "$abc",
                        "content": {"body": "hello from matrix"},
                        "origin_server_ts": 1700000000000,
                    }
                ],
                "end": "",
            },
        )

    result = backfill_platform_history(
        eng,
        platform="matrix",
        channel_id="!room:example.org",
        transport=httpx.MockTransport(_handler),
    )

    assert result["recovered"] == 1
    (row,) = eng.nodes.values()
    assert row["text"] == "hello from matrix"


def test_nextcloud_basic_auth_uses_user_and_token(monkeypatch):
    monkeypatch.setenv("NEXTCLOUD_TOKEN", "fake-nc-token")
    monkeypatch.setenv("NEXTCLOUD_URL", "https://cloud.example.org")
    monkeypatch.setenv("NEXTCLOUD_USER", "botuser")
    eng = _FakeEngine()

    def _handler(request: httpx.Request) -> httpx.Response:
        import base64

        auth = request.headers.get("authorization", "")
        assert auth.startswith("Basic ")
        decoded = base64.b64decode(auth.removeprefix("Basic ")).decode()
        assert decoded == "botuser:fake-nc-token"
        return httpx.Response(
            200,
            json={
                "ocs": {"data": [{"id": 1, "message": "nc message", "timestamp": 1}]}
            },
        )

    result = backfill_platform_history(
        eng,
        platform="nextcloud",
        channel_id="talk-room",
        transport=httpx.MockTransport(_handler),
    )
    assert result["recovered"] == 1


def test_voicecall_twilio_basic_auth_single_page(monkeypatch):
    """Voicecall is deliberately single-page (see the preset's comment on a real
    ``next_url_field``/httpx query-merge bug this backfill lane discovered but does not
    fix); this proves the basic-auth (SID:token) shape + a full page of history."""
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "fake-auth-token")
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    eng = _FakeEngine()

    def _handler(request: httpx.Request) -> httpx.Response:
        import base64

        auth = request.headers.get("authorization", "")
        assert auth.startswith("Basic ")
        decoded = base64.b64decode(auth.removeprefix("Basic ")).decode()
        assert decoded == "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx:fake-auth-token"
        return httpx.Response(
            200,
            json={
                "messages": [
                    {"sid": "SM1", "body": "sms one", "date_sent": "2026-01-01"},
                    {"sid": "SM2", "body": "sms two", "date_sent": "2026-01-02"},
                ],
                "next_page_uri": None,
            },
        )

    result = backfill_platform_history(
        eng,
        platform="voicecall",
        channel_id="unused",
        transport=httpx.MockTransport(_handler),
    )
    assert result["recovered"] == 2
    texts = sorted(v["text"] for v in eng.nodes.values())
    assert texts == ["sms one", "sms two"]
