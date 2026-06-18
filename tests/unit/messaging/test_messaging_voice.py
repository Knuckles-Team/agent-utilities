"""Tests for voice input transcription wiring (CONCEPT:ECO-4.68)."""

from __future__ import annotations

import pytest

from agent_utilities.messaging import router, voice
from agent_utilities.messaging.models import (
    EventType,
    InboundEvent,
    MediaAttachment,
    MediaType,
    Message,
)


@pytest.mark.asyncio
async def test_transcribe_attachments_uses_voice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake(url: str) -> str:
        return "hello from voice"

    monkeypatch.setattr(voice, "transcribe_voice", _fake)
    ev = InboundEvent(
        event_type=EventType.MESSAGE,
        platform="telegram",
        channel_id="42",
        message=Message(
            attachments=[MediaAttachment(media_type=MediaType.VOICE_NOTE, url="u")]
        ),
    )
    assert await router._transcribe_attachments(ev) == "hello from voice"


@pytest.mark.asyncio
async def test_transcribe_attachments_none_without_audio() -> None:
    ev = InboundEvent(
        event_type=EventType.MESSAGE,
        platform="telegram",
        channel_id="42",
        message=Message(
            attachments=[MediaAttachment(media_type=MediaType.IMAGE, url="img")]
        ),
    )
    assert await router._transcribe_attachments(ev) == ""


@pytest.mark.asyncio
async def test_transcribe_voice_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MESSAGING_VOICE", "0")
    assert await voice.transcribe_voice("http://x/a.ogg") == ""
