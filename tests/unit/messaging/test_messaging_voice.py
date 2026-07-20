"""Tests for voice input transcription wiring (CONCEPT:AU-ECO.messaging.telegram-voice-note)."""

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


def test_get_backend_falls_back_to_faster_whisper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CONCEPT:AU-ECO.messaging.telegram-voice-note — when the full audio-transcriber package (pyaudio-bound) is absent,
    _get_backend uses faster-whisper's WhisperModel directly: the lean messaging path."""
    import sys
    import types

    from agent_utilities.messaging import voice

    voice._backend = None
    # Force `from audio_transcriber.audio_transcriber import ...` to ImportError.
    monkeypatch.setitem(sys.modules, "audio_transcriber", None)
    # Fake faster_whisper.WhisperModel returning two segments.
    segs = [types.SimpleNamespace(text=" hello"), types.SimpleNamespace(text=" world")]

    class _Model:
        def __init__(self, *a: object, **k: object) -> None: ...
        def transcribe(self, path: str):
            return (segs, object())

    fw = types.ModuleType("faster_whisper")
    fw.WhisperModel = _Model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", fw)

    backend = voice._get_backend()
    try:
        assert isinstance(backend, voice._FasterWhisper)
        assert backend.transcribe("x.ogg")["text"] == "hello world"
    finally:
        voice._backend = None


def test_sniff_image_media_type() -> None:
    """CONCEPT:AU-ECO.messaging.image-attachment-fallback — magic-byte sniff for generic/absent content-types (Telegram)."""
    from agent_utilities.messaging.router import _sniff_image_media_type

    assert _sniff_image_media_type(b"\xff\xd8\xff\xe0junk") == "image/jpeg"
    assert _sniff_image_media_type(b"\x89PNG\r\n\x1a\nrest") == "image/png"
    assert _sniff_image_media_type(b"GIF89a...") == "image/gif"
    assert _sniff_image_media_type(b"RIFF\x00\x00\x00\x00WEBPxx") == "image/webp"
    assert _sniff_image_media_type(b"not-an-image") is None
