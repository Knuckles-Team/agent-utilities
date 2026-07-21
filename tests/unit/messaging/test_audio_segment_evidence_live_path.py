"""Live-path proof for the ``AudioSegment`` evidence-locus producer
(CONCEPT:AU-KG.identity.evidence-spine-convergence, W2 wire-up #2).

``docs/architecture/evidence_spine_convergence.md`` documented all eleven
``eg_modality::EvidenceSpan`` loci as having a tested :class:`MediaStore`
method but explicitly flagged that NONE had a live AU producer call — every
``store_<locus>_evidence`` method was reachable but never actually invoked
off a real ingestion/enrichment path. This test proves the gap is closed for
the ``AudioSegment`` locus: the messaging durable-media persist path
(``router._persist_media``), given a real voice-note attachment, transcribes
it WITH segment timing and calls
``MediaStore.store_audio_segment_evidence`` for every timed segment — as a
side effect of the EXISTING entry point (``_persist_media``, invoked from
``_persist_and_enrich`` on every inbound message with attachments), not a
new opt-in call site a caller must remember to use.

Offline: HTTP download is faked (mirrors ``test_persist_media_occurrence.py``),
:class:`MediaStore` is faked, and Whisper transcription is faked via
monkeypatching ``agent_utilities.messaging.voice.transcribe_bytes_with_segments``
— no network, no engine, no real ASR model.
"""

from __future__ import annotations

import httpx
import pytest

from agent_utilities.messaging import router, voice
from agent_utilities.messaging.models import (
    EventType,
    InboundEvent,
    MediaAttachment,
    MediaType,
    Message,
)


class _FakeResp:
    def __init__(self, content: bytes = b"OGGBYTES") -> None:
        self.content = content
        self.headers = {"content-type": "audio/ogg"}

    def raise_for_status(self) -> None:
        return None


class _FakeHttpxClient:
    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self) -> "_FakeHttpxClient":
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    async def get(self, url: str) -> _FakeResp:
        return _FakeResp()


class _StoredMedia:
    def __init__(self, occurrence_id: str) -> None:
        self.occurrence_id = occurrence_id
        self.digest = "deadbeef"
        self.blob_id = f"blob:{self.digest}"


class _FakeStore:
    """Fakes the two :class:`MediaStore` methods the persist path calls."""

    def __init__(self) -> None:
        self.media_calls: list[tuple[bytes, dict]] = []
        self.segment_calls: list[tuple[bytes, dict]] = []

    def store_media(self, data: bytes, **kwargs):
        self.media_calls.append((data, kwargs))
        return _StoredMedia("occ-voice-1")

    def store_audio_segment_evidence(self, data: bytes, **kwargs):
        self.segment_calls.append((data, kwargs))
        return object()


def _voice_event() -> InboundEvent:
    return InboundEvent(
        event_type=EventType.MESSAGE,
        platform="slack",
        channel_id="C1",
        thread_id="T1",
        user_id="u1",
        message=Message(
            id="m1",
            author_id="u1",
            channel_id="C1",
            attachments=[
                MediaAttachment(media_type=MediaType.VOICE_NOTE, url="http://x/voice.ogg")
            ],
        ),
    )


@pytest.mark.asyncio
async def test_persist_media_writes_audio_segment_evidence_for_voice_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(httpx, "AsyncClient", _FakeHttpxClient)
    store = _FakeStore()
    monkeypatch.setattr(router, "_resolve_media_store", lambda engine: store)

    async def _fake_transcribe(content: bytes):
        assert content == b"OGGBYTES"
        return "hello world", [
            {"start": 0.0, "end": 1.2, "text": "hello"},
            {"start": 1.2, "end": 2.4, "text": " world"},
        ]

    monkeypatch.setattr(voice, "transcribe_bytes_with_segments", _fake_transcribe)

    ev = _voice_event()
    # The EXISTING durable-media entry point — no new call site.
    await router._persist_media(object(), ev, message_memory_id="mem:1")

    # store_media ran once (the occurrence + blob write, unchanged behavior).
    assert len(store.media_calls) == 1

    # AND the AudioSegment evidence locus was populated — one call per
    # transcribed segment, each anchored to the SAME occurrence store_media
    # just created, with the segment's timing converted to milliseconds.
    assert len(store.segment_calls) == 2
    (data1, kw1), (data2, kw2) = store.segment_calls
    assert data1 == data2 == b"OGGBYTES"
    assert kw1["audio_id"] == kw2["audio_id"] == "occ-voice-1"
    assert kw1["start_ms"] == 0 and kw1["end_ms"] == 1200
    assert kw2["start_ms"] == 1200 and kw2["end_ms"] == 2400
    assert kw1["source"] == "slack"


@pytest.mark.asyncio
async def test_persist_media_skips_segment_evidence_for_non_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An image attachment never triggers transcription/segment-evidence writes."""
    monkeypatch.setattr(httpx, "AsyncClient", _FakeHttpxClient)
    store = _FakeStore()
    monkeypatch.setattr(router, "_resolve_media_store", lambda engine: store)

    called = False

    async def _fake_transcribe(content: bytes):
        nonlocal called
        called = True
        return "", []

    monkeypatch.setattr(voice, "transcribe_bytes_with_segments", _fake_transcribe)

    ev = InboundEvent(
        event_type=EventType.MESSAGE,
        platform="slack",
        channel_id="C1",
        message=Message(
            id="m1",
            author_id="u1",
            channel_id="C1",
            attachments=[MediaAttachment(media_type=MediaType.IMAGE, url="http://x/img.png")],
        ),
    )
    await router._persist_media(object(), ev, message_memory_id="mem:1")

    assert len(store.media_calls) == 1
    assert store.segment_calls == []
    assert called is False


@pytest.mark.asyncio
async def test_persist_media_skips_segment_evidence_when_store_media_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No occurrence (``store_media`` returned ``None``) → no orphaned evidence write."""
    monkeypatch.setattr(httpx, "AsyncClient", _FakeHttpxClient)

    class _FailingStore(_FakeStore):
        def store_media(self, data: bytes, **kwargs):
            self.media_calls.append((data, kwargs))
            return None

    store = _FailingStore()
    monkeypatch.setattr(router, "_resolve_media_store", lambda engine: store)

    called = False

    async def _fake_transcribe(content: bytes):
        nonlocal called
        called = True
        return "text", [{"start": 0.0, "end": 1.0, "text": "x"}]

    monkeypatch.setattr(voice, "transcribe_bytes_with_segments", _fake_transcribe)

    ev = _voice_event()
    await router._persist_media(object(), ev, message_memory_id="mem:1")

    assert store.segment_calls == []
    assert called is False
