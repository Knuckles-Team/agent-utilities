"""Voice-note transcription for messaging (CONCEPT:AU-ECO.messaging.whisper-transcription).

When a user sends a voice note / audio instead of text, transcribe it to text with the
audio-transcriber's Whisper backend and feed the transcript into the normal message flow —
so you can just talk to the agent. Best-effort + opt-out (``MESSAGING_VOICE=0``); the model
loads lazily once and transcription runs off the event loop.

CONCEPT:AU-ECO.messaging.whisper-transcription — Voice input → transcript via Whisper
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from typing import Any

from agent_utilities.core.config import setting

logger = logging.getLogger(__name__)

_backend: Any = None  # cached Whisper backend (model loaded once)


def _enabled() -> bool:
    return str(setting("MESSAGING_VOICE", "1")).strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


class _FasterWhisper:
    """Thin in-process faster-whisper transcriber — the lean path when the full
    ``audio-transcriber`` package (which hard-deps pyaudio/portaudio) isn't installed.
    The messaging container ships only ``faster-whisper``."""

    def __init__(self, model_name: str) -> None:
        from faster_whisper import WhisperModel

        # CPU-safe defaults: the messaging host has no GPU allocated; int8 is the
        # fastest CPU compute type. The model downloads + caches on first use.
        self.model = WhisperModel(model_name, device="cpu", compute_type="int8")

    def transcribe(self, path: str) -> dict:
        # faster-whisper segments already carry their own leading space, so concatenate
        # directly (joining with a space would double them).
        segments, _info = self.model.transcribe(path)
        seg_list = list(segments)
        return {
            "text": "".join(seg.text for seg in seg_list).strip(),
            # Timed segments (CONCEPT:AU-KG.identity.evidence-spine-convergence) —
            # the raw start/end faster-whisper already computes per segment, kept
            # here (not just concatenated away) so a caller with the source bytes
            # can locate a transcript's `AudioSegment` evidence loci.
            "segments": [
                {
                    "start": float(getattr(seg, "start", 0.0) or 0.0),
                    "end": float(getattr(seg, "end", 0.0) or 0.0),
                    "text": seg.text,
                }
                for seg in seg_list
            ],
        }


def _get_backend() -> Any:
    global _backend
    if _backend is not None:
        return _backend
    model_name = str(setting("MESSAGING_VOICE_MODEL", "base"))
    try:
        # Prefer the full audio-transcriber backend when the package is present.
        from audio_transcriber.audio_transcriber import FasterWhisperBackend

        backend: Any = FasterWhisperBackend(logger)
        backend.load_model(model_name)
    except ImportError:
        # Lean path: faster-whisper directly (no pyaudio-bound full package).
        backend = _FasterWhisper(model_name)
    _backend = backend
    return backend


def _sync_transcribe_full(path: str) -> dict:
    """Run the backend transcribe and always return the full ``{"text", "segments"}`` shape.

    The full ``audio_transcriber`` package's backend already returns ``segments``
    (each carrying ``start``/``end``); the lean in-process ``_FasterWhisper`` now
    does too (see its ``transcribe``). Any other backend shape degrades to an
    empty ``segments`` list rather than raising.
    """
    result = _get_backend().transcribe(path)
    if isinstance(result, dict):
        return {
            "text": str(result.get("text", "")).strip(),
            "segments": list(result.get("segments") or []),
        }
    return {"text": str(result).strip(), "segments": []}


def _sync_transcribe(path: str) -> str:
    return str(_sync_transcribe_full(path).get("text", "")).strip()


async def _transcribe_bytes_full(content: bytes, *, suffix: str = ".ogg") -> dict:
    """Transcribe already-in-memory audio ``content``, returning ``{"text", "segments"}``.

    Shared by :func:`transcribe_voice` (which downloads its own bytes) and
    :func:`transcribe_bytes_with_segments` (for a caller — e.g. the messaging
    durable-media persist path — that already downloaded the bytes for storage
    and would otherwise fetch them a second time just to transcribe).
    """
    if not _enabled() or not content:
        return {"text": "", "segments": []}
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
        fh.write(content)
        path = fh.name
    try:
        # Whisper load/transcribe is blocking — run off the event loop.
        result = await asyncio.to_thread(_sync_transcribe_full, path)
        text = str(result.get("text", "")).strip()
        if text:
            logger.info(
                "[CONCEPT:AU-ECO.messaging.whisper-transcription] Transcribed voice note (%d chars).",
                len(text),
            )
        return result
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


async def transcribe_voice(url: str) -> str:
    """Download a voice/audio attachment and return its transcript ("" on failure)."""
    if not _enabled() or not url:
        return ""
    try:
        from agent_utilities.protocols.source_connectors.http_safety import (
            configured_source_http_policy,
            safe_get_bytes_async,
        )

        content, _encoding = await safe_get_bytes_async(
            url, timeout=30.0, **configured_source_http_policy()
        )
        result = await _transcribe_bytes_full(content)
        return str(result.get("text", "")).strip()
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "[CONCEPT:AU-ECO.messaging.whisper-transcription] voice transcription failed (%s)",
            type(e).__name__,
        )
        return ""


async def transcribe_bytes_with_segments(
    content: bytes, *, suffix: str = ".ogg"
) -> tuple[str, list[dict]]:
    """Transcribe already-downloaded audio ``content``, returning ``(text, segments)``.

    CONCEPT:AU-KG.identity.evidence-spine-convergence — the ASR-with-timing half of the
    ``AudioSegment`` evidence-locus producer (``MediaStore.store_audio_segment_evidence``'s
    own docstring names this exact shape as its "natural producer"). Each segment dict
    carries ``start``/``end`` in FLOATING-POINT SECONDS (the Whisper-native unit); a
    caller writing evidence loci converts to the locus's millisecond fields. Returns
    ``("", [])`` on any failure or when transcription is disabled — never raises, so a
    best-effort background persist path can call this unconditionally.
    """
    try:
        result = await _transcribe_bytes_full(content, suffix=suffix)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "[CONCEPT:AU-KG.identity.evidence-spine-convergence] segment transcription failed (%s)",
            type(e).__name__,
        )
        return "", []
    return str(result.get("text", "")).strip(), list(result.get("segments") or [])
