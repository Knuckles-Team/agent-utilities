"""Voice-note transcription for messaging (CONCEPT:ECO-4.68).

When a user sends a voice note / audio instead of text, transcribe it to text with the
audio-transcriber's Whisper backend and feed the transcript into the normal message flow —
so you can just talk to the agent. Best-effort + opt-out (``MESSAGING_VOICE=0``); the model
loads lazily once and transcription runs off the event loop.

CONCEPT:ECO-4.68 — Voice input → transcript via Whisper
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


def _get_backend() -> Any:
    global _backend
    if _backend is not None:
        return _backend
    from audio_transcriber.audio_transcriber import FasterWhisperBackend

    backend = FasterWhisperBackend(logger)
    backend.load_model(str(setting("MESSAGING_VOICE_MODEL", "base")))
    _backend = backend
    return backend


def _sync_transcribe(path: str) -> str:
    result = _get_backend().transcribe(path)
    if isinstance(result, dict):
        return str(result.get("text", "")).strip()
    return str(result).strip()


async def transcribe_voice(url: str) -> str:
    """Download a voice/audio attachment and return its transcript ("" on failure)."""
    if not _enabled() or not url:
        return ""
    try:
        import httpx

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as fh:
            fh.write(resp.content)
            path = fh.name
        try:
            # Whisper load/transcribe is blocking — run off the event loop.
            text = await asyncio.to_thread(_sync_transcribe, path)
            if text:
                logger.info(
                    "[CONCEPT:ECO-4.68] Transcribed voice note (%d chars).", len(text)
                )
            return text
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
    except Exception as e:  # noqa: BLE001
        logger.warning("[CONCEPT:ECO-4.68] voice transcription failed: %s", e)
        return ""
