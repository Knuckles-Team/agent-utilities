from __future__ import annotations

"""Media transcription bridge — speech → text via faster-whisper.

CONCEPT:ECO-4.31 — Media Transcription Bridge

A thin client for the ``faster-whisper`` service (``fedirz/faster-whisper-server``),
which exposes an **OpenAI-compatible** transcription API. This gives the harness
speech-to-text to complement the speech *generation* in :mod:`gateway`
(CONCEPT:ECO-4.30); transcribed text can then flow into the KG through the
document-source connector framework like any other document.

``httpx`` is lazy; ``WHISPER_URL`` (or ``FASTER_WHISPER_URL``) sets the endpoint;
an unreachable service raises :class:`~.gateway.MediaServiceError`.
"""

import logging
import os
from typing import Any

from .gateway import MediaServiceError, _request

logger = logging.getLogger(__name__)

__all__ = ["Transcriber", "transcribe"]

DEFAULT_WHISPER_URL = os.environ.get(
    "WHISPER_URL",
    os.environ.get("FASTER_WHISPER_URL", "http://faster-whisper.arpa:8000"),
)


class Transcriber:
    """Speech-to-text via the faster-whisper OpenAI-compatible API (ECO-4.31).

    Args:
        base_url: faster-whisper base URL (``WHISPER_URL`` env by default).
        model: Whisper model name advertised by the server (default ``whisper-1``,
            the OpenAI-compatible alias the server maps to its loaded model).
        http_fn: Optional injected HTTP caller for offline tests.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        model: str = "whisper-1",
        http_fn: Any = None,
    ) -> None:
        self.base_url = (base_url or DEFAULT_WHISPER_URL).rstrip("/")
        self.model = model
        self._http_fn = http_fn

    def transcribe(
        self,
        audio: bytes,
        *,
        filename: str = "audio.wav",
        language: str | None = None,
        translate: bool = False,
    ) -> str:
        """Transcribe ``audio`` bytes → text.

        Posts multipart to ``/v1/audio/transcriptions`` (or
        ``/v1/audio/translations`` when ``translate`` is set), the OpenAI-
        compatible endpoints faster-whisper serves.

        Raises:
            MediaServiceError: on empty input or an unreachable/erroring service.
        """
        if not audio:
            raise MediaServiceError("transcribe() requires non-empty audio bytes")
        endpoint = "/v1/audio/translations" if translate else "/v1/audio/transcriptions"
        data: dict[str, Any] = {"model": self.model}
        if language:
            data["language"] = language
        resp = _request(
            self._http_fn,
            "POST",
            f"{self.base_url}{endpoint}",
            files={"file": (filename, audio, "application/octet-stream")},
            data=data,
        )
        try:
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise MediaServiceError(f"whisper returned non-JSON: {exc}") from exc
        text = payload.get("text") if isinstance(payload, dict) else None
        if text is None:
            raise MediaServiceError("whisper response carried no 'text'")
        return str(text)


def transcribe(audio: bytes, *, language: str | None = None) -> str:
    """Convenience: transcribe ``audio`` bytes to text (CONCEPT:ECO-4.31)."""
    return Transcriber().transcribe(audio, language=language)
