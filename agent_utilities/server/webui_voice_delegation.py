"""The host-injected voice-transcription delegation the WebUI backend delegates through.

CONCEPT:AU-ECO.mcp.webui-voice-transcription-delegation

``agent_webui.api_extensions``'s ``POST /voice/transcribe`` route never talks to a
transcription backend itself. It calls one host-supplied workspace helper,
``transcribe_voice``, and reports 501 when it is absent (GOC-07's honest-501
contract — see ``agent-webui``'s ``tests/test_destubbed_endpoints.py::
test_voice_transcribe_honest_without_whisper``). This module is that host side,
mirroring :mod:`~agent_utilities.server.webui_mcp_delegation`'s shape.

It delegates through the SAME governed audio sidecar
(``audio-transcriber-mcp``, :func:`agent_utilities.media.sidecar_delegate.delegate_extract`)
that ``graph_media_sidecar action=ingest_audio`` uses — but calls only the
transcription primitive, never the heavier
:func:`~agent_utilities.media.audio_sidecar.ingest_audio_via_sidecar` ingest
pipeline. That distinction is deliberate: ``ingest_audio_via_sidecar`` stores
the raw audio as a permanent ``:AssetOccurrence`` blob and writes
``AudioSegment`` evidence loci for every call — right for a genuine media
ingestion, wrong for an ephemeral mic-dictation clip, which should become
text in a chat box, not permanent KG evidence on every utterance.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Protocol, TypedDict

logger = logging.getLogger(__name__)

__all__ = ["WebUiVoiceDelegation", "webui_voice_delegation_helpers"]


class _TranscribeVoice(Protocol):
    def __call__(self, *, content: bytes, content_type: str) -> Any: ...


class WebUiVoiceDelegation(TypedDict):
    """The one workspace-helper key the WebUI looks up by name.

    A typed seam on purpose, matching ``WebUiMcpDelegation``: the WebUI
    resolves this with ``get_helper('transcribe_voice')`` and answers 501
    when it is missing, so a rename here must be a type error rather than a
    silently-unwired route.
    """

    transcribe_voice: _TranscribeVoice


async def _transcribe_voice(*, content: bytes, content_type: str) -> dict[str, str]:
    """Transcribe one bounded audio clip via the governed audio sidecar.

    Computes the clip's own SHA-256 digest (this call never persists the
    clip, so there is no stored blob to derive one from) and hands it
    straight to :func:`~agent_utilities.media.sidecar_delegate.delegate_extract`
    for modality ``"audio"`` — the SAME sidecar
    (``audio-transcriber-mcp``'s ``transcribe_media`` tool,
    action=``transcribe_segments``) ``ingest_audio_via_sidecar`` calls, minus
    its blob-store + ``AudioSegment`` write-back (see the module docstring).

    Raises on any failure (sidecar unreachable, no capability, empty
    transcript-bearing segments) rather than fabricating a transcript —
    ``api_extensions.transcribe_voice_chunk`` catches this and answers a
    genuine 500, never inventing text. Never returns a canned/placeholder
    string.
    """
    from agent_utilities.media.sidecar_delegate import delegate_extract

    digest = hashlib.sha256(content).hexdigest()
    result = delegate_extract(
        content, digest=digest, media_type=content_type, modality="audio"
    )
    if not result.available:
        raise RuntimeError(result.error or "audio transcription sidecar unavailable")

    segments = result.raw.get("segments")
    segments = segments if isinstance(segments, list) else []
    text = " ".join(
        str(segment.get("text") or "").strip()
        for segment in segments
        if isinstance(segment, dict) and str(segment.get("text") or "").strip()
    ).strip()
    return {"text": text}


def webui_voice_delegation_helpers() -> WebUiVoiceDelegation:
    """The ``transcribe_voice`` workspace helper."""
    return WebUiVoiceDelegation(transcribe_voice=_transcribe_voice)
