"""Wiring tests for the WebUI's host-injected voice-transcription delegation.

CONCEPT:AU-ECO.mcp.webui-voice-transcription-delegation

Mirrors ``test_webui_mcp_delegation.py``'s shape: verify the helper's shape,
drive ``_transcribe_voice`` against a mocked
``media.sidecar_delegate.delegate_extract`` (both success and the honest
degrade path), and statically prove BOTH production callers of
``create_agent_web_app`` actually inject it -- this repo's own
``webui_mcp_delegation.py`` module docstring documents that a control wired
at only one of those two call sites (this repo's ``app.py`` embedded mount,
and ``agent-webui``'s own ``server.py`` standalone CLI entrypoint) is exactly
the GOC-60-W04b failure mode.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from agent_utilities.media.sidecar_delegate import SidecarDelegationResult
from agent_utilities.server.webui_voice_delegation import (
    webui_voice_delegation_helpers,
)


def test_helpers_expose_transcribe_voice() -> None:
    helpers = webui_voice_delegation_helpers()
    assert set(helpers) == {"transcribe_voice"}


@pytest.mark.asyncio
async def test_transcribe_voice_joins_segment_text_in_order() -> None:
    transcribe_voice = webui_voice_delegation_helpers()["transcribe_voice"]

    fake_result = SidecarDelegationResult(
        available=True,
        modality="audio",
        provider="audio-transcriber-mcp",
        tool="transcribe_media",
        action="transcribe_segments",
        activity_id="activity-1",
        raw={
            "segments": [
                {"start_ms": 0, "end_ms": 500, "text": " hello ", "confidence": 0.9},
                {"start_ms": 500, "end_ms": 1000, "text": "world", "confidence": 0.8},
            ],
            "provider": "faster-whisper",
        },
    )

    with patch(
        "agent_utilities.media.sidecar_delegate.delegate_extract",
        return_value=fake_result,
    ) as mock_delegate:
        result = await transcribe_voice(content=b"clip-bytes", content_type="audio/webm")

    assert result == {"text": "hello world"}
    # Called with modality="audio" -- the SAME sidecar ingest_audio_via_sidecar
    # uses -- and never persists (no session/media_store kwargs, unlike
    # ingest_audio_via_sidecar's heavier call).
    _, kwargs = mock_delegate.call_args
    assert kwargs["modality"] == "audio"
    assert kwargs["media_type"] == "audio/webm"


@pytest.mark.asyncio
async def test_transcribe_voice_computes_its_own_digest_never_persists() -> None:
    import hashlib

    transcribe_voice = webui_voice_delegation_helpers()["transcribe_voice"]
    content = b"a dictation clip"
    expected_digest = hashlib.sha256(content).hexdigest()

    fake_result = SidecarDelegationResult(
        available=True,
        modality="audio",
        provider="audio-transcriber-mcp",
        tool="transcribe_media",
        action="transcribe_segments",
        activity_id=None,
        raw={"segments": []},
    )
    with patch(
        "agent_utilities.media.sidecar_delegate.delegate_extract",
        return_value=fake_result,
    ) as mock_delegate:
        await transcribe_voice(content=content, content_type="audio/wav")

    assert mock_delegate.call_args.kwargs["digest"] == expected_digest
    assert mock_delegate.call_args.args[0] == content


@pytest.mark.asyncio
async def test_transcribe_voice_raises_never_fabricates_on_sidecar_unavailable() -> None:
    """The sidecar being unavailable (e.g. audio-transcriber-mcp unreachable)
    must raise so the WebUI route answers a genuine 500 -- never a canned or
    empty-string "transcript"."""
    transcribe_voice = webui_voice_delegation_helpers()["transcribe_voice"]

    fake_result = SidecarDelegationResult(
        available=False,
        modality="audio",
        provider="audio-transcriber-mcp",
        tool="transcribe_media",
        action="transcribe_segments",
        activity_id=None,
        raw={},
        error="RuntimeError: sidecar unreachable",
    )
    with patch(
        "agent_utilities.media.sidecar_delegate.delegate_extract",
        return_value=fake_result,
    ):
        with pytest.raises(RuntimeError, match="sidecar unreachable"):
            await transcribe_voice(content=b"clip-bytes", content_type="audio/webm")


def test_app_py_injects_the_voice_delegation_helper_before_mounting() -> None:
    """``server/app.py``'s embedded-mount caller must actually add it to the
    helper bundle before ``create_agent_web_app`` mounts."""
    import agent_utilities.server.app as app_module

    source = Path(app_module.__file__).read_text(encoding="utf-8")
    assert "webui_voice_delegation_helpers()" in source
    injected = source.index("helpers.update(webui_voice_delegation_helpers())")
    created = source.index("web_app = create_agent_web_app(")
    assert injected < created, (
        "the voice delegation helper must be merged into `helpers` BEFORE it "
        "is handed to create_agent_web_app"
    )


## NOTE: the OTHER production caller of ``create_agent_web_app``
## (``agent-webui``'s own standalone CLI entrypoint, ``agent/agent_webui/
## server.py::main`` -- confirmed live by the k8s Deployment's Dockerfile
## CMD, ``python -m agent_webui.server``) is a SEPARATE repository this one
## cannot import or statically scan without coupling this repo's test suite
## to that repo's unmerged branch state. It is fixed too (same PR wave), and
## verified directly there: agent-webui's own
## ``test_workspace_helpers_chokepoint.py`` (the authoritative AST gate for
## every ``create_agent_web_app`` call site in ITS repo) passes against the
## edit, plus a direct read of ``main()`` confirms
## ``webui_voice_delegation_helpers()`` is merged into ``workspace_helpers``
## before the app is built -- see that repo's own test run in this change's
## report. Deliberately not re-asserted here as a cross-repo file-path check
## (that pattern breaks CI the moment either repo's directory layout shifts,
## and passes vacuously via ``skip`` in exactly the case that matters least).
