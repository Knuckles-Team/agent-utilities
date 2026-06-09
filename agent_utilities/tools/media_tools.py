from __future__ import annotations

"""Agent tools for media generation + transcription (CONCEPT:ECO-4.30 / ECO-4.31).

Exposes the :mod:`agent_utilities.ecosystem.media` gateway as agent tools so an
agent can synthesize speech, generate images/video, and transcribe audio via the
self-hosted services. Outputs (binary) are written to an output directory and the
tool returns a JSON description (path + bytes) — agents handle paths, not raw
bytes. Registered under the ``MEDIA_TOOLS`` env gate in ``tool_registry``.
"""

import base64
import json
import logging
import os
import tempfile
import time
from pathlib import Path

from pydantic_ai import RunContext

from ..models import AgentDeps

logger = logging.getLogger(__name__)


def _output_dir() -> Path:
    """Resolve the media output directory (``MEDIA_OUTPUT_DIR`` or a temp dir)."""
    base = os.environ.get("MEDIA_OUTPUT_DIR") or os.path.join(
        tempfile.gettempdir(), "agent_utilities_media"
    )
    path = Path(base)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _stamp(prefix: str, ext: str) -> Path:
    return _output_dir() / f"{prefix}_{int(time.time() * 1000)}.{ext}"


async def synthesize_speech(
    ctx: RunContext[AgentDeps], text: str, speaker: str = "", language: str = "en"
) -> str:
    """Synthesize speech audio from text via the xtts service (CONCEPT:ECO-4.30).

    Args:
        ctx: The agent run context.
        text: The text to speak.
        speaker: Optional studio speaker name (server default when empty).
        language: Language code (default ``en``).

    Returns:
        JSON ``{path, bytes}`` describing the written WAV file, or ``{error}``.
    """
    from ..ecosystem.media import SpeechSynthesizer

    try:
        wav = SpeechSynthesizer().synthesize(
            text, speaker=speaker or None, language=language
        )
        out = _stamp("speech", "wav")
        out.write_bytes(wav)
        return json.dumps({"path": str(out), "bytes": len(wav)})
    except Exception as e:  # noqa: BLE001
        return json.dumps({"error": str(e)})


async def generate_image(
    ctx: RunContext[AgentDeps],
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    steps: int = 4,
    backend: str = "flux",
) -> str:
    """Generate an image from a prompt (CONCEPT:ECO-4.30).

    Args:
        ctx: The agent run context.
        prompt: The text prompt to render.
        width: Image width in pixels.
        height: Image height in pixels.
        steps: Diffusion steps (4 = fast schnell-style; raise for quality).
        backend: Image service — ``flux`` (flux.2) or ``sd35`` (Stable Diffusion 3.5).

    Returns:
        JSON ``{path, bytes}`` describing the written PNG, or ``{error}``.
    """
    from ..ecosystem.media import generate_image as _gen

    try:
        img = _gen(prompt, backend=backend, width=width, height=height, steps=steps)
        out = _stamp("image", "png")
        out.write_bytes(img)
        return json.dumps({"path": str(out), "bytes": len(img)})
    except Exception as e:  # noqa: BLE001
        return json.dumps({"error": str(e)})


async def generate_video(
    ctx: RunContext[AgentDeps], prompt: str, num_frames: int = 65, fps: int = 16
) -> str:
    """Generate a short video from a prompt via hunyuanvideo (CONCEPT:ECO-4.30).

    Args:
        ctx: The agent run context.
        prompt: The text prompt to animate.
        num_frames: Number of frames (kept small to stay light on the GPU host).
        fps: Frames per second.

    Returns:
        JSON describing the written MP4 (``{path, bytes}``), a pending job
        (``{job_id}``), or a fetch URL (``{url}``); or ``{error}``.
    """
    from ..ecosystem.media import VideoGenerator

    try:
        result = VideoGenerator().generate(prompt, num_frames=num_frames, fps=fps)
        if result.get("video"):
            out = _stamp("video", "mp4")
            out.write_bytes(result["video"])
            return json.dumps({"path": str(out), "bytes": len(result["video"])})
        return json.dumps({k: v for k, v in result.items() if k != "video"})
    except Exception as e:  # noqa: BLE001
        return json.dumps({"error": str(e)})


async def transcribe_audio(
    ctx: RunContext[AgentDeps], audio_path: str, language: str = ""
) -> str:
    """Transcribe an audio file to text via faster-whisper (CONCEPT:ECO-4.31).

    Args:
        ctx: The agent run context.
        audio_path: Path to an audio file, or base64 audio bytes.
        language: Optional language hint.

    Returns:
        JSON ``{text}`` with the transcription, or ``{error}``.
    """
    from ..ecosystem.media import Transcriber

    try:
        p = Path(audio_path)
        if p.exists():
            audio = p.read_bytes()
            fname = p.name
        else:
            audio = base64.b64decode(audio_path)
            fname = "audio.wav"
        text = Transcriber().transcribe(
            audio, filename=fname, language=language or None
        )
        return json.dumps({"text": text})
    except Exception as e:  # noqa: BLE001
        return json.dumps({"error": str(e)})


media_tools = [synthesize_speech, generate_image, generate_video, transcribe_audio]
