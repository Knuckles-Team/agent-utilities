from __future__ import annotations

"""Bounded, workspace-confined media generation and transcription tools."""

import base64
import binascii
import json
import os
import secrets
from pathlib import Path

from pydantic_ai import RunContext

from agent_utilities.security.persistence_privacy import (
    PersistencePrivacyGuard,
    persistence_reference,
)

from ..models import AgentDeps

_MAX_PROMPT_BYTES = 64 * 1024
_MAX_AUDIO_BYTES = 64 * 1024 * 1024
_MAX_OUTPUT_BYTES = 256 * 1024 * 1024


def _workspace_root(ctx: RunContext[AgentDeps]) -> Path:
    candidate = Path(ctx.deps.workspace_path)
    if candidate.is_symlink():
        raise ValueError("workspace unavailable")
    root = candidate.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("workspace unavailable")
    return root


def _output_dir(ctx: RunContext[AgentDeps]) -> Path:
    path = _workspace_root(ctx) / ".agents" / "media"
    path.mkdir(parents=True, mode=0o700, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise ValueError("media output directory is unsafe")
    if os.name == "posix":
        path.chmod(0o700)
    return path


def _write_asset(
    ctx: RunContext[AgentDeps], prefix: str, extension: str, data: bytes
) -> str:
    if not isinstance(data, bytes) or not data or len(data) > _MAX_OUTPUT_BYTES:
        raise ValueError("media output exceeds the supported limit")
    name = f"{prefix}_{secrets.token_hex(12)}.{extension}"
    destination = _output_dir(ctx) / name
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return f"media://{name}"


def _bounded_text(value: str, *, required: bool = True) -> str:
    if not isinstance(value, str) or (required and not value.strip()):
        raise ValueError("media text is invalid")
    if len(value.encode("utf-8")) > _MAX_PROMPT_BYTES:
        raise ValueError("media text exceeds the supported limit")
    return value


async def synthesize_speech(
    ctx: RunContext[AgentDeps], text: str, speaker: str = "", language: str = "en"
) -> str:
    """Synthesize speech and return a logical asset reference, never a host path."""
    from ..ecosystem.media import SpeechSynthesizer

    try:
        payload = _bounded_text(text)
        speaker = _bounded_text(speaker, required=False)
        if not isinstance(language, str) or not 1 <= len(language) <= 16:
            raise ValueError("language is invalid")
        wav = SpeechSynthesizer().synthesize(
            payload,
            speaker=speaker or None,
            language=language,
        )
        reference = _write_asset(ctx, "speech", "wav", wav)
        return json.dumps({"asset_ref": reference, "bytes": len(wav)})
    except Exception:
        return json.dumps({"error": "speech generation failed"})


async def generate_image(
    ctx: RunContext[AgentDeps],
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    steps: int = 4,
    backend: str = "flux",
) -> str:
    """Generate a bounded image into the private workspace media store."""
    from ..ecosystem.media import generate_image as _generate

    try:
        prompt = _bounded_text(prompt)
        if backend not in {"flux", "sd35"}:
            raise ValueError("image backend is invalid")
        if not 64 <= width <= 4096 or not 64 <= height <= 4096:
            raise ValueError("image dimensions are invalid")
        if width * height > 16_777_216 or not 1 <= steps <= 100:
            raise ValueError("image resource request is invalid")
        image = _generate(
            prompt,
            backend=backend,
            width=width,
            height=height,
            steps=steps,
        )
        reference = _write_asset(ctx, "image", "png", image)
        return json.dumps({"asset_ref": reference, "bytes": len(image)})
    except Exception:
        return json.dumps({"error": "image generation failed"})


async def generate_video(
    ctx: RunContext[AgentDeps], prompt: str, num_frames: int = 65, fps: int = 16
) -> str:
    """Generate bounded video or return a pseudonymous asynchronous job reference."""
    from ..ecosystem.media import VideoGenerator

    try:
        prompt = _bounded_text(prompt)
        if not 1 <= num_frames <= 300 or not 1 <= fps <= 120:
            raise ValueError("video resource request is invalid")
        result = VideoGenerator().generate(prompt, num_frames=num_frames, fps=fps)
        if not isinstance(result, dict):
            raise ValueError("video response is invalid")
        video = result.get("video")
        if isinstance(video, bytes):
            reference = _write_asset(ctx, "video", "mp4", video)
            return json.dumps({"asset_ref": reference, "bytes": len(video)})
        job = result.get("job_id") or result.get("id")
        if job:
            return json.dumps(
                {
                    "status": "pending",
                    "job_ref": persistence_reference("media_job", job),
                }
            )
        return json.dumps({"status": "pending"})
    except Exception:
        return json.dumps({"error": "video generation failed"})


def _audio_input(ctx: RunContext[AgentDeps], value: str) -> tuple[bytes, str]:
    if not isinstance(value, str) or not value:
        raise ValueError("audio input is invalid")
    if value.startswith("base64:"):
        encoded = value.removeprefix("base64:")
        if len(encoded) > ((_MAX_AUDIO_BYTES + 2) // 3) * 4:
            raise ValueError("audio input exceeds the supported limit")
        try:
            decoded = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("audio input is invalid") from exc
        if not decoded or len(decoded) > _MAX_AUDIO_BYTES:
            raise ValueError("audio input exceeds the supported limit")
        return decoded, "audio.wav"
    root = _workspace_root(ctx)
    supplied = Path(value)
    candidate = supplied if supplied.is_absolute() else root / supplied
    resolved = candidate.resolve(strict=True)
    resolved.relative_to(root)
    if candidate.is_symlink() or not resolved.is_file():
        raise ValueError("audio input is unavailable")
    if resolved.stat().st_size > _MAX_AUDIO_BYTES:
        raise ValueError("audio input exceeds the supported limit")
    return resolved.read_bytes(), resolved.name


async def transcribe_audio(
    ctx: RunContext[AgentDeps], audio_path: str, language: str = ""
) -> str:
    """Transcribe workspace-relative or explicitly prefixed base64 audio."""
    from ..ecosystem.media import Transcriber

    try:
        audio, filename = _audio_input(ctx, audio_path)
        if not isinstance(language, str) or len(language) > 16:
            raise ValueError("language is invalid")
        text = Transcriber().transcribe(
            audio,
            filename=filename,
            language=language or None,
        )
        clean, _ = PersistencePrivacyGuard().sanitize_text(str(text or ""))
        return json.dumps({"text": clean})
    except Exception:
        return json.dumps({"error": "audio transcription failed"})


media_tools = [synthesize_speech, generate_image, generate_video, transcribe_audio]
