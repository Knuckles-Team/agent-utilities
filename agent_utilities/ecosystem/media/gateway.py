from __future__ import annotations

"""Media generation gateway — universal, multi-backend image / video / speech.

CONCEPT:ECO-4.30 — Media Generation Gateway

A backend-abstracted media layer: each modality (image / video / speech /
transcription) is served by one of several interchangeable **backends**, selected
by name, so the harness is never tied to a single model or server. Backends are
either a self-hosted **ComfyUI** engine (workflow API — the consolidated,
on-demand engine deployed on GB10), an **OpenAI-compatible** images/audio server,
or a **generic REST** ``/generate`` server.

Named backends (override endpoint/model via env or config):
  * **image** — ``stable-diffusion`` (SD3.5), ``flux``, ``qwen-image``,
    ``hunyuanimage``, ``comfyui`` (generic).
  * **video** — ``hunyuanvideo``, ``ltx``, ``comfyui``.
  * **speech** — ``xtts`` (Coqui ``xtts-streaming-server``), ``openai`` (OpenAI-
    compatible ``/v1/audio/speech``).

Transports are thin, lazy-``httpx`` clients: :class:`ImageGenerator` (REST/OpenAI
txt2img), :class:`VideoGenerator` (REST), :class:`SpeechSynthesizer` (xtts), and
:class:`ComfyUIClient` (workflow API). Endpoints come from ``{SERVICE}_URL`` env
vars; ``httpx`` is imported lazily; an unreachable service raises a clear
:class:`MediaServiceError` rather than hanging or returning an empty result.
"""

import base64
import logging
import time
from typing import Any

from agent_utilities.core.config import setting

logger = logging.getLogger(__name__)

__all__ = [
    "MediaServiceError",
    "SpeechSynthesizer",
    "ImageGenerator",
    "VideoGenerator",
    "ComfyUIClient",
    "synthesize_speech",
    "generate_image",
    "generate_video",
    "list_image_backends",
    "list_video_backends",
    "list_speech_backends",
]

# Consolidated ComfyUI engine (image + video on demand) — the GB10 default.
DEFAULT_COMFYUI_URL = setting("COMFYUI_URL", "http://comfyui.arpa:8188")
DEFAULT_XTTS_URL = setting("XTTS_URL", "http://xtts.arpa:5002")
DEFAULT_OPENAI_TTS_URL = setting("OPENAI_TTS_URL", "http://xtts.arpa:5002")
# Optional per-model REST servers (used when a backend is run standalone).
DEFAULT_FLUX_URL = setting("FLUX_URL", DEFAULT_COMFYUI_URL)
DEFAULT_SD35_URL = setting("SD35_URL", DEFAULT_COMFYUI_URL)
DEFAULT_QWEN_IMAGE_URL = setting("QWEN_IMAGE_URL", DEFAULT_COMFYUI_URL)
DEFAULT_HUNYUAN_IMAGE_URL = setting("HUNYUAN_IMAGE_URL", DEFAULT_COMFYUI_URL)
DEFAULT_HUNYUAN_URL = setting("HUNYUAN_URL", DEFAULT_COMFYUI_URL)
DEFAULT_LTX_URL = setting("LTX_URL", DEFAULT_COMFYUI_URL)

# Named image backends → (transport, config). ``comfyui`` transport routes through
# the consolidated engine with the named checkpoint; ``rest`` hits a standalone
# txt2img server. Add a row to support a new model — no new code.
_IMAGE_BACKENDS: dict[str, dict[str, Any]] = {
    "comfyui": {"transport": "comfyui", "url": DEFAULT_COMFYUI_URL, "checkpoint": ""},
    "flux": {
        "transport": "comfyui",
        "url": DEFAULT_FLUX_URL,
        "checkpoint": "flux1-schnell",
    },
    "stable-diffusion": {
        "transport": "comfyui",
        "url": DEFAULT_SD35_URL,
        "checkpoint": "sd3.5_medium",
    },
    "sd35": {
        "transport": "comfyui",
        "url": DEFAULT_SD35_URL,
        "checkpoint": "sd3.5_medium",
    },
    "qwen-image": {
        "transport": "comfyui",
        "url": DEFAULT_QWEN_IMAGE_URL,
        "checkpoint": "qwen-image",
    },
    "hunyuanimage": {
        "transport": "comfyui",
        "url": DEFAULT_HUNYUAN_IMAGE_URL,
        "checkpoint": "hunyuan-image",
    },
}

# Named video backends.
_VIDEO_BACKENDS: dict[str, dict[str, Any]] = {
    "comfyui": {"transport": "comfyui", "url": DEFAULT_COMFYUI_URL, "checkpoint": ""},
    "hunyuanvideo": {
        "transport": "comfyui",
        "url": DEFAULT_HUNYUAN_URL,
        "checkpoint": "hunyuan-video",
    },
    "ltx": {"transport": "comfyui", "url": DEFAULT_LTX_URL, "checkpoint": "ltx-video"},
}

# Named speech backends.
_SPEECH_BACKENDS: dict[str, dict[str, Any]] = {
    "xtts": {"transport": "xtts", "url": DEFAULT_XTTS_URL},
    "openai": {"transport": "openai", "url": DEFAULT_OPENAI_TTS_URL},
}


def list_image_backends() -> list[str]:
    """Return the registered image backend names (CONCEPT:ECO-4.30)."""
    return sorted(_IMAGE_BACKENDS)


def list_video_backends() -> list[str]:
    """Return the registered video backend names (CONCEPT:ECO-4.30)."""
    return sorted(_VIDEO_BACKENDS)


def list_speech_backends() -> list[str]:
    """Return the registered speech backend names (CONCEPT:ECO-4.30)."""
    return sorted(_SPEECH_BACKENDS)


# HTTP caller signature: (method, url, **kwargs) -> (status_code, json|bytes|text).
HttpFn = Any


class MediaServiceError(RuntimeError):
    """Raised when a media-generation service is unreachable or errors.

    CONCEPT:ECO-4.30 — the clear, non-silent failure path.
    """


def _httpx():
    try:
        import httpx

        return httpx
    except ImportError as exc:  # pragma: no cover - environment without httpx
        raise MediaServiceError(
            "media gateway needs 'httpx'. Install it, or pass http_fn for offline use."
        ) from exc


def _request(http_fn: HttpFn | None, method: str, url: str, **kwargs: Any) -> Any:
    """Issue an HTTP request via the injected ``http_fn`` or lazy ``httpx``.

    The injected ``http_fn(method, url, **kwargs)`` returns an ``httpx.Response``
    -like object (with ``.status_code`` / ``.json()`` / ``.content`` / ``.text``)
    — used by tests with ``httpx.MockTransport`` semantics or a fake.
    """
    timeout = kwargs.pop("timeout", 600.0)
    if http_fn is not None:
        return http_fn(method, url, **kwargs)
    httpx = _httpx()
    try:
        resp = httpx.request(method, url, timeout=timeout, **kwargs)
        resp.raise_for_status()
        return resp
    except Exception as exc:  # noqa: BLE001 — surface as a clear gateway error
        raise MediaServiceError(f"{method} {url} failed: {exc}") from exc


class SpeechSynthesizer:
    """Text-to-speech via the Coqui ``xtts-streaming-server`` (CONCEPT:ECO-4.30).

    Args:
        base_url: xtts base URL (``XTTS_URL`` env by default).
        http_fn: Optional injected HTTP caller for offline tests.
    """

    def __init__(
        self, base_url: str | None = None, *, http_fn: HttpFn | None = None
    ) -> None:
        self.base_url = (base_url or DEFAULT_XTTS_URL).rstrip("/")
        self._http_fn = http_fn

    def studio_speakers(self) -> dict[str, Any]:
        """Return the server's preset studio speakers (name → conditioning)."""
        resp = _request(self._http_fn, "GET", f"{self.base_url}/studio_speakers")
        return resp.json()

    def synthesize(
        self,
        text: str,
        *,
        speaker: str | None = None,
        language: str = "en",
        speaker_embedding: list[float] | None = None,
        gpt_cond_latent: list[list[float]] | None = None,
    ) -> bytes:
        """Synthesize ``text`` to WAV bytes.

        Resolves a studio ``speaker``'s conditioning when explicit
        ``speaker_embedding`` / ``gpt_cond_latent`` are not supplied, then calls
        ``POST /tts`` (which returns base64-encoded WAV).

        Raises:
            MediaServiceError: if the service is unreachable or no speaker can be
                resolved.
        """
        if not text.strip():
            raise MediaServiceError("synthesize() requires non-empty text")
        if speaker_embedding is None or gpt_cond_latent is None:
            speakers = self.studio_speakers()
            if not speakers:
                raise MediaServiceError("xtts returned no studio speakers")
            chosen = speakers.get(speaker) if speaker else next(iter(speakers.values()))
            if chosen is None:
                raise MediaServiceError(f"xtts speaker {speaker!r} not found")
            speaker_embedding = chosen["speaker_embedding"]
            gpt_cond_latent = chosen["gpt_cond_latent"]
        resp = _request(
            self._http_fn,
            "POST",
            f"{self.base_url}/tts",
            json={
                "text": text,
                "language": language,
                "speaker_embedding": speaker_embedding,
                "gpt_cond_latent": gpt_cond_latent,
            },
        )
        payload = resp.json()
        wav_b64 = (
            payload
            if isinstance(payload, str)
            else payload.get("wav") or payload.get("audio")
        )
        if not wav_b64:
            raise MediaServiceError("xtts /tts returned no audio payload")
        return base64.b64decode(wav_b64)


class ImageGenerator:
    """Text-to-image via the ``flux.2`` service (CONCEPT:ECO-4.30).

    The request/response shape is configurable to match the deployed flux server:
    a JSON ``POST {endpoint}`` with the prompt under ``prompt_field``, returning
    either raw image bytes, a base64 field, or a URL field.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        endpoint: str = "/generate",
        prompt_field: str = "prompt",
        image_field: str = "image",
        http_fn: HttpFn | None = None,
    ) -> None:
        self.base_url = (base_url or DEFAULT_FLUX_URL).rstrip("/")
        self.endpoint = endpoint
        self.prompt_field = prompt_field
        self.image_field = image_field
        self._http_fn = http_fn

    def generate(
        self,
        prompt: str,
        *,
        width: int = 1024,
        height: int = 1024,
        steps: int = 4,
        seed: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> bytes:
        """Generate an image from ``prompt`` → image bytes (PNG/JPEG).

        Defaults to a 4-step schnell-style request (fast, light). Decodes a
        base64/URL response or returns raw bytes.
        """
        if not prompt.strip():
            raise MediaServiceError("generate() requires a non-empty prompt")
        body: dict[str, Any] = {
            self.prompt_field: prompt,
            "width": width,
            "height": height,
            "steps": steps,
            **(extra or {}),
        }
        if seed is not None:
            body["seed"] = seed
        resp = _request(
            self._http_fn, "POST", f"{self.base_url}{self.endpoint}", json=body
        )
        return self._decode_image(resp)

    def _decode_image(self, resp: Any) -> bytes:
        ctype = ""
        try:
            ctype = resp.headers.get("content-type", "")
        except Exception:  # noqa: BLE001
            ctype = ""
        if ctype.startswith("image/"):
            return resp.content
        try:
            payload = resp.json()
        except Exception:  # noqa: BLE001 — not JSON → assume raw bytes
            return resp.content
        b64 = (
            payload.get(self.image_field)
            or payload.get("b64_json")
            or payload.get("image_base64")
        )
        if b64:
            return base64.b64decode(b64)
        url = payload.get("url") or payload.get("image_url")
        if url:
            img = _request(self._http_fn, "GET", url)
            return img.content
        raise MediaServiceError("flux response carried no image bytes/base64/url")


class VideoGenerator:
    """Text-to-video via the ``hunyuanvideo`` service (CONCEPT:ECO-4.30).

    Video generation is long-running; the client supports a synchronous response
    (bytes / base64 / url) and a simple job-poll shape (``{job_id}`` → poll
    ``status_endpoint`` until a ``video``/``url`` is present).
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        endpoint: str = "/generate",
        status_endpoint: str = "/status",
        prompt_field: str = "prompt",
        video_field: str = "video",
        http_fn: HttpFn | None = None,
    ) -> None:
        self.base_url = (base_url or DEFAULT_HUNYUAN_URL).rstrip("/")
        self.endpoint = endpoint
        self.status_endpoint = status_endpoint
        self.prompt_field = prompt_field
        self.video_field = video_field
        self._http_fn = http_fn

    def generate(
        self,
        prompt: str,
        *,
        num_frames: int = 65,
        fps: int = 16,
        width: int = 512,
        height: int = 320,
        steps: int = 30,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Submit a text-to-video job → ``{"video": bytes|None, "url": str|None, "job_id": str|None}``.

        Conservative defaults (short clip, 512×320) to keep the GB10 footprint
        light. Returns either inline video bytes, a URL to fetch, or a job id the
        caller can poll via :meth:`status`.
        """
        if not prompt.strip():
            raise MediaServiceError("generate() requires a non-empty prompt")
        body: dict[str, Any] = {
            self.prompt_field: prompt,
            "num_frames": num_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "steps": steps,
            **(extra or {}),
        }
        resp = _request(
            self._http_fn, "POST", f"{self.base_url}{self.endpoint}", json=body
        )
        return self._decode(resp)

    def status(self, job_id: str) -> dict[str, Any]:
        """Poll a running video job by id."""
        resp = _request(
            self._http_fn, "GET", f"{self.base_url}{self.status_endpoint}/{job_id}"
        )
        return self._decode(resp)

    def _decode(self, resp: Any) -> dict[str, Any]:
        try:
            payload = resp.json()
        except Exception:  # noqa: BLE001 — raw bytes response
            return {"video": resp.content, "url": None, "job_id": None}
        b64 = payload.get(self.video_field) or payload.get("video_base64")
        return {
            "video": base64.b64decode(b64) if b64 else None,
            "url": payload.get("url") or payload.get("video_url"),
            "job_id": payload.get("job_id") or payload.get("id"),
            "status": payload.get("status"),
        }


class ComfyUIClient:
    """ComfyUI workflow-API transport for image + video (CONCEPT:ECO-4.30).

    The consolidated GB10 engine. Submits a prompt workflow (``POST /prompt``),
    polls ``GET /history/{id}`` until the run completes, then fetches the output
    via ``GET /view``. A minimal default txt2img / txt2video workflow is built from
    the checkpoint + prompt; callers may pass a full ``workflow`` to override.

    Args:
        base_url: ComfyUI base URL (``COMFYUI_URL`` env by default).
        http_fn: Optional injected HTTP caller for offline tests.
        poll_interval / timeout: history polling cadence + ceiling (seconds).
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        http_fn: HttpFn | None = None,
        poll_interval: float = 1.0,
        timeout: float = 600.0,
    ) -> None:
        self.base_url = (base_url or DEFAULT_COMFYUI_URL).rstrip("/")
        self._http_fn = http_fn
        self.poll_interval = poll_interval
        self.timeout = timeout

    @staticmethod
    def default_txt2img_workflow(
        prompt: str, checkpoint: str, *, steps: int, width: int, height: int
    ) -> dict[str, Any]:
        """A minimal ComfyUI txt2img graph (checkpoint→clip→ksampler→decode→save)."""
        return {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": 0,
                    "steps": steps,
                    "cfg": 7.0,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0],
                },
            },
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": checkpoint or "sd3.5_medium.safetensors"},
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": width, "height": height, "batch_size": 1},
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": prompt, "clip": ["4", 1]},
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "", "clip": ["4", 1]},
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {"filename_prefix": "agentutil", "images": ["8", 0]},
            },
        }

    def run_workflow(self, workflow: dict[str, Any]) -> list[bytes]:
        """Submit a workflow, await completion, and return output file bytes."""
        resp = _request(
            self._http_fn, "POST", f"{self.base_url}/prompt", json={"prompt": workflow}
        )
        prompt_id = (resp.json() or {}).get("prompt_id")
        if not prompt_id:
            raise MediaServiceError("ComfyUI /prompt returned no prompt_id")
        deadline = self.timeout
        waited = 0.0
        while waited <= deadline:
            hist = (
                _request(
                    self._http_fn, "GET", f"{self.base_url}/history/{prompt_id}"
                ).json()
                or {}
            )
            entry = hist.get(prompt_id)
            if entry and entry.get("outputs"):
                return self._collect_outputs(entry["outputs"])
            if self._http_fn is not None:
                # Deterministic in tests: a fake history is returned immediately.
                raise MediaServiceError("ComfyUI run produced no outputs")
            time.sleep(self.poll_interval)
            waited += self.poll_interval
        raise MediaServiceError(f"ComfyUI run {prompt_id} timed out after {deadline}s")

    def _collect_outputs(self, outputs: dict[str, Any]) -> list[bytes]:
        files: list[bytes] = []
        for node in outputs.values():
            for key in ("images", "gifs", "videos"):
                for item in node.get(key, []) or []:
                    params = {
                        k: item.get(k, "") for k in ("filename", "subfolder", "type")
                    }
                    view = _request(
                        self._http_fn, "GET", f"{self.base_url}/view", params=params
                    )
                    files.append(view.content)
        if not files:
            raise MediaServiceError("ComfyUI outputs carried no files")
        return files

    def generate_image(
        self,
        prompt: str,
        *,
        checkpoint: str = "",
        steps: int = 20,
        width: int = 1024,
        height: int = 1024,
        workflow: dict[str, Any] | None = None,
    ) -> bytes:
        wf = workflow or self.default_txt2img_workflow(
            prompt, checkpoint, steps=steps, width=width, height=height
        )
        return self.run_workflow(wf)[0]

    def generate_video(
        self,
        prompt: str,
        *,
        workflow: dict[str, Any],
    ) -> dict[str, Any]:
        files = self.run_workflow(workflow)
        return {"video": files[0], "url": None, "job_id": None, "status": "done"}


def _resolve_image(backend: str, http_fn: HttpFn | None):
    spec = _IMAGE_BACKENDS.get(backend.lower())
    if spec is None:
        raise MediaServiceError(
            f"unknown image backend {backend!r}; choose from {list_image_backends()}"
        )
    return spec


def synthesize_speech(
    text: str,
    *,
    backend: str = "xtts",
    speaker: str | None = None,
    language: str = "en",
) -> bytes:
    """Convenience: synthesize ``text`` to audio bytes (CONCEPT:ECO-4.30).

    ``backend`` selects the speech engine (``xtts`` or ``openai``).
    """
    spec = _SPEECH_BACKENDS.get(backend.lower())
    if spec is None:
        raise MediaServiceError(
            f"unknown speech backend {backend!r}; choose from {list_speech_backends()}"
        )
    if spec["transport"] == "openai":
        resp = _request(
            None,
            "POST",
            f"{spec['url'].rstrip('/')}/v1/audio/speech",
            json={"model": "tts-1", "input": text, "voice": speaker or "alloy"},
        )
        return resp.content
    return SpeechSynthesizer(spec["url"]).synthesize(
        text, speaker=speaker, language=language
    )


def generate_image(
    prompt: str, *, backend: str = "stable-diffusion", **kwargs: Any
) -> bytes:
    """Convenience: generate an image from ``prompt`` (CONCEPT:ECO-4.30).

    ``backend`` is one of :func:`list_image_backends` — ``stable-diffusion``,
    ``flux``, ``qwen-image``, ``hunyuanimage``, or ``comfyui``. The default routes
    through the consolidated ComfyUI engine with the model's checkpoint; set
    ``{MODEL}_URL`` to point a backend at a standalone REST server instead.
    """
    spec = _resolve_image(backend, None)
    if spec["transport"] == "comfyui":
        return ComfyUIClient(spec["url"]).generate_image(
            prompt, checkpoint=spec.get("checkpoint", ""), **kwargs
        )
    return ImageGenerator(spec["url"]).generate(prompt, **kwargs)


def generate_video(
    prompt: str,
    *,
    backend: str = "hunyuanvideo",
    workflow: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience: submit a text-to-video job (CONCEPT:ECO-4.30).

    ``backend`` is one of :func:`list_video_backends` — ``hunyuanvideo``, ``ltx``,
    or ``comfyui``. ComfyUI-backed video requires a ``workflow`` graph for the
    chosen model; a standalone REST backend uses the simple ``/generate`` shape.
    """
    spec = _VIDEO_BACKENDS.get(backend.lower())
    if spec is None:
        raise MediaServiceError(
            f"unknown video backend {backend!r}; choose from {list_video_backends()}"
        )
    if spec["transport"] == "comfyui":
        if workflow is None:
            raise MediaServiceError(
                f"video backend {backend!r} (ComfyUI) requires a 'workflow' graph"
            )
        return ComfyUIClient(spec["url"]).generate_video(prompt, workflow=workflow)
    return VideoGenerator(spec["url"]).generate(prompt, **kwargs)
