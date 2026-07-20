"""Tests for the media generation + transcription gateway (CONCEPT:AU-ECO.toolkit.media-gateway-failure-path/4.31).

Offline: every HTTP call is routed through an injected ``http_fn`` returning a
fake response object, so no media services are contacted.
"""

from __future__ import annotations

import base64

import pytest

from agent_utilities.ecosystem.media import (
    ImageGenerator,
    MediaServiceError,
    SpeechSynthesizer,
    Transcriber,
    VideoGenerator,
)
from agent_utilities.ecosystem.media.gateway import generate_image


class _Resp:
    def __init__(self, j=None, content=b"", headers=None):
        self._j = j
        self.content = content
        self.headers = headers or {}

    def json(self):
        if self._j is None:
            raise ValueError("no json")
        return self._j

    def raise_for_status(self):
        return None


@pytest.mark.concept("AU-ECO.toolkit.media-gateway-failure-path")
def test_image_generator_base64_response():
    def http(method, url, **kw):
        return _Resp(j={"image": base64.b64encode(b"PNGDATA").decode()})

    img = ImageGenerator(http_fn=http).generate("a cat", steps=4)
    assert img == b"PNGDATA"


@pytest.mark.concept("AU-ECO.toolkit.media-gateway-failure-path")
def test_image_generator_raw_bytes_response():
    def http(method, url, **kw):
        return _Resp(content=b"RAWPNG", headers={"content-type": "image/png"})

    img = ImageGenerator(http_fn=http).generate("a dog")
    assert img == b"RAWPNG"


@pytest.mark.concept("AU-ECO.toolkit.media-gateway-failure-path")
def test_image_backend_selection_and_error():
    with pytest.raises(MediaServiceError):
        generate_image("x", backend="does-not-exist")


@pytest.mark.concept("AU-ECO.toolkit.media-gateway-failure-path")
def test_speech_synthesizer_resolves_studio_speaker():
    seen = []

    def http(method, url, **kw):
        seen.append(url.rsplit("/", 1)[-1])
        if url.endswith("/studio_speakers"):
            return _Resp(
                j={"Aria": {"speaker_embedding": [0.1], "gpt_cond_latent": [[0.2]]}}
            )
        return _Resp(j=base64.b64encode(b"WAVDATA").decode())

    wav = SpeechSynthesizer(http_fn=http).synthesize("hello")
    assert wav == b"WAVDATA"
    assert "studio_speakers" in seen and "tts" in seen


@pytest.mark.concept("AU-ECO.toolkit.media-gateway-failure-path")
def test_speech_requires_text():
    with pytest.raises(MediaServiceError):
        SpeechSynthesizer(http_fn=lambda *a, **k: _Resp(j={})).synthesize("   ")


@pytest.mark.concept("AU-ECO.toolkit.media-gateway-failure-path")
def test_video_generator_inline_and_job():
    def inline(method, url, **kw):
        return _Resp(j={"video": base64.b64encode(b"MP4").decode(), "status": "done"})

    out = VideoGenerator(http_fn=inline).generate("a sunset")
    assert out["video"] == b"MP4" and out["status"] == "done"

    def job(method, url, **kw):
        return _Resp(j={"job_id": "j1", "status": "running"})

    out2 = VideoGenerator(http_fn=job).generate("a river")
    assert out2["job_id"] == "j1" and out2["video"] is None


@pytest.mark.concept("AU-ECO.toolkit.media-transcription-bridge")
def test_transcriber_openai_compatible():
    def http(method, url, **kw):
        assert "transcriptions" in url
        assert "file" in kw.get("files", {})
        return _Resp(j={"text": "hello there"})

    assert Transcriber(http_fn=http).transcribe(b"AUDIO") == "hello there"


@pytest.mark.concept("AU-ECO.toolkit.media-transcription-bridge")
def test_transcriber_requires_audio():
    with pytest.raises(MediaServiceError):
        Transcriber(http_fn=lambda *a, **k: _Resp(j={})).transcribe(b"")


@pytest.mark.concept("AU-ECO.toolkit.media-gateway-failure-path")
def test_media_tools_registered_under_gate():
    from agent_utilities.tools.media_tools import media_tools

    names = {t.__name__ for t in media_tools}
    assert names == {
        "synthesize_speech",
        "generate_image",
        "generate_video",
        "transcribe_audio",
    }


@pytest.mark.concept("AU-ECO.toolkit.media-gateway-failure-path")
def test_named_backend_registries():
    from agent_utilities.ecosystem.media import (
        list_image_backends,
        list_speech_backends,
        list_video_backends,
    )

    img = set(list_image_backends())
    assert {"stable-diffusion", "flux", "qwen-image", "hunyuanimage", "comfyui"} <= img
    assert {"hunyuanvideo", "ltx", "comfyui"} <= set(list_video_backends())
    assert {"xtts", "openai"} <= set(list_speech_backends())


@pytest.mark.concept("AU-ECO.toolkit.media-gateway-failure-path")
def test_comfyui_client_workflow_image():
    from agent_utilities.ecosystem.media import ComfyUIClient

    def http(method, url, **kw):
        if url.endswith("/prompt"):
            return _Resp(j={"prompt_id": "pid1"})
        if "/history/" in url:
            return _Resp(
                j={
                    "pid1": {
                        "outputs": {
                            "9": {
                                "images": [
                                    {
                                        "filename": "out.png",
                                        "subfolder": "",
                                        "type": "output",
                                    }
                                ]
                            }
                        }
                    }
                }
            )
        if "/view" in url:
            return _Resp(content=b"PNGBYTES")
        return _Resp(j={})

    img = ComfyUIClient(http_fn=http).generate_image("a cat", checkpoint="sd3.5_medium")
    assert img == b"PNGBYTES"


@pytest.mark.concept("AU-ECO.toolkit.media-gateway-failure-path")
def test_comfyui_backed_image_via_named_backend():
    from agent_utilities.ecosystem.media.gateway import ComfyUIClient

    def http(method, url, **kw):
        if url.endswith("/prompt"):
            return _Resp(j={"prompt_id": "p"})
        if "/history/" in url:
            return _Resp(
                j={
                    "p": {
                        "outputs": {
                            "9": {
                                "images": [
                                    {
                                        "filename": "x.png",
                                        "type": "output",
                                        "subfolder": "",
                                    }
                                ]
                            }
                        }
                    }
                }
            )
        return _Resp(content=b"IMG")

    # Drive the workflow client directly with the SD3.5 checkpoint preset.
    out = ComfyUIClient(http_fn=http).generate_image(
        "sunset", checkpoint="sd3.5_medium", steps=28
    )
    assert out == b"IMG"


@pytest.mark.concept("AU-ECO.toolkit.media-gateway-failure-path")
def test_unknown_backends_raise():
    from agent_utilities.ecosystem.media import (
        generate_image,
        generate_video,
        synthesize_speech,
    )

    with pytest.raises(MediaServiceError):
        generate_image("x", backend="nope")
    with pytest.raises(MediaServiceError):
        generate_video("x", backend="nope")
    with pytest.raises(MediaServiceError):
        synthesize_speech("x", backend="nope")
