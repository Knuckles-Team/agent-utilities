"""Media generation + transcription gateway (CONCEPT:ECO-4.30 / ECO-4.31).

Self-hosted media for the harness: image (``flux.2``), video (``hunyuanvideo``),
speech synthesis (``xtts``), and transcription (``faster-whisper``). All clients
are lazy-``httpx``, env-configured (``{SERVICE}_URL``), and fail loudly via
:class:`MediaServiceError` rather than hanging or returning empty results.
"""

from .gateway import (
    ComfyUIClient,
    ImageGenerator,
    MediaServiceError,
    SpeechSynthesizer,
    VideoGenerator,
    generate_image,
    generate_video,
    list_image_backends,
    list_speech_backends,
    list_video_backends,
    synthesize_speech,
)
from .transcription import Transcriber, transcribe

__all__ = [
    "MediaServiceError",
    "SpeechSynthesizer",
    "ImageGenerator",
    "VideoGenerator",
    "ComfyUIClient",
    "Transcriber",
    "synthesize_speech",
    "generate_image",
    "generate_video",
    "transcribe",
    "list_image_backends",
    "list_video_backends",
    "list_speech_backends",
]
