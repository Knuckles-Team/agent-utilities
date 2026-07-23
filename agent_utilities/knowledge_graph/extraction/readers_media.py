"""Media modality readers — audio transcripts + scanned/image OCR (CONCEPT:AU-KG.backend.mirror-health-repair).

Part of the universal multi-modal ingestion funnel
(``reader -> structure-router -> {open|schema} extraction -> ontology grounding``).
The reader stage turns a *non-text* file into plain text so the existing
text-centric seam (``IngestionEngine._enrich_text`` → concepts + facts, and
``extract_document`` → ``Document`` node) can run unchanged. These readers cover
two modalities that ``enrichment/extractors/document.py`` does not:

* **Audio** (``.wav`` / ``.mp3`` / ``.m4a`` / ``.flac`` / ``.ogg`` / ``.aac``) →
  speech-to-text transcript via optional ``faster-whisper`` (CTranslate2). Model
  size comes from ``KG_ASR_MODEL`` (default ``"base"``).
* **Images / scanned pages** (``.png`` / ``.jpg`` / ``.jpeg`` / ``.tiff`` /
  ``.bmp`` / ``.webp``) → OCR text via optional ``pytesseract`` (preferred) or
  ``rapidocr-onnxruntime`` (fallback, pure-onnx, no system binary).

Conventions mirror ``document.py``: heavy deps are **strictly import-guarded and
lazy** — nothing here imports ``faster_whisper`` / ``pytesseract`` / ``rapidocr``
/ ``PIL`` at module load. A reader runs only when its dependency (and, for ASR,
the model) is importable; otherwise it returns ``""`` and logs a one-line note,
never raising — ingest paths are best-effort. The model/engine handles are cached
per process so repeated reads don't re-pay load cost.

Readers self-register via the ``@register_reader(*exts)`` decorator into a small
extension→callable registry; the structure-router / ingestion engine dispatches
through :func:`read_media` (or :func:`reader_for` to test capability first). Adding
a new modality is one decorated function — no edits to the dispatch site.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

logger = logging.getLogger(__name__)

# A reader takes an absolute file path and returns extracted plain text (or "").
ReaderFn = Callable[[str], str]

# extension (lower, with leading dot) -> reader callable. Populated by the
# @register_reader decorator at import time.
_READERS: dict[str, ReaderFn] = {}

# Process-level caches for the heavy, slow-to-construct backends. None = unset;
# False = probed and unavailable (so we don't re-probe a missing dep every call).
_ASR_MODEL: object | None = None
_ASR_PROBED = False
_OCR_ENGINE: object | None = None
_OCR_PROBED = False

# Default Whisper model size when KG_ASR_MODEL is unset. "base" balances speed and
# accuracy on CPU; tiny/small/medium/large-v3 are the other faster-whisper sizes.
_DEFAULT_ASR_MODEL = "base"


def register_reader(*extensions: str) -> Callable[[ReaderFn], ReaderFn]:
    """Register ``fn`` as the reader for each file ``extension`` (e.g. ``".mp3"``).

    Extensions are normalised to lowercase with a leading dot. The decorator
    returns the function unchanged so it stays directly callable/testable.
    """

    def _decorate(fn: ReaderFn) -> ReaderFn:
        for ext in extensions:
            norm = ext.lower()
            if not norm.startswith("."):
                norm = "." + norm
            _READERS[norm] = fn
        return fn

    return _decorate


def supported_extensions() -> frozenset[str]:
    """The set of file extensions a media reader is registered for."""
    return frozenset(_READERS)


def reader_for(file_path: str) -> ReaderFn | None:
    """Return the registered reader for ``file_path``'s extension, or ``None``.

    Lets the structure-router ask "is this a media modality I can read?" without
    actually invoking the (potentially heavy) reader.
    """
    return _READERS.get(os.path.splitext(file_path)[1].lower())


def read_media(file_path: str, max_chars: int = 8_000_000) -> str:
    """Best-effort: dispatch ``file_path`` to its registered media reader.

    Returns extracted plain text (capped at ``max_chars``, same generous bound as
    :func:`document.read_document_text` — size is governed downstream by chunking),
    or ``""`` when the extension has no reader, the optional dependency/model is
    unavailable, or extraction fails. Never raises: media reading sits on the
    best-effort ingest path.
    """
    fn = reader_for(file_path)
    if fn is None:
        return ""
    try:
        text = fn(file_path)
    except Exception as exc:  # never break ingest on a bad/corrupt media file
        logger.warning(
            "media reader failed (exception_type=%s)",
            type(exc).__name__,
        )
        return ""
    return (text or "")[:max_chars]


# --------------------------------------------------------------------------- #
# Audio transcript reader (optional faster-whisper)                           #
# --------------------------------------------------------------------------- #


def _asr_model_size() -> str:
    """Whisper model size from ``KG_ASR_MODEL`` (default ``"base"``)."""
    # Live read (operators retune ASR size per deployment) via the sanctioned
    # accessor — never a bare os.environ read (Configuration discipline).
    from agent_utilities.core.config import setting

    size = setting("KG_ASR_MODEL", _DEFAULT_ASR_MODEL, cast=str)
    return (size or _DEFAULT_ASR_MODEL).strip() or _DEFAULT_ASR_MODEL


def _get_asr_model() -> object | None:
    """Lazily build + cache a faster-whisper model, or ``None`` if unavailable.

    Imports ``faster_whisper`` only here (never at module load). A missing dep or
    a failed model load is probed once and memoised so we don't retry every file.
    """
    global _ASR_MODEL, _ASR_PROBED
    if _ASR_PROBED:
        return _ASR_MODEL
    _ASR_PROBED = True
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        logger.info(
            "audio reader: faster-whisper not installed — audio ingest is a no-op "
            "(pip install faster-whisper)"
        )
        _ASR_MODEL = None
        return None
    size = _asr_model_size()
    try:
        # int8 on CPU keeps the model light; auto device picks GPU if present.
        _ASR_MODEL = WhisperModel(size, device="auto", compute_type="int8")
    except Exception as exc:
        logger.warning(
            "audio reader initialization failed; audio ingest is a no-op "
            "(exception_type=%s)",
            type(exc).__name__,
        )
        _ASR_MODEL = None
    return _ASR_MODEL


@register_reader(".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac")
def read_audio_transcript(file_path: str) -> str:
    """Transcribe an audio file to text via optional faster-whisper.

    Returns the concatenated transcript, or ``""`` with a logged note when
    ``faster-whisper`` (or its model) is unavailable. Lazy + best-effort.
    """
    model = _get_asr_model()
    if model is None:
        return ""
    try:
        segments, _info = model.transcribe(file_path)  # type: ignore[attr-defined]  # faster-whisper untyped model
        return " ".join(seg.text.strip() for seg in segments if seg.text).strip()
    except Exception as exc:
        logger.warning(
            "audio transcription failed (exception_type=%s)",
            type(exc).__name__,
        )
        return ""


# --------------------------------------------------------------------------- #
# Image / scanned-page OCR reader (optional pytesseract or rapidocr)          #
# --------------------------------------------------------------------------- #


def _ocr_with_pytesseract(file_path: str) -> str | None:
    """OCR via pytesseract + Pillow. ``None`` if either dep (or the tesseract
    binary) is unavailable; ``""``/text on a successful run."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(file_path) as img:
            return pytesseract.image_to_string(img) or ""
    except Exception as exc:
        # pytesseract raises TesseractNotFoundError when the system binary is
        # missing — treat as "engine unavailable" so we fall back to rapidocr.
        logger.info(
            "image reader backend unavailable; trying fallback (exception_type=%s)",
            type(exc).__name__,
        )
        return None


def _get_rapidocr_engine() -> object | None:
    """Lazily build + cache a rapidocr engine, or ``None`` if unavailable."""
    global _OCR_ENGINE, _OCR_PROBED
    if _OCR_PROBED:
        return _OCR_ENGINE
    _OCR_PROBED = True
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        _OCR_ENGINE = None
        return None
    try:
        _OCR_ENGINE = RapidOCR()
    except Exception as exc:
        logger.warning(
            "image reader initialization failed (exception_type=%s)",
            type(exc).__name__,
        )
        _OCR_ENGINE = None
    return _OCR_ENGINE


def _persist_image_region_evidence(file_path: str, rows: list) -> None:
    """Through-write an ``ImageRegion`` evidence locus per recognised text line
    (CONCEPT:AU-KG.identity.evidence-spine-convergence).

    RapidOCR already computes a quadrilateral box per line (``row[0]``)
    alongside its text — before this it was discarded (only ``row[1]`` text
    survived the join below). The box's four points are collapsed to the axis-
    aligned rectangle ``ImageRegion`` expects (min/max over the real, already-
    computed points — no new detection). Stores the whole source image once per
    line (content-addressed dedup collapses the repeats to one blob) so the
    region's coordinates stay locatable within the actual bytes cited, mirroring
    ``store_document_page_evidence``'s page-bytes-plus-box convention. Best-
    effort: an unreadable file or an engine-unavailable store is logged and
    skipped, never raised — this sits on the best-effort OCR read path.
    """
    try:
        with open(file_path, "rb") as fh:
            data = fh.read()
    except OSError:
        return
    if not data:
        return
    try:
        from ..memory.native_ingest import media_store

        store = media_store()
    except Exception as e:  # noqa: BLE001
        logger.debug("image region evidence store unavailable: %s", e)
        return
    import mimetypes

    mime_type = mimetypes.guess_type(file_path)[0] or "image/png"
    for row in rows:
        if len(row) < 2 or not row[1]:
            continue
        try:
            xs = [float(pt[0]) for pt in row[0]]
            ys = [float(pt[1]) for pt in row[0]]
        except (TypeError, ValueError, IndexError):
            continue
        if not xs or not ys:
            continue
        x, y = min(xs), min(ys)
        width, height = max(xs) - x, max(ys) - y
        if width <= 0 or height <= 0:
            continue
        try:
            store.store_image_region_evidence(
                data,
                image_id=file_path,
                x=x,
                y=y,
                width=width,
                height=height,
                mime_type=mime_type,
                source="rapidocr",
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("image region evidence write failed: %s", e)


def _ocr_with_rapidocr(file_path: str) -> str | None:
    """OCR via rapidocr-onnxruntime. ``None`` if unavailable; text otherwise."""
    engine = _get_rapidocr_engine()
    if engine is None:
        return None
    try:
        result, _elapsed = engine(file_path)  # type: ignore[operator]  # optional ASR engine untyped
        if not result:
            return ""
        _persist_image_region_evidence(file_path, result)
        # result rows are [box, text, score]; join recognised text lines.
        return "\n".join(row[1] for row in result if len(row) > 1 and row[1]).strip()
    except Exception as exc:
        logger.warning(
            "image OCR failed (exception_type=%s)",
            type(exc).__name__,
        )
        return None


@register_reader(".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp")
def read_image_ocr(file_path: str) -> str:
    """OCR an image / scanned page to text via optional pytesseract or rapidocr.

    Tries pytesseract first (fast, ubiquitous), then rapidocr-onnxruntime (pure
    onnx, no system binary). Returns ``""`` with a logged note when neither OCR
    backend is available. Lazy + best-effort.
    """
    text = _ocr_with_pytesseract(file_path)
    if text is None:
        text = _ocr_with_rapidocr(file_path)
    if text is None:
        logger.info(
            "image reader: no OCR backend available — image ingest is a no-op "
            "(pip install pytesseract+Pillow, or rapidocr-onnxruntime)"
        )
        return ""
    return text
