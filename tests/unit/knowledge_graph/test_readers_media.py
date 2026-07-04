"""Tests for the media modality readers (CONCEPT:AU-KG.backend.mirror-health-repair).

Heavy deps (faster-whisper, pytesseract, rapidocr) are optional and absent in CI,
so these tests assert the registry + dispatch contract and the lazy/no-op/never-
raise behaviour, and use monkeypatching to exercise the success paths without the
real backends.
"""

from __future__ import annotations

import sys
import types

import pytest

from agent_utilities.knowledge_graph.extraction import readers_media as m


@pytest.fixture(autouse=True)
def _reset_caches(monkeypatch):
    """Reset the per-process backend caches so probes re-run each test."""
    monkeypatch.setattr(m, "_ASR_MODEL", None)
    monkeypatch.setattr(m, "_ASR_PROBED", False)
    monkeypatch.setattr(m, "_OCR_ENGINE", None)
    monkeypatch.setattr(m, "_OCR_PROBED", False)
    yield


# --------------------------------------------------------------------------- #
# Registry / dispatch contract                                                #
# --------------------------------------------------------------------------- #


def test_audio_extensions_registered():
    exts = m.supported_extensions()
    for ext in (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"):
        assert ext in exts
        assert m.reader_for(f"/tmp/clip{ext}") is m.read_audio_transcript


def test_image_extensions_registered():
    exts = m.supported_extensions()
    for ext in (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"):
        assert ext in exts
        assert m.reader_for(f"/tmp/scan{ext}") is m.read_image_ocr


def test_reader_for_is_case_insensitive():
    assert m.reader_for("/x/A.WAV") is m.read_audio_transcript
    assert m.reader_for("/x/B.PNG") is m.read_image_ocr


def test_reader_for_unknown_extension_is_none():
    assert m.reader_for("/x/notes.txt") is None
    assert m.reader_for("/x/paper.pdf") is None


def test_read_media_unknown_extension_returns_empty():
    assert m.read_media("/x/notes.txt") == ""


def test_register_reader_normalises_extension():
    sentinel = "REGISTERED"

    @m.register_reader("XyZ")  # no dot, mixed case
    def _r(_path: str) -> str:
        return sentinel

    try:
        assert m.reader_for("/a/b.xyz") is _r
        assert m.read_media("/a/b.xyz") == sentinel
    finally:
        m._READERS.pop(".xyz", None)


# --------------------------------------------------------------------------- #
# Lazy + no-op degradation (deps absent)                                      #
# --------------------------------------------------------------------------- #


def test_module_imports_without_heavy_deps():
    # The module must not import faster_whisper / pytesseract / rapidocr / PIL at
    # load time. If it did, importing it under a clean env would have raised.
    assert "faster_whisper" not in _module_level_imports()


def _module_level_imports() -> set[str]:
    import ast
    import inspect

    src = inspect.getsource(m)
    tree = ast.parse(src)
    names: set[str] = set()
    for node in tree.body:  # module level only
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_audio_reader_no_op_when_faster_whisper_missing(monkeypatch):
    # Force the import to fail regardless of whether the dep is installed.
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    assert m._get_asr_model() is None
    assert m.read_audio_transcript("/tmp/clip.mp3") == ""
    assert m.read_media("/tmp/clip.mp3") == ""


def test_image_reader_no_op_when_no_ocr_backend(monkeypatch):
    monkeypatch.setitem(sys.modules, "pytesseract", None)
    monkeypatch.setitem(sys.modules, "PIL", None)
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", None)
    assert m.read_image_ocr("/tmp/scan.png") == ""
    assert m.read_media("/tmp/scan.png") == ""


def test_read_media_never_raises_on_reader_error(monkeypatch):
    def _boom(_path: str) -> str:
        raise RuntimeError("corrupt media")

    monkeypatch.setitem(m._READERS, ".boom", _boom)
    assert m.read_media("/tmp/x.boom") == ""


def test_asr_probe_is_memoised(monkeypatch):
    # Trigger the ImportError path by making the symbol import fail.
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    assert m._get_asr_model() is None
    assert m._ASR_PROBED is True
    # Second call must not re-probe (cached None).
    assert m._get_asr_model() is None


# --------------------------------------------------------------------------- #
# Success paths (backends faked)                                              #
# --------------------------------------------------------------------------- #


def test_audio_transcribe_success(monkeypatch):
    seg = types.SimpleNamespace(text=" hello world ")
    seg2 = types.SimpleNamespace(text="from audio")

    class _FakeModel:
        def transcribe(self, _path):
            return [seg, seg2], object()

    monkeypatch.setattr(m, "_get_asr_model", lambda: _FakeModel())
    out = m.read_audio_transcript("/tmp/clip.wav")
    assert out == "hello world from audio"


def test_asr_model_size_from_env(monkeypatch):
    monkeypatch.setenv("KG_ASR_MODEL", "small")
    assert m._asr_model_size() == "small"


def test_asr_model_size_default(monkeypatch):
    monkeypatch.delenv("KG_ASR_MODEL", raising=False)
    assert m._asr_model_size() == "base"


def test_image_ocr_prefers_pytesseract(monkeypatch):
    monkeypatch.setattr(m, "_ocr_with_pytesseract", lambda _p: "tesseract text")

    def _should_not_run(_p):
        raise AssertionError("rapidocr should not be tried when tesseract works")

    monkeypatch.setattr(m, "_ocr_with_rapidocr", _should_not_run)
    assert m.read_image_ocr("/tmp/scan.png") == "tesseract text"


def test_image_ocr_falls_back_to_rapidocr(monkeypatch):
    monkeypatch.setattr(m, "_ocr_with_pytesseract", lambda _p: None)
    monkeypatch.setattr(m, "_ocr_with_rapidocr", lambda _p: "rapid text")
    assert m.read_image_ocr("/tmp/scan.png") == "rapid text"


def test_read_media_caps_output(monkeypatch):
    monkeypatch.setitem(m._READERS, ".big", lambda _p: "x" * 100)
    assert m.read_media("/tmp/a.big", max_chars=10) == "x" * 10
