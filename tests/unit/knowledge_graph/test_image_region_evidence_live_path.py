"""Live-path proof for the ``ImageRegion`` evidence-locus producer
(CONCEPT:AU-KG.identity.evidence-spine-convergence, Evidence seam completion).

The seam audit (``reports/seam-closure-audit-2026-07-22.md``) found
``readers_media.py``'s RapidOCR branch already computes a box per recognised
text line (``result`` rows are ``[box, text, score]``) and discards the box,
keeping only the text — confirming ``store_image_region_evidence`` was
generalized-but-unwired. This test proves the gap is closed: the EXISTING OCR
entry point (``read_image_ocr`` → ``_ocr_with_rapidocr``, reached from every
image ingested through ``read_media``/``read_any``) now also writes an
``ImageRegion`` evidence locus per recognised line, as a side effect — not a
new opt-in call site a caller must remember to use.

Offline: the RapidOCR engine and the media store are both faked; no real
onnxruntime model, no engine round-trip.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.extraction import readers_media as m
from agent_utilities.knowledge_graph.memory import native_ingest


class _FakeStore:
    def __init__(self) -> None:
        self.region_calls: list[tuple[bytes, dict]] = []

    def store_image_region_evidence(self, data: bytes, **kwargs):
        self.region_calls.append((data, kwargs))
        return object()


@pytest.fixture(autouse=True)
def _reset_caches(monkeypatch):
    monkeypatch.setattr(m, "_OCR_ENGINE", None)
    monkeypatch.setattr(m, "_OCR_PROBED", False)
    yield


def test_rapidocr_writes_image_region_evidence_per_line(tmp_path, monkeypatch):
    img = tmp_path / "scan.png"
    img.write_bytes(b"PNGBYTES")

    # Two recognised lines, each with a quadrilateral box (RapidOCR's own shape).
    fake_result = [
        ([[10.0, 20.0], [110.0, 20.0], [110.0, 40.0], [10.0, 40.0]], "hello", 0.99),
        ([[10.0, 50.0], [90.0, 50.0], [90.0, 70.0], [10.0, 70.0]], "world", 0.95),
    ]

    class _FakeEngine:
        def __call__(self, _path):
            return fake_result, 0.01

    monkeypatch.setattr(m, "_get_rapidocr_engine", lambda: _FakeEngine())

    store = _FakeStore()
    monkeypatch.setattr(native_ingest, "media_store", lambda: store)

    # The EXISTING OCR entry point — no new call site.
    text = m._ocr_with_rapidocr(str(img))

    # Text extraction is unchanged.
    assert text == "hello\nworld"

    # AND an ImageRegion evidence locus was written per recognised line, box
    # coordinates collapsed to the axis-aligned rectangle RapidOCR's own
    # quadrilateral bounds.
    assert len(store.region_calls) == 2
    (data1, kw1), (data2, kw2) = store.region_calls
    assert data1 == data2 == b"PNGBYTES"
    assert kw1["image_id"] == kw2["image_id"] == str(img)
    assert kw1["x"] == 10.0 and kw1["y"] == 20.0
    assert kw1["width"] == 100.0 and kw1["height"] == 20.0
    assert kw2["x"] == 10.0 and kw2["y"] == 50.0
    assert kw2["width"] == 80.0 and kw2["height"] == 20.0
    assert kw1["source"] == "rapidocr"


def test_rapidocr_skips_evidence_when_store_unavailable(tmp_path, monkeypatch):
    """No engine reachable → OCR text still returned, no evidence write attempted."""
    img = tmp_path / "scan.png"
    img.write_bytes(b"PNGBYTES")

    fake_result = [
        ([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], "x", 0.9),
    ]

    class _FakeEngine:
        def __call__(self, _path):
            return fake_result, 0.01

    monkeypatch.setattr(m, "_get_rapidocr_engine", lambda: _FakeEngine())

    def _boom():
        raise native_ingest.NativeIngestError("no engine")

    monkeypatch.setattr(native_ingest, "media_store", _boom)

    text = m._ocr_with_rapidocr(str(img))
    assert text == "x"  # OCR itself never breaks on a store failure


def test_rapidocr_skips_evidence_for_unreadable_file(monkeypatch):
    """A file that vanished between OCR and the evidence read → no write, no raise."""
    fake_result = [
        ([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], "x", 0.9),
    ]

    class _FakeEngine:
        def __call__(self, _path):
            return fake_result, 0.01

    monkeypatch.setattr(m, "_get_rapidocr_engine", lambda: _FakeEngine())

    store = _FakeStore()
    monkeypatch.setattr(native_ingest, "media_store", lambda: store)

    text = m._ocr_with_rapidocr("/nonexistent/path/scan.png")
    assert text == "x"
    assert store.region_calls == []
