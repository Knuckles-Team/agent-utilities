"""Tests for the self-registering multi-modal reader registry (CONCEPT:AU-KG.enrichment.multimodal-readers)."""

from __future__ import annotations

import importlib

import pytest

readers = importlib.import_module("agent_utilities.knowledge_graph.extraction.readers")


@pytest.fixture(autouse=True)
def _fresh_discovery(monkeypatch):
    """Reset the one-shot discovery latch AND snapshot/restore the global registry
    so a test that overwrites a built-in extension (e.g. the MIME-key test) cannot
    leak that override into later tests."""
    monkeypatch.setattr(readers, "_DISCOVERED", False, raising=False)
    saved = dict(readers._READER_REGISTRY)
    yield
    readers._READER_REGISTRY.clear()
    readers._READER_REGISTRY.update(saved)


def test_builtin_readers_registered_at_import():
    keys = readers.list_readers()
    # Inline built-ins must be present after discovery.
    for ext in (".csv", ".tsv", ".html", ".pptx", ".xlsx", ".png"):
        assert ext in keys, f"{ext} not registered"
    # Document-family extensions are served by the fallback, NOT registered.
    for ext in (".pdf", ".txt", ".md"):
        assert ext not in keys


def test_register_reader_decorator_and_dispatch():
    @readers.register_reader(".zzz", "FOO")
    def _reader(path: str) -> str:  # noqa: ARG001
        return "custom-text"

    # Extension key normalised with leading dot; bare ext normalised too.
    assert readers.get_reader(".zzz") is _reader
    assert readers.get_reader(".foo") is _reader
    assert readers.get_reader("zzz") is _reader


def test_register_reader_mime_key_maps_to_extension():
    @readers.register_reader("application/pdf")  # maps to nothing in _MIME_TO_EXT
    def _noop(path: str) -> str:  # noqa: ARG001
        return ""

    # application/pdf isn't in the tiny MIME map -> no key registered, no crash.
    @readers.register_reader("text/csv")
    def _csv_like(path: str) -> str:  # noqa: ARG001
        return "csv-mime"

    assert readers.get_reader("text/csv") is _csv_like
    assert readers.get_reader(".csv") is _csv_like  # overwrote builtin, idempotent path


def test_read_delimited_csv(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    out = readers.read_delimited(str(p))
    assert "a | b | c" in out
    assert "1 | 2 | 3" in out


def test_read_delimited_tsv(tmp_path):
    p = tmp_path / "data.tsv"
    p.write_text("x\ty\n10\t20\n", encoding="utf-8")
    out = readers.read_delimited(str(p))
    assert "x | y" in out
    assert "10 | 20" in out


def test_read_html_stdlib_fallback(tmp_path):
    p = tmp_path / "page.html"
    p.write_text(
        "<html><head><style>.x{}</style></head>"
        "<body><script>ignore()</script><h1>Hello</h1><p>World</p></body></html>",
        encoding="utf-8",
    )
    out = readers.read_html(str(p))
    assert "Hello" in out
    assert "World" in out
    assert "ignore" not in out


def test_read_any_dispatches_registered_modality(tmp_path):
    p = tmp_path / "table.csv"
    p.write_text("name,role\nada,eng\n", encoding="utf-8")
    out = readers.read_any(str(p))
    assert "name | role" in out
    assert "ada | eng" in out


def test_read_any_routes_document_family_to_fallback(tmp_path, monkeypatch):
    p = tmp_path / "note.txt"
    p.write_text("plain body", encoding="utf-8")

    called = {}

    def _fake_read_document_text(path, max_chars=8_000_000):
        called["path"] = path
        return "FROM-FALLBACK"

    import agent_utilities.knowledge_graph.enrichment.extractors.document as doc

    monkeypatch.setattr(doc, "read_document_text", _fake_read_document_text)
    out = readers.read_any(str(p))
    assert out == "FROM-FALLBACK"
    assert called["path"] == str(p)


def test_read_any_unknown_extension_falls_back(tmp_path, monkeypatch):
    p = tmp_path / "mystery.xyz"
    p.write_text("data", encoding="utf-8")

    import agent_utilities.knowledge_graph.enrichment.extractors.document as doc

    monkeypatch.setattr(doc, "read_document_text", lambda *a, **k: "")
    # Unknown modality -> document fallback -> "" (best-effort no-op).
    assert readers.read_any(str(p)) == ""


def test_read_any_never_raises_on_reader_failure(tmp_path):
    @readers.register_reader(".boom")
    def _boom(path: str) -> str:  # noqa: ARG001
        raise RuntimeError("kaboom")

    p = tmp_path / "x.boom"
    p.write_text("", encoding="utf-8")
    assert readers.read_any(str(p)) == ""  # swallowed, not raised


def test_heavy_readers_degrade_to_noop_when_dep_absent(tmp_path, monkeypatch):
    # Simulate python-pptx / openpyxl / faster-whisper / pytesseract absent.
    real_import = __import__

    def _blocked_import(name, *args, **kwargs):
        if name in ("pptx", "openpyxl", "faster_whisper", "pytesseract", "PIL"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _blocked_import)
    pptx = tmp_path / "deck.pptx"
    pptx.write_bytes(b"PK\x03\x04")  # not a real pptx; reader shouldn't get that far
    assert readers.read_pptx(str(pptx)) == ""
    assert readers.read_xlsx(str(tmp_path / "book.xlsx")) == ""
    assert readers.read_audio(str(tmp_path / "clip.mp3")) == ""
    assert readers.read_image_ocr(str(tmp_path / "scan.png")) == ""


def test_discover_is_idempotent_and_cached():
    first = readers.discover()
    second = readers.discover()
    assert set(first) == set(second)
    assert readers._DISCOVERED is True


def test_read_any_calls_discover_live_path(monkeypatch, tmp_path):
    # Wire-First: read_any must trigger discover() (the registration mechanism).
    hits = {"n": 0}
    orig = readers.discover

    def _spy():
        hits["n"] += 1
        return orig()

    monkeypatch.setattr(readers, "discover", _spy)
    p = tmp_path / "t.csv"
    p.write_text("a,b\n", encoding="utf-8")
    readers.read_any(str(p))
    assert hits["n"] >= 1
