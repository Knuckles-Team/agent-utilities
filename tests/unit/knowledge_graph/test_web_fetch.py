"""Unit tests for the unified web-fetch resolver (CONCEPT:AU-KG.query.vendor-agnostic-traversal)."""

from __future__ import annotations

import agent_utilities.knowledge_graph.ingestion.web_fetch as wf
from agent_utilities.knowledge_graph.ingestion.web_fetch import (
    FetchedPage,
    _archivebox_text,
    _looks_like_path,
    _records,
    resolve_web_fetch,
)


def _page(backend: str) -> FetchedPage:
    return FetchedPage(url="https://x.io", markdown="hello body", backend=backend)


def test_precedence_requests_floor(monkeypatch):
    """No ArchiveBox + no crawl4ai → requests is the floor."""
    monkeypatch.setattr(wf, "archivebox_configured", lambda: False)
    monkeypatch.setattr(wf, "_fetch_via_crawl4ai", lambda u, t: None)
    monkeypatch.setattr(wf, "_fetch_via_requests", lambda u, t: _page("requests"))
    page = resolve_web_fetch("https://x.io")
    assert page is not None and page.backend == "requests"


def test_archivebox_preferred_when_configured(monkeypatch):
    monkeypatch.setattr(wf, "archivebox_configured", lambda: True)
    monkeypatch.setattr(wf, "_fetch_via_archivebox", lambda u, t: _page("archivebox"))
    monkeypatch.setattr(
        wf, "_fetch_via_crawl4ai", lambda u, t: _page("crawl4ai")
    )  # would win if reached
    page = resolve_web_fetch("https://x.io")
    assert page is not None and page.backend == "archivebox"


def test_falls_through_on_backend_failure(monkeypatch):
    """ArchiveBox miss + crawl4ai miss → requests serves it."""
    monkeypatch.setattr(wf, "archivebox_configured", lambda: True)
    monkeypatch.setattr(wf, "_fetch_via_archivebox", lambda u, t: None)
    monkeypatch.setattr(wf, "_fetch_via_crawl4ai", lambda u, t: None)
    monkeypatch.setattr(wf, "_fetch_via_requests", lambda u, t: _page("requests"))
    page = resolve_web_fetch("https://x.io")
    assert page is not None and page.backend == "requests"


def test_prefer_forces_single_backend(monkeypatch):
    monkeypatch.setattr(wf, "_fetch_via_crawl4ai", lambda u, t: _page("crawl4ai"))
    monkeypatch.setattr(wf, "_fetch_via_requests", lambda u, t: _page("requests"))
    page = resolve_web_fetch("https://x.io", prefer="crawl4ai")
    assert page is not None and page.backend == "crawl4ai"


def test_returns_none_when_all_fail(monkeypatch):
    monkeypatch.setattr(wf, "archivebox_configured", lambda: False)
    monkeypatch.setattr(wf, "_fetch_via_crawl4ai", lambda u, t: None)
    monkeypatch.setattr(wf, "_fetch_via_requests", lambda u, t: None)
    assert resolve_web_fetch("https://x.io") is None


def test_records_normalization():
    assert _records(None) == []
    assert _records({"results": [{"a": 1}, "skip"]}) == [{"a": 1}]
    assert _records([{"a": 1}, 2]) == [{"a": 1}]
    assert _records({"id": "s1"}) == [{"id": "s1"}]  # bare dict → single record


def test_looks_like_path():
    assert _looks_like_path("archive/snap/output.txt")
    assert _looks_like_path("/data/x.html")
    assert not _looks_like_path("This is real extracted body text.\nWith newlines.")


def test_archivebox_text_inline_then_archiveresult():
    # inline markdown on the snapshot wins immediately
    text, title = _archivebox_text(
        lambda *a, **k: {}, {"markdown": "# Inline", "title": "T"}, "https://x.io"
    )
    assert text == "# Inline" and title == "T"

    # else: ask for the text archiveresult; skip path-like outputs, take real body
    calls = {"n": 0}

    def call(tool, action, params):
        calls["n"] += 1
        if params.get("extractor") == "markdown":
            return {"results": [{"output": "snapshots/abc/output.md"}]}  # path → skip
        return {"results": [{"text": "real preserved body text\nmore"}]}

    text, title = _archivebox_text(
        call, {"abid": "abc", "title": "Doc"}, "https://x.io"
    )
    assert "real preserved body" in text and title == "Doc"
