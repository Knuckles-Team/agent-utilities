"""Live-path test: a research-roundup page auto-acquires + links its papers.

CONCEPT:KG-2.7 — exercises the IngestionEngine wiring (``_acquire_referenced_papers``),
not just the helpers, per the Wire-First discipline.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

import agent_utilities.knowledge_graph.ingestion.research_acquisition as ra
from agent_utilities.knowledge_graph.ingestion.engine import (
    ContentType,
    IngestionEngine,
    IngestionManifest,
    IngestionResult,
)

ROUNDUP = (
    "Papers: https://arxiv.org/abs/2603.09022 , https://arxiv.org/abs/2604.12345 , "
    "and arxiv 2605.00001 are great."
)


@pytest.mark.asyncio
async def test_roundup_page_acquires_and_links_papers(monkeypatch):
    kg = MagicMock()
    kg.backend = MagicMock()
    engine = IngestionEngine(kg_engine=kg)

    # Fake the download → three local PDFs (no network).
    monkeypatch.setattr(
        ra, "acquire_papers", lambda refs, **k: [Path(f"/tmp/p{i}.pdf") for i in range(3)]
    )

    # Fake the recursive per-paper ingest → each becomes a Document with a doc_id.
    async def fake_batch(manifests):
        assert all(m.metadata.get("source_kind") == "research_paper" for m in manifests)
        return [
            IngestionResult(
                manifest=m, status="success", details={"doc_id": f"doc:{i}"}
            )
            for i, m in enumerate(manifests)
        ]

    monkeypatch.setattr(engine, "ingest_batch", fake_batch)

    page_result = IngestionResult(
        manifest=IngestionManifest(content_type=ContentType.DOCUMENT, source_uri="u"),
        status="success",
        details={"doc_id": "page:1"},
    )
    await engine._acquire_referenced_papers(
        page_result, ROUNDUP, "https://blog/x", force=False
    )

    assert page_result.details["papers_acquired"] == 3
    assert page_result.details["papers_ingested"] == 3
    # page → each paper MENTIONS edge written
    edges = {
        (c.args[0], c.args[1], c.kwargs.get("rel_type"))
        for c in kg.backend.add_edge.call_args_list
    }
    assert ("page:1", "doc:0", "MENTIONS") in edges
    assert len(edges) == 3


@pytest.mark.asyncio
async def test_pdf_url_downloads_bytes_not_text(monkeypatch):
    """A .pdf URL is fetched as bytes → temp .pdf → file unit (not the text path)."""
    kg = MagicMock()
    kg.backend = MagicMock()
    engine = IngestionEngine(kg_engine=kg)

    class FakeResp:
        content = b"%PDF-1.7 fake bytes"

        def raise_for_status(self):
            pass

    monkeypatch.setattr("requests.get", lambda *a, **k: FakeResp())
    seen = {}

    def fake_file(manifest, path):
        seen["suffix"] = path.suffix
        seen["source_url"] = manifest.metadata.get("source_url")
        return IngestionResult(manifest=manifest, status="success")

    monkeypatch.setattr(engine, "_ingest_document_file", fake_file)
    # resolve_web_fetch must NOT be used for a binary doc URL
    import agent_utilities.knowledge_graph.ingestion.web_fetch as wf

    monkeypatch.setattr(
        wf, "resolve_web_fetch", lambda *a, **k: pytest.fail("text path used for PDF")
    )

    m = IngestionManifest(
        content_type=ContentType.DOCUMENT,
        source_uri="https://x.io/release-notes.pdf",
    )
    res = await engine._ingest_document_url(m, m.source_uri)
    assert res.status == "success"
    assert seen["suffix"] == ".pdf"
    assert seen["source_url"] == "https://x.io/release-notes.pdf"


@pytest.mark.asyncio
async def test_non_roundup_page_skips(monkeypatch):
    """A page with too few scholarly links does not trigger acquisition."""
    kg = MagicMock()
    kg.backend = MagicMock()
    engine = IngestionEngine(kg_engine=kg)
    called = {"n": 0}
    monkeypatch.setattr(ra, "acquire_papers", lambda refs, **k: called.__setitem__("n", 1))

    page_result = IngestionResult(
        manifest=IngestionManifest(content_type=ContentType.DOCUMENT, source_uri="u"),
        status="success",
        details={"doc_id": "page:1"},
    )
    await engine._acquire_referenced_papers(
        page_result, "one link https://arxiv.org/abs/2603.09022 only", "u", force=False
    )
    assert called["n"] == 0  # below ROUNDUP_MIN_PAPERS → skipped
    assert "papers_acquired" not in page_result.details
