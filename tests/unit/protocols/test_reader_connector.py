"""Tests for the readability reader connector (CONCEPT:AU-KG.enrichment.multimodal-readers).

Offline: a ``fetch_fn`` is injected so the local readability tier runs without a
network call or a JINA key.
"""

from __future__ import annotations

import pytest

from agent_utilities.protocols.source_connectors.registry import (
    discover,
    get_connector_class,
)

_HTML = (
    "<html><head><title>Jina v5</title></head>"
    "<body><script>x()</script><p>Jina AI released jina-embeddings-v5 in 2025.</p>"
    "</body></html>"
)


def test_reader_is_discovered() -> None:
    discover()
    assert get_connector_class("reader") is not None


def test_reader_loads_single_clean_document_offline() -> None:
    discover()
    cls = get_connector_class("reader")
    conn = cls(url="http://example.com/post", fetch_fn=lambda _u: _HTML)
    docs = list(conn.load())
    assert len(docs) == 1
    doc = docs[0]
    assert doc.source_uri == "http://example.com/post"
    assert doc.doc_type == "article"
    assert doc.metadata["reader"] == "local"  # offline → local tier
    # boilerplate <script> stripped; body content retained
    assert "x()" not in doc.text
    assert "jina-embeddings-v5" in doc.text


def test_reader_requires_url() -> None:
    discover()
    cls = get_connector_class("reader")
    with pytest.raises(ValueError, match="url"):
        cls()


def test_reader_skips_empty_pages() -> None:
    discover()
    cls = get_connector_class("reader")
    conn = cls(url="http://x", fetch_fn=lambda _u: "   ")
    assert list(conn.load()) == []
