import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Create a mock module for pymupdf4llm
mock_pymupdf4llm = MagicMock()
mock_pymupdf4llm.to_markdown.return_value = "PDF content"
sys.modules["pymupdf4llm"] = mock_pymupdf4llm

# Create a mock module for crawl4ai
mock_crawl4ai = MagicMock()
mock_browser = AsyncMock()
mock_browser.start = AsyncMock()
mock_browser.close = AsyncMock()
mock_result = MagicMock()
mock_result.success = True
mock_result.url = "https://example.com/article"
mock_result.markdown_v2 = None
mock_result.markdown = "Web content"
mock_browser.arun_many.return_value = [mock_result]
mock_browser.__aenter__.return_value = mock_browser
mock_browser.__aexit__.return_value = False
mock_crawl4ai.AsyncWebCrawler = MagicMock(return_value=mock_browser)
sys.modules["crawl4ai"] = mock_crawl4ai

from agent_utilities.models.knowledge_pack import (
    KnowledgePackBundle,
    KnowledgePackHydrator,
)


@pytest.fixture
def sample_bundle():
    return KnowledgePackBundle(
        name="test_pack",
        version="1.0.0",
        description="A test pack",
        domain="testing",
        nodes=[
            {
                "id": "node_pdf",
                "type": "DOCUMENT",
                "url": "https://example.com/doc.pdf",
            },
            {"id": "node_web", "type": "ARTICLE", "url": "https://example.com/article"},
            {"id": "node_no_url", "type": "SOFTWARE_PROJECT", "name": "No URL Project"},
        ],
        edges=[],
    )


@pytest.mark.asyncio
async def test_hydrate_bundle(sample_bundle):
    with (
        patch("requests.get") as mock_get,
        patch("tempfile.NamedTemporaryFile") as mock_tempfile,
        # No searxng server configured — exercise the zero-infra crawl4ai path.
        patch(
            "agent_utilities.models.knowledge_pack._searxng_connector_for",
            return_value=None,
        ),
    ):
        mock_tempfile.return_value.__enter__.return_value.name = ".tmp/fake.pdf"
        mock_get.return_value.content = b"fake pdf bytes"
        mock_get.return_value.raise_for_status = MagicMock()

        await KnowledgePackHydrator.hydrate(sample_bundle)
        hydrated_bundle = sample_bundle

        # Verify PDF node
        pdf_node = next(n for n in hydrated_bundle.nodes if n["id"] == "node_pdf")
        assert pdf_node["content"] == "PDF content"

        # Verify Web node
        web_node = next(n for n in hydrated_bundle.nodes if n["id"] == "node_web")
        assert web_node["content"] == "Web content"

        # Verify node with no url
        no_url_node = next(n for n in hydrated_bundle.nodes if n["id"] == "node_no_url")
        assert "content" not in no_url_node


# ---------------------------------------------------------------------------
# KG-2.59 reuse policy: web retrieval routes through the searxng-mcp
# mcp_tool source preset when configured; crawl4ai stays the final fallback.
# ---------------------------------------------------------------------------


class FakeSearxngConnector:
    """Fake mcp_tool searxng source: yields canned SourceDocuments."""

    def __init__(self, docs):
        self._docs = docs

    def load(self):
        yield from self._docs


def _searxng_doc(url: str, text: str):
    from agent_utilities.protocols.source_connectors.base import SourceDocument

    return SourceDocument(id=url, source_uri=url, title="result", text=text)


@pytest.mark.asyncio
async def test_hydrate_web_via_searxng_mcp_tool(sample_bundle):
    """When searxng is configured, web content comes from the mcp_tool source."""
    url = "https://example.com/article"
    fake = FakeSearxngConnector([_searxng_doc(url, "Searxng content")])

    with (
        patch("requests.get") as mock_get,
        patch("tempfile.NamedTemporaryFile") as mock_tempfile,
        patch(
            "agent_utilities.models.knowledge_pack._searxng_connector_for",
            return_value=fake,
        ) as mock_factory,
    ):
        mock_tempfile.return_value.__enter__.return_value.name = ".tmp/fake.pdf"
        mock_get.return_value.content = b"fake pdf bytes"
        mock_get.return_value.raise_for_status = MagicMock()

        await KnowledgePackHydrator.hydrate(sample_bundle)

    web_node = next(n for n in sample_bundle.nodes if n["id"] == "node_web")
    assert web_node["content"] == "Searxng content"
    mock_factory.assert_called_once_with(url)


def test_hydrate_via_searxng_returns_unmatched_urls_for_fallback():
    """URLs the searxng source cannot serve fall through to crawl4ai."""
    matched = "https://example.com/found"
    unmatched = "https://example.com/missing"
    node_map = {matched: {"id": "a"}, unmatched: {"id": "b"}}

    def factory(query):
        return FakeSearxngConnector([_searxng_doc(matched, "Found body")])

    with patch(
        "agent_utilities.models.knowledge_pack._searxng_connector_for",
        side_effect=factory,
    ):
        remaining = KnowledgePackHydrator._hydrate_via_searxng(
            [matched, unmatched], node_map
        )

    assert remaining == [unmatched]
    assert node_map[matched]["content"] == "Found body"
    assert "content" not in node_map[unmatched]


def test_hydrate_via_searxng_unconfigured_keeps_all_urls():
    """No searxng server configured: every URL is left for crawl4ai."""
    urls = ["https://example.com/a", "https://example.com/b"]
    node_map = {u: {} for u in urls}

    with patch(
        "agent_utilities.models.knowledge_pack._searxng_connector_for",
        return_value=None,
    ):
        remaining = KnowledgePackHydrator._hydrate_via_searxng(urls, node_map)

    assert remaining == urls
    assert all("content" not in node_map[u] for u in urls)


def test_searxng_connector_factory_unconfigured(monkeypatch):
    """Factory returns None when mcp_config has no searxng server."""
    from agent_utilities.models import knowledge_pack as kp
    from agent_utilities.protocols.source_connectors.connectors import mcp_package

    monkeypatch.setattr(mcp_package, "_load_mcp_config", lambda: {"other-mcp": {}})
    assert kp._searxng_connector_for("https://example.com") is None


def test_searxng_connector_factory_builds_preset(monkeypatch):
    """Factory builds an mcp_tool connector bound to the searxng preset."""
    from agent_utilities.models import knowledge_pack as kp
    from agent_utilities.protocols.source_connectors.connectors import mcp_package

    monkeypatch.setattr(
        mcp_package,
        "_load_mcp_config",
        lambda: {"searxng-mcp": {"url": "http://searxng-mcp.test/mcp"}},
    )
    conn = kp._searxng_connector_for("site query")
    assert conn is not None
    assert conn.tool == "web_search"
    assert conn.params == {"query": "site query"}
    assert conn.records_path == "results"
    assert conn.text_field == "content"
