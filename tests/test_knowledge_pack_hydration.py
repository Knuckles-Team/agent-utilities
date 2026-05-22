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
    ):
        mock_tempfile.return_value.__enter__.return_value.name = "/tmp/fake.pdf"
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
