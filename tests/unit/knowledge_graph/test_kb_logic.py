import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from agent_utilities.knowledge_graph.kb.extractor import KBExtractor
from agent_utilities.knowledge_graph.kb.parser import KBDocumentParser
from agent_utilities.models.knowledge_base import DocumentChunk, ExtractedArticle

@pytest.fixture
def mock_chunk():
    return DocumentChunk(
        content="This is a test document content about Pydantic AI.",
        source_path="test.md",
        source_type="md",
        chunk_index=0,
        content_hash="hash123",
        word_count=10
    )

@pytest.mark.asyncio
async def test_kb_extractor_fallback(mock_chunk):
    extractor = KBExtractor()
    # Mocking _get_article_agent to return None to force fallback
    with patch.object(KBExtractor, "_get_article_agent", return_value=None):
        article = await extractor.extract_article([mock_chunk], "test topic")
        assert article is not None
        assert article.title == "test topic"
        assert "Pydantic AI" in article.content

@pytest.mark.asyncio
async def test_kb_extractor_agent_success(mock_chunk):
    extractor = KBExtractor()
    mock_agent = AsyncMock()

    mock_result = MagicMock()
    mock_result.data = ExtractedArticle(
        title="Success",
        summary="A successful extraction",
        content="Full content",
        concepts=["success"],
        facts=[],
        backlinks=[],
        tags=["test"]
    )
    mock_agent.run.return_value = mock_result

    with patch.object(KBExtractor, "_get_article_agent", return_value=mock_agent):
        article = await extractor.extract_article([mock_chunk], "test topic")
        assert article is not None
        assert article.title == "Success"
        mock_agent.run.assert_called_once()

def test_kb_parser_markdown(tmp_path):
    parser = KBDocumentParser(chunk_size=10)
    md_file = tmp_path / "test.md"
    md_file.write_text("Word1 Word2 Word3 Word4 Word5 Word6 Word7 Word8 Word9 Word10 Word11 Word12")

    source = parser.parse_file(md_file)
    assert source is not None
    assert source.name == "test"
    assert len(source.chunks) > 1 # Chunked because chunk_size is 10
    assert source.chunks[0].word_count >= 9 # Roughly

def test_kb_parser_directory(tmp_path):
    parser = KBDocumentParser()
    (tmp_path / "file1.md").write_text("content1")
    (tmp_path / "file2.txt").write_text("content2")
    (tmp_path / "file3.jpg").write_text("content3") # Unsupported

    sources = parser.parse_directory(tmp_path)
    assert len(sources) == 2
    names = [s.name for s in sources]
    assert "file1" in names
    assert "file2" in names

def test_read_skill_graph_metadata(tmp_path):
    parser = KBDocumentParser()
    skill_md = tmp_path / "SKILL.md"
    skill_md.write_text("---\nname: Test Graph\ndescription: A graph\ntags: [tag1]\n---")

    meta = parser.read_skill_graph_metadata(tmp_path)
    assert meta["name"] == "Test Graph"
    assert meta["tags"] == ["tag1"]
