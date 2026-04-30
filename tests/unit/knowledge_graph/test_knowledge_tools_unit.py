from unittest.mock import MagicMock

import pytest
from pydantic_ai import RunContext

from agent_utilities.models import AgentDeps
from agent_utilities.tools.knowledge_tools import (
    add_knowledge_memory,
    get_code_impact,
    get_knowledge_memory,
    search_knowledge_graph,
)


@pytest.fixture
def mock_ctx():
    deps = MagicMock(spec=AgentDeps)
    deps.knowledge_engine = MagicMock()
    ctx = MagicMock(spec=RunContext)
    ctx.deps = deps
    return ctx

@pytest.mark.asyncio
async def test_search_knowledge_graph(mock_ctx):
    mock_ctx.deps.knowledge_engine.search_hybrid.return_value = [
        {"id": "node1", "type": "agent", "name": "Test Agent", "description": "desc"}
    ]
    result = await search_knowledge_graph(mock_ctx, "query")
    assert "[AGENT]" in result
    assert "Test Agent" in result

@pytest.mark.asyncio
async def test_add_knowledge_memory(mock_ctx):
    mock_ctx.deps.knowledge_engine.add_memory.return_value = "mem:123"
    result = await add_knowledge_memory(mock_ctx, "content", name="name")
    assert "mem:123" in result
    mock_ctx.deps.knowledge_engine.add_memory.assert_called_once_with(
        "content", name="name", category="general", tags=None
    )

@pytest.mark.asyncio
async def test_get_knowledge_memory(mock_ctx):
    mock_ctx.deps.knowledge_engine.get_memory.return_value = {
        "id": "mem:123",
        "name": "Memory",
        "timestamp": "2026",
        "category": "fact",
        "description": "content"
    }
    result = await get_knowledge_memory(mock_ctx, "mem:123")
    assert "content" in result
    assert "Memory" in result

@pytest.mark.asyncio
async def test_get_code_impact(mock_ctx):
    mock_ctx.deps.knowledge_engine.query_impact.return_value = [
        {"id": "file.py", "type": "file", "file_path": "path/file.py"}
    ]
    result = await get_code_impact(mock_ctx, "entity")
    assert "file.py" in result
    assert "Impact Set" in result
