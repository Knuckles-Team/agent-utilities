"""CONCEPT:AU-KG.query.object-graph-mapper"""

from unittest.mock import MagicMock

import pytest
from pydantic_ai import RunContext

from agent_utilities.models import AgentDeps
from agent_utilities.tools.knowledge_tools import (
    add_knowledge_memory,
    get_code_impact,
    get_knowledge_memory,
    search_knowledge_graph,
    sync_feature_to_memory,
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
        "description": "content",
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


def _spec_stub(title: str = "Feature Title") -> MagicMock:
    spec = MagicMock()
    spec.title = title
    story = MagicMock()
    story.description = "Goal description"
    spec.user_stories = [story]
    return spec


@pytest.mark.asyncio
async def test_sync_feature_to_memory_updates_existing_memory_not_duplicates(
    mock_ctx, monkeypatch
):
    """Regression test: real engine nodes carry the CURRENT ``node_type``
    property (the schema retired the bare ``type`` key ``add_memory``/
    ``_serialize_node`` used to write). A lookup keyed on bare ``type`` alone
    always misses against real data, so every sync silently created a
    duplicate memory node instead of updating the existing one — this proves
    the fixed lookup finds the existing node and calls ``update_memory``."""
    mock_ctx.deps.workspace_path = "/tmp/ws"
    engine = mock_ctx.deps.knowledge_engine
    feature_id = "feature-42"
    mem_name = f"SDD Feature Memory: {feature_id}"

    engine.graph.node_ids.return_value = ["other:1", "mem:existing"]
    engine.graph._get_node_properties.side_effect = lambda nid: {
        "other:1": {"node_type": "agent", "name": "unrelated"},
        "mem:existing": {"node_type": "memory", "name": mem_name},
    }[nid]

    monkeypatch.setattr(
        "agent_utilities.tools.knowledge_tools.SDDManager",
        lambda *_a, **_k: MagicMock(
            load=lambda cls, feature_id: (
                _spec_stub() if cls.__name__ == "Spec" else None
            )
        ),
    )

    result = await sync_feature_to_memory(mock_ctx, feature_id)

    engine.update_memory.assert_called_once()
    assert engine.update_memory.call_args.args[0] == "mem:existing"
    engine.add_memory.assert_not_called()
    assert "updated" in result.lower()
