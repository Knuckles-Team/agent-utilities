"""CONCEPT:AU-KG.query.object-graph-mapper"""

from unittest.mock import MagicMock

import pytest

from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
from agent_utilities.knowledge_graph.pipeline.phases.centrality import (
    execute_centrality,
)
from agent_utilities.knowledge_graph.pipeline.types import PipelineContext


@pytest.fixture
def mock_pipeline_ctx():
    ctx = MagicMock(spec=PipelineContext)
    ctx.graph = GraphComputeEngine(backend_type="rust")
    ctx.backend = MagicMock()
    ctx.config = MagicMock()
    return ctx


@pytest.mark.asyncio
async def test_execute_centrality(mock_pipeline_ctx):
    # Add some nodes and edges
    mock_pipeline_ctx.graph.add_edge("A", "B", relationship="connects_to")
    mock_pipeline_ctx.graph.add_edge("B", "C", relationship="connects_to")

    result = await execute_centrality(mock_pipeline_ctx, {})

    assert result["centrality_calculated"] is True
    # The rust graph backend doesn't support direct node property access via .nodes["A"] like NetworkX
    # It returns nodes through graph._get_node_properties("A") or via get_nodes()
    props = mock_pipeline_ctx.graph._get_node_properties("A")
    assert "centrality" in props
    assert result["top_node"] is not None
