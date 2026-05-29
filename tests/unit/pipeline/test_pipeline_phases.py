"""CONCEPT:KG-2.0"""

from unittest.mock import AsyncMock, MagicMock, patch

from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
import pytest

from agent_utilities.knowledge_graph.pipeline.phases.centrality import (
    execute_centrality,
)
from agent_utilities.knowledge_graph.pipeline.phases.workspace_sync import (
    execute_workspace_sync,
)
from agent_utilities.knowledge_graph.pipeline.types import PipelineContext


@pytest.fixture
def mock_pipeline_ctx():
    import sys
    sys.modules["repository_manager"] = MagicMock()
    sys.modules["repository_manager.repository_manager"] = MagicMock()

    ctx = MagicMock(spec=PipelineContext)
    ctx.graph = GraphComputeEngine(backend_type="rust")
    ctx.backend = MagicMock()
    ctx.config = MagicMock()
    return ctx


@pytest.mark.asyncio
async def test_execute_centrality(mock_pipeline_ctx):
    # Add some nodes and edges
    mock_pipeline_ctx.graph.add_edge("A", "B")
    mock_pipeline_ctx.graph.add_edge("B", "C")

    result = await execute_centrality(mock_pipeline_ctx, {})

    assert result["centrality_calculated"] is True
    # The rust graph backend doesn't support direct node property access via .nodes["A"] like NetworkX
    # It returns nodes through graph._get_node_properties("A") or via get_nodes()
    props = mock_pipeline_ctx.graph._get_node_properties("A")
    assert "centrality" in props
    assert result["top_node"] is not None


@pytest.mark.asyncio
async def test_execute_workspace_sync_no_yml(mock_pipeline_ctx, tmp_path):
    with patch(
        "agent_utilities.knowledge_graph.pipeline.phases.workspace_sync.get_agent_workspace",
        return_value=tmp_path,
    ):
        result = await execute_workspace_sync(mock_pipeline_ctx, {})
        assert result["status"] == "skipped"
        assert "workspace.yml missing" in result["reason"]


@pytest.mark.asyncio
async def test_execute_workspace_sync_success(mock_pipeline_ctx, tmp_path):
    workspace_dir = tmp_path / "agent_data"
    workspace_dir.mkdir()
    yml_path = tmp_path / "workspace.yml"
    project_dir = tmp_path / "Workspace" / "repo"
    project_dir.mkdir(parents=True)
    yml_path.write_text(
        f"projects:\n  - url: https://github.com/test/repo.git\n    name: repo\n    path: {project_dir}"
    )

    with (
        patch(
            "agent_utilities.knowledge_graph.pipeline.phases.workspace_sync.get_agent_workspace",
            return_value=workspace_dir,
        ),
        patch("repository_manager.repository_manager.Git") as MockGit,
        patch(
            "agent_utilities.knowledge_graph.kb.ingestion.KBIngestionEngine"
        ) as MockIngest,
    ):
        MockIngest.return_value.ingest_directory = AsyncMock()

        result = await execute_workspace_sync(mock_pipeline_ctx, {})

        assert result["projects_synced"] == 1
        assert result["auto_ingested"] == 1
        MockGit.return_value.clone_projects.assert_called_once()
