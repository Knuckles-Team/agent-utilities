from unittest.mock import MagicMock, patch
import ladybug

from agent_utilities.graph import builder

def test_build_tag_env_map():
    tags = ["incidents", "git-ops"]
    env_map = builder.build_tag_env_map(tags)
    assert env_map["incidents"] == "INCIDENTSTOOL"
    assert env_map["git-ops"] == "GIT_OPSTOOL"

@patch("agent_utilities.graph.builder.get_discovery_registry")
@patch("agent_utilities.graph.builder.load_mcp_config")
@patch("agent_utilities.graph.builder.GraphBuilder")
@patch("agent_utilities.graph.client.get_graph_client")
@patch("agent_utilities.graph.builder.ingest_prompts_to_graph")
@patch("agent_utilities.graph.builder.get_agent_workspace")
@patch("agent_utilities.graph.builder.RegistryPipeline")
@patch("agent_utilities.graph.builder.RegistryGraphEngine")
@patch("agent_utilities.graph.builder.PipelineConfig")
@patch("ladybug.Connection")
@patch("ladybug.Database")
@patch("agent_utilities.graph.builder.sync_mcp_agents")
@patch("agent_utilities.graph.builder.should_sync")
@patch("agent_utilities.graph.builder.resolve_mcp_config_path")
def test_initialize_graph_from_workspace(
    mock_resolve,
    mock_should_sync,

    mock_sync,
    mock_db,
    mock_conn,
    mock_engine,
    mock_config,
    mock_pipeline,
    mock_ws,
    mock_ingest,
    mock_client,
    mock_gb,
    mock_load_mcp,
    mock_registry,
):



    from pathlib import Path
    mock_registry.return_value.agents = []
    mock_load_mcp.return_value = []
    mock_client.return_value = MagicMock()
    mock_ws.return_value = Path("/tmp/agent_test_workspace")
    mock_resolve.return_value = Path("/tmp/mcp_config.json")
    mock_should_sync.return_value = True



    # Mock GraphBuilder and its methods
    builder_instance = mock_gb.return_value
    builder_instance.node.return_value = lambda x: x

    # Ensure background sync is disabled and validation mode is off for this test
    with patch.dict("os.environ", {"KNOWLEDGE_GRAPH_SYNC_BACKGROUND": "false", "VALIDATION_MODE": "false"}), \
         patch("agent_utilities.graph.builder.DEFAULT_VALIDATION_MODE", False):
        graph, config = builder.initialize_graph_from_workspace()

    assert "tag_prompts" in config
    assert "tag_env_vars" in config
    assert mock_registry.called
    assert mock_load_mcp.called
    assert mock_should_sync.called
    assert mock_sync.called
    assert mock_pipeline.called
    assert mock_ingest.called


def test_pydantic_graph_availability():
    assert builder._PYDANTIC_GRAPH_AVAILABLE is True
