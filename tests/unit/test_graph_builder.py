import pytest
from unittest.mock import MagicMock, patch
from agent_utilities.graph import builder

def test_build_tag_env_map():
    tags = ["incidents", "git-ops"]
    env_map = builder.build_tag_env_map(tags)
    assert env_map["incidents"] == "INCIDENTSTOOL"
    assert env_map["git-ops"] == "GIT_OPSTOOL"

@patch("agent_utilities.graph.builder.get_discovery_registry")
@patch("agent_utilities.graph.builder.load_mcp_config")
@patch("agent_utilities.graph.builder.GraphBuilder")
def test_initialize_graph_from_workspace(mock_gb, mock_load_mcp, mock_registry):
    mock_registry.return_value.agents = []
    mock_load_mcp.return_value = []
    
    # Mock GraphBuilder and its methods
    builder_instance = mock_gb.return_value
    builder_instance.node.return_value = lambda x: x
    
    graph, config = builder.initialize_graph_from_workspace()
    
    assert "tag_prompts" in config
    assert "tag_env_vars" in config
    assert mock_registry.called
    assert mock_load_mcp.called

def test_pydantic_graph_availability():
    assert builder._PYDANTIC_GRAPH_AVAILABLE is True
