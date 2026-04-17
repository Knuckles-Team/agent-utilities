import pytest
import asyncio
from unittest.mock import patch
from agent_utilities.graph.steps import fetch_unified_context, _emit_node_lifecycle
from agent_utilities.graph.config_helpers import emit_graph_event, load_mcp_config, load_node_agents_registry
from agent_utilities.models import MCPConfigModel, MCPAgentRegistryModel

@pytest.mark.asyncio
async def test_fetch_unified_context_mocked():
    """Test fetch_unified_context with mocked file reads."""
    with patch("agent_utilities.graph.steps.load_workspace_file", return_value="some content"):
        with patch("subprocess.check_output", return_value=b"M somefile.py"):
            context = await fetch_unified_context()
            assert "PROJECT CONTEXT" in context
            assert "some content" in context
            assert "M somefile.py" in context

def test_emit_graph_event():
    """Test emit_graph_event logic."""
    eq = asyncio.Queue()
    with patch("agent_utilities.graph.config_helpers._log_graph_trace") as mock_log:
        emit_graph_event(eq, "test_event", foo="bar")
        assert eq.qsize() == 1
        event = eq.get_nowait()
        assert event["type"] == "data-graph-event"
        assert event["data"]["event"] == "test_event"
        assert event["data"]["foo"] == "bar"
        mock_log.assert_called_once()

def test_emit_node_lifecycle():
    """Test _emit_node_lifecycle helper."""
    eq = asyncio.Queue()
    with patch("agent_utilities.graph.steps.emit_graph_event") as mock_emit:
        _emit_node_lifecycle(eq, "node1", "node_start", extra="data")
        mock_emit.assert_called_once_with(eq, "node_start", node_id="node1", extra="data")

@pytest.mark.asyncio
async def test_load_config_fallbacks():
    """Test configuration loading fallbacks on missing files."""
    with patch("agent_utilities.graph.config_helpers.get_workspace_path") as mock_path:
        mock_path.return_value.exists.return_value = False
        config = load_mcp_config()
        assert isinstance(config, MCPConfigModel)

        registry = load_node_agents_registry()
        assert isinstance(registry, MCPAgentRegistryModel)
