import json
import os
import tempfile
from unittest.mock import MagicMock, patch

from agent_utilities import mcp_utilities
from agent_utilities.mcp.kg_coordinator import KGCoordinator


def test_kg_coordinator_is_port_open():
    with patch("socket.socket") as mock_socket:
        mock_conn = mock_socket.return_value.__enter__.return_value
        # If connect works:
        mock_conn.connect.return_value = None
        assert KGCoordinator.is_port_open("127.0.0.1", 8100) is True

        # If connect raises:
        mock_conn.connect.side_effect = ConnectionRefusedError()
        assert KGCoordinator.is_port_open("127.0.0.1", 8100) is False


@patch("agent_utilities.mcp.kg_coordinator.KGCoordinator.is_port_open")
@patch("httpx.get")
def test_kg_coordinator_is_server_healthy(mock_get, mock_port_open):
    # Port closed -> unhealthy
    mock_port_open.return_value = False
    assert KGCoordinator.is_server_healthy("127.0.0.1", 8100) is False

    # Port open but http raises -> unhealthy
    mock_port_open.return_value = True
    mock_get.side_effect = Exception("http failed")
    assert KGCoordinator.is_server_healthy("127.0.0.1", 8100) is False

    # Port open, http returns 200 -> healthy
    mock_get.side_effect = None
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_get.return_value = mock_response
    assert KGCoordinator.is_server_healthy("127.0.0.1", 8100) is True


@patch("psutil.net_connections")
@patch("psutil.Process")
def test_kg_coordinator_cleanup_rogue(mock_process, mock_net_conns):
    # Mocking a connection listening on 8100
    mock_conn = MagicMock()
    mock_conn.status = "LISTEN"
    mock_conn.laddr = ("127.0.0.1", 8100)
    mock_conn.pid = 9999
    mock_net_conns.return_value = [mock_conn]

    mock_proc = MagicMock()
    mock_proc.name.return_value = "python"
    mock_process.return_value = mock_proc

    KGCoordinator.cleanup_rogue_instances(8100)

    # Verifies it cleaned up gracefully
    mock_proc.terminate.assert_called_once()


@patch("subprocess.Popen")
@patch("agent_utilities.mcp.kg_coordinator.KGCoordinator.is_server_healthy")
@patch("agent_utilities.mcp.kg_coordinator.KGCoordinator.cleanup_rogue_instances")
@patch("time.sleep")
def test_kg_coordinator_spawn_server(
    mock_sleep, mock_cleanup, mock_healthy, mock_popen
):
    mock_healthy.side_effect = [
        False,
        True,
    ]  # first check unhealthy, second check healthy

    success = KGCoordinator.spawn_server("127.0.0.1", 8100)

    assert success is True
    mock_cleanup.assert_called_once_with(8100)
    mock_popen.assert_called_once()


@patch("pydantic_ai.mcp.load_mcp_servers")
@patch("agent_utilities.mcp.kg_coordinator.KGCoordinator.get_kg_client")
def test_load_config_intercepts_kg(mock_get_client, mock_load_mcp):
    config_data = {
        "mcpServers": {
            "test-server": {
                "command": "python",
                "args": ["-m", "test"],
            },
            "agent-utilities-kg": {
                "command": "uv",
                "args": ["run", "python", "-m", "agent_utilities.mcp.kg_server"],
            },
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(config_data, tmp)
        tmp_path = tmp.name

    try:
        mock_server = MagicMock()
        mock_load_mcp.return_value = [mock_server]

        # Let's ensure validation mode is off for testing interception
        with patch("agent_utilities.core.config.DEFAULT_VALIDATION_MODE", False):
            servers = mcp_utilities.load_mcp_servers_from_config(tmp_path)

        # "test-server" was loaded via standard file; "agent-utilities-kg" was intercepted
        assert len(servers) == 2
        assert mock_get_client.called
        assert servers[0].id == "test-server"
        assert servers[1].id == "agent-utilities-kg"

        # Check that it is MCPServerSSE
        assert type(servers[1]).__name__ == "MCPServerSSE"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
