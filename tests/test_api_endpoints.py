import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.name = "TestAgent"
    # Mock to_a2a to return a FastAPI app
    from fastapi import FastAPI

    a2a_app = FastAPI()

    @a2a_app.get("/info")
    async def info():
        return {"name": "TestAgent"}

    agent.to_a2a.return_value = a2a_app
    return agent


@pytest.fixture
def client(mock_agent):
    with patch("agent_utilities.server.create_agent") as mock_create:
        mock_create.return_value = (mock_agent, [])
        # We need to mock initialize_workspace and other workspace calls
        with (
            patch("agent_utilities.server.initialize_workspace"),
            patch(
                "agent_utilities.server.load_identity",
                return_value={"name": "TestAgent"},
            ),
            patch("agent_utilities.server.get_skills_path", return_value=None),
        ):
            from agent_utilities.server import build_agent_app

            app = build_agent_app(enable_web_ui=False)
            return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "OK"
    assert response.json()["agent"] == "Agent"  # Default name if not overridden


def test_a2a_info_endpoint(client):
    response = client.get("/a2a/info")
    assert response.status_code == 200
    assert response.json()["name"] == "TestAgent"


def test_mcp_config_endpoint(client, tmp_path):
    with patch("agent_utilities.server.get_workspace_path") as mock_ws_path:
        config_file = tmp_path / "mcp_config.json"
        config_file.write_text('{"mcpServers": {"test": {}}}')
        mock_ws_path.return_value = config_file

        response = client.get("/mcp/config")
        assert response.status_code == 200
        assert "test" in response.json()["mcpServers"]


def test_list_chats_endpoint(client):
    with patch(
        "agent_utilities.server.list_chats_from_disk", return_value=["chat1", "chat2"]
    ):
        response = client.get("/chats")
        assert response.status_code == 200
        assert len(response.json()) == 2
