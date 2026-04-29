import pytest
import json
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient
from agent_utilities.server import build_agent_app

@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.toolsets = []
    agent.to_a2a.return_value = MagicMock()
    return agent

@pytest.fixture
def client(mock_agent):
    # Mocking create_agent to return our mock_agent
    with patch("agent_utilities.server.create_agent", return_value=(mock_agent, [])):
        app = build_agent_app(
            provider="test-provider",
            model_id="test-model",
            enable_web_ui=False,
            enable_acp=False,
            enable_otel=False,
            graph_bundle=("graph", "config")
        )
        return TestClient(app)

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "OK"
    assert "agent" in data

def test_list_chats(client):
    with patch("agent_utilities.server.list_chats_from_disk", return_value=["chat1", "chat2"]):
        response = client.get("/chats")
        assert response.status_code == 200
        assert response.json() == ["chat1", "chat2"]

def test_get_chat_success(client):
    with patch("agent_utilities.server.get_chat_from_disk", return_value={"id": "chat1", "messages": []}):
        response = client.get("/chats/chat1")
        assert response.status_code == 200
        assert response.json()["id"] == "chat1"

def test_get_chat_not_found(client):
    with patch("agent_utilities.server.get_chat_from_disk", return_value=None):
        response = client.get("/chats/missing")
        assert response.status_code == 404
        assert "error" in response.json()

def test_mcp_config(client):
    with patch("agent_utilities.server.get_workspace_path") as mock_path:
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.read_text.return_value = json.dumps({"mcpServers": {"test": {}}})

        response = client.get("/mcp/config")
        assert response.status_code == 200
        assert "mcpServers" in response.json()

def test_resolve_approval_missing_rid(client):
    response = client.post("/api/approve", json={})
    assert response.status_code == 400
    assert "request_id is required" in response.json()["error"]

def test_resolve_approval_success(client):
    with patch("agent_utilities.server._approval_manager") as mock_mgr:
        mock_mgr.resolve.return_value = True
        response = client.post("/api/approve", json={"request_id": "req1", "decisions": {}})
        assert response.status_code == 200
        assert response.json()["status"] == "resolved"

def test_resolve_approval_not_found(client):
    with patch("agent_utilities.server._approval_manager") as mock_mgr:
        mock_mgr.resolve.return_value = False
        response = client.post("/api/approve", json={"request_id": "missing", "decisions": {}})
        assert response.status_code == 404

@pytest.mark.asyncio
async def test_reload_mcp_config(client):
    # We must patch where it's used if it's a local import, or use the full path
    with patch("agent_utilities.mcp_agent_manager.sync_mcp_agents", new_callable=AsyncMock) as mock_sync, \
         patch("agent_utilities.graph_orchestration.load_node_agents_registry") as mock_reg:

        mock_reg.return_value.agents = [1, 2]
        mock_reg.return_value.tools = [1, 2, 3]

        response = client.post("/mcp/reload")
        print(f"DEBUG: response={response.json()}")
        assert response.status_code == 200
        assert response.json()["status"] == "reloaded"
        assert response.json()["agents"] == 2

def test_generate_codemap_not_initialized(client):
    with patch("agent_utilities.knowledge_graph.engine.IntelligenceGraphEngine.get_active", return_value=None):
        response = client.post("/api/codemap", json={"prompt": "test"})
        assert response.status_code == 503
        assert "Knowledge Graph not initialized" in response.json()["message"]

@pytest.mark.asyncio
async def test_generate_codemap_success(client):
    mock_kg = MagicMock()
    mock_artifact = MagicMock()
    mock_artifact.id = "map1"
    mock_artifact.model_dump.return_value = {"id": "map1", "nodes": []}

    with patch("agent_utilities.knowledge_graph.engine.IntelligenceGraphEngine.get_active", return_value=mock_kg), \
         patch("agent_utilities.knowledge_graph.codemaps.CodemapGenerator") as mock_gen_class:

        mock_gen = mock_gen_class.return_value
        mock_gen.create = AsyncMock(return_value=mock_artifact)

        response = client.post("/api/codemap", json={"prompt": "analyze this", "mode": "smart"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["codemap_id"] == "map1"
        assert "artifact" in data

@pytest.mark.asyncio
async def test_ag_ui_stream(client):
    with patch("pydantic_ai.ui.ag_ui.AGUIAdapter") as mock_adapter_class:
        mock_adapter = mock_adapter_class.return_value
        mock_adapter.dispatch_request = AsyncMock()

        # Mocking StreamingResponse
        from fastapi.responses import StreamingResponse
        async def mock_iterator():
            yield b"data: test\n\n"

        mock_adapter.dispatch_request.return_value = StreamingResponse(mock_iterator())

        response = client.post("/ag-ui", json={"query": "hello"})
        assert response.status_code == 200
        # Check if we can read the stream
        content = b"".join(response.iter_bytes())
        assert b"data: test" in content

def test_stream_endpoint(client, mock_agent):
    # Mocking run_graph_stream from its source module
    with patch("agent_utilities.graph_orchestration.run_graph_stream") as mock_run:
        async def mock_stream_gen(*args, **kwargs):
            yield "event: message\ndata: hello\n\n"

        mock_run.return_value = mock_stream_gen()

        response = client.post("/stream", json={"query": "test"})
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        content = b"".join(response.iter_bytes())
        assert b"data: hello" in content

def test_list_mcp_tools(client, mock_agent):
    mock_tool = MagicMock()
    mock_tool.name = "test_tool"
    mock_tool.description = "test desc"

    mock_ts = MagicMock()
    mock_ts.get_tools.return_value = [mock_tool]
    mock_ts.name = "test_server"

    mock_agent.toolsets = [mock_ts]

    response = client.get("/mcp/tools")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["name"] == "test_tool"
    assert data[0]["tag"] == "test_server"
