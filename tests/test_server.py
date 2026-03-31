import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
import os
from agent_utilities import agent_utilities
from agent_utilities.agent_utilities import (
    create_agent_server,
    create_agent,
    initialize_workspace,
)
from agent_webui.server import create_agent_web_app as create_enhanced_web_app


@pytest.fixture
def dummy_workspace(tmp_path, monkeypatch):
    def mock_get_workspace_path(subpath=""):
        res = tmp_path / subpath if subpath else tmp_path
        if subpath and not res.exists() and "." not in subpath:


            if not res.suffix:
                res.mkdir(parents=True, exist_ok=True)
        return res

    monkeypatch.setattr(agent_utilities, "get_workspace_path", mock_get_workspace_path)

    yield tmp_path


@pytest.fixture
def dummy_agent(dummy_workspace):
    os.environ["OPENAI_API_KEY"] = "sk-dummy"


    initialize_workspace()


    agent = create_agent(
        name="TestBot", system_prompt="Test description", model_id="gpt-4o"
    )
    return agent


def test_server_creation_logic(dummy_agent):


    import uvicorn

    original_run = uvicorn.run
    uvicorn.run = MagicMock()

    os.environ["OPENAI_API_KEY"] = "sk-dummy"


    create_agent_server(enable_web_ui=True)
    app = uvicorn.run.call_args[0][0]
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "Dashboard" in response.text

    uvicorn.run = original_run


def test_enhanced_api_endpoints(dummy_agent):

    helpers = {
        "agent_name": "TestBot",
        "agent_emoji": "🤖",
        "get_workspace_path": lambda x="": x,
        "list_workspace_files": lambda: ["IDENTITY.md"],
    }
    app = create_enhanced_web_app(dummy_agent, workspace_helpers=helpers)
    client = TestClient(app)


    response = client.get("/api/enhanced/info")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "TestBot"
    assert data["emoji"] == "🤖"


    response = client.get("/api/enhanced/files")
    assert response.status_code == 200
    files = response.json()
    assert "IDENTITY.md" in files


def test_cli_parser(monkeypatch):
    from agent_utilities.agent_utilities import create_agent_parser


    monkeypatch.setenv("ENABLE_WEB_UI", "False")
    monkeypatch.setenv("ENABLE_OTEL", "False")
    parser = create_agent_parser()
    args = parser.parse_args([])
    assert args.web is False
    assert args.otel is False


    args = parser.parse_args(["--web", "--otel"])
    assert args.web is True
    assert args.otel is True


    monkeypatch.setenv("ENABLE_WEB_UI", "True")
    monkeypatch.setenv("ENABLE_OTEL", "True")
    parser = create_agent_parser()
    args = parser.parse_args([])
    assert args.web is True
    assert args.otel is True


    args = parser.parse_args(["--no-web", "--no-otel"])
    assert args.web is False
    assert args.otel is False
