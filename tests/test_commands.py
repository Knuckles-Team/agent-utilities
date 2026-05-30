import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from agent_utilities.server.routers.commands import router


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_execute_clear_chat(client):
    resp = client.post("/api/enhanced/commands/execute", json={"command": "/clear"})
    assert resp.status_code == 200
    data = resp.json()
    assert "response_markdown" in data
    assert "Chat session cleared." in data["response_markdown"]
    assert "client_actions" in data
    assert any(act["action"] == "clear_chat" for act in data["client_actions"])


def test_execute_help(client):
    resp = client.post("/api/enhanced/commands/execute", json={"command": "/help"})
    assert resp.status_code == 200
    data = resp.json()
    assert "response_markdown" in data
    assert "Available Commands:" in data["response_markdown"]


def test_autocomplete_empty(client):
    resp = client.get("/api/enhanced/commands/autocomplete?query=")
    assert resp.status_code == 200
    data = resp.json()
    assert "suggestions" in data
    assert len(data["suggestions"]) > 0
    assert "/help" in data["suggestions"]


def test_autocomplete_filter(client):
    resp = client.get("/api/enhanced/commands/autocomplete?query=/graph")
    assert resp.status_code == 200
    data = resp.json()
    assert "suggestions" in data
    assert all(suggestion.startswith("/graph") for suggestion in data["suggestions"])
