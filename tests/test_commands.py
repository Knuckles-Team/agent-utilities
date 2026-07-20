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


# ---------------------------------------------------------------------------
# Real-output regression guards: the /graph, /kb, /sdd, /resources subcommands
# must never emit the old fabricated placeholder data. With no live engine in the
# test process they must return an HONEST "not active" message instead.
# ---------------------------------------------------------------------------

_FABRICATED_MARKERS = [
    "Online (LadybugDB)",
    "Unified Parallel Engine Scheduler",
    "Relevance: 95%",
    "Successfully initiated background KB ingestion",
    "workspace-docs",
    "mcp-servers-index",
    "All indexes updated",
    "Successfully spawned background agent subtask",
    "agent-research-01",
    "agent-tui-helper",
    "Zero-Trust Security Alignment",
]


@pytest.mark.parametrize(
    "command",
    [
        "/graph stats",
        "/graph search widget",
        "/graph impact some_symbol",
        "/kb list",
        "/kb search widget",
        "/kb ingest /tmp/example",
        "/sdd specs",
        "/sdd constitution",
        "/sdd sync",
        "/resources",
        "/resources spawn helper",
    ],
)
def test_subcommands_emit_no_fabricated_data(client, command):
    """Every de-stubbed subcommand returns real-or-honest output, never fakes."""
    resp = client.post("/api/enhanced/commands/execute", json={"command": command})
    assert resp.status_code == 200
    md = resp.json()["response_markdown"]
    for marker in _FABRICATED_MARKERS:
        assert marker not in md, f"{command!r} leaked fabricated marker {marker!r}"
    # Hardcoded 42/89 graph counts must never appear in graph stats output.
    if command == "/graph stats":
        assert "42" not in md and "89" not in md


def test_graph_stats_endpoint_no_fabricated_counts():
    """GET /api/enhanced/graph/stats never returns the old hardcoded 42/89."""
    from fastapi import FastAPI
    from starlette.testclient import TestClient as _TC

    from agent_utilities.server.routers.enhanced import router as enhanced_router

    app = FastAPI()
    app.include_router(enhanced_router)
    c = _TC(app)
    data = c.get("/api/enhanced/graph/stats").json()
    # Either real counts from a live backend, or an honest unavailable status —
    # but never the fabricated {nodes: 42, edges: 89}.
    assert not (data.get("nodes") == 42 and data.get("edges") == 89)
    assert data.get("status") in {"ok", "unavailable", "error"}
