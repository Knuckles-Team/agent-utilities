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


# ---------------------------------------------------------------------------
# CXA-AU-01-02 characterization: pin every remaining branch of
# execute_slash_command (CCN 126) BEFORE any decomposition. These assertions
# were confirmed to fail against a deliberately mutated implementation before
# being committed (two-commit discipline, see lane report).
# ---------------------------------------------------------------------------


def test_execute_no_leading_slash(client):
    """A command string without a leading '/' is rejected before any dispatch."""
    resp = client.post(
        "/api/enhanced/commands/execute", json={"command": "help"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["response_markdown"] == (
        "Error: Command must start with a slash `/`."
    )
    assert data["client_actions"] == []


def test_execute_empty_command(client):
    """An empty command string also fails the leading-slash check."""
    resp = client.post("/api/enhanced/commands/execute", json={"command": ""})
    assert resp.status_code == 200
    assert resp.json()["response_markdown"] == (
        "Error: Command must start with a slash `/`."
    )


def test_execute_model_no_args_reports_unknown_with_no_registry(client):
    """With no model_registry on app.state, current model reports as unknown."""
    resp = client.post("/api/enhanced/commands/execute", json={"command": "/model"})
    assert resp.status_code == 200
    data = resp.json()
    assert "Current active model: `unknown`." in data["response_markdown"]
    assert data["client_actions"] == []


def test_execute_model_with_args_emits_set_model_action(client):
    resp = client.post(
        "/api/enhanced/commands/execute", json={"command": "/model gpt-5"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "Switched model to `gpt-5`." in data["response_markdown"]
    assert data["client_actions"] == [{"action": "set_model", "value": "gpt-5"}]


def test_execute_tools_empty_registry(client):
    resp = client.post("/api/enhanced/commands/execute", json={"command": "/tools"})
    assert resp.status_code == 200
    assert (
        resp.json()["response_markdown"] == "No tools currently registered."
    )


def test_execute_skills_empty_registry(client):
    resp = client.post("/api/enhanced/commands/execute", json={"command": "/skills"})
    assert resp.status_code == 200
    assert (
        resp.json()["response_markdown"] == "No custom skills currently active."
    )


def test_execute_unknown_command_fallback(client):
    resp = client.post(
        "/api/enhanced/commands/execute", json={"command": "/unknownxyz"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["response_markdown"] == (
        "Unknown slash command: `/unknownxyz`. Type `/help` for a list of "
        "available commands."
    )
    assert data["client_actions"] == []


def test_execute_quit_aliases_to_exit_which_has_no_handler(client):
    """BUG (see BUGS FOUND in lane report): 'quit' is standardized to 'exit',
    but no branch of the dispatch ever handles cmd_name == 'exit', so both
    /quit and /exit fall through to the generic unknown-command response —
    and the unknown-command text names `exit`, not the command the caller
    typed. Pinned here exactly as observed so a future fix is a deliberate,
    reviewed change rather than an accidental behaviour drift.
    """
    resp_quit = client.post(
        "/api/enhanced/commands/execute", json={"command": "/quit"}
    )
    resp_exit = client.post(
        "/api/enhanced/commands/execute", json={"command": "/exit"}
    )
    assert resp_quit.status_code == 200
    assert resp_exit.status_code == 200
    expected = (
        "Unknown slash command: `/exit`. Type `/help` for a list of "
        "available commands."
    )
    assert resp_quit.json()["response_markdown"] == expected
    assert resp_exit.json()["response_markdown"] == expected


def test_execute_graph_nodes_falls_through_to_stats(client):
    """BUG (see BUGS FOUND in lane report): /help advertises
    `/graph nodes [type]` as a distinct subcommand, but the `/graph` dispatch
    only special-cases `search` and `impact`; anything else (including
    `nodes` and any `[type]` argument) silently falls through to the
    default `stats` branch. Pinned exactly as observed.
    """
    resp = client.post(
        "/api/enhanced/commands/execute", json={"command": "/graph nodes"}
    )
    resp_typed = client.post(
        "/api/enhanced/commands/execute",
        json={"command": "/graph nodes SomeType"},
    )
    assert resp.status_code == 200
    assert resp_typed.status_code == 200
    # Both collapse to the exact same generic stats response — the "SomeType"
    # filter argument is silently discarded.
    assert resp.json()["response_markdown"] == resp_typed.json()["response_markdown"]
    assert "Knowledge Graph Statistics" in resp.json()["response_markdown"]


def test_execute_cron_default_and_subcommands_return_operation_envelope(client):
    """With no configured scheduler control authority (the unit-test
    environment), /cron and its subcommands all surface the same structured
    operation-failed envelope via public_error_text rather than raising.
    """
    for command in ("/cron", "/cron calendar", "/cron logs"):
        resp = client.post(
            "/api/enhanced/commands/execute", json={"command": command}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["client_actions"] == []
        assert '"status": "failed"' in data["response_markdown"]
        assert '"error_class": "RuntimeError"' in data["response_markdown"]


def test_execute_cron_unknown_subcommand(client):
    """An unrecognized /cron subcommand still needs the calendar() call to
    succeed first (it is fetched unconditionally before subcommand dispatch);
    in this environment that call fails closed, so the response is the same
    operation-failed envelope rather than the 'Unknown `/cron` subcommand'
    text — pinned so a future change to that ordering is visible.
    """
    resp = client.post(
        "/api/enhanced/commands/execute", json={"command": "/cron bogus"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert '"status": "failed"' in data["response_markdown"]


def test_execute_kb_unknown_subcommand(client):
    resp = client.post(
        "/api/enhanced/commands/execute", json={"command": "/kb bogus"}
    )
    assert resp.status_code == 200
    assert (
        resp.json()["response_markdown"] == "Unknown `/kb` subcommand: `bogus`"
    )


def test_execute_sdd_unknown_subcommand(client):
    resp = client.post(
        "/api/enhanced/commands/execute", json={"command": "/sdd bogus"}
    )
    assert resp.status_code == 200
    assert (
        resp.json()["response_markdown"] == "Unknown `/sdd` subcommand: `bogus`"
    )


def test_execute_resources_unknown_subcommand(client):
    resp = client.post(
        "/api/enhanced/commands/execute", json={"command": "/resources bogus"}
    )
    assert resp.status_code == 200
    assert (
        resp.json()["response_markdown"]
        == "Unknown `/resources` subcommand: `bogus`"
    )


def test_execute_resources_spawn_no_name(client):
    resp = client.post(
        "/api/enhanced/commands/execute", json={"command": "/resources spawn"}
    )
    assert resp.status_code == 200
    assert (
        resp.json()["response_markdown"] == "Usage: `/resources spawn <name>`"
    )


def test_autocomplete_no_match(client):
    resp = client.get("/api/enhanced/commands/autocomplete?query=/zzz")
    assert resp.status_code == 200
    assert resp.json()["suggestions"] == []
