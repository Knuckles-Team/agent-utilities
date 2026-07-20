"""CONCEPT:AU-OS.scaling.bridge-developer-workspace-mutating / ORCH-1.46 — /api/runtime/* HTTP surface: session -> act -> status -> delete."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_utilities.server.routers import runtime as runtime_router


@pytest.fixture()
def client(monkeypatch):
    from agent_utilities.core.config import config
    from agent_utilities.runtime import LocalWorkspace

    class _IsolatedTestWorkspace(LocalWorkspace):
        name = "docker"

        def __init__(self, *, run_id, image, network):
            del run_id, image, network
            super().__init__()

    monkeypatch.setattr(runtime_router, "DockerWorkspace", _IsolatedTestWorkspace)
    monkeypatch.setattr(config, "runtime_workspace_images", ["test-image"])
    app = FastAPI()
    app.include_router(runtime_router.router)
    runtime_router._SESSIONS.clear()
    return TestClient(app)


def test_session_lifecycle_and_act(client):
    # isolated backend contract, using an in-process fake in this unit test
    resp = client.post(
        "/api/runtime/sessions",
        json={"prefer_docker": True, "image": "test-image"},
    )
    assert resp.status_code == 200
    sid = resp.json()["session_id"]
    assert resp.json()["backend"] == "docker"

    # a shell action
    act = client.post(
        f"/api/runtime/sessions/{sid}/act",
        json={"kind": "cmd_run", "command": "echo wired"},
    )
    assert act.status_code == 200
    assert act.json()["kind"] == "cmd_output"
    assert "wired" in act.json()["stdout"]

    # a file write then read
    client.post(
        f"/api/runtime/sessions/{sid}/act",
        json={"kind": "file_write", "path": "a.txt", "content": "hello"},
    )
    read = client.post(
        f"/api/runtime/sessions/{sid}/act",
        json={"kind": "file_read", "path": "a.txt"},
    )
    assert read.json()["content"] == "hello"

    # status reflects recorded events
    st = client.get(f"/api/runtime/sessions/{sid}")
    assert st.json()["steps"] == 3

    # teardown
    assert client.delete(f"/api/runtime/sessions/{sid}").json()["stopped"] is True
    assert client.get(f"/api/runtime/sessions/{sid}").status_code == 404


def test_invalid_action_is_422(client):
    sid = client.post(
        "/api/runtime/sessions",
        json={"prefer_docker": True, "image": "test-image"},
    ).json()["session_id"]
    bad = client.post(f"/api/runtime/sessions/{sid}/act", json={"kind": "nonsense"})
    assert bad.status_code == 422
    client.delete(f"/api/runtime/sessions/{sid}")


def test_act_on_unknown_session_is_404(client):
    resp = client.post(
        "/api/runtime/sessions/doesnotexist/act",
        json={"kind": "cmd_run", "command": "true"},
    )
    assert resp.status_code == 404


def test_provenance_panel_data(client, monkeypatch):
    """CONCEPT:AU-OS.scaling.kg-provenance-panel-data — the webui provenance panel reads action + mutated-symbol rows."""

    class _Backend:
        def execute(self, q, params=None):
            if "MUTATED" in q:
                return [{"action_id": "wsaction:r:2", "symbol_id": "Code:m.py::foo"}]
            return [
                {
                    "id": "wsaction:r:1",
                    "kind": "file_edit",
                    "step": 1,
                    "payload_ref": "pref_workspace_payload_" + "0" * 64,
                    "obs_kind": "file_edit_ok",
                    "obs_payload_ref": "pref_workspace_payload_" + "1" * 64,
                }
            ]

    class _Engine:
        backend = _Backend()

    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    monkeypatch.setattr(
        IntelligenceGraphEngine, "get_active", classmethod(lambda cls: _Engine())
    )
    sid = client.post(
        "/api/runtime/sessions",
        json={"prefer_docker": True, "image": "test-image"},
    ).json()["session_id"]
    resp = client.get(f"/api/runtime/sessions/{sid}/provenance")
    assert resp.status_code == 200
    data = resp.json()
    assert data["actions"][0]["kind"] == "file_edit"
    assert data["mutated"][0]["symbol_id"] == "Code:m.py::foo"
    client.delete(f"/api/runtime/sessions/{sid}")
