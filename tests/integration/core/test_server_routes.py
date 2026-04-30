"""Integration tests for the agent-utilities FastAPI server routes.

These tests are the pytest migration of the ad-hoc Phase 1 smoke tests
(``p1_smoke_test.py`` and ``p1_smoke_test_webui.py``) plus the Phase 6
Scenario 5 ("cross-module imports") check.

They boot ``build_agent_app`` in-process twice (with and without
``enable_web_ui=True``), probe the documented HTTP surface using
``TestClient``, and assert that the enhanced ``/api/enhanced/*`` router
is only mounted when the web UI is enabled.

No external network, no subprocesses, no live LLM required — everything
runs against a fake "dummy-model" provider and a tempfile-backed Ladybug
graph database.
"""
from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.routing import Mount

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_app(workspace: Path, db_path: Path, *, enable_web_ui: bool) -> FastAPI:
    """Construct a ``build_agent_app`` instance isolated to ``workspace``.

    All dummy LLM credentials and a tempfile-backed LadybugDB path are set
    on the process environment so the resulting app is fully self-contained.
    """
    os.environ["WORKSPACE_DIR"] = str(workspace)
    os.environ["GRAPH_DB_PATH"] = str(db_path)
    os.environ.setdefault("DEFAULT_PROVIDER", "openai")
    os.environ.setdefault("DEFAULT_MODEL_ID", "dummy-model")

    # Import lazily so test collection doesn't pull in the whole server stack.
    from agent_utilities.server import build_agent_app

    return build_agent_app(
        provider="openai",
        model_id="dummy-model",
        base_url=None,
        api_key="sk-test-not-real",
        mcp_url="",
        mcp_config=None,
        custom_skills_directory=None,
        debug=False,
        enable_web_ui=enable_web_ui,
        workspace=str(workspace),
        ssl_verify=False,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def app_with_web_ui(tmp_path_factory: pytest.TempPathFactory) -> FastAPI:
    """A ``build_agent_app`` instance with ``enable_web_ui=True``."""
    ws = tmp_path_factory.mktemp("server_routes_webui_ws")
    db = tmp_path_factory.mktemp("server_routes_webui_db") / "kg.db"
    return _build_app(ws, db, enable_web_ui=True)


@pytest.fixture(scope="module")
def app_no_web_ui(tmp_path_factory: pytest.TempPathFactory) -> FastAPI:
    """A ``build_agent_app`` instance with ``enable_web_ui=False``."""
    ws = tmp_path_factory.mktemp("server_routes_bare_ws")
    db = tmp_path_factory.mktemp("server_routes_bare_db") / "kg.db"
    return _build_app(ws, db, enable_web_ui=False)


@pytest.fixture(scope="module")
def client_with_web_ui(app_with_web_ui: FastAPI) -> Iterator[TestClient]:
    """TestClient bound to the web-UI-enabled app."""
    with TestClient(app_with_web_ui, raise_server_exceptions=False) as client:
        yield client


@pytest.fixture(scope="module")
def client_no_web_ui(app_no_web_ui: FastAPI) -> Iterator[TestClient]:
    """TestClient bound to the bare (no web UI) app."""
    with TestClient(app_no_web_ui, raise_server_exceptions=False) as client:
        yield client


# ---------------------------------------------------------------------------
# Core routes (always present, web UI or not)
# ---------------------------------------------------------------------------


def test_health(client_with_web_ui: TestClient) -> None:
    """``/health`` returns 200 with status=OK, agent, and version fields."""
    resp = client_with_web_ui.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "OK"
    assert "agent" in data
    assert "version" in data


def test_mcp_config(client_with_web_ui: TestClient) -> None:
    """``/mcp/config`` returns 200 with a JSON object containing ``mcpServers``."""
    resp = client_with_web_ui.get("/mcp/config")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    assert "mcpServers" in data


def test_mcp_tools(client_with_web_ui: TestClient) -> None:
    """``/mcp/tools`` returns 200 with a list body."""
    resp = client_with_web_ui.get("/mcp/tools")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_chats(client_with_web_ui: TestClient) -> None:
    """``/chats`` returns 200 with a list body."""
    resp = client_with_web_ui.get("/chats")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_a2a_mount_present(app_with_web_ui: FastAPI) -> None:
    """The ``/a2a`` ``Mount`` route is always registered on the app."""
    mounts = [
        r for r in app_with_web_ui.routes
        if isinstance(r, Mount) and getattr(r, "path", None) == "/a2a"
    ]
    assert len(mounts) == 1, (
        "Expected exactly one /a2a Mount; got "
        f"{[getattr(m, 'path', '?') for m in mounts]}"
    )


def test_acp_available_when_acp_installed() -> None:
    """``is_acp_available()`` returns True when ``pydantic-acp`` is installed."""
    from agent_utilities.protocols.acp_adapter import is_acp_available

    assert is_acp_available() is True


# ---------------------------------------------------------------------------
# Enhanced routes (Phase 1 finding: only mounted with enable_web_ui=True)
# ---------------------------------------------------------------------------

ENHANCED_ROUTES: list[str] = [
    "/api/enhanced/info",
    "/api/enhanced/graph/stats",
    "/api/enhanced/kb/list",
    "/api/enhanced/sdd/specs",
    "/api/enhanced/resources",
    "/api/enhanced/maintenance/status",
    "/api/enhanced/pipeline/status",
    "/api/enhanced/agents",
    "/api/enhanced/skills",
]


@pytest.mark.parametrize("path", ENHANCED_ROUTES)
def test_enhanced_routes_available_with_web_ui(
    client_with_web_ui: TestClient, path: str
) -> None:
    """Enhanced routes are mounted (200 OK) when ``enable_web_ui=True``."""
    resp = client_with_web_ui.get(path)
    assert resp.status_code == 200, (
        f"Expected 200 at {path} with web UI enabled, got {resp.status_code}: "
        f"{resp.text[:200]}"
    )


@pytest.mark.parametrize("path", ENHANCED_ROUTES)
def test_enhanced_routes_absent_without_web_ui(
    client_no_web_ui: TestClient, path: str
) -> None:
    """Enhanced routes 404 when ``enable_web_ui=False`` (Phase 1 finding)."""
    resp = client_no_web_ui.get(path)
    assert resp.status_code == 404, (
        f"Expected 404 at {path} without web UI, got {resp.status_code}: "
        f"{resp.text[:200]}"
    )


# ---------------------------------------------------------------------------
# Deep imports (Phase 6 S5 cross-module integrity)
# ---------------------------------------------------------------------------


def test_deep_imports() -> None:
    """All key public symbols must import cleanly.

    Mirrors the Phase 6 Scenario 5 import-integrity smoke test.
    """
    from agent_utilities import create_agent_server, create_graph_agent_server
    from agent_utilities.knowledge_graph.backends import create_backend
    from agent_utilities.knowledge_graph.engine import IntelligenceGraphEngine
    from agent_utilities.knowledge_graph.kb.ingestion import KBIngestionEngine
    from agent_utilities.knowledge_graph.maintainer import GraphMaintainer
    from agent_utilities.knowledge_graph.pipeline.runner import PipelineRunner
    from agent_utilities.protocols.a2a import A2AClient, register_a2a_peer
    from agent_utilities.protocols.acp_adapter import (
        create_acp_app,
        create_graph_acp_app,
        is_acp_available,
    )
    from agent_utilities.sdd import SDDManager

    # Every symbol must be a real object (not a stub/None). We don't need to
    # call them — the mere successful import is the integrity check.
    imported = [
        create_agent_server,
        create_graph_agent_server,
        A2AClient,
        register_a2a_peer,
        create_acp_app,
        create_graph_acp_app,
        is_acp_available,
        create_backend,
        IntelligenceGraphEngine,
        KBIngestionEngine,
        GraphMaintainer,
        PipelineRunner,
        SDDManager,
    ]
    assert all(obj is not None for obj in imported), (
        "One or more deep imports resolved to None"
    )
