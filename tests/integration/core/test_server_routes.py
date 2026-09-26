from __future__ import annotations

"""CONCEPT:AU-ECO.messaging.native-backend-abstraction"""

"""Integration tests for the agent-utilities FastAPI server routes.

These tests are the pytest migration of the ad-hoc Phase 1 smoke tests
(``p1_smoke_test.py`` and ``p1_smoke_test_webui.py``) plus the Phase 6
Scenario 5 ("cross-module imports") check.

They boot ``build_agent_app`` in-process, probe its headless HTTP surface
using ``TestClient``, and assert that the former WebUI enhanced routes are
absent from AU.

No external network, no subprocesses, no live LLM required — everything
runs against a fake "dummy-model" provider and a tempfile-backed Ladybug
graph database.
"""


import os
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.routing import Mount

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_app(workspace: Path, db_path: Path) -> FastAPI:
    """Construct a ``build_agent_app`` instance isolated to ``workspace``.

    All dummy LLM credentials and a tempfile-backed LadybugDB path are set
    on the process environment so the resulting app is fully self-contained.
    """
    os.environ["WORKSPACE_DIR"] = str(workspace)
    os.environ["AGENT_UTILITIES_DATA_DIR"] = str(db_path.parent / "agent-data")
    os.environ.setdefault("DEFAULT_PROVIDER", "openai")
    os.environ.setdefault("DEFAULT_MODEL_ID", "dummy-model")

    # Import lazily so test collection doesn't pull in the whole server stack.
    from agent_utilities.server import build_agent_app

    # Route coverage is independent from the durable A2A runtime. Networked
    # A2A startup correctly requires a configured process identity, so replace
    # only that mounted application with an inert in-process ASGI peer here.
    with patch(
        "agent_utilities.protocols.a2a_epistemic.agent_to_epistemic_a2a",
        return_value=FastAPI(),
    ):
        return build_agent_app(
            provider="openai",
            model_id="dummy-model",
            base_url=None,
            api_key="sk-test-not-real",
            mcp_url="",
            mcp_config=None,
            custom_skills_directory=None,
            debug=False,
            enable_acp=False,
            workspace=str(workspace),
        )


def _registered_route_paths(app: FastAPI) -> set[str]:
    """Collect concrete paths recursively without crossing the auth boundary."""

    paths: set[str] = set()

    def visit(routes: list[object], prefix: str = "") -> None:
        for route in routes:
            effective_contexts = getattr(route, "effective_route_contexts", None)
            if callable(effective_contexts):
                paths.update(str(context.path) for context in effective_contexts())
                continue
            route_path = str(getattr(route, "path", "") or "")
            full_path = f"{prefix.rstrip('/')}/{route_path.lstrip('/')}" or "/"
            if isinstance(route, Mount):
                nested = getattr(route.app, "routes", ())
                visit(list(nested), full_path)
            else:
                paths.add(full_path)

    visit(list(app.routes))
    return paths


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def app_headless(tmp_path_factory: pytest.TempPathFactory) -> FastAPI:
    """A headless ``build_agent_app`` instance."""
    ws = tmp_path_factory.mktemp("server_routes_headless_ws")
    db = tmp_path_factory.mktemp("server_routes_headless_db") / "kg.db"
    return _build_app(ws, db)


@pytest.fixture(scope="module")
def client_headless(app_headless: FastAPI) -> Iterator[TestClient]:
    """TestClient bound to the headless app."""
    with TestClient(app_headless, raise_server_exceptions=False) as client:
        yield client


# ---------------------------------------------------------------------------
# Core routes
# ---------------------------------------------------------------------------


def test_health(client_headless: TestClient) -> None:
    """``/health`` is dependency-free and non-fingerprinting."""
    resp = client_headless.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
    assert resp.headers["cache-control"] == "no-store"


def test_health_ready_reflects_the_same_report_in_its_status_code(
    client_headless: TestClient,
) -> None:
    """``/health/ready`` exposes only the readiness decision."""
    resp = client_headless.get("/health/ready")
    body = resp.json()
    assert body["status"] in {"ready", "not_ready"}
    assert set(body) == {"status"}
    assert resp.status_code == (200 if body["status"] == "ready" else 503)


def test_rest_health_routes_share_the_reserved_async_collector() -> None:
    """Top-level liveness/readiness and the dashboard route all invoke the
    owning async health seam, so no REST consumer can drift back to the shared
    default executor.
    """
    report = {
        "status": "healthy",
        "checks": [{"name": "engine", "status": "ok", "latency_ms": 0.0}],
        "generated_at": "synthetic",
    }
    from agent_utilities.gateway.api import dashboard_router
    from agent_utilities.server.routers.core import router as core_router

    app = FastAPI()
    app.include_router(core_router)
    app.include_router(dashboard_router, prefix="/api/dashboard")

    with patch(
        "agent_utilities.observability.runtime_health.collect_health_async",
        new_callable=AsyncMock,
        return_value=report,
    ) as collector:
        with TestClient(app) as client:
            liveness = client.get("/health")
            readiness = client.get("/health/ready")
            dashboard = client.get("/api/dashboard/health")

    assert liveness.json() == {"status": "ok"}
    assert readiness.json() == {"status": "ready"}
    assert dashboard.json() == report
    assert collector.await_count == 2


@pytest.mark.parametrize(
    "path",
    ["/api/chat", "/api/configure"],
)
def test_current_pydantic_web_routes_are_registered_but_not_anonymous(
    app_headless: FastAPI,
    client_headless: TestClient,
    path: str,
) -> None:
    """Current Pydantic AI web routes exist behind verified identity."""

    assert path in _registered_route_paths(app_headless)
    resp = client_headless.get(path)
    assert resp.status_code == 401
    assert resp.json() == {"error": "Verified Bearer identity required"}


def test_a2a_mount_present(app_headless: FastAPI) -> None:
    """The ``/a2a`` ``Mount`` route is always registered on the app."""
    mounts = [
        r
        for r in app_headless.routes
        if isinstance(r, Mount) and getattr(r, "path", None) == "/a2a"
    ]
    assert len(mounts) == 1, (
        "Expected exactly one /a2a Mount; got "
        f"{[getattr(m, 'path', '?') for m in mounts]}"
    )


def test_acp_available_when_acp_installed() -> None:
    """``is_acp_available()`` follows the first-party Harness ACP extra."""
    pytest.importorskip("pydantic_ai_harness.experimental.acp")
    from agent_utilities.protocols.acp_adapter import is_acp_available

    assert is_acp_available() is True


def test_acp_is_not_a_fake_http_mount(app_headless: FastAPI) -> None:
    """Agent Client Protocol is stdio JSON-RPC, not a Starlette mount."""
    mounts = [
        route
        for route in app_headless.routes
        if isinstance(route, Mount) and getattr(route, "path", None) == "/acp"
    ]
    assert mounts == []


# ---------------------------------------------------------------------------
# WebUI enhanced routes are absent from AU
# ---------------------------------------------------------------------------

ENHANCED_ROUTES: list[str] = [
    "/api/enhanced/tools",
    "/api/enhanced/chats",
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
def test_enhanced_routes_absent_from_au(
    app_headless: FastAPI,
    client_headless: TestClient,
    path: str,
) -> None:
    """Absent routes remain non-fingerprinting behind the same auth boundary."""

    assert path not in _registered_route_paths(app_headless)
    resp = client_headless.get(path)
    assert resp.status_code == 401, (
        f"Expected 401 at {path} in headless AU, got {resp.status_code}: "
        f"{resp.text[:200]}"
    )
    assert resp.json() == {"error": "Verified Bearer identity required"}


# ---------------------------------------------------------------------------
# Deep imports (Phase 6 S5 cross-module integrity)
# ---------------------------------------------------------------------------


def test_deep_imports() -> None:
    """All key public symbols must import cleanly.

    Mirrors the Phase 6 Scenario 5 import-integrity smoke test.
    """
    from agent_utilities import create_agent_server
    from agent_utilities.knowledge_graph.backends import create_backend
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
    from agent_utilities.knowledge_graph.core.maintainer import GraphMaintainer
    from agent_utilities.knowledge_graph.kb.ingestion import KBIngestionEngine
    from agent_utilities.knowledge_graph.pipeline.runner import PipelineRunner
    from agent_utilities.protocols.a2a import A2AClient, register_a2a_peer
    from agent_utilities.protocols.acp_adapter import (
        create_acp_agent,
        create_graph_acp_agent,
        is_acp_available,
        run_acp_agent_sync,
    )
    from agent_utilities.sdd import SDDManager

    # Every symbol must be a real object (not a placeholder/None). We don't need to
    # call them — the mere successful import is the integrity check.
    imported = [
        create_agent_server,
        create_agent_server,
        A2AClient,
        register_a2a_peer,
        create_acp_agent,
        create_graph_acp_agent,
        run_acp_agent_sync,
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
