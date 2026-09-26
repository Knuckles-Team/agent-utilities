"""AU's server no longer publishes GraphOS-owned gateway routes."""

from __future__ import annotations

from fastapi import FastAPI

from agent_utilities.server.app import _include_gateway_routers


def test_au_server_does_not_mount_retired_gateway_routes() -> None:
    app = FastAPI()
    _include_gateway_routers(app)
    paths = set(app.openapi()["paths"])

    assert "/api/dashboard/full" not in paths
    assert "/api/artifacts" not in paths
    assert "/api/observability/usage" not in paths
    assert "/api/graph/query" not in paths
    assert any(path.startswith("/api/") for path in paths)
