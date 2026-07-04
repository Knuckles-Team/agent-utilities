"""Tests for the Python-tier gateway Prometheus metrics (CONCEPT:AU-OS.observability.no-op-without-metrics).

Covers: the no-op fallback contract (prometheus_client is an optional
extra), exposition rendering, the route-TEMPLATE label discipline (bounded
cardinality), the pure-ASGI middleware recording (count/duration/in-flight,
including the exception path), and the /metrics endpoint + register wiring.
"""

from __future__ import annotations

import pytest

from agent_utilities.observability import gateway_metrics as gm
from agent_utilities.observability.gateway_metrics import (
    GatewayMetricsMiddleware,
    _NoopMetric,
    _route_template,
    metrics_endpoint,
    render_metrics,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class RecordingMetric:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict, float | None]] = []
        self._labels: dict = {}

    def labels(self, **kwargs):
        clone = RecordingMetric()
        clone.events = self.events
        clone._labels = kwargs
        return clone

    def inc(self, amount: float = 1.0):
        self.events.append(("inc", self._labels, amount))

    def dec(self, amount: float = 1.0):
        self.events.append(("dec", self._labels, amount))

    def observe(self, value: float):
        self.events.append(("observe", self._labels, value))

    def set(self, value: float):
        self.events.append(("set", self._labels, value))


@pytest.fixture
def fakes(monkeypatch):
    out = {
        "requests": RecordingMetric(),
        "duration": RecordingMetric(),
        "in_flight": RecordingMetric(),
    }
    monkeypatch.setattr(gm, "GATEWAY_REQUESTS", out["requests"])
    monkeypatch.setattr(gm, "GATEWAY_REQUEST_DURATION", out["duration"])
    monkeypatch.setattr(gm, "GATEWAY_IN_FLIGHT", out["in_flight"])
    return out


def _scope(path="/api/graph/query", method="GET"):
    return {"type": "http", "method": method, "path": path, "headers": []}


async def _call(mw, scope):
    sent: list[dict] = []

    async def send(msg):
        sent.append(msg)

    async def receive():
        return {"type": "http.request"}

    await mw(scope, receive, send)
    return sent


def _make_app(status=200, route_template=None, raises=None):
    async def app(scope, receive, send):  # noqa: ARG001
        if route_template is not None:
            scope["route"] = type("R", (), {"path_format": route_template})()
        if raises is not None:
            raise raises
        await send({"type": "http.response.start", "status": status, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    return app


# ---------------------------------------------------------------------------
# No-op fallback + rendering
# ---------------------------------------------------------------------------


class TestFallback:
    def test_noop_metric_chains(self):
        m = _NoopMetric()
        assert m.labels(route="/x", method="GET") is m
        m.inc()
        m.dec()
        m.observe(1.5)
        m.set(2.0)  # nothing raises

    def test_render_metrics_without_prometheus(self, monkeypatch):
        monkeypatch.setattr(gm, "PROMETHEUS_AVAILABLE", False)
        body, content_type = render_metrics()
        assert b"metrics unavailable" in body
        assert content_type.startswith("text/plain")

    async def test_metrics_endpoint_returns_response(self):
        response = await metrics_endpoint()
        assert response.status_code == 200
        assert response.body


# ---------------------------------------------------------------------------
# Route-template label (bounded cardinality)
# ---------------------------------------------------------------------------


class TestRouteTemplate:
    def test_uses_fastapi_route_path_format(self):
        route = type("R", (), {"path_format": "/api/things/{thing_id}"})()
        scope = {"type": "http", "path": "/api/things/12345", "route": route}
        assert _route_template(scope, 200) == "/api/things/{thing_id}"

    def test_404_collapses_to_unmatched(self):
        scope = {"type": "http", "path": "/wp-admin/setup.php"}
        assert _route_template(scope, 404) == "unmatched"
        assert _route_template(scope, 405) == "unmatched"

    def test_unrouted_non_404_buckets_by_first_segment(self):
        scope = {"type": "http", "path": "/assets/app/v1/main.js"}
        assert _route_template(scope, 200) == "/assets/*"
        assert _route_template({"type": "http", "path": "/"}, 200) == "/"


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class TestMetricsMiddleware:
    async def test_records_count_and_duration(self, fakes):
        mw = GatewayMetricsMiddleware(
            _make_app(status=200, route_template="/api/graph/{name}")
        )
        sent = await _call(mw, _scope(path="/api/graph/concepts", method="POST"))
        assert sent[0]["status"] == 200
        incs = [e for e in fakes["requests"].events if e[0] == "inc"]
        assert incs == [
            (
                "inc",
                {"route": "/api/graph/{name}", "method": "POST", "status": "200"},
                1.0,
            )
        ]
        observes = [e for e in fakes["duration"].events if e[0] == "observe"]
        assert len(observes) == 1
        assert observes[0][1] == {"route": "/api/graph/{name}"}
        assert observes[0][2] >= 0

    async def test_in_flight_inc_then_dec(self, fakes):
        mw = GatewayMetricsMiddleware(_make_app())
        await _call(mw, _scope())
        kinds = [e[0] for e in fakes["in_flight"].events]
        assert kinds == ["inc", "dec"]

    async def test_exception_recorded_as_500_and_propagates(self, fakes):
        mw = GatewayMetricsMiddleware(_make_app(raises=RuntimeError("boom")))
        with pytest.raises(RuntimeError, match="boom"):
            await _call(mw, _scope())
        incs = [e for e in fakes["requests"].events if e[0] == "inc"]
        assert len(incs) == 1
        assert incs[0][1]["status"] == "500"
        # the in-flight gauge is still balanced
        kinds = [e[0] for e in fakes["in_flight"].events]
        assert kinds == ["inc", "dec"]

    async def test_metrics_path_not_instrumented(self, fakes):
        mw = GatewayMetricsMiddleware(_make_app())
        await _call(mw, _scope(path="/metrics"))
        assert fakes["requests"].events == []
        assert fakes["in_flight"].events == []

    async def test_non_http_scope_passes_through(self, fakes):
        called = {}

        async def ws_app(scope, receive, send):  # noqa: ARG001
            called["yes"] = True

        mw = GatewayMetricsMiddleware(ws_app)
        await mw({"type": "websocket", "path": "/ws"}, None, None)
        assert called.get("yes")
        assert fakes["requests"].events == []


# ---------------------------------------------------------------------------
# Gateway wiring (register_graph_routes mounts middleware + /metrics)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestGatewayWiring:
    def test_register_graph_routes_mounts_metrics_and_limiter(self, monkeypatch):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from agent_utilities.core.config import config
        from agent_utilities.gateway.graph_api import register_graph_routes

        monkeypatch.setattr(config, "gateway_metrics", True)
        monkeypatch.setattr(config, "gateway_rate_limit", 1.0)
        monkeypatch.setattr(config, "gateway_rate_burst", 2.0)

        app = FastAPI()
        register_graph_routes(app, prefix="/api")
        client = TestClient(app)

        # /metrics is mounted, returns exposition (or the no-op placeholder)
        # and is NEVER rate limited
        for _ in range(5):
            response = client.get("/metrics")
            assert response.status_code == 200

        # the limiter kicks in after the burst on normal routes (keyed by
        # client IP for unauthenticated calls) with a Retry-After header
        statuses = [client.get("/api/unknown-route-xyz").status_code for _ in range(4)]
        assert statuses[:2] == [404, 404]
        assert 429 in statuses[2:]
        limited = client.get("/api/unknown-route-xyz")
        assert limited.status_code == 429
        assert "retry-after" in limited.headers
        assert limited.json()["error"] == "rate limit exceeded"
