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
    render_metrics,
    render_metrics_with_engine,
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
        # CONCEPT:AU-OS.observability.no-op-without-metrics — `.labels(...)`
        # returns a per-labelset child (cached on repeat calls with the same
        # labels), not the parent itself: callers assert on recorded state
        # (test_workitem_claim_metrics.py) so distinct label combinations must
        # track independently, exactly like a real prometheus_client series.
        m = _NoopMetric()
        child = m.labels(route="/x", method="GET")
        assert m.labels(route="/x", method="GET") is child
        assert m.labels(route="/y", method="GET") is not child
        child.inc()
        child.dec()
        child.observe(1.5)
        child.set(2.0)  # nothing raises
        assert child._value.get() == 2.0

    def test_render_metrics_without_prometheus(self, monkeypatch):
        monkeypatch.setattr(gm, "PROMETHEUS_AVAILABLE", False)
        body, content_type = render_metrics()
        assert b"metrics unavailable" in body
        assert content_type.startswith("text/plain")

    async def test_render_metrics_with_engine_returns_body(self):
        body, content_type = await render_metrics_with_engine()
        assert body
        assert content_type

    async def test_render_metrics_with_engine_unreachable_signals_down(
        self, monkeypatch
    ):
        # Force the fetch to fail regardless of what happens to be listening
        # on 127.0.0.1:9101 in the ambient environment (dev hosts run
        # node-exporter on that same port outside the pod netns, so a "just
        # don't run anything" test would be environment-dependent). The
        # gateway's own metrics must still be served in full, plus an
        # explicit down signal — never a silently partial scrape.
        import httpx

        async def raise_connect_error(self, *args, **kwargs):
            raise httpx.ConnectError("connection refused (test)")

        monkeypatch.setattr(httpx.AsyncClient, "get", raise_connect_error)
        body, _content_type = await render_metrics_with_engine()
        assert b"graph_os_engine_metrics_up 0" in body
        assert b"engine metrics unavailable" in body

    async def test_render_metrics_with_engine_reachable_merges_series(
        self, monkeypatch
    ):
        async def fake_fetch():
            return b"epistemic_graph_fake_metric 42\n", True

        monkeypatch.setattr(gm, "_fetch_engine_metrics", fake_fetch)
        body, _content_type = await render_metrics_with_engine()
        assert b"graph_os_engine_metrics_up 1" in body
        assert b"epistemic_graph_fake_metric 42" in body


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
