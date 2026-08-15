"""Wiring tests for the graph-os metrics + OTLP trace surface.

These are deliberately NOT existence tests. Per this repo's wiring-test
standard, proving that a function or a route object exists proves nothing about
whether the system is observable. Every test here either

* drives a **real ASGI request** through a **real factory-built** MCP server and
  reads the response body, or
* runs the **real underlying operation** and asserts a metric's value in the
  live ``prometheus_client`` registry actually **moved**.

Covers:
  * ``GET /metrics`` reachability on a factory-built server, and the intact
    security gate that keeps it OFF an unauthenticated network listener
    (CONCEPT:AU-OS.observability.no-op-without-metrics).
  * Delegation run metrics moving when ``Orchestrator.execute_agent`` /
    ``execute_workflow`` actually run (CONCEPT:AU-OS.observability.delegation-run-metrics).
  * The running-vs-dispatchable child metrics, including that a scrape samples
    them (CONCEPT:AU-ECO.multiplexer.running-vs-dispatchable-metrics).
  * The ``PROMETHEUS_MULTIPROC_DIR`` guard (CONCEPT:AU-OS.observability.multiprocess-registry-guard).
  * The OTLP trace-signal endpoint override and loud-but-soft export failure
    (CONCEPT:AU-OS.observability.otlp-trace-fanout).
"""

from __future__ import annotations

import logging

import pytest

from agent_utilities.observability import gateway_metrics as gm

prometheus_client = pytest.importorskip(
    "prometheus_client", reason="metrics extra not installed"
)


# ---------------------------------------------------------------------------
# Helpers — read real values back out of the live registry
# ---------------------------------------------------------------------------


def _sample(name: str, labels: dict[str, str] | None = None) -> float:
    """Current value of one series in the default registry (0.0 if absent)."""
    value = prometheus_client.REGISTRY.get_sample_value(name, labels or {})
    return float(value or 0.0)


async def _asgi_get(app, path: str, headers: list[tuple[bytes, bytes]] | None = None):
    """Drive one real ASGI GET and return ``(status, headers, body)``."""
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": headers or [],
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 8000),
    }
    messages: list[dict] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    await app(scope, receive, send)

    start = next(m for m in messages if m["type"] == "http.response.start")
    body = b"".join(
        m.get("body", b"") for m in messages if m["type"] == "http.response.body"
    )
    return start["status"], dict(start.get("headers", [])), body


#: A REAL network-listener argv, satisfying the factory's fail-closed exposure
#: guard (`_validate_network_exposure`) exactly the way the live deployment does:
#: authenticated, behind a TLS-terminating ingress restricted to known peers.
#: Without these the factory refuses to build at all, so a metrics test that
#: skipped them would not be exercising a realistic remote server.
_STATIC_TOKENS_ENV = "WIRING_TEST_STATIC_TOKENS"
_NETWORK_ARGV = [
    "test-mcp",
    "--transport",
    "streamable-http",
    "--host",
    "0.0.0.0",
    "--auth-type",
    "static",
    "--static-tokens-ref",
    f"env://{_STATIC_TOKENS_ENV}",
    "--tls-terminated",
    "--trusted-proxy-cidrs",
    "127.0.0.0/8",
]


def _build_server(monkeypatch, argv: list[str], *, metrics_ref: str = ""):
    """Build a REAL factory server (no mocks) with a controlled CLI/env."""
    import json

    from agent_utilities.mcp import server_factory

    monkeypatch.setenv(
        _STATIC_TOKENS_ENV,
        json.dumps({"c" * 40: {"client_id": "wiring-test", "scopes": []}}),
    )
    monkeypatch.setenv("MCP_ALLOWED_HOSTS", "graph-os.test,127.0.0.1")
    monkeypatch.setattr("sys.argv", argv)
    real_setting = server_factory.setting

    def _setting(key, default=None):
        if key == "MCP_METRICS_TOKEN_REF":
            return metrics_ref
        return real_setting(key, default)

    monkeypatch.setattr(server_factory, "setting", _setting)
    _args, mcp, _mw = server_factory.create_mcp_server(
        name="wiring-test", version="0.0.0", instructions="t"
    )
    return mcp


def _orchestrator():
    """A real ``Orchestrator`` with only the collaborators these paths touch."""
    from agent_utilities.orchestration import manager as manager_mod

    class _NoThreat:
        is_malicious = False
        explanation = ""

    class _PassthroughScanner:
        def scan_text(self, _task):
            return _NoThreat()

    orchestrator = manager_mod.Orchestrator.__new__(manager_mod.Orchestrator)
    orchestrator.engine = None
    orchestrator.scanner = _PassthroughScanner()
    return orchestrator


# ---------------------------------------------------------------------------
# 1. GET /metrics is reachable through a real ASGI request
# ---------------------------------------------------------------------------


class TestMetricsRouteReachable:
    @pytest.mark.asyncio
    async def test_loopback_listener_serves_real_exposition_body(self, monkeypatch):
        """A real ASGI GET /metrics returns 200 with real exposition text.

        This is the wiring assertion the route-table check could not make: the
        handler runs, ``render_metrics`` executes, and the body carries a metric
        family this package actually defines.
        """
        gm.MCP_TOOL_CALLS.labels(tool="wiring_probe", outcome="ok").inc()

        mcp = _build_server(
            monkeypatch,
            ["test-mcp", "--transport", "streamable-http", "--host", "127.0.0.1"],
        )
        status, headers, body = await _asgi_get(mcp.http_app(), "/metrics")

        assert status == 200
        assert b"text/plain" in headers[b"content-type"]
        text = body.decode()
        assert "agent_utilities_mcp_tool_calls_total" in text
        assert 'tool="wiring_probe"' in text

    @pytest.mark.asyncio
    async def test_network_listener_without_token_does_not_expose_metrics(
        self, monkeypatch
    ):
        """The security gate stays intact: no token ⇒ no route on 0.0.0.0.

        Regression guard. Exposing an unauthenticated /metrics on a
        network-reachable listener is a deliberate NON-goal; the supported fix
        is to provision ``MCP_METRICS_TOKEN_REF``.
        """
        mcp = _build_server(monkeypatch, _NETWORK_ARGV)
        status, _headers, _body = await _asgi_get(mcp.http_app(), "/metrics")
        assert status == 404

    @pytest.mark.asyncio
    async def test_network_listener_with_token_requires_and_accepts_bearer(
        self, monkeypatch
    ):
        """With the token provisioned the route mounts and enforces the bearer.

        This is the exact configuration the deployment now runs, exercised end
        to end: wrong/absent credential ⇒ 401, correct bearer ⇒ real exposition.
        """
        token = "z" * 48
        monkeypatch.setenv("WIRING_TEST_METRICS_TOKEN", token)
        mcp = _build_server(
            monkeypatch,
            _NETWORK_ARGV,
            metrics_ref="env://WIRING_TEST_METRICS_TOKEN",
        )
        app = mcp.http_app()

        status, _h, _b = await _asgi_get(app, "/metrics")
        assert status == 401

        status, _h, _b = await _asgi_get(
            app, "/metrics", headers=[(b"authorization", b"Bearer wrong-token")]
        )
        assert status == 401

        status, _h, body = await _asgi_get(
            app,
            "/metrics",
            headers=[(b"authorization", f"Bearer {token}".encode())],
        )
        assert status == 200
        assert b"agent_utilities_" in body


# ---------------------------------------------------------------------------
# 2. Delegation metrics move when a delegation actually runs
# ---------------------------------------------------------------------------


class TestDelegationMetricsLivePath:
    @pytest.mark.asyncio
    async def test_execute_agent_moves_the_counter_and_histogram(self, monkeypatch):
        """Call the REAL ``Orchestrator.execute_agent``; assert the value moved."""
        from agent_utilities.orchestration import manager as manager_mod

        async def _fake_run_agent(**kwargs):
            return "done"

        monkeypatch.setattr(manager_mod, "run_agent", _fake_run_agent)

        labels = {"kind": "agent", "target": "wiring-agent", "outcome": "ok"}
        before = _sample("agent_utilities_delegation_runs_total", labels)
        before_hist = _sample(
            "agent_utilities_delegation_duration_seconds_count",
            {"kind": "agent", "target": "wiring-agent"},
        )

        result = await manager_mod.Orchestrator.execute_agent(
            _orchestrator(), agent_name="wiring-agent", task="probe"
        )

        assert result == "done"
        assert _sample("agent_utilities_delegation_runs_total", labels) == before + 1
        assert (
            _sample(
                "agent_utilities_delegation_duration_seconds_count",
                {"kind": "agent", "target": "wiring-agent"},
            )
            == before_hist + 1
        )

    @pytest.mark.asyncio
    async def test_failed_delegation_is_counted_as_error_and_still_raises(
        self, monkeypatch
    ):
        """A failing delegation records outcome=error and re-raises unchanged."""
        from agent_utilities.orchestration import manager as manager_mod

        async def _boom(**kwargs):
            raise RuntimeError("delegation blew up")

        monkeypatch.setattr(manager_mod, "run_agent", _boom)

        labels = {"kind": "agent", "target": "failing-agent", "outcome": "error"}
        before = _sample("agent_utilities_delegation_runs_total", labels)

        with pytest.raises(RuntimeError, match="delegation blew up"):
            await manager_mod.Orchestrator.execute_agent(
                _orchestrator(), agent_name="failing-agent", task="probe"
            )

        assert _sample("agent_utilities_delegation_runs_total", labels) == before + 1

    @pytest.mark.asyncio
    async def test_cancellation_is_not_counted_as_an_error(self):
        """A cancelled run is ``cancelled``, never ``error`` — it is not a failure."""
        import asyncio

        labels = {"kind": "agent", "target": "cancelled-agent", "outcome": "cancelled"}
        before = _sample("agent_utilities_delegation_runs_total", labels)

        with pytest.raises(asyncio.CancelledError):  # noqa: PT012 - span must wrap raise
            with gm.delegation_span("agent", "cancelled-agent"):
                raise asyncio.CancelledError()

        assert _sample("agent_utilities_delegation_runs_total", labels) == before + 1

    def test_in_flight_gauge_returns_to_zero(self):
        """The in-flight gauge is symmetric even on the failure path."""
        before = _sample("agent_utilities_delegation_in_flight", {"kind": "workflow"})
        with pytest.raises(ValueError):  # noqa: PT012 - span must wrap raise
            with gm.delegation_span("workflow", "wf"):
                raise ValueError("x")
        assert (
            _sample("agent_utilities_delegation_in_flight", {"kind": "workflow"})
            == before
        )


# ---------------------------------------------------------------------------
# 3. Running vs dispatchable — the distinction must survive into the metrics
# ---------------------------------------------------------------------------


class TestRunningVsDispatchable:
    def test_running_child_with_open_breaker_is_running_but_not_dispatchable(self):
        """The whole point: process-up must not imply tools-callable."""
        gm.publish_multiplexer_child_gauges(
            {
                "children": {
                    "broken-mcp": {
                        "server": "broken-mcp",
                        "state": "up",
                        "breaker": "open",
                        "mounted_tools": 7,
                    }
                }
            }
        )
        assert (
            _sample(
                "agent_utilities_mcp_child_process_running", {"server": "broken-mcp"}
            )
            == 1
        )
        assert (
            _sample("agent_utilities_mcp_child_tools_mounted", {"server": "broken-mcp"})
            == 7
        )
        assert (
            _sample(
                "agent_utilities_mcp_child_tools_dispatchable",
                {"server": "broken-mcp"},
            )
            == 0
        )
        assert _sample("agent_utilities_mcp_servers_running") == 1
        assert _sample("agent_utilities_mcp_servers_dispatchable") == 0

    def test_mounted_but_down_child_is_neither_running_nor_dispatchable(self):
        gm.publish_multiplexer_child_gauges(
            {
                "children": {
                    "restarting-mcp": {
                        "state": "restarting",
                        "breaker": "closed",
                        "mounted_tools": 3,
                    }
                }
            }
        )
        assert (
            _sample(
                "agent_utilities_mcp_child_process_running",
                {"server": "restarting-mcp"},
            )
            == 0
        )
        assert (
            _sample(
                "agent_utilities_mcp_child_tools_dispatchable",
                {"server": "restarting-mcp"},
            )
            == 0
        )

    def test_healthy_child_is_running_and_dispatchable(self):
        gm.publish_multiplexer_child_gauges(
            {
                "children": {
                    "good-mcp": {
                        "state": "up",
                        "breaker": "closed",
                        "mounted_tools": 5,
                    }
                }
            }
        )
        assert (
            _sample("agent_utilities_mcp_child_process_running", {"server": "good-mcp"})
            == 1
        )
        assert (
            _sample(
                "agent_utilities_mcp_child_tools_dispatchable", {"server": "good-mcp"}
            )
            == 5
        )
        assert _sample("agent_utilities_mcp_servers_dispatchable") == 1

    def test_status_snapshot_publishes_the_gauges_and_a_scrape_samples_them(self):
        """Live path: the real ``status_snapshot`` feeds the real ``/metrics`` body.

        Proves the two halves of the wiring at once — that the snapshot the
        ``multiplexer_status`` tool renders is the same data the scrape sees,
        and that a scrape triggers a fresh sample rather than serving stale
        gauges.
        """
        from agent_utilities.mcp import multiplexer as mux_mod

        class _FakeRuntime:
            def status(self):
                return {"server": "sampled-mcp", "state": "up", "breaker": "closed"}

        class _FakeMux:
            children = {"sampled-mcp": _FakeRuntime()}
            aggregated_tools: list = []
            tool_to_server = {"sampled-mcp__probe": ("sampled-mcp", "probe")}
            _exposed = {"sampled-mcp__probe"}
            _child_catalog_fingerprints: dict = {}
            # status_snapshot also reads _child_schema_revisions (per-child
            # catalog revision counter) and _child_schema_refresh_errors --
            # both real MCPMultiplexer.__init__ attributes this fake must
            # mirror for the real (unmocked) status_snapshot to run.
            _child_schema_revisions: dict = {}
            _child_schema_refresh_errors: dict = {}

            _mounted_tool_counts = mux_mod.MCPMultiplexer._mounted_tool_counts
            status_snapshot = mux_mod.MCPMultiplexer.status_snapshot

        fake = _FakeMux()
        snapshot = fake.status_snapshot()
        assert snapshot["children"]["sampled-mcp"]["mounted_tools"] == 1

        mux_mod._LIVE_MULTIPLEXERS.add(fake)
        mux_mod._register_child_health_sampler()
        try:
            body, _content_type = gm.render_metrics()
        finally:
            mux_mod._LIVE_MULTIPLEXERS.discard(fake)
        text = body.decode()
        assert (
            'agent_utilities_mcp_child_tools_dispatchable{server="sampled-mcp"} 1'
            in (text)
        )
        assert 'agent_utilities_mcp_child_process_running{server="sampled-mcp"} 1' in (
            text
        )

    def test_a_failing_sampler_never_blanks_the_scrape(self, caplog):
        def _bad_sampler():
            raise RuntimeError("sampler exploded")

        gm.register_scrape_sampler(_bad_sampler)
        try:
            with caplog.at_level(logging.WARNING):
                body, _ct = gm.render_metrics()
        finally:
            gm._SCRAPE_SAMPLERS.remove(_bad_sampler)
        assert b"agent_utilities_" in body
        assert "sampler exploded" in caplog.text


# ---------------------------------------------------------------------------
# 4. The prometheus_client multiprocess footgun
# ---------------------------------------------------------------------------


class TestMultiprocessGuard:
    def test_default_is_the_single_process_registry(self, monkeypatch):
        monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
        assert gm._collection_registry() is prometheus_client.REGISTRY

    def test_multiproc_dir_is_honoured_when_usable(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
        registry = gm._collection_registry()
        assert registry is not prometheus_client.REGISTRY

    def test_unusable_multiproc_dir_falls_back_loudly_not_silently(
        self, monkeypatch, caplog, tmp_path
    ):
        """A broken multiprocess setup must degrade AND say so at ERROR."""
        missing = tmp_path / "does-not-exist"
        monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(missing))
        with caplog.at_level(logging.ERROR):
            registry = gm._collection_registry()
        assert registry is prometheus_client.REGISTRY
        assert "UNDER-REPORTS" in caplog.text

    def test_scrape_still_renders_under_a_broken_multiproc_setting(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path / "nope"))
        body, content_type = gm.render_metrics()
        assert b"agent_utilities_" in body
        assert "text/plain" in content_type


# ---------------------------------------------------------------------------
# 5. OTLP traces — endpoint override + loud-but-soft failure
# ---------------------------------------------------------------------------


class TestOtlpTraceWiring:
    def test_traces_endpoint_defaults_to_the_base_collector(self, monkeypatch):
        from agent_utilities import observability as obs

        monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
        assert (
            obs._resolve_traces_endpoint("http://collector:4318")
            == "http://collector:4318/v1/traces"
        )

    def test_standard_signal_var_overrides_with_a_complete_url(self, monkeypatch):
        """``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`` is a COMPLETE url, per spec."""
        from agent_utilities import observability as obs

        monkeypatch.setenv(
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
            "http://tempo.apps.svc.cluster.local:4318/v1/traces",
        )
        assert (
            obs._resolve_traces_endpoint("https://langfuse.arpa/api/public/otel")
            == "http://tempo.apps.svc.cluster.local:4318/v1/traces"
        )

    def test_export_failure_is_soft_but_loud(self, caplog):
        """An unreachable collector must not raise — and must not be silent."""
        from opentelemetry.sdk.trace.export import SpanExportResult

        from agent_utilities import observability as obs

        class _Unreachable:
            def export(self, spans):
                raise ConnectionError("tempo unreachable")

            def shutdown(self):
                return None

        exporter = obs._LoudFailureSpanExporter(
            _Unreachable(), endpoint="http://tempo:4318/v1/traces"
        )
        with caplog.at_level(logging.ERROR):
            result = exporter.export([])

        assert result is SpanExportResult.FAILURE
        assert "tempo unreachable" in caplog.text
        # The endpoint itself is redacted by this repo's logging privacy filter;
        # what must survive is that the failure is LOUD (ERROR) and names the
        # cause, so a dead trace pipeline cannot pass for an idle one.
        assert caplog.records and caplog.records[0].levelno == logging.ERROR
        assert "Traces are being dropped" in caplog.text

    def test_repeated_failures_are_throttled_then_recovery_is_logged(self, caplog):
        from opentelemetry.sdk.trace.export import SpanExportResult

        from agent_utilities import observability as obs

        class _Flaky:
            def __init__(self):
                self.ok = False

            def export(self, spans):
                if not self.ok:
                    raise ConnectionError("down")
                return SpanExportResult.SUCCESS

            def shutdown(self):
                return None

        inner = _Flaky()
        exporter = obs._LoudFailureSpanExporter(inner, endpoint="http://tempo:4318")
        with caplog.at_level(logging.ERROR):
            exporter.export([])
            exporter.export([])
            exporter.export([])
        assert caplog.text.count("OTLP span export") == 1  # throttled, not spammed

        caplog.clear()
        inner.ok = True
        with caplog.at_level(logging.INFO):
            assert exporter.export([]) is SpanExportResult.SUCCESS
        assert "recovered" in caplog.text
