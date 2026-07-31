"""Wiring tests for the standalone gateway daemon's metrics listener (D-OG-4).

CONCEPT:AU-OS.observability.daemon-metrics-listener. The `graph-os-host`
container runs `python3 -m agent_utilities.gateway.daemon` with NO HTTP server
of any kind, so the `SCHEDULED_JOB_*`/`LANE_*`/`KG_INGEST_*`/`DISPATCH_*`
families it records were structurally uncollectable. These tests drive a REAL
HTTP request against a REAL `prometheus_client.start_http_server` listener and
assert a real metric's exposed value moves — not merely that the function
exists or returns True.
"""

from __future__ import annotations

import socket
import sys
import urllib.request

import pytest

from agent_utilities.gateway import daemon

pytestmark = pytest.mark.concept("AU-OS.observability.daemon-metrics-listener")


def _free_port() -> int:
    """Ask the OS for a currently-unused loopback port (test-only)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(autouse=True)
def _reset_daemon_metrics_state(monkeypatch: pytest.MonkeyPatch):
    """Each test gets a clean singleton + clean env — the listener is a
    process-wide side effect (a real bound socket), so state must not leak
    between tests."""
    monkeypatch.setattr(daemon, "_metrics_started", False)
    monkeypatch.delenv("KG_DAEMON_METRICS", raising=False)
    monkeypatch.delenv("KG_DAEMON_METRICS_HOST", raising=False)
    monkeypatch.delenv("KG_DAEMON_METRICS_PORT", raising=False)
    yield


def test_listener_serves_a_real_metric_value_that_actually_moves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The core wiring proof: start the real listener, record a real counter
    increment through the SAME registry `gateway_metrics.py` defines, and
    prove the scraped HTTP body reflects it — end to end, no mocks."""
    pytest.importorskip("prometheus_client")
    from agent_utilities.observability.gateway_metrics import SCHEDULED_JOB_RUNS

    port = _free_port()
    monkeypatch.setenv("KG_DAEMON_METRICS_HOST", "127.0.0.1")
    monkeypatch.setenv("KG_DAEMON_METRICS_PORT", str(port))

    assert daemon.start_daemon_metrics_listener() is True

    # Baseline scrape — establish the family is present and read its current
    # value before mutating it.
    before = urllib.request.urlopen(
        f"http://127.0.0.1:{port}/metrics", timeout=5
    ).read().decode()
    assert "agent_utilities_scheduled_job_runs_total" in before

    # Drive a REAL increment on a schedule/outcome label pair unlikely to
    # collide with anything else the process recorded this run.
    SCHEDULED_JOB_RUNS.labels(
        schedule="__test_d_og_4_wiring", outcome="ok"
    ).inc()

    after = urllib.request.urlopen(
        f"http://127.0.0.1:{port}/metrics", timeout=5
    ).read().decode()
    target_line = [
        line
        for line in after.splitlines()
        if line.startswith("agent_utilities_scheduled_job_runs_total")
        and 'schedule="__test_d_og_4_wiring"' in line
        and 'outcome="ok"' in line
    ]
    assert target_line, "the incremented series must appear in the real scrape"
    assert target_line[0].strip().endswith(" 1.0")


def test_listener_starts_once_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("prometheus_client")
    port = _free_port()
    monkeypatch.setenv("KG_DAEMON_METRICS_HOST", "127.0.0.1")
    monkeypatch.setenv("KG_DAEMON_METRICS_PORT", str(port))

    assert daemon.start_daemon_metrics_listener() is True
    # A second call must not attempt a second bind (which would raise) — it
    # short-circuits on the singleton flag and reports success.
    assert daemon.start_daemon_metrics_listener() is True

    body = urllib.request.urlopen(
        f"http://127.0.0.1:{port}/metrics", timeout=5
    ).read()
    assert body  # still genuinely serving


def test_disabled_via_config_never_binds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KG_DAEMON_METRICS", "false")
    port = _free_port()
    monkeypatch.setenv("KG_DAEMON_METRICS_HOST", "127.0.0.1")
    monkeypatch.setenv("KG_DAEMON_METRICS_PORT", str(port))

    assert daemon.start_daemon_metrics_listener() is False

    # Prove nothing is actually listening — a real socket-level check, not
    # just trusting the return value.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(1)
        with pytest.raises(ConnectionRefusedError):
            probe.connect(("127.0.0.1", port))


def test_bind_failure_is_loud_and_never_crashes_the_daemon(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A port already in use must degrade to a loud ERROR log and a clean
    ``False`` return — never an unhandled exception that would take the
    daemon process down (the daemon's whole reason to keep running through
    a metrics-listener failure)."""
    pytest.importorskip("prometheus_client")
    port = _free_port()
    monkeypatch.setenv("KG_DAEMON_METRICS_HOST", "127.0.0.1")
    monkeypatch.setenv("KG_DAEMON_METRICS_PORT", str(port))

    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    blocker.bind(("127.0.0.1", port))
    blocker.listen(1)
    try:
        with caplog.at_level("ERROR"):
            result = daemon.start_daemon_metrics_listener()
        assert result is False
        assert any(
            record.levelname == "ERROR"
            and "Daemon metrics listener NOT started" in record.message
            for record in caplog.records
        )
    finally:
        blocker.close()


def test_missing_metrics_extra_logs_error_and_returns_false(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """When the optional ``metrics`` extra is absent, the daemon must still
    start (this only proves the function's own contract) — degrading LOUDLY,
    never via a silent ``except ImportError: pass``."""
    real_import = __import__

    def _blocking_import(name, *args, **kwargs):
        if name == "prometheus_client" or name.startswith("prometheus_client."):
            raise ImportError("simulated: metrics extra not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "prometheus_client", raising=False)
    monkeypatch.setattr("builtins.__import__", _blocking_import)

    with caplog.at_level("ERROR"):
        result = daemon.start_daemon_metrics_listener()
    assert result is False
    assert any(
        record.levelname == "ERROR"
        and "optional 'metrics' extra" in record.message
        for record in caplog.records
    )
