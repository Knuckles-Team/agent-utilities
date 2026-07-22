"""Proof that graph-os's ``/health`` reports the TRUTH (CONCEPT:AU-OS.observability.no-op-without-metrics).

The historical bug: ``/health`` was an unconditional stub returning
``{"status": "ok"}`` regardless of reality — it would report healthy with the
epistemic-graph engine completely gone. ``observability.runtime_health.
collect_health`` is the ONE shared core that replaces it (dispatched into by
both the REST gateway and the MCP server's ``/health``/``/health/ready`` routes
and ``graph_configure(action="health")``).

The tests below prove the three semantics the whole endpoint exists for:

1. **Engine genuinely unreachable → the report is UNHEALTHY.** Not "healthy
   regardless" — a real, live socket probe against a real (bound but
   never-listened, or plain absent) endpoint.
2. **A co-service/dependency that IS configured but down → unhealthy.**
3. **A co-service/dependency that is simply NOT configured → healthy/
   informational (``not_configured``), never dragging the rollup down.**

Plus the two hard structural guarantees: a check that raises or hangs is
NEVER swallowed into "ok" (:func:`_run_bounded`), and the whole endpoint is
therefore bounded and cannot hang past its per-check ceiling.
"""

from __future__ import annotations

import contextlib
import os
import socket
import threading
import time

import pytest

from agent_utilities.core.config import AgentConfig
from agent_utilities.knowledge_graph.core import engine_resolver as er
from agent_utilities.knowledge_graph.core import shard_topology as st
from agent_utilities.observability import runtime_health as rh


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
class _UDSServer:
    """A trivial UDS listener so a transport probe sees an engine "running"."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(path)
        self._sock.listen(8)
        self._stop = False
        self._t = threading.Thread(target=self._serve, daemon=True)
        self._t.start()

    def _serve(self) -> None:
        self._sock.settimeout(0.2)
        while not self._stop:
            try:
                conn, _ = self._sock.accept()
                conn.close()
            except OSError:
                continue

    def close(self) -> None:
        self._stop = True
        try:
            self._sock.close()
        finally:
            with contextlib.suppress(OSError):
                os.unlink(self.path)


class _TCPServer:
    """A trivial TCP listener bound to an OS-assigned free port."""

    def __init__(self, host: str = "127.0.0.1") -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, 0))
        self._sock.listen(8)
        self.host = host
        self.port = self._sock.getsockname()[1]
        self._stop = False
        self._t = threading.Thread(target=self._serve, daemon=True)
        self._t.start()

    def _serve(self) -> None:
        self._sock.settimeout(0.2)
        while not self._stop:
            try:
                conn, _ = self._sock.accept()
                conn.close()
            except OSError:
                continue

    def close(self) -> None:
        self._stop = True
        self._sock.close()


def _patch_resolve_endpoints(
    monkeypatch: pytest.MonkeyPatch, endpoints: list[str]
) -> None:
    """Make BOTH ``shard_topology_status`` and ``resolve_engine`` agree on the
    same (unconfigured / default-local) coordinator contact — mirrors
    ``tests/unit/knowledge_graph/core/test_engine_resolver.py``'s convention.

    The suite's own session fixture exports a real ``GRAPH_SERVICE_ENDPOINTS``
    for engine-backed tests, which would otherwise force every resolution to
    ``mode="remote"`` regardless of what we patch here — clear it so THIS
    test's local topology decides the mode instead.
    """
    monkeypatch.delenv("GRAPH_SERVICE_ENDPOINTS", raising=False)
    monkeypatch.setattr(st, "resolve_endpoints", lambda _cfg=None: list(endpoints))
    monkeypatch.setattr(er, "resolve_endpoints", lambda _cfg: list(endpoints))


@pytest.fixture(autouse=True)
def _isolated_breaker_registry():
    """Every test starts with a clean circuit-breaker registry."""
    from agent_utilities.knowledge_graph.core import engine_breaker as eb

    eb.reset_breakers()
    yield
    eb.reset_breakers()


# --------------------------------------------------------------------------- #
# 1. engine reachability — the core bug this task fixes
# --------------------------------------------------------------------------- #
def test_engine_unreachable_reports_unhealthy(monkeypatch, tmp_path):
    """THE bug fix: nothing listening on the resolved engine endpoint → the
    engine check (and the overall rollup) MUST be unhealthy — not "ok
    regardless", which is what the stub always returned.
    """
    dead_sock = str(tmp_path / "nobody-home.sock")  # never bound/listened
    _patch_resolve_endpoints(monkeypatch, [f"unix://{dead_sock}"])

    result = rh._check_engine(AgentConfig())

    assert result["status"] == "unhealthy"
    assert "reason" in result and result["reason"]
    assert result["detail"]["reachable_count"] == 0


def test_engine_unreachable_remote_mode_is_unhealthy_fail_loud(monkeypatch):
    """An explicitly configured (``GRAPH_SERVICE_ENDPOINTS``) remote engine that
    is down is unhealthy — remote mode never silently autostarts a stand-in.
    """
    monkeypatch.setenv(
        "GRAPH_SERVICE_ENDPOINTS", "tcp://127.0.0.1:1"
    )  # port 1: nobody listens
    cfg = AgentConfig()

    result = rh._check_engine(cfg)

    assert result["status"] == "unhealthy"
    assert result["detail"]["resolved_mode"] == "remote"
    assert "remote" in result["reason"]


def test_engine_reachable_reports_ok(monkeypatch, tmp_path):
    """A REAL listening engine endpoint → the engine check is ok, with
    resolved_mode="shared" (an already-running local engine is reused)."""
    sock_path = str(tmp_path / "engine-up.sock")
    server = _UDSServer(sock_path)
    try:
        _patch_resolve_endpoints(monkeypatch, [f"unix://{sock_path}"])
        result = rh._check_engine(AgentConfig())
    finally:
        server.close()

    assert result["status"] == "ok"
    assert result["detail"]["resolved_mode"] == "shared"
    assert result["detail"]["reachable_count"] == 1


def test_engine_open_breaker_is_unhealthy_even_when_transport_reachable(
    monkeypatch, tmp_path
):
    """A transport-reachable engine whose circuit breaker is OPEN (real calls
    are currently failing fast) must still report unhealthy — raw connectivity
    alone is not enough evidence the engine is actually serving.
    """
    from agent_utilities.knowledge_graph.core import engine_breaker as eb

    sock_path = str(tmp_path / "engine-flaky.sock")
    server = _UDSServer(sock_path)
    try:
        endpoint = f"unix://{sock_path}"
        _patch_resolve_endpoints(monkeypatch, [endpoint])
        cfg = AgentConfig()
        breaker = eb.get_breaker(endpoint)
        for _ in range(cfg.engine_breaker_threshold + 1):
            breaker.record_failure()
        assert breaker.state == "open"

        result = rh._check_engine(cfg)
    finally:
        server.close()

    assert result["status"] == "unhealthy"
    assert "breaker" in result["reason"]
    assert result["detail"]["open_breaker_count"] == 1


# --------------------------------------------------------------------------- #
# 2. co-service semantics: not_configured (healthy) vs configured-but-down
# --------------------------------------------------------------------------- #
def test_kg_host_daemon_not_configured_for_remote_engine(monkeypatch):
    monkeypatch.setenv("GRAPH_SERVICE_ENDPOINTS", "tcp://remote-engine.internal:9100")
    result = rh._check_kg_host_daemon(AgentConfig())
    assert result["status"] == "not_configured"


def test_kg_host_daemon_configured_but_down_is_unhealthy(monkeypatch):
    from agent_utilities.knowledge_graph.core import host_lock

    monkeypatch.delenv("GRAPH_SERVICE_ENDPOINTS", raising=False)
    monkeypatch.setattr(host_lock, "host_daemon_running", lambda: False)

    result = rh._check_kg_host_daemon(AgentConfig())

    assert result["status"] == "unhealthy"
    assert "host-daemon lock" in result["reason"]


def test_kg_host_daemon_configured_and_up_is_ok(monkeypatch):
    from agent_utilities.knowledge_graph.core import host_lock

    monkeypatch.delenv("GRAPH_SERVICE_ENDPOINTS", raising=False)
    monkeypatch.setattr(host_lock, "host_daemon_running", lambda: True)

    result = rh._check_kg_host_daemon(AgentConfig())

    assert result["status"] == "ok"


def test_messaging_not_configured_is_healthy_informational(monkeypatch):
    """No platform credentials anywhere → not_configured, never unhealthy."""
    from agent_utilities.messaging.registry import MessagingRegistry

    monkeypatch.setattr(
        MessagingRegistry.instance(), "configured_backend_ids", lambda: []
    )

    result = rh._check_messaging(AgentConfig())

    assert result["status"] == "not_configured"


def test_messaging_configured_but_not_connected_is_unhealthy(monkeypatch):
    """Credentials present, but no connected backend instance in this process
    → unhealthy (the exact "configured but DOWN" semantic the task requires).
    """
    from agent_utilities.messaging.registry import MessagingRegistry

    registry = MessagingRegistry.instance()
    monkeypatch.setattr(registry, "configured_backend_ids", lambda: ["telegram"])
    monkeypatch.setattr(registry, "get_backend", lambda _id: None)

    result = rh._check_messaging(AgentConfig())

    assert result["status"] == "unhealthy"
    assert "telegram" in result["reason"]


def test_messaging_configured_and_connected_is_ok(monkeypatch):
    from agent_utilities.messaging.registry import MessagingRegistry

    class _Connected:
        is_connected = True

    registry = MessagingRegistry.instance()
    monkeypatch.setattr(registry, "configured_backend_ids", lambda: ["telegram"])
    monkeypatch.setattr(registry, "get_backend", lambda _id: _Connected())

    result = rh._check_messaging(AgentConfig())

    assert result["status"] == "ok"


def test_state_store_not_configured_by_default():
    """``STATE_DB_URI`` unset (the zero-infra default) → not_configured."""
    result = rh._check_state_store(AgentConfig())
    assert result["status"] == "not_configured"


def test_state_store_configured_but_unreachable_is_unhealthy(monkeypatch):
    """A configured Postgres state store that cannot actually be reached must
    be unhealthy, never silently treated as fine.
    """
    monkeypatch.setenv("STATE_DB_URI", "postgresql://u:p@127.0.0.1:1/doesnotmatter")

    result = rh._check_state_store(AgentConfig())

    assert result["status"] == "unhealthy"
    assert "state store" in result["reason"] or "Postgres" in result["reason"]


def test_kafka_bus_not_configured_by_default():
    result = rh._check_kafka_bus(AgentConfig())
    assert result["status"] == "not_configured"


def test_kafka_bus_configured_but_unreachable_is_unhealthy(monkeypatch):
    monkeypatch.setenv("TASK_QUEUE_BACKEND", "kafka")
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:1")

    result = rh._check_kafka_bus(AgentConfig())

    assert result["status"] == "unhealthy"


def test_kafka_bus_configured_and_reachable_is_ok(monkeypatch):
    server = _TCPServer()
    try:
        monkeypatch.setenv("AGENT_BUS_LOG_BACKEND", "kafka")
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", f"{server.host}:{server.port}")
        result = rh._check_kafka_bus(AgentConfig())
    finally:
        server.close()

    assert result["status"] == "ok"


def test_stardog_mirror_not_configured_by_default():
    result = rh._check_stardog_mirror(AgentConfig())
    assert result["status"] == "not_configured"


def test_stardog_mirror_configured_but_unreachable_is_unhealthy(monkeypatch):
    monkeypatch.setenv("CONTINUOUS_STARDOG_MIRROR", "true")
    monkeypatch.setenv("STARDOG_ENDPOINT", "http://127.0.0.1:1")

    result = rh._check_stardog_mirror(AgentConfig())

    assert result["status"] == "unhealthy"


def test_stardog_mirror_configured_and_reachable_is_ok(monkeypatch):
    server = _TCPServer()
    try:
        monkeypatch.setenv("CONTINUOUS_STARDOG_MIRROR", "true")
        monkeypatch.setenv("STARDOG_ENDPOINT", f"http://{server.host}:{server.port}")
        result = rh._check_stardog_mirror(AgentConfig())
    finally:
        server.close()

    assert result["status"] == "ok"


# --------------------------------------------------------------------------- #
# 3. the hard structural guarantee: a broken/hung probe is NEVER "ok"
# --------------------------------------------------------------------------- #
def test_a_probe_that_raises_is_unhealthy_never_swallowed_to_ok():
    def _boom(_cfg):
        raise RuntimeError("driver exploded")

    result = rh._run_bounded("boom", lambda: _boom(None))

    assert result["status"] == "unhealthy"
    assert "RuntimeError" in result["reason"]


def test_a_probe_that_hangs_is_bounded_and_reported_unhealthy(monkeypatch):
    """``/health`` must never hang: a check that blocks past its ceiling is
    abandoned and reported unhealthy, not awaited forever.
    """
    monkeypatch.setattr(rh, "_CHECK_WALL_TIMEOUT_S", 0.2)

    def _hang():
        time.sleep(5.0)
        return {"name": "hang", "status": "ok"}

    started = time.monotonic()
    result = rh._run_bounded("hang", _hang)
    elapsed = time.monotonic() - started

    assert elapsed < 2.0, "a hung probe must not block the caller past its bound"
    assert result["status"] == "unhealthy"
    assert "exceeded" in result["reason"]


# --------------------------------------------------------------------------- #
# 4. overall rollup semantics
# --------------------------------------------------------------------------- #
def test_collect_health_rollup_is_unhealthy_if_any_check_is_unhealthy(monkeypatch):
    monkeypatch.setattr(
        rh,
        "_CHECKS",
        (
            ("a", lambda _cfg: {"name": "a", "status": "ok"}),
            ("b", lambda _cfg: {"name": "b", "status": "not_configured"}),
            ("c", lambda _cfg: {"name": "c", "status": "unhealthy", "reason": "down"}),
        ),
    )

    report = rh.collect_health()

    assert report["status"] == "unhealthy"
    assert not rh.is_overall_healthy(report)
    assert {c["name"] for c in report["checks"]} == {"a", "b", "c"}
    assert "generated_at" in report


def test_collect_health_rollup_is_healthy_when_only_ok_and_not_configured(monkeypatch):
    monkeypatch.setattr(
        rh,
        "_CHECKS",
        (
            ("a", lambda _cfg: {"name": "a", "status": "ok"}),
            ("b", lambda _cfg: {"name": "b", "status": "not_configured"}),
        ),
    )

    report = rh.collect_health()

    assert report["status"] == "healthy"
    assert rh.is_overall_healthy(report)


def test_collect_health_never_raises_even_when_agentconfig_is_broken(monkeypatch):
    def _boom():
        raise RuntimeError("config load exploded")

    monkeypatch.setattr("agent_utilities.core.config.AgentConfig", lambda: _boom())

    report = rh.collect_health()

    assert report["status"] == "unhealthy"
    assert report["checks"][0]["status"] == "unhealthy"


def test_collect_health_payload_never_carries_raw_endpoint_strings(
    monkeypatch, tmp_path
):
    """No hostnames/paths/DSNs leak — only counts/booleans/resolved-mode/ids
    (the engine check's redaction convention, matching deployment/doctor.py).
    """
    dead_sock = str(tmp_path / "nobody.sock")
    _patch_resolve_endpoints(monkeypatch, [f"unix://{dead_sock}"])
    monkeypatch.setenv(
        "STATE_DB_URI", "postgresql://secretuser:hunter2@10.0.0.5:5432/prod"
    )

    report = rh.collect_health()

    payload_text = str(report)
    assert "hunter2" not in payload_text
    assert dead_sock not in payload_text
    assert "secretuser" not in payload_text
