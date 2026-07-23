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


@pytest.fixture(autouse=True)
def _isolated_mirror_build_status():
    """``get_mirror_build_status()`` reads a process-wide registry other test
    modules (e.g. ``test_mirror_set.py``) also populate — clear it so a
    ``kg_mirrors`` test never sees another test's mirror names."""
    from agent_utilities.knowledge_graph import backends as B

    B._MIRROR_BUILD_STATUS.clear()
    yield
    B._MIRROR_BUILD_STATUS.clear()


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
# 2b. kg_mirrors — optional kg_connections mirrors (CONCEPT:AU-KG.backend.mirror-health-repair)
#
# The availability bug this check exists for: a broken OPTIONAL mirror
# (missing driver / unreachable host / bad credentials) used to crash the
# whole graph-os server. The fix isolates the failure at its source
# (``backends/__init__.py``'s ``_build_mirror_set``); this check makes that
# failure OBSERVABLE. Critically, it must never report ``unhealthy`` — that
# would pull graph-os out of Service routing over a dependency the
# epistemic-graph engine authority does not need. It reports ``degraded``
# instead: visible, but informational-only (never fails the rollup) — same
# semantics ``not_configured`` already gets elsewhere in this module.
# --------------------------------------------------------------------------- #
def _configure_one_mirror(monkeypatch):
    """``_resolve_mirror_target_names`` (like ``_build_mirror_set``) reads the
    process-wide config singleton directly, not an injected ``cfg`` — mirrors
    the exact pattern ``tests/unit/knowledge_graph/test_mirror_set.py`` already
    uses for the same module.
    """
    from agent_utilities.core.config import config as live_cfg

    monkeypatch.delenv("GRAPH_MIRROR_TARGETS", raising=False)
    monkeypatch.setattr(live_cfg, "graph_mirror_targets", ["prod-neo4j"], raising=False)
    monkeypatch.setattr(
        live_cfg,
        "kg_connections",
        [{"name": "prod-neo4j", "backend": "neo4j"}],
        raising=False,
    )


def test_kg_mirrors_not_configured_by_default():
    """No ``GRAPH_MIRROR_TARGETS`` / mirror-role ``kg_connections`` entries at
    all → informational ``not_configured``, matching every other optional
    co-service check in this module."""
    result = rh._check_kg_mirrors(AgentConfig())
    assert result["status"] == "not_configured"


def test_kg_mirrors_all_healthy_is_ok(monkeypatch):
    from agent_utilities.knowledge_graph import backends as B

    _configure_one_mirror(monkeypatch)
    monkeypatch.setattr(
        B, "_MIRROR_BUILD_STATUS", {"prod-neo4j": {"backend_type": "neo4j", "ok": True}}
    )

    result = rh._check_kg_mirrors(AgentConfig())

    assert result["status"] == "ok"
    assert result["detail"]["healthy"] == ["prod-neo4j"]
    assert result["detail"]["failed"] == {}


def test_kg_mirrors_broken_mirror_is_degraded_never_unhealthy(monkeypatch):
    """THE requirement: a mirror that failed to construct is reported as
    ``degraded`` detail — informational — NEVER ``unhealthy``. An optional
    mirror going down must not pull graph-os out of Service routing.
    """
    from agent_utilities.knowledge_graph import backends as B

    _configure_one_mirror(monkeypatch)
    monkeypatch.setattr(
        B,
        "_MIRROR_BUILD_STATUS",
        {
            "prod-neo4j": {
                "backend_type": "neo4j",
                "ok": False,
                "reason": "ImportError: Neo4j driver is not installed",
            }
        },
    )

    result = rh._check_kg_mirrors(AgentConfig())

    assert result["status"] == "degraded"
    assert result["status"] != "unhealthy"
    assert "prod-neo4j" in result["detail"]["failed"]
    assert "Neo4j driver is not installed" in result["detail"]["failed"]["prod-neo4j"]


def test_kg_mirrors_degraded_check_never_flips_the_overall_rollup(monkeypatch):
    """The whole point: even with a broken optional mirror, ``collect_health``'s
    overall status must stay ``healthy`` (readiness is unaffected)."""
    from agent_utilities.knowledge_graph import backends as B

    _configure_one_mirror(monkeypatch)
    monkeypatch.setattr(
        B,
        "_MIRROR_BUILD_STATUS",
        {
            "prod-neo4j": {
                "backend_type": "neo4j",
                "ok": False,
                "reason": "ImportError: Neo4j driver is not installed",
            }
        },
    )
    monkeypatch.setattr(
        rh,
        "_CHECKS",
        (
            ("kg_mirrors", rh._check_kg_mirrors),
            ("a", lambda _cfg: {"name": "a", "status": "ok"}),
        ),
    )

    report = rh.collect_health()

    assert report["status"] == "healthy"
    assert rh.is_overall_healthy(report)
    kg_check = next(c for c in report["checks"] if c["name"] == "kg_mirrors")
    assert kg_check["status"] == "degraded"


def test_kg_mirrors_never_builds_or_reconnects_anything(monkeypatch):
    """Pure read-only registry scan — the check itself must never trigger a
    mirror (re)build (no driver import, no network I/O), matching the
    ``messaging`` check's convention."""
    from agent_utilities.knowledge_graph import backends as B

    _configure_one_mirror(monkeypatch)

    def _boom(*_a, **_k):
        raise AssertionError("kg_mirrors health check must not build anything")

    monkeypatch.setattr(B, "_build_mirror_set", _boom)
    monkeypatch.setattr(B, "_build_member", _boom)

    result = rh._check_kg_mirrors(AgentConfig())

    # Configured, but no build has been observed yet in this process.
    assert result["status"] == "ok"
    assert result["detail"]["unobserved"] == ["prod-neo4j"]


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

    # The REPORT stays truthful: any unhealthy check makes the rollup unhealthy.
    assert report["status"] == "unhealthy"
    assert {c["name"] for c in report["checks"]} == {"a", "b", "c"}
    assert "generated_at" in report
    # READINESS is deliberately narrower than the rollup. It answers only "can
    # this process serve requests", which depends on the engine it reads and
    # writes through — not on optional co-services that run in their own
    # deployments. Gating rotation on those turned one component's outage into a
    # total one (a down messaging daemon pulled a serving graph-os out of the
    # Service). Here "c" is unhealthy but is not the engine, so the process
    # stays ready while the report still says unhealthy.
    assert rh.is_overall_healthy(report)


def test_readiness_is_false_when_the_engine_itself_is_unhealthy(monkeypatch):
    monkeypatch.setattr(
        rh,
        "_CHECKS",
        (
            ("engine", lambda _cfg: {"name": "engine", "status": "unhealthy", "reason": "x"}),
            ("messaging", lambda _cfg: {"name": "messaging", "status": "ok"}),
        ),
    )

    report = rh.collect_health()

    assert report["status"] == "unhealthy"
    assert not rh.is_overall_healthy(report)


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


# --------------------------------------------------------------------------- #
# 2c. embedding_endpoint — CONCEPT:AU-KG.retrieval.embedding-fast-fail
#
# Semantic search is OPTIONAL relative to the engine's mandatory keyword/
# lexical fallback, so a tripped embedding-endpoint breaker must surface as
# ``degraded`` (visible) — NEVER ``unhealthy`` (which would pull graph-os out
# of Service routing over an optional dependency), matching the ``kg_mirrors``
# precedent above.
# --------------------------------------------------------------------------- #
def test_embedding_endpoint_not_configured_when_no_embedding_model():
    from types import SimpleNamespace

    cfg = SimpleNamespace(default_embedding_model=None)
    result = rh._check_embedding_endpoint(cfg)
    assert result["status"] == "not_configured"


def test_embedding_endpoint_ok_when_breaker_closed(monkeypatch):
    from types import SimpleNamespace

    from agent_utilities.core import embedding_failover as ef

    cfg = SimpleNamespace(default_embedding_model=object())
    monkeypatch.setattr(
        ef,
        "embedding_endpoint_status",
        lambda: {
            "active_model_key": "embedding",
            "active_base_url": "http://vllm-embed.arpa/v1",
            "active_gpu_group": "gb10",
            "is_fallback": False,
            "fallback_configured": False,
            "primary_breaker": {"state": "closed", "trips": 0},
            "failover_count": 0,
            "recovery_count": 0,
        },
    )

    result = rh._check_embedding_endpoint(cfg)

    assert result["status"] == "ok"
    assert result["detail"]["breaker_state"] == "closed"
    # No raw endpoint URL leaks into the health payload.
    assert "vllm-embed.arpa" not in str(result)


def test_embedding_endpoint_open_breaker_is_degraded_never_unhealthy(monkeypatch):
    """THE requirement this check exists for: a tripped embedding breaker
    (the exact scenario a misconfigured MODEL_HTTP_ALLOWED_PRIVATE_HOSTS
    produces) is reported as degraded — informational — never unhealthy."""
    from types import SimpleNamespace

    from agent_utilities.core import embedding_failover as ef

    cfg = SimpleNamespace(default_embedding_model=object())
    monkeypatch.setattr(
        ef,
        "embedding_endpoint_status",
        lambda: {
            "active_model_key": "embedding",
            "active_base_url": "http://vllm-embed.arpa/v1",
            "active_gpu_group": "gb10",
            "is_fallback": False,
            "fallback_configured": False,
            "primary_breaker": {"state": "open", "trips": 3},
            "failover_count": 0,
            "recovery_count": 0,
        },
    )

    result = rh._check_embedding_endpoint(cfg)

    assert result["status"] == "degraded"
    assert result["status"] != "unhealthy"
    assert "keyword" in result["reason"] or "lexical" in result["reason"]
    assert result["detail"]["breaker_trips"] == 3


def test_embedding_endpoint_degraded_check_never_flips_the_overall_rollup(
    monkeypatch,
):
    from agent_utilities.core import embedding_failover as ef

    monkeypatch.setattr(
        ef,
        "embedding_endpoint_status",
        lambda: {
            "active_model_key": "embedding",
            "active_base_url": None,
            "active_gpu_group": None,
            "is_fallback": False,
            "fallback_configured": False,
            "primary_breaker": {"state": "open", "trips": 1},
            "failover_count": 0,
            "recovery_count": 0,
        },
    )
    monkeypatch.setattr(
        rh,
        "_CHECKS",
        (
            ("embedding_endpoint", rh._check_embedding_endpoint),
            ("a", lambda _cfg: {"name": "a", "status": "ok"}),
        ),
    )
    monkeypatch.setattr(
        "agent_utilities.core.config.AgentConfig.default_embedding_model",
        property(lambda self: object()),
        raising=False,
    )

    report = rh.collect_health()

    assert report["status"] == "healthy"
    assert rh.is_overall_healthy(report)
    check = next(c for c in report["checks"] if c["name"] == "embedding_endpoint")
    assert check["status"] == "degraded"


def test_embedding_endpoint_fallback_routing_is_degraded(monkeypatch):
    from types import SimpleNamespace

    from agent_utilities.core import embedding_failover as ef

    cfg = SimpleNamespace(default_embedding_model=object())
    monkeypatch.setattr(
        ef,
        "embedding_endpoint_status",
        lambda: {
            "active_model_key": "embedding:fallback",
            "active_base_url": "http://fallback.internal/v1",
            "active_gpu_group": "shared",
            "is_fallback": True,
            "fallback_configured": True,
            "primary_breaker": {"state": "closed", "trips": 0},
            "failover_count": 1,
            "recovery_count": 0,
        },
    )

    result = rh._check_embedding_endpoint(cfg)

    assert result["status"] == "degraded"
    assert "FALLBACK" in result["reason"]


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
