"""Tests for the self-composing ``graph-os`` co-service supervisor.

Covers the three things the entrypoint promises:

1. Config-driven detection (no new env flag — messaging via real credential
   presence, the host daemon via ``KG_DAEMON_ROLE`` + the live host-lock file).
2. Bounded-restart supervision + clean shutdown for a co-service thread.
3. STDOUT PURITY: a co-service that logs/prints while running must never leak
   a byte onto stdout once ``protect_stdio_jsonrpc()`` has been applied — the
   critical stdio-transport invariant (stdout IS the JSON-RPC channel).

Plus a LIVE-PATH test (:func:`test_start_co_services_live_path_starts_messaging`)
that drives the real ``start_co_services`` entry point end to end and asserts
messaging actually started serving — not merely that a helper exists.
"""

from __future__ import annotations

import asyncio
import builtins
import io
import logging
import sys
import threading
import time
import warnings

import pytest

from agent_utilities.knowledge_graph.core.session import GraphSession
from agent_utilities.mcp import co_service_supervisor as cosvc
from agent_utilities.mcp import server_factory
from agent_utilities.messaging import daemon as messaging_daemon
from agent_utilities.messaging.service import MessagingService
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext


def _verified_session(actor_id: str = "co-service-test") -> GraphSession:
    actor = ActorContext(
        actor_id=actor_id,
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("system",),
        tenant_id="test-tenant",
        authenticated=True,
    )
    return GraphSession(
        actor=actor,
        tenant=actor.tenant_id,
        scopes=frozenset({"kg:admin"}),
        policy_version="current",
        audience="graph-runtime",
    )


@pytest.fixture(autouse=True)
def _reset_messaging_singleton():
    MessagingService._instance = None
    yield
    MessagingService._instance = None


# ── Detection ─────────────────────────────────────────────────────────────


def test_host_daemon_needed_true_only_when_client_and_unhosted(monkeypatch):
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.engine_tasks.daemon_role",
        lambda: "client",
    )
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.host_lock.host_daemon_running",
        lambda: False,
    )
    assert cosvc.host_daemon_needed() is True


@pytest.mark.parametrize(
    ("role", "hosted"),
    [
        ("auto", False),  # default — the engine self-elects on its own construction
        ("host", False),  # explicit host — same reason
        ("client", True),  # a real host already exists — nothing to do
    ],
)
def test_host_daemon_needed_false_cases(monkeypatch, role, hosted):
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.engine_tasks.daemon_role",
        lambda: role,
    )
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.host_lock.host_daemon_running",
        lambda: hosted,
    )
    assert cosvc.host_daemon_needed() is False


def test_detect_composition_reads_existing_config_only(monkeypatch):
    """No new flag: messaging comes from real credential presence, webui from
    the existing ENABLE_WEB_UI field, host-daemon from the existing
    KG_DAEMON_ROLE + live lock file."""

    class _Cfg:
        enable_web_ui = True

    monkeypatch.setattr("agent_utilities.core.config.config", _Cfg())
    monkeypatch.setattr(
        messaging_daemon, "configured_platforms", lambda engine=None: ["telegram"]
    )
    monkeypatch.setattr(cosvc, "host_daemon_needed", lambda: True)

    plan = cosvc.detect_composition()
    assert plan.messaging_platforms == ("telegram",)
    assert plan.messaging_configured is True
    assert plan.host_daemon_needed is True
    assert plan.web_ui_enabled is True
    assert plan.co_service_names() == ("messaging", "graph-os-daemon", "agent-webui")


def test_detect_composition_nothing_configured(monkeypatch):
    class _Cfg:
        enable_web_ui = False

    monkeypatch.setattr("agent_utilities.core.config.config", _Cfg())
    monkeypatch.setattr(
        messaging_daemon, "configured_platforms", lambda engine=None: []
    )
    monkeypatch.setattr(cosvc, "host_daemon_needed", lambda: False)

    plan = cosvc.detect_composition()
    assert plan.co_service_names() == ()


# ── Supervision: bounded restart + clean shutdown ──────────────────────────


def test_supervisor_restarts_a_crashing_co_service_up_to_the_bound(monkeypatch):
    monkeypatch.setattr(cosvc, "_RESTART_WINDOW_SECONDS", 300.0)
    monkeypatch.setattr(cosvc, "_MAX_BACKOFF_SECONDS", 0.01)

    attempts: list[int] = []

    def _always_crashes(stop_event: threading.Event) -> None:
        attempts.append(1)
        raise RuntimeError("boom")

    supervisor = cosvc.CoServiceSupervisor()
    session = _verified_session()
    supervisor.start_service("flaky", _always_crashes, session)

    # _MAX_RESTARTS + 1 initial attempt, each backing off ~0.01s -> settles fast.
    deadline = time.monotonic() + 10.0
    while len(attempts) <= cosvc._MAX_RESTARTS and time.monotonic() < deadline:
        time.sleep(0.05)

    # Give the loop one more beat to observe it gave up (no more attempts appear).
    count_after_wait = len(attempts)
    time.sleep(0.2)
    assert len(attempts) == count_after_wait, (
        "supervisor kept restarting past the bound"
    )
    assert count_after_wait == cosvc._MAX_RESTARTS + 1
    assert "flaky" not in supervisor.running()  # gave up — thread exited
    supervisor.stop_all(timeout=5.0)  # hygiene; the thread already exited


def test_supervisor_clean_shutdown_joins_a_well_behaved_service():
    started = threading.Event()

    def _well_behaved(stop_event: threading.Event) -> None:
        started.set()
        stop_event.wait()  # blocks until told to stop, like the real messaging loop

    supervisor = cosvc.CoServiceSupervisor()
    session = _verified_session()
    supervisor.start_service("polite", _well_behaved, session)

    assert started.wait(timeout=5.0)
    assert supervisor.running() == ("polite",)

    supervisor.stop_all(timeout=5.0)
    assert supervisor.running() == ()


def test_supervisor_thread_carries_the_verified_session(monkeypatch):
    """The co-service thread must run under the SAME verified actor/session for
    its whole lifetime (via ``_authorized_background_thread``), including
    restarts — proving the identity fix threads all the way through."""
    from agent_utilities.security.brain_context import current_actor

    seen: list[str] = []
    done = threading.Event()

    def _reads_actor(stop_event: threading.Event) -> None:
        seen.append(current_actor().actor_id)
        done.set()
        stop_event.wait()

    session = _verified_session(actor_id="supervised-actor")
    supervisor = cosvc.CoServiceSupervisor()
    supervisor.start_service("identity-check", _reads_actor, session)
    assert done.wait(timeout=5.0)
    supervisor.stop_all(timeout=5.0)
    assert seen == ["supervised-actor"]


# ── STDOUT purity (the critical stdio-transport invariant) ─────────────────


@pytest.fixture
def _stdio_protection_guard():
    """Save/restore the process-global stdio protection state around the test.

    ``protect_stdio_jsonrpc`` is a one-time, process-wide monkeypatch of
    ``builtins.print``/``warnings.showwarning`` by design (see its docstring) —
    tests must not leave that patch applied for the rest of the suite.
    """
    original_print = builtins.print
    original_showwarning = warnings.showwarning
    original_flag = server_factory._STDIO_PROTECTED
    yield
    builtins.print = original_print
    warnings.showwarning = original_showwarning
    server_factory._STDIO_PROTECTED = original_flag


def test_stdout_stays_pure_jsonrpc_while_a_co_service_logs_and_prints(
    _stdio_protection_guard,
):
    """The exact invariant this feature depends on: once
    ``protect_stdio_jsonrpc()`` runs (as ``kg_server.mcp_server()`` does before
    starting any co-service on stdio), a co-service thread that both ``print``s
    (bare AND with an explicit ``file=sys.stdout``) and logs a warning must
    produce ZERO bytes on stdout."""
    captured_stdout = io.StringIO()
    real_stdout = sys.stdout
    sys.stdout = captured_stdout
    try:
        server_factory.protect_stdio_jsonrpc()

        logged = threading.Event()

        def _leaky_co_service(stop_event: threading.Event) -> None:
            print("bare print from a co-service")
            print("explicit stdout print", file=sys.stdout)
            logging.getLogger("test.co-service-leak").warning("a warning")
            warnings.warn("a python warning", UserWarning, stacklevel=1)
            logged.set()
            stop_event.wait()

        supervisor = cosvc.CoServiceSupervisor()
        session = _verified_session()
        supervisor.start_service("leaky", _leaky_co_service, session)
        try:
            assert logged.wait(timeout=5.0)
            # Give the (now-redirected) writes a beat to land wherever they land.
            time.sleep(0.1)
        finally:
            supervisor.stop_all(timeout=5.0)

        assert captured_stdout.getvalue() == "", (
            "a co-service leaked onto stdout — this would corrupt the MCP "
            "JSON-RPC stream on the stdio transport"
        )
    finally:
        sys.stdout = real_stdout


# ── LIVE-PATH: starting the real entrypoint actually starts messaging ──────


def test_start_co_services_live_path_starts_messaging(monkeypatch):
    """Drive the REAL ``start_co_services`` (what ``kg_server.mcp_server()``
    calls) with a messaging-configured detection and assert the messaging
    co-service actually reaches the backend-connect step and keeps running —
    not merely that ``start_co_services``/``run_forever`` exist."""
    monkeypatch.setattr(
        messaging_daemon, "configured_platforms", lambda engine=None: ["fake"]
    )
    # Keep this test scoped to messaging — the host-daemon decision is covered
    # separately and touches the real engine singleton/lock.
    monkeypatch.setattr(cosvc, "host_daemon_needed", lambda: False)

    class _Cfg:
        enable_web_ui = False

    monkeypatch.setattr("agent_utilities.core.config.config", _Cfg())

    reached_get_backend = threading.Event()

    class _FakeBackend:
        id = "fake"
        is_connected = True

        async def register_commands(self, specs):
            return None

        async def listen(self):
            while True:  # pragma: no branch — cancelled on shutdown
                await asyncio.sleep(3600)
                yield {}

    async def _fake_get_backend(self, platform):
        reached_get_backend.set()
        return _FakeBackend()

    async def _fake_planner_handler(engine):
        async def _handler(event):
            return None

        return _handler

    monkeypatch.setattr(MessagingService, "get_backend", _fake_get_backend)
    monkeypatch.setattr(
        "agent_utilities.messaging.router.create_planner_handler",
        _fake_planner_handler,
    )

    session = _verified_session()
    engine = object()  # opaque — never dereferenced before this point in _serve

    supervisor = cosvc.start_co_services(session, engine)
    try:
        assert reached_get_backend.wait(timeout=10.0), (
            "start_co_services() did not actually start messaging serving "
            "(get_backend was never reached)"
        )
        assert "messaging" in supervisor.running()
    finally:
        supervisor.stop_all(timeout=10.0)
    assert supervisor.running() == ()
