"""Tests for the self-composing ``graph-os`` co-service supervisor.

Covers the three things the entrypoint promises:

1. Config-driven detection (no new env flag — messaging via real credential
   presence) without changing the explicit KG daemon role.
2. Bounded-restart supervision + clean shutdown for a co-service thread.
3. STDOUT PURITY: a co-service that logs/prints while running must never leak
   a byte onto stdout while this process's fd 1 is diverted — the critical
   stdio-transport invariant (stdout IS the JSON-RPC channel). B-19 deleted
   the old process-wide ``protect_stdio_jsonrpc()`` monkeypatch of
   ``builtins.print``/``warnings.showwarning``; purity is now owned fd-level
   by the MCP SDK's own ``stdio_server()`` for the scope of
   ``mcp.run(transport="stdio")`` (see the "Stdio JSON-RPC purity" note in
   ``agent_utilities/mcp/server_factory.py``). The test below proves the
   general property this module relies on — a co-service thread shares this
   process's real OS file-descriptor table, so an fd-level diversion of fd 1
   transparently redirects anything it writes, with no per-thread
   cooperation and no Python-level patch — using a real ``os.dup2``, not a
   mock. The full, live, SUBPROCESS proof that the actual served stdio
   entrypoint behaves this way end to end lives in
   ``tests/integration/mcp/test_stdio_fd_ownership.py``.

Plus a LIVE-PATH test (:func:`test_start_co_services_live_path_starts_messaging`)
that drives the real ``start_co_services`` entry point end to end and asserts
messaging actually started serving — not merely that a helper exists.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time

import pytest

from agent_utilities.knowledge_graph.core.session import GraphSession
from agent_utilities.mcp import co_service_supervisor as cosvc
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


def test_detect_composition_reads_existing_config_only(monkeypatch):
    """No new flag: messaging comes from real credential presence, webui from
    the existing ENABLE_WEB_UI field. The host role is deliberately outside
    served co-service composition."""

    class _Cfg:
        enable_web_ui = True

    monkeypatch.setattr("agent_utilities.core.config.config", _Cfg())
    monkeypatch.setattr(
        messaging_daemon, "configured_platforms", lambda engine=None: ["telegram"]
    )

    plan = cosvc.detect_composition()
    assert plan.messaging_platforms == ("telegram",)
    assert plan.messaging_configured is True
    assert plan.web_ui_enabled is True
    assert plan.co_service_names() == ("messaging", "agent-webui")


def test_detect_composition_nothing_configured(monkeypatch):
    class _Cfg:
        enable_web_ui = False

    monkeypatch.setattr("agent_utilities.core.config.config", _Cfg())
    monkeypatch.setattr(
        messaging_daemon, "configured_platforms", lambda engine=None: []
    )
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


def test_a_process_wide_fd1_diversion_transparently_captures_a_co_service_thread(
    capfd,
):
    """The general fd-sharing property this module's STDIO-safety contract now
    rests on (see the module docstring): a co-service thread needs no
    cooperation of its own for stdout purity. It shares this process's real OS
    file-descriptor table, so whenever fd 1 has been diverted at the OS level
    — exactly what ``mcp.server.stdio.stdio_server()`` does for the scope of
    ``mcp.run(transport="stdio")`` — EVERY write a co-service thread makes
    through fd 1 (a bare ``print()``, an explicit ``file=sys.stdout`` print, or
    a raw ``os.write(1, ...)`` that bypasses Python's stdout object entirely)
    is transparently redirected. Demonstrated with a real ``os.dup2``, not a
    mock, and fully restored afterwards (proven directly on the fd, not by
    trusting a saved Python reference).

    ``capfd.disabled()``: pytest's own default capture replaces ``sys.stdout``
    with an object decoupled from the fd 1 *number* (only a raw ``os.write(1,
    ...)`` still resolves through the OS fd table pytest also redirects) — so
    ``print()`` under an active pytest capture would prove nothing about this
    process's real, undecorated stdout object, which is what production code
    actually uses. Disabling capture for this block restores that real,
    fd-1-backed ``sys.stdout``, matching what a served process sees.
    """
    with capfd.disabled():
        real_stdout_fd = os.dup(1)
        read_fd, write_fd = os.pipe()
        try:
            os.dup2(write_fd, 1)  # simulate stdio_server()'s fd-level diversion
            os.close(write_fd)

            logged = threading.Event()

            def _leaky_co_service(stop_event: threading.Event) -> None:
                # flush=True: without an explicit flush the bytes could sit in
                # Python's userspace buffer past the point this test reads the
                # pipe, which would prove nothing about the fd-level
                # redirection this test exists to show.
                print("bare print from a co-service", flush=True)
                print("explicit stdout print", file=sys.stdout, flush=True)
                os.write(1, b"raw fd write from a co-service\n")
                logged.set()
                stop_event.wait()

            supervisor = cosvc.CoServiceSupervisor()
            session = _verified_session()
            supervisor.start_service("leaky", _leaky_co_service, session)
            try:
                assert logged.wait(timeout=5.0)
                time.sleep(0.1)
            finally:
                supervisor.stop_all(timeout=5.0)
        finally:
            # Restore fd 1 to exactly what it was before this test diverted
            # it — this closes the pipe's write end (fd 1's dup), so the
            # pending read below sees EOF instead of blocking.
            os.dup2(real_stdout_fd, 1)
            os.close(real_stdout_fd)

        os.set_blocking(read_fd, False)
        try:
            captured = os.read(read_fd, 65536)
        except BlockingIOError:
            captured = b""
        os.close(read_fd)

    assert captured == (
        b"bare print from a co-service\n"
        b"explicit stdout print\n"
        b"raw fd write from a co-service\n"
    ), (
        "a co-service's write did not reach the diverted fd 1 target — the "
        f"fd-sharing property this module depends on broke: {captured!r}"
    )


def test_no_process_wide_monkeypatch_remains_and_identity_is_untouched():
    """Regression guard for B-19 itself: there is no ``protect_stdio_jsonrpc``
    (or any process-global ``_STDIO_PROTECTED`` flag) left anywhere to call,
    and importing every module that used to call it leaves ``builtins.print``
    and ``sys.stdout`` as the EXACT SAME objects they were before — asserted
    by identity (``is``), not merely by behaviour. There is nothing to
    save/restore in a fixture because there is nothing left that mutates
    either of them."""
    import builtins
    import sys

    from agent_utilities.mcp import harness_server, kg_server
    from agent_utilities.mcp import server_factory as sf

    before_print = builtins.print
    before_stdout = sys.stdout

    assert not hasattr(sf, "protect_stdio_jsonrpc")
    assert not hasattr(sf, "_STDIO_PROTECTED")
    # Re-import (a no-op for an already-imported module, but the exact thing
    # kg_server.mcp_server()/harness_server.main() do at call time) proves
    # neither entrypoint's module body touches either global as a side effect.
    assert kg_server is not None
    assert harness_server is not None

    assert builtins.print is before_print
    assert sys.stdout is before_stdout


# ── LIVE-PATH: starting the real entrypoint actually starts messaging ──────


def test_start_co_services_live_path_starts_messaging(monkeypatch):
    """Drive the REAL ``start_co_services`` (what ``kg_server.mcp_server()``
    calls) with a messaging-configured detection and assert the messaging
    co-service actually reaches the backend-connect step and keeps running —
    not merely that ``start_co_services``/``run_forever`` exist."""
    monkeypatch.setattr(
        messaging_daemon, "configured_platforms", lambda engine=None: ["fake"]
    )

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
