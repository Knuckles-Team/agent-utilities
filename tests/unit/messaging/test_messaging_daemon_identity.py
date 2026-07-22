"""Regression: the messaging daemon must mint a verified process identity.

Previously ``messaging/daemon.py`` called
``IntelligenceGraphEngine.get_or_create()`` with no process identity wired at
all — unlike ``kg_server.py`` (``_mint_process_session``) and
``knowledge_graph/ingest_worker.py`` (``acquire_process_identity_token`` →
``mint_actor_from_token_sync`` → ``mint_graph_session`` →
``engine_verified_context()``), it never acquired a token, minted an actor, or
verified a session before constructing the engine — so in any real deployment
(every engine construction requires a verified :class:`GraphSession` outside
``AGENT_UTILITIES_TESTING``) the messaging daemon could never actually reach
the engine. This proves the fix copies the exact same established pattern and
that ``main()`` actually uses it (not just that the helper exists).
"""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import MagicMock

from agent_utilities.messaging import daemon as messaging_daemon


def test_mint_process_identity_follows_the_established_pattern(monkeypatch):
    """``mint_process_identity`` must call acquire → mint_actor → mint_session →
    verify, in that order — the exact chain ``ingest_worker.py``/``kg_server.py``
    use, never a bare ``IntelligenceGraphEngine()``/``get_or_create()``."""
    calls: list[str] = []

    fake_token = "token-xyz"
    fake_actor = MagicMock(name="actor")
    fake_session = MagicMock(name="session")

    def _acquire(config):
        calls.append("acquire_process_identity_token")
        return fake_token

    def _mint_actor(token):
        assert token == fake_token
        calls.append("mint_actor_from_token_sync")
        return fake_actor

    def _mint_session(actor):
        assert actor is fake_actor
        calls.append("mint_graph_session")
        return fake_session

    def _verify(self=None):
        calls.append("engine_verified_context")

    fake_session.engine_verified_context = _verify

    monkeypatch.setattr(
        "agent_utilities.security.request_identity.acquire_process_identity_token",
        _acquire,
    )
    monkeypatch.setattr(
        "agent_utilities.security.request_identity.mint_actor_from_token_sync",
        _mint_actor,
    )
    monkeypatch.setattr(
        "agent_utilities.security.request_identity.mint_graph_session",
        _mint_session,
    )

    result = messaging_daemon.mint_process_identity()

    assert result is fake_session
    assert calls == [
        "acquire_process_identity_token",
        "mint_actor_from_token_sync",
        "mint_graph_session",
        "engine_verified_context",
    ]


def test_main_never_constructs_the_engine_without_minting_identity_first(
    monkeypatch,
):
    """LIVE-PATH: run the real ``main()`` body and assert identity is minted
    BEFORE the engine is touched — not merely that ``mint_process_identity``
    exists somewhere in the module."""
    order: list[str] = []

    fake_session = MagicMock(name="session")
    fake_session.actor = MagicMock(name="actor")

    def _mint_identity():
        order.append("mint_process_identity")
        return fake_session

    class _FakeEngine:
        @classmethod
        def get_or_create(cls):
            order.append("engine_get_or_create")
            return MagicMock(name="engine")

    monkeypatch.setattr(messaging_daemon, "mint_process_identity", _mint_identity)
    monkeypatch.setattr(messaging_daemon, "_validate_fleet_auth", lambda: None)
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.engine.IntelligenceGraphEngine",
        _FakeEngine,
    )
    # No platforms configured -> main() returns immediately after the engine
    # check, which is all this test needs to prove ordering.
    monkeypatch.setattr(
        messaging_daemon, "configured_platforms", lambda engine=None: []
    )

    entered: list[str] = []

    import contextlib

    @contextlib.contextmanager
    def _use_actor(actor):
        entered.append("use_actor")
        yield actor

    @contextlib.contextmanager
    def _use_session(session):
        entered.append("use_session")
        yield session

    monkeypatch.setattr("agent_utilities.security.brain_context.use_actor", _use_actor)
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.session.use_session", _use_session
    )

    messaging_daemon.main()

    assert order == ["mint_process_identity", "engine_get_or_create"]
    assert entered == ["use_actor", "use_session"]


def test_run_forever_propagates_actor_and_session_into_the_serve_task(monkeypatch):
    """LIVE-PATH: the actor/session set around ``run_forever`` must be visible
    INSIDE the async ``_serve`` task it schedules (contextvars propagate across
    the thread's own event loop) — proving the identity fix actually reaches
    the code that talks to the engine, not just the outer synchronous call."""
    from agent_utilities.knowledge_graph.core.session import GraphSession, use_session
    from agent_utilities.models.company_brain import ActorType
    from agent_utilities.security.brain_context import (
        ActorContext,
        current_actor,
        use_actor,
    )

    actor = ActorContext(
        actor_id="messaging-test",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("system",),
        tenant_id="test-tenant",
        authenticated=True,
    )
    session = GraphSession(
        actor=actor,
        tenant=actor.tenant_id,
        scopes=frozenset({"kg:read"}),
        policy_version="current",
        audience="graph-runtime",
    )

    seen_actor_id: list[str] = []
    stop_event = threading.Event()

    async def _fake_serve(engine, platforms):
        # Proves the actor set by the caller's `with use_actor(...)` block is
        # visible from INSIDE the scheduled asyncio task, then asks
        # ``run_forever`` to stop — deterministic (no reliance on thread
        # scheduling order): the assertion is guaranteed to have run before
        # shutdown, because shutdown is triggered BY this coroutine.
        seen_actor_id.append(current_actor().actor_id)
        stop_event.set()
        # Keep the task alive (like the real ``_serve``, which blocks on
        # listener tasks) until the stop watcher tears the loop down.
        await asyncio.sleep(3600)

    monkeypatch.setattr(messaging_daemon, "_serve", _fake_serve)

    with use_actor(actor), use_session(session):
        messaging_daemon.run_forever(
            engine=object(), platforms=["fake"], stop_event=stop_event
        )

    assert seen_actor_id == ["messaging-test"]


def test_configured_platforms_is_a_pure_config_read(monkeypatch):
    """``configured_platforms`` must not require an engine (composition
    detection runs before any engine/process identity exists)."""

    class _Svc:
        def configured_platforms(self):
            return ["telegram"]

    monkeypatch.setattr(
        "agent_utilities.messaging.service.MessagingService.instance",
        classmethod(lambda cls, engine=None: _Svc()),
    )
    assert messaging_daemon.configured_platforms() == ["telegram"]
    assert messaging_daemon.configured_platforms(engine=None) == ["telegram"]
