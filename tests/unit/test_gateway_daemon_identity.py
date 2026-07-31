"""Standalone GraphOS host daemon process-identity regression tests."""

from __future__ import annotations

import contextlib
from unittest.mock import MagicMock

import pytest

from agent_utilities.gateway import daemon


def test_mint_process_identity_is_bounded_and_verified(monkeypatch):
    from agent_utilities.models.company_brain import ActorType
    from agent_utilities.security.brain_context import ActorContext

    calls: list[str] = []
    actor = ActorContext(
        actor_id="host",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("kg:admin",),
        tenant_id="service",
        authenticated=True,
        groups=(),
        credential_expires_at=4_000_000_000,
    )
    session = MagicMock(actor=actor)

    def _mint_session(minted_actor):
        calls.append("session")
        session.actor = minted_actor
        return session

    monkeypatch.setattr(
        "agent_utilities.security.request_identity.acquire_process_identity_token",
        lambda config: calls.append("acquire") or "token",
    )
    monkeypatch.setattr(
        "agent_utilities.security.request_identity.mint_actor_from_token_sync",
        lambda token: calls.append("actor") or actor,
    )
    monkeypatch.setattr(
        "agent_utilities.security.request_identity.mint_graph_session",
        _mint_session,
    )

    assert daemon.mint_process_identity() is session
    assert calls == ["acquire", "actor", "session"]
    assert session.actor.credential_lease.expires_at == 4_000_000_000
    session.ensure_authority_current.assert_called_once_with(minimum_ttl_seconds=30)
    session.engine_verified_context.assert_called_once_with()


def test_refresh_process_identity_rejects_authority_change(monkeypatch):
    from agent_utilities.knowledge_graph.core.session import SessionExpiredError
    from agent_utilities.security.brain_context import CredentialLease

    actor = MagicMock(
        actor_id="host",
        actor_type="service",
        roles=("kg:admin",),
        tenant_id="service",
        authenticated=True,
        groups=(),
        credential_lease=CredentialLease(1),
    )
    session = MagicMock(actor=actor)
    session.ensure_authority_current.side_effect = SessionExpiredError("renew")
    changed = MagicMock(
        actor_id="other-host",
        actor_type=actor.actor_type,
        roles=actor.roles,
        tenant_id=actor.tenant_id,
        authenticated=True,
        groups=actor.groups,
        credential_expires_at=4_000_000_000,
    )
    monkeypatch.setattr(
        "agent_utilities.security.request_identity.acquire_process_identity_token",
        lambda config: "token",
    )
    monkeypatch.setattr(
        "agent_utilities.security.request_identity.mint_actor_from_token_sync",
        lambda token: changed,
    )

    with pytest.raises(RuntimeError, match="changed during renewal"):
        daemon.refresh_process_identity(session)


def test_main_binds_identity_before_engine_start(monkeypatch):
    order: list[str] = []
    session = MagicMock()
    session.actor = MagicMock()

    monkeypatch.setattr(
        daemon,
        "mint_process_identity",
        lambda: order.append("mint") or session,
    )

    def _start():
        order.append("start")
        daemon._engine = object()
        return daemon._engine

    monkeypatch.setattr(daemon, "start_host_daemon", _start)
    monkeypatch.setattr(
        daemon,
        "stop_host_daemon",
        lambda: order.append("stop"),
    )
    # This test is about identity-binding ORDER, not the D-OG-4 metrics
    # listener; stub it out so it never touches a real socket/thread here
    # (a real prometheus_client server thread would also pick up this test's
    # ``threading.Event`` monkeypatch below, since ``daemon.threading`` IS
    # the process-wide ``threading`` module, not a per-file copy).
    monkeypatch.setattr(
        daemon,
        "start_daemon_metrics_listener",
        lambda: order.append("metrics") or False,
    )

    entered: list[str] = []

    @contextlib.contextmanager
    def _use_actor(actor):
        entered.append("actor")
        yield actor

    @contextlib.contextmanager
    def _use_session(bound_session):
        entered.append("session")
        yield bound_session

    monkeypatch.setattr(
        "agent_utilities.security.brain_context.use_actor",
        _use_actor,
    )
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.session.use_session",
        _use_session,
    )

    class _Stop:
        def wait(self, timeout):
            return True

        def set(self):
            return None

    monkeypatch.setattr(daemon.threading, "Event", _Stop)
    monkeypatch.setattr("signal.signal", lambda *args: None)

    daemon.main()

    assert order == ["mint", "start", "metrics", "stop"]
    assert entered == ["actor", "session"]
