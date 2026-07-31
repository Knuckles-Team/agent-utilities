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

    assert order == ["mint", "start", "stop"]
    assert entered == ["actor", "session"]


class _StubEngine:
    """Minimal engine stand-in: has no ``start_task_workers``, so the pool
    branch is a no-op and only the two sink installs can fail."""


def _drive_host_daemon(monkeypatch, *, failing_import: str, boom: Exception):
    """Run ``start_host_daemon`` with one best-effort sink install raising.

    Returns the ``logging`` records emitted by the daemon module.
    """
    monkeypatch.setattr(daemon, "_engine", None, raising=False)
    monkeypatch.setattr(
        "agent_utilities.security.run_token.require_token_secret",
        lambda: None,
    )
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.engine.IntelligenceGraphEngine.get_or_create",
        classmethod(lambda cls, **kw: _StubEngine()),
    )

    def _raise(*args, **kwargs):
        raise boom

    # Both installs are imported INSIDE the try block, so patching the source
    # module attribute is what the daemon actually resolves.
    monkeypatch.setattr(
        "agent_utilities.harness.tracing.set_kg_trace_sink",
        _raise if failing_import == "trace" else (lambda sink: None),
    )
    monkeypatch.setattr(
        "agent_utilities.harness.trace_backend.KGTraceBackend",
        (lambda **kw: object()),
    )
    monkeypatch.setattr(
        "agent_utilities.security.error_surface.register_detail_persistence_sink",
        _raise if failing_import == "detail" else (lambda sink: None),
    )
    monkeypatch.setattr(
        "agent_utilities.observability.error_detail_sink.GraphErrorDetailSink",
        (lambda **kw: object()),
    )
    return daemon


@pytest.mark.parametrize(
    ("failing_import", "fragment"),
    [
        ("trace", "KG trace sink install failed"),
        ("detail", "durable error-detail sink install failed"),
    ],
)
def test_sink_install_failure_logs_the_real_cause_not_just_its_type(
    monkeypatch, caplog, failing_import, fragment
):
    """A best-effort sink install that fails must log the exception's MESSAGE.

    Falsifying guard for the reconciliation-gate-2 fix: these handlers used to
    log only ``type(exc).__name__`` ("RuntimeError"), which told an operator a
    sink was missing but never *why* — the exact cause-dropping shape
    ``scripts/check_swallowed_errors.py`` classifies as ``log_type_name_only``.
    The distinctive message below is absent from the old formatting, so this
    test fails against it and passes against a cause-preserving log.
    """
    # NB: no ``host:port`` substring — the observability layer scrubs those to
    # ``<endpoint>``, which would mask the very thing this test asserts.
    marker = "signing bundle rejected by the custody policy"
    _drive_host_daemon(
        monkeypatch, failing_import=failing_import, boom=RuntimeError(marker)
    )

    with caplog.at_level("WARNING", logger=daemon.logger.name):
        engine = daemon.start_host_daemon()

    assert isinstance(engine, _StubEngine), "daemon must still start (best-effort)"
    matching = [r for r in caplog.records if fragment in r.getMessage()]
    assert matching, f"no warning matched {fragment!r}: {caplog.text}"
    record = matching[0]
    assert marker in record.getMessage(), (
        "the real cause was dropped — only its type survived: "
        f"{record.getMessage()!r}"
    )
    # NB: ``exc_info`` is deliberately NOT asserted. ``core/log_privacy.py``'s
    # record factory unconditionally nulls ``exc_info``/``exc_text``/``stack_info``
    # on every ``agent_utilities.*`` record (tracebacks embed host filesystem
    # paths), so ``exc_info=exc`` is a no-op in this package — as
    # ``scripts/check_swallowed_errors.py`` itself documents. The interpolated
    # message is therefore the ONLY channel that carries the cause, which is
    # exactly what is asserted above.
