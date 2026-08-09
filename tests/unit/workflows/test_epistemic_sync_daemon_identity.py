"""BUG-061 regression: ``start_epistemic_sync_daemon`` must bind a verified actor.

``start_epistemic_sync_daemon`` used to spawn a bare ``threading.Thread`` (the
BUG-055 shape): ``contextvars.ContextVar``s do not cross a thread boundary, so
every graph write the daemon's recurring ``run_sync_cycle()`` makes would run
with no bound actor the moment anything actually called this entrypoint (it
has zero callers repo-wide today -- the live ``epistemic_sync`` MCP action
calls ``EpistemicSyncWorkflow.run_sync_cycle()`` directly instead, never this
daemon path).

This proves, with a known-bad input at each layer, the same property
``test_bug033_actor_binding.py`` proves for its three write paths:

1. With a verified session/actor ambient at call time, the spawned daemon
   thread's sync-cycle work observes that SAME real, authenticated actor.
2. With NOTHING ambient (a bare cron/script invocation), the call raises
   BEFORE any thread is spawned -- it must never silently start an
   unauthorized background worker.

Revert the fix (restore the bare ``threading.Thread(...)`` in
``start_epistemic_sync_daemon``) and both tests fail: test 1 observes no
captured actor (``IdentityRequiredError`` inside the thread, swallowed by
nothing catching it -- the stub simply never records), and test 2 no longer
raises -- the thread is spawned regardless of identity.
"""

from __future__ import annotations

import contextvars
import threading

import pytest


def _stub_workflow_class(monkeypatch, captured: list) -> None:
    """Replace ``EpistemicSyncWorkflow`` with a stub that records the actor
    ambient when its (one-shot) sync cycle would run, and returns
    immediately instead of looping forever -- so the test thread joins fast
    and never touches a real engine/backend/SPARQL endpoint."""
    import agent_utilities.workflows.epistemic_sync as es
    from agent_utilities.security.brain_context import current_actor

    class _StubWorkflow:
        def __init__(self, *_a, **_kw) -> None:
            pass

        async def run_forever(self, interval_seconds: int = 3600) -> None:
            captured.append(current_actor())

    monkeypatch.setattr(es, "EpistemicSyncWorkflow", _StubWorkflow)


def test_daemon_thread_binds_the_caller_ambient_session(monkeypatch) -> None:
    """A verified session/actor ambient at call time is captured BEFORE the
    thread boundary and re-bound inside the daemon thread, so the thread's
    own sync-cycle work observes that SAME real, authenticated actor --
    never nothing and never a different/synthesized identity."""
    from agent_utilities.knowledge_graph.core.session import GraphSession, use_session
    from agent_utilities.models.company_brain import ActorType
    from agent_utilities.security.brain_context import ActorContext, use_actor
    import agent_utilities.workflows.epistemic_sync as es

    captured: list = []
    _stub_workflow_class(monkeypatch, captured)

    actor = ActorContext(
        actor_id="system:epistemic-sync-caller",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=(),
        tenant_id="tenant-a",
        authenticated=True,
    )
    session = GraphSession(
        actor=actor,
        tenant="tenant-a",
        scopes=frozenset({"kg:write"}),
        graph="tenant-a-graph",
        policy_version="v1",
        audience="test-audience",
    )

    def isolated() -> None:
        with use_actor(actor), use_session(session):
            es.start_epistemic_sync_daemon()

    contextvars.Context().run(isolated)

    spawned = [t for t in threading.enumerate() if t.name == "EpistemicSyncWorker"]
    assert len(spawned) == 1
    spawned[0].join(timeout=5)
    assert not spawned[0].is_alive()

    assert len(captured) == 1
    assert captured[0].authenticated is True
    assert captured[0].actor_id == "system:epistemic-sync-caller"


def test_daemon_fails_rather_than_spawn_an_unauthorized_thread() -> None:
    """No ambient session/actor at all (a bare cron/script invocation, the
    reproduced BUG-055-shaped gap) must raise BEFORE any thread is spawned
    -- never silently start an unauthorized background worker."""
    import agent_utilities.workflows.epistemic_sync as es
    from agent_utilities.knowledge_graph.core.session import SessionRequiredError

    def isolated() -> None:
        with pytest.raises(SessionRequiredError):
            es.start_epistemic_sync_daemon()

    contextvars.Context().run(isolated)

    spawned = [t for t in threading.enumerate() if t.name == "EpistemicSyncWorker"]
    assert spawned == []
