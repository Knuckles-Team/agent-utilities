"""``WorkItemLeaseGuard``'s background renewal thread must inherit the ambient
``GraphSession`` (CONCEPT:AU-P0-1 session propagation).

A bare ``threading.Thread`` does NOT inherit :mod:`contextvars` the way
``asyncio.Task`` does. ``WorkItemLeaseGuard.start()`` spawns
``_heartbeat_loop`` (via ``require_current`` -> ``_work_item_fence_still_valid``
-> ``work_item.heartbeat``) on exactly such a bare thread, so the renewal
thread used to run with an EMPTY context -- losing the ambient ``GraphSession``
the caller entered via ``use_session()`` before starting the guard, even
though the guard's initial synchronous ``require_current()`` call (still on
the calling thread) worked fine. This is the same bug shape (and fix) already
landed for the messaging intake lease renewal thread; see
``tests/unit/messaging/test_intake_lease.py::test_renewal_worker_inherits_ambient_graph_session``
and ``tests/unit/orchestration/test_agent_activation.py::test_start_heartbeat_worker_inherits_ambient_graph_session``.
"""

from __future__ import annotations

import threading

from agent_utilities.knowledge_graph.core.session import (
    GraphSession,
    SessionRequiredError,
    current_session,
    use_session,
)
from agent_utilities.orchestration import agent_dispatch_worker as worker
from agent_utilities.orchestration import work_item as wi
from agent_utilities.security.actor_identity import ActorType
from agent_utilities.security.brain_context import ActorContext


def _verified_session() -> GraphSession:
    actor = ActorContext(
        actor_id="lease-guard-test",
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


def test_lease_guard_heartbeat_thread_inherits_ambient_graph_session(monkeypatch):
    """The background renewal thread's ``heartbeat()`` call must see the SAME
    session the calling thread entered via ``use_session()``, not just the
    guard's synchronous ``require_current()`` call made on ``start()`` itself
    (that one already runs on the calling thread and would pass even without
    the fix) -- so only calls observed from a non-main thread are asserted.
    """
    session = _verified_session()
    observed_from_worker_thread: list[GraphSession] = []
    worker_heartbeat_called = threading.Event()

    def _heartbeat(engine, item_id, claim, *, lease_ttl_s):
        ambient = current_session()
        if ambient is None:
            raise SessionRequiredError(
                "no ambient GraphSession reached the lease-guard heartbeat thread"
            )
        if threading.current_thread() is not threading.main_thread():
            observed_from_worker_thread.append(ambient)
            worker_heartbeat_called.set()
        return True

    monkeypatch.setattr(wi, "heartbeat", _heartbeat)

    claim = {"work_item_id": "wi-lease-guard-test"}
    with use_session(session):
        guard = worker.WorkItemLeaseGuard(
            object(),
            "wi-lease-guard-test",
            claim,
            lease_ttl_s=0.05,
            heartbeat_interval_s=0.01,
        )
        guard.start()
    try:
        assert worker_heartbeat_called.wait(timeout=2.0)
    finally:
        guard.close()

    assert observed_from_worker_thread
    assert all(seen is session for seen in observed_from_worker_thread)
