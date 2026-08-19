"""Focused tests for the durable messaging intake lease boundary."""

from __future__ import annotations

import threading

from agent_utilities.knowledge_graph.core.session import GraphSession
from agent_utilities.messaging import daemon as messaging_daemon
from agent_utilities.messaging import intake_lease
from agent_utilities.models.company_brain import ActorType
from agent_utilities.orchestration.work_item import NativeWorkItemRequired
from agent_utilities.security.brain_context import ActorContext


def _verified_session() -> GraphSession:
    actor = ActorContext(
        actor_id="intake-lease-test",
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


def test_two_contenders_cannot_both_enter_intake(monkeypatch):
    """The native claim result, not process-local state, selects one owner."""
    monkeypatch.setattr(intake_lease, "_identity_digest", lambda platform: "identity")
    rows: dict[str, dict[str, object]] = {}
    submit_calls: list[dict[str, object]] = []
    claim_calls: list[str] = []
    lock = threading.Lock()
    owner: str | None = None

    def _submit(engine, **kwargs):
        submit_calls.append(kwargs)
        item_id = str(kwargs["work_item_id"])
        rows.setdefault(
            item_id,
            {
                "kind": kwargs["kind"],
                "queue": kwargs["queue"],
                "tenant": kwargs["tenant"],
                "metadata": kwargs["metadata"],
            },
        )
        return item_id, len(submit_calls) == 1

    def _get_work_item(engine, item_id):
        return rows.get(item_id)

    def _claim(engine, item_id, *, token, lease_ttl_s):
        nonlocal owner
        with lock:
            claim_calls.append(token)
            if owner is not None:
                return None
            owner = token
            return {
                "_native": True,
                "tenant": "test-tenant",
                "lease_owner": token,
                "lease_epoch": 1,
                "fencing_token": 1,
            }

    monkeypatch.setattr(intake_lease, "submit_work_item_atomic", _submit)
    monkeypatch.setattr(intake_lease, "get_work_item", _get_work_item)
    monkeypatch.setattr(intake_lease, "claim_specific", _claim)

    results: list[intake_lease.IntakeLease | None] = [None, None]

    def _contend(index: int) -> None:
        results[index] = intake_lease.acquire_intake_lease(
            object(), "telegram", _verified_session()
        )

    threads = [threading.Thread(target=_contend, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)

    assert sum(result is not None for result in results) == 1
    assert len(claim_calls) == 2
    assert len(submit_calls) == 2
    assert all("token" not in call["metadata"] for call in submit_calls)


def test_two_entrypoints_cannot_both_start_the_same_poller(monkeypatch):
    """The public standalone/embedded boundary admits one native owner."""
    monkeypatch.setattr(intake_lease, "_identity_digest", lambda platform: "identity")
    rows: dict[str, dict[str, object]] = {}
    owner: str | None = None
    lock = threading.Lock()

    def _submit(engine, **kwargs):
        item_id = str(kwargs["work_item_id"])
        rows.setdefault(
            item_id,
            {
                "kind": kwargs["kind"],
                "queue": kwargs["queue"],
                "tenant": kwargs["tenant"],
                "metadata": kwargs["metadata"],
            },
        )
        return item_id, True

    def _get_work_item(engine, item_id):
        return rows.get(item_id)

    def _claim(engine, item_id, *, token, lease_ttl_s):
        nonlocal owner
        with lock:
            if owner is not None:
                return None
            owner = token
            return {
                "_native": True,
                "tenant": "test-tenant",
                "lease_owner": token,
                "lease_epoch": 1,
                "fencing_token": 1,
            }

    monkeypatch.setattr(intake_lease, "submit_work_item_atomic", _submit)
    monkeypatch.setattr(intake_lease, "get_work_item", _get_work_item)
    monkeypatch.setattr(intake_lease, "claim_specific", _claim)
    monkeypatch.setattr(intake_lease, "defer_work_item", lambda *args, **kwargs: True)

    started = threading.Event()
    first_stop = threading.Event()
    polling_entries: list[tuple[str, ...]] = []

    def _poll(engine, platforms, stop_event):
        polling_entries.append(tuple(platforms))
        started.set()
        stop_event.wait(timeout=2.0)

    monkeypatch.setattr(messaging_daemon, "_run_poll_loop", _poll)
    session = _verified_session()
    first = threading.Thread(
        target=messaging_daemon.run_forever,
        args=(object(), ["telegram"], first_stop),
        kwargs={"session": session, "intake_intent": True},
    )
    first.start()
    assert started.wait(timeout=2.0)

    second_stop = threading.Event()
    second = threading.Thread(
        target=messaging_daemon.run_forever,
        args=(object(), ["telegram"], second_stop),
        kwargs={"session": session, "intake_intent": True},
    )
    second.start()
    second.join(timeout=2.0)

    assert not second.is_alive()
    assert second_stop.is_set()
    assert polling_entries == [("telegram",)]

    first_stop.set()
    first.join(timeout=2.0)
    assert not first.is_alive()


def test_missing_native_lease_capability_fails_closed(monkeypatch):
    monkeypatch.setattr(intake_lease, "_identity_digest", lambda platform: "identity")

    def _unsupported(engine, **kwargs):
        raise NativeWorkItemRequired("claim_work_item unavailable")

    monkeypatch.setattr(intake_lease, "submit_work_item_atomic", _unsupported)
    assert (
        intake_lease.acquire_intake_leases(object(), ["telegram"], _verified_session())
        == ()
    )


def test_lost_renewal_stops_the_listener(monkeypatch):
    lease = intake_lease.IntakeLease(
        platform="telegram",
        item_id="workitem:messaging-intake:test",
        claim={
            "_native": True,
            "tenant": "test-tenant",
            "lease_owner": "owner",
            "lease_epoch": 1,
            "fencing_token": 1,
        },
        lease_ttl_s=0.03,
    )
    served = threading.Event()
    monkeypatch.setattr(intake_lease, "heartbeat", lambda *args, **kwargs: False)
    monkeypatch.setattr(intake_lease, "defer_work_item", lambda *args, **kwargs: True)

    def _serve(platforms, stop_event):
        assert platforms == ["telegram"]
        served.set()
        assert stop_event.wait(timeout=2.0)

    intake_lease.run_with_intake_leases(object(), (lease,), threading.Event(), _serve)
    assert served.is_set()


def test_lost_renewal_stops_the_actual_public_poll_loop(monkeypatch):
    """A lease fence loss reaches and stops the shared polling entrypoint."""
    lease = intake_lease.IntakeLease(
        platform="telegram",
        item_id="workitem:messaging-intake:test",
        claim={
            "_native": True,
            "tenant": "test-tenant",
            "lease_owner": "owner",
            "lease_epoch": 1,
            "fencing_token": 1,
        },
        lease_ttl_s=0.03,
    )
    started = threading.Event()
    stopped = threading.Event()
    monkeypatch.setattr(
        intake_lease,
        "acquire_intake_leases",
        lambda engine, platforms, session, **kwargs: (lease,),
    )
    monkeypatch.setattr(intake_lease, "heartbeat", lambda *args, **kwargs: False)
    monkeypatch.setattr(intake_lease, "defer_work_item", lambda *args, **kwargs: True)

    def _poll(engine, platforms, stop_event):
        started.set()
        assert stop_event.wait(timeout=2.0)
        stopped.set()

    monkeypatch.setattr(messaging_daemon, "_run_poll_loop", _poll)
    messaging_daemon.run_forever(
        object(),
        ["telegram"],
        threading.Event(),
        session=_verified_session(),
        intake_intent=True,
    )

    assert started.is_set()
    assert stopped.is_set()
