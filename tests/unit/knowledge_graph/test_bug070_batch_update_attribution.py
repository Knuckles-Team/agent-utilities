"""BUG-070 regression tests: caller/session attribution for ``lifecycle.batch_update``.

BUG-064: a caller-supplied Cypher payload reached ``lifecycle.batch_update``
and mutated 45,478 of 47,465 live nodes with NO caller or session identity
recorded at any layer -- not the engine's own slow-query log
(``op=lifecycle.batch_update ... duration=57.88s``), not the sidecar. Three
separate investigations found nothing because nothing was ever committed --
the mutation arrived as a runtime payload, almost certainly via a queued
WorkItem's claimed executor, not code.

This file proves, with a known-bad input:

1. A ``lifecycle.batch_update`` call made with NO bound ``GraphSession`` and
   NO bound WorkItem-execution context is LOUD (an ``ERROR``-level
   "UNATTRIBUTED" log line), not silently indistinguishable from any other
   call -- the audit counterpart to the write-chokepoint fail-closed
   governance stamping already enforced upstream (BUG-033/058/062).
2. A ``lifecycle.batch_update`` call made through the WorkItem-claimed path
   (``bind_work_item_context``, the seam ``execute_work_item_turn`` /
   ``execute_agent_task_turn`` bind around their pluggable ``executor(...)``
   call) emits the claimed WorkItem's id/agent/lease/capability in its log
   record -- answering "who ran this" from logs alone.
3. A call made under a bound ``GraphSession`` (an authenticated actor, no
   WorkItem in flight -- e.g. a direct ``engine_lifecycle_batch_update`` MCP
   call) emits the actor identity instead of the loud/unattributed line.

Revert the fix (drop ``_log_mutation_attribution``/``_mutation_attribution``
and their call sites in ``engine_breaker._guard``, and/or drop
``work_item_context.bind_work_item_context``'s use in
``agent_dispatch_worker``) and every assertion in this file that inspects
``caplog`` for attribution content fails; the "UNATTRIBUTED" test instead
finds no distinguishing log line at all for the actor-less/work-item-less
call, reproducing BUG-064's exact blind spot.
"""

from __future__ import annotations

import logging

import pytest

from agent_utilities.knowledge_graph.core import engine_breaker
from agent_utilities.knowledge_graph.core.engine_breaker import (
    CircuitBreaker,
    reset_breakers,
    wrap_client_with_breaker,
)
from agent_utilities.knowledge_graph.core.session import (
    GraphSession,
    reset_session,
    set_session,
    suspend_session,
)
from agent_utilities.orchestration.work_item_context import bind_work_item_context
from agent_utilities.security.brain_context import ActorContext, ActorType

_LOGGER_NAME = "agent_utilities.knowledge_graph.core.engine_breaker"


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_breakers()
    yield
    reset_breakers()


class _FakeLifecycle:
    """Stand-in for the generated client's ``lifecycle`` sub-client."""

    def __init__(self) -> None:
        self.calls: list[list[dict]] = []

    def batch_update(self, operations):
        self.calls.append(operations)
        return {"applied": len(operations)}


class _FakeClient:
    def __init__(self) -> None:
        self.lifecycle = _FakeLifecycle()


def _proxy() -> tuple[_FakeClient, object]:
    client = _FakeClient()
    breaker = CircuitBreaker("bug070-ep", threshold=5, cooldown=10)
    return client, wrap_client_with_breaker(client, breaker)


def _verified_session(
    actor_id: str = "agent-42", tenant: str = "tenant-a"
) -> GraphSession:
    return GraphSession(
        actor=ActorContext(
            actor_id=actor_id,
            actor_type=ActorType.AUTOMATED_SERVICE,
            tenant_id=tenant,
            authenticated=True,
        ),
        tenant=tenant,
        scopes=frozenset({"kg:write"}),
        graph=tenant,
    )


# The BUG-064 payload shape: a raw, caller-supplied mutation with no
# structural hint of who authored it.
_BUG064_SHAPED_OPERATIONS = [
    {
        "cypher": (
            "MATCH (n) WHERE n._visibility IS NULL "
            "SET n._visibility = 'public' RETURN count(n)"
        )
    }
]


class TestActorlessBatchUpdateIsLoud:
    """Requirement: a mutation with no bound actor must be loud, not silent.

    ``tests/conftest.py``'s repo-wide autouse ``isolate_graph_compute_engine``
    fixture binds an authenticated ``GraphSession`` for every test by default
    (the normal, correct posture for the suite) -- so proving the actor-less
    case requires explicitly stepping outside it with ``suspend_session()``,
    the same helper production code uses at genuinely unauthenticated
    boundaries (e.g. liveness probes).
    """

    def test_no_session_no_work_item_logs_unattributed_error(self, caplog):
        _client, proxy = _proxy()
        with suspend_session(), caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            result = proxy.lifecycle.batch_update(_BUG064_SHAPED_OPERATIONS)

        assert result == {"applied": 1}
        unattributed = [
            r
            for r in caplog.records
            if r.name == _LOGGER_NAME and "UNATTRIBUTED" in r.message
        ]
        assert unattributed, (
            "expected one loud UNATTRIBUTED log line for an actor-less, "
            f"work-item-less batch_update; got records: {[r.message for r in caplog.records]}"
        )
        record = unattributed[0]
        assert record.levelno == logging.ERROR
        assert "lifecycle.batch_update" in record.message
        assert "outcome=ok" in record.message

    def test_unattributed_call_still_reflects_operation_count(self, caplog):
        _client, proxy = _proxy()
        with suspend_session(), caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            proxy.lifecycle.batch_update(_BUG064_SHAPED_OPERATIONS * 3)

        unattributed = [
            r
            for r in caplog.records
            if r.name == _LOGGER_NAME and "UNATTRIBUTED" in r.message
        ]
        assert unattributed
        assert "operations=3" in unattributed[0].message


class TestWorkItemClaimedBatchUpdateIsAttributed:
    """Requirement: cover the WorkItem-claimed path specifically."""

    def test_work_item_context_emits_claim_identity(self, caplog):
        _client, proxy = _proxy()
        with (
            caplog.at_level(logging.INFO, logger=_LOGGER_NAME),
            bind_work_item_context(
                work_item_id="workitem:bug070-repro",
                agent_id="agent-dispatch-worker-7",
                lease_id="lease:abc123",
                capability="work_item.execute",
            ),
        ):
            proxy.lifecycle.batch_update(_BUG064_SHAPED_OPERATIONS)

        attributed = [
            r
            for r in caplog.records
            if r.name == _LOGGER_NAME and "batch_update attribution" in r.message
        ]
        assert attributed, (
            "expected an attributed log line naming the claimed WorkItem; got "
            f"records: {[r.message for r in caplog.records]}"
        )
        record = attributed[0]
        assert record.levelno == logging.INFO
        assert "workitem:bug070-repro" in record.message
        assert "lease:abc123" in record.message
        assert "work_item.execute" in record.message
        # No UNATTRIBUTED line should also fire for the same call.
        assert not [r for r in caplog.records if "UNATTRIBUTED" in r.message]

    def test_work_item_context_is_scoped_to_the_executor_call(self, caplog):
        """The binding must not leak past its ``with`` block."""
        _client, proxy = _proxy()
        with bind_work_item_context(
            work_item_id="workitem:scoped", agent_id="a", lease_id="l"
        ):
            pass  # context active, but no engine call made inside it

        with suspend_session(), caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            proxy.lifecycle.batch_update(_BUG064_SHAPED_OPERATIONS)

        unattributed = [
            r
            for r in caplog.records
            if r.name == _LOGGER_NAME and "UNATTRIBUTED" in r.message
        ]
        assert unattributed, "work-item context must not leak outside its block"


class TestSessionBoundBatchUpdateIsAttributed:
    """A direct (non-WorkItem) caller with a verified GraphSession also attributes."""

    def test_bound_actor_emits_actor_identity(self, caplog):
        _client, proxy = _proxy()
        token = set_session(_verified_session(actor_id="agent-42", tenant="tenant-a"))
        try:
            with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
                proxy.lifecycle.batch_update(_BUG064_SHAPED_OPERATIONS)
        finally:
            reset_session(token)

        attributed = [
            r
            for r in caplog.records
            if r.name == _LOGGER_NAME and "batch_update attribution" in r.message
        ]
        assert attributed
        record = attributed[0]
        assert record.levelno == logging.INFO
        # The actor id is pseudonymized (bus_reference), never logged raw --
        # but it must be present and non-empty, and distinct across actors.
        assert "actor=busref_agent_" in record.message
        assert not [r for r in caplog.records if "UNATTRIBUTED" in r.message]

    def test_different_actors_produce_different_attribution_tags(self, caplog):
        _client, proxy = _proxy()
        seen: list[str] = []
        for actor_id in ("agent-42", "agent-99"):
            token = set_session(_verified_session(actor_id=actor_id, tenant="tenant-a"))
            try:
                with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
                    proxy.lifecycle.batch_update(_BUG064_SHAPED_OPERATIONS)
            finally:
                reset_session(token)
            attributed = [
                r
                for r in caplog.records
                if r.name == _LOGGER_NAME and "batch_update attribution" in r.message
            ]
            seen.append(attributed[-1].message)
            caplog.clear()
        assert seen[0] != seen[1]


class TestAttributionIsScopedToBatchUpdate:
    """Non-mutating/other ops are unaffected -- no attribution-log noise."""

    def test_unrelated_op_emits_no_attribution_log(self, caplog):
        class _Namespace:
            def add(self, *args, **kwargs):
                return "added"

        class _Client:
            def __init__(self) -> None:
                self.nodes = _Namespace()

        breaker = CircuitBreaker("bug070-other-ep", threshold=5, cooldown=10)
        proxy = wrap_client_with_breaker(_Client(), breaker)

        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            proxy.nodes.add("n1")

        assert not [
            r
            for r in caplog.records
            if "UNATTRIBUTED" in r.message or "batch_update attribution" in r.message
        ]


def test_attributed_mutation_ops_is_scoped_not_global(monkeypatch):
    """Sanity check on the op allowlist itself, so a future op rename is caught."""
    assert engine_breaker._ATTRIBUTED_MUTATION_OPS == frozenset(
        {"lifecycle.batch_update"}
    )
