"""CONCEPT:AU-KG.backend.schedule-on-control-graph — the scheduler's ``:Schedule``
reads/writes must run under a session actually bound to the ``__control__``
graph, not a graph-scoped view retargeting the ambient tenant session.

Production symptom this closes: every 60s tick logged
``collapse scan: active=0 schedules=0 over=0`` and
``CypherEngineError: ... error_type=PermissionError`` -- none of the 9
``deploy/schedules.yml`` entries (including ``fleet-tool-schema-sync``, the
only writer of the SQL fleet-catalog tables) ever fired. Root cause:
``_control_backend(engine)`` returns a graph-scoped view pinned to
``__control__`` (``EpistemicGraphBackend.for_graph``), but the scheduler's
ambient ``GraphSession`` is bound to the tenant graph it actually runs
under (e.g. ``homelab``). ``graph_compute._send_routed`` fails closed on
that mismatch: ``if self._fixed_graph and session.graph != self._fixed_graph:
raise PermissionError(...)``.

The fix (``schedule_engine._control_session_scope``) mirrors the SAME
sanctioned ``GraphSession.with_graph()`` + ``use_session()`` narrowing
``TaskManagerMixin._control_session_scope`` /
``_ControlPlaneWorkItemEngine._control_session_scope`` already use for the
identical shape of problem (``knowledge_graph/core/engine_tasks.py``).

These tests reproduce the production ``PermissionError`` in a fake backend
(mirroring ``graph_compute._send_routed``'s own check) rather than mocking
the retargeting seam under test -- ``_EnforcingControlBackend.execute``/
``add_node`` raise exactly like the real engine when driven by a session
whose ``graph`` disagrees with the backend's own ``graph_name``.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from agent_utilities.core import schedule_engine as se
from agent_utilities.knowledge_graph.core.session import (
    GraphSession,
    current_session,
    use_session,
)
from agent_utilities.security.brain_context import ActorContext, ActorType


def _at(h: int, m: int) -> datetime:
    return datetime(2026, 6, 15, h, m)  # a Monday


def _session(graph: str) -> GraphSession:
    actor = ActorContext(
        actor_id="scheduler-daemon",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("test",),
        tenant_id="test-tenant",
        authenticated=True,
    )
    return GraphSession(
        actor=actor,
        tenant="test-tenant",
        scopes=frozenset({"kg:read", "kg:write"}),
        graph=graph,
        policy_version="test-policy",
        audience="test-audience",
    )


class _EnforcingControlBackend:
    """Fake control-plane backend reproducing the EXACT production check
    ``graph_compute._send_routed`` performs for a graph-scoped view pinned
    to a fixed graph::

        if self._fixed_graph and session.graph != self._fixed_graph:
            raise PermissionError(
                "A graph-scoped view cannot retarget the verified GraphSession"
            )

    ``graph_name`` mirrors the attribute the real fix reads
    (``EpistemicGraphBackend.graph_name``) to learn what graph to retarget
    onto -- so a genuine mismatch between the ambient session and this
    graph raises, and a correctly-retargeted session does not.
    """

    def __init__(self, graph_name: str = "__control__") -> None:
        self.graph_name = graph_name
        self.nodes: dict[str, dict] = {}

    def _enforce(self) -> None:
        session = current_session()
        if session is None or session.graph != self.graph_name:
            raise PermissionError(
                "A graph-scoped view cannot retarget the verified GraphSession"
            )

    def add_node(self, node_id: str, *, node_type: str, **props) -> None:
        self._enforce()
        self.nodes[node_id] = {"id": node_id, "node_type": node_type, **props}

    def execute(self, query: str, params=None):
        self._enforce()
        if "{id: $id}" in query:
            row = self.nodes.get((params or {}).get("id"))
            return [dict(row)] if row is not None else []
        return [dict(row) for row in self.nodes.values()]


class _FakeEngine:
    def __init__(self, *, schedules_seeded: bool = True) -> None:
        self.control_backend = _EnforcingControlBackend()
        self.submitted: list[dict] = []
        self._schedules_seeded = schedules_seeded

    def submit_task(self, **kw):
        self.submitted.append(kw)
        return kw.get("job_id", "job-x")

    def _ingest_work_item_index(self):
        return {
            str(row.get("job_id")): {
                "status": "ready",
                "metadata": {
                    **dict(row.get("extra_meta") or {}),
                    "type": row.get("task_type") or "scheduled_job",
                },
            }
            for row in self.submitted
        }


def test_fake_control_backend_rejects_a_mismatched_session_unretargeted() -> None:
    """Negative control: proves the fake faithfully reproduces the real
    ``PermissionError`` a graph-scoped control-plane view raises when driven
    by an ambient session bound to a different graph, so the positive test
    below is proving something real -- calls the backend directly,
    bypassing ``_control_session_scope`` entirely.
    """
    backend = _EnforcingControlBackend()
    with use_session(_session(graph="homelab")):
        with pytest.raises(PermissionError, match="cannot retarget"):
            backend.execute("MATCH (s:Schedule) RETURN s.id as id")


def test_tick_loads_schedules_when_ambient_session_targets_a_different_graph() -> (
    None
):
    """THE production condition: the scheduler's ambient ``GraphSession`` is
    bound to the tenant graph it runs under (``homelab``), never
    ``__control__``. Before the fix, every ``:Schedule`` read/write raised
    from inside ``_EnforcingControlBackend._enforce`` (reproducing
    ``graph_compute._send_routed``), ``_load_all`` returned nothing, and the
    tick fired nothing -- exactly the reported ``schedules=0`` symptom.
    """
    eng = _FakeEngine()
    with use_session(_session(graph="homelab")):
        se.register_schedule(
            eng,
            se.ScheduleSpec(
                name="fleet-tool-schema-sync",
                payload={"kind": "skill", "ref": "mcp-fleet", "action": "sync"},
                trigger="cron",
                cron="5 * * * *",
            ),
        )

        specs = se._load_all(eng)
        assert [s.name for s in specs] == ["fleet-tool-schema-sync"]

        res = se.run_scheduler_tick(eng, now=_at(0, 5))

    assert res["fired"] == ["fleet-tool-schema-sync"]
    assert eng.submitted, "the due schedule must actually enqueue a WorkItem"


def test_failed_seed_does_not_mark_seeded_and_the_next_tick_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AGENTS.md *Fail closed* rule 2 (verified-write-state-advance):
    ``run_scheduler_tick`` previously set ``engine._schedules_seeded = True``
    unconditionally after attempting ``seed_schedules``, even when it
    raised -- a single transient failure (e.g. the control-graph session not
    yet available at boot) permanently disabled seeding, and therefore every
    schedule in ``deploy/schedules.yml``, for the rest of the process's
    life, since this branch never runs again once the flag is set. The flag
    must advance ONLY on a confirmed seed.
    """
    eng = _FakeEngine(schedules_seeded=False)
    calls = {"n": 0}

    def _boom(_engine):
        calls["n"] += 1
        raise RuntimeError("control-graph session not ready yet")

    monkeypatch.setattr(se, "seed_schedules", _boom)

    with use_session(_session(graph="homelab")):
        se.run_scheduler_tick(eng, now=_at(0, 0))
        assert calls["n"] == 1
        assert eng._schedules_seeded is False  # NOT advanced on a failed seed

        se.run_scheduler_tick(eng, now=_at(0, 1))
        assert calls["n"] == 2  # retried on the next tick, not skipped forever
        assert eng._schedules_seeded is False


def test_successful_seed_marks_seeded_and_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive counterpart: a CONFIRMED seed marks the flag and is not
    re-attempted on later ticks -- the fix narrows exactly the failure path,
    it does not turn seeding into a per-tick operation.
    """
    eng = _FakeEngine(schedules_seeded=False)
    calls = {"n": 0}

    def _ok(_engine):
        calls["n"] += 1
        return 0

    monkeypatch.setattr(se, "seed_schedules", _ok)

    with use_session(_session(graph="homelab")):
        se.run_scheduler_tick(eng, now=_at(0, 0))
        assert calls["n"] == 1
        assert eng._schedules_seeded is True

        se.run_scheduler_tick(eng, now=_at(0, 1))
        assert calls["n"] == 1  # not re-attempted once seeded
