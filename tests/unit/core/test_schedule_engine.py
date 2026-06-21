"""CONCEPT:OS-5.44 — unified scheduling engine.

The one durable, dynamic scheduler: ``:Schedule`` nodes (cron | interval |
adaptive) evaluated by a single tick that ENQUEUES ``scheduled_job`` tasks
(never runs them inline), with node-backed idempotency and failure backoff.
Exercised against a real ``EpistemicGraphBackend`` with a fake engine that
records ``submit_task`` calls.
"""

from __future__ import annotations

from datetime import datetime

from agent_utilities.core import schedule_engine as se
from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
    EpistemicGraphBackend,
)


def _at(h: int, m: int) -> datetime:
    return datetime(2026, 6, 15, h, m)  # a Monday


class _FakeEngine:
    def __init__(self):
        self.backend = EpistemicGraphBackend()
        self.submitted: list[dict] = []
        self._schedules_seeded = True  # skip yaml seeding in unit tests
        self.ticked: list[str] = []

    def submit_task(self, **kw):
        self.submitted.append(kw)
        return kw.get("job_id", "job-x")

    def query_cypher(self, _q, params=None):
        # Emulate the coalesce probe: count un-consumed ticks for a schedule.
        # The fake never completes tasks, so every submitted tick counts as
        # pending until a test clears ``submitted`` to simulate consumption.
        name = (params or {}).get("name")
        n = sum(
            1
            for s in self.submitted
            if (s.get("extra_meta") or {}).get("schedule") == name
        )
        return [{"n": n}]

    def _tick_demo(self):
        self.ticked.append("demo")


def test_cron_matcher_fields() -> None:
    assert se.cron_matches("0 */4 * * *", _at(4, 0))
    assert not se.cron_matches("0 */4 * * *", _at(5, 0))
    assert not se.cron_matches("0 */4 * * *", _at(4, 30))
    assert se.cron_matches("* * * * *", _at(4, 30))
    assert se.cron_matches("30 9 * * 1", _at(9, 30))  # Monday 09:30
    assert not se.cron_matches("30 9 * * 0", _at(9, 30))  # Sunday only
    assert se.cron_matches("0 0,12 * * *", _at(12, 0))
    assert se.cron_matches("0 9-17 * * *", _at(13, 0))


def test_cron_due_enqueues_and_no_double_fire() -> None:
    eng = _FakeEngine()
    se.register_schedule(
        eng,
        se.ScheduleSpec(
            name="job-a",
            payload={"kind": "skill", "ref": "x", "action": "delta"},
            trigger="cron",
            cron="0 */4 * * *",
        ),
    )
    se.register_schedule(
        eng,
        se.ScheduleSpec(
            name="job-b",
            payload={"kind": "skill", "ref": "y", "action": "delta"},
            trigger="cron",
            cron="0 5 * * *",
        ),
    )
    # 04:00 fires only job-a (job-b is 05:00)
    res = se.run_scheduler_tick(eng, now=_at(4, 0))
    assert res["fired"] == ["job-a"]
    assert eng.submitted[-1]["task_type"] == "scheduled_job"
    assert eng.submitted[-1]["job_id"].startswith("sched:job-a:")
    # same minute again → no double-fire (node last_minute persisted)
    assert se.run_scheduler_tick(eng, now=_at(4, 0))["fired"] == []
    # 05:00 fires job-b
    assert se.run_scheduler_tick(eng, now=_at(5, 0))["fired"] == ["job-b"]


def test_interval_schedule_fires_then_waits() -> None:
    eng = _FakeEngine()
    se.register_schedule(
        eng,
        se.ScheduleSpec(
            name="ivl",
            payload={"kind": "maint", "ref": "demo"},
            trigger="interval",
            interval_s=3600.0,
            next_run_unix=0.0,  # due immediately
        ),
    )
    assert se.run_scheduler_tick(eng)["fired"] == ["ivl"]
    # next_run_unix advanced ~1h → not due again right away
    assert se.run_scheduler_tick(eng)["fired"] == []


def test_tick_coalesces_unconsumed_prior() -> None:
    # CONCEPT:OS-5.44 — an interval/cron tick must NOT pile a new task while the
    # previous tick for the same schedule is still un-consumed. Otherwise a slow
    # or stalled consumer (e.g. an engine outage) accumulates an unbounded
    # backlog of stale ticks (the file_watch/enrichment/analysis backlog leak).
    eng = _FakeEngine()
    se.register_schedule(
        eng,
        se.ScheduleSpec(
            name="cw",
            payload={"kind": "maint", "ref": "demo"},
            trigger="cron",
            cron="* * * * *",  # due every minute
        ),
    )
    assert se.run_scheduler_tick(eng, now=_at(4, 0))["fired"] == ["cw"]
    assert len(eng.submitted) == 1
    # next minute: due again, but the 04:00 tick is still pending → coalesced
    assert se.run_scheduler_tick(eng, now=_at(4, 1))["fired"] == []
    assert len(eng.submitted) == 1
    # once the prior tick is consumed, the schedule fires again
    eng.submitted.clear()
    assert se.run_scheduler_tick(eng, now=_at(4, 2))["fired"] == ["cw"]
    assert len(eng.submitted) == 1


def test_disabled_schedule_does_not_fire() -> None:
    eng = _FakeEngine()
    se.register_schedule(
        eng,
        se.ScheduleSpec(
            name="off",
            payload={"kind": "maint", "ref": "demo"},
            trigger="cron",
            cron="* * * * *",
            enabled=False,
        ),
    )
    assert se.run_scheduler_tick(eng, now=_at(1, 1))["fired"] == []


def test_run_scheduled_job_maint_dispatch() -> None:
    eng = _FakeEngine()
    res = se.run_scheduled_job(eng, {"kind": "maint", "ref": "demo"})
    assert res["status"] == "ok" and eng.ticked == ["demo"]


def test_run_scheduled_job_unknown_maint() -> None:
    eng = _FakeEngine()
    res = se.run_scheduled_job(eng, {"kind": "maint", "ref": "nope"})
    assert res["status"] == "skipped"


def test_set_enabled_and_run_now() -> None:
    eng = _FakeEngine()
    se.register_schedule(
        eng,
        se.ScheduleSpec(
            name="ctl",
            payload={"kind": "maint", "ref": "demo"},
            trigger="cron",
            cron="0 0 1 1 *",  # almost never
        ),
    )
    assert se.set_enabled(eng, "ctl", False)["enabled"] is False
    assert se.set_enabled(eng, "missing", True)["status"] == "error"
    # run_now clears the gate; combined with re-enable it fires on a cron-matching tick
    se.set_enabled(eng, "ctl", True)
    se.set_interval(eng, "ctl", 1.0)  # convert to interval so it is due now
    se.run_now(eng, "ctl")
    assert se.run_scheduler_tick(eng)["fired"] == ["ctl"]


def test_record_schedule_result_backoff() -> None:
    eng = _FakeEngine()
    se.register_schedule(
        eng,
        se.ScheduleSpec(
            name="flaky",
            payload={"kind": "maint", "ref": "demo"},
            trigger="interval",
            interval_s=60.0,
        ),
    )
    se.record_schedule_result(eng, "flaky", ok=False)
    spec = se._load_one(eng, "flaky")
    assert spec.consecutive_failures == 1 and spec.backoff_until > 0
    se.record_schedule_result(eng, "flaky", ok=True)
    spec = se._load_one(eng, "flaky")
    assert spec.consecutive_failures == 0 and spec.backoff_until == 0
