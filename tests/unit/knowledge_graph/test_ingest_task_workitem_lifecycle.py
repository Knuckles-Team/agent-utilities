"""End-to-end ingestion-queue lifecycle through the WorkItem shadow (AU-P1-CL).

AU-P1-1 built the engine-native ``WorkItem`` state machine
(:mod:`agent_utilities.orchestration.work_item`) and migrated the
``:AgentTask`` claim path onto it; the ingestion queue
(:mod:`agent_utilities.knowledge_graph.core.engine_tasks`) was left SHIMMED
(read-only observability only). AU-P1-CL makes WorkItem authoritative there
too: ``_claim_next_task`` claims a deterministic shadow WorkItem (not a raw
``:Task.status`` field CAS), and ``_update_task_status``/``_fail_or_retry_task``
commit through it (fenced on the claim-time epoch), mirroring the outcome
onto the legacy ``:Task`` node for unmigrated readers.

These exercise the full claim -> complete / claim -> fail -> retry ->
dead-letter / lease-expiry-reap lifecycle against a real in-memory
``EpistemicGraphBackend`` (no isolated ``__control__`` graph in these tests
— the same pattern ``test_task_reaper.py`` uses).
"""

from unittest.mock import patch

from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
    EpistemicGraphBackend,
)
from agent_utilities.knowledge_graph.core.engine_tasks import (
    TaskManagerMixin,
    _decode_metadata,
    _encode_metadata,
)
from agent_utilities.orchestration import work_item as wi

TOKEN = "livehost:1:1700000001"


class _LifecycleHarness:
    """Minimal object exposing exactly what the claim/commit/reap lifecycle
    touches, bound to the REAL ``TaskManagerMixin`` methods under test."""

    def __init__(self, backend: EpistemicGraphBackend):
        self.backend = backend
        self._tok = TOKEN
        self._candidates: list[dict] = []

    def _select_pending_task(self, admit=None):
        return self._candidates.pop(0) if self._candidates else None

    def _make_admission(self):
        return None

    def _get_host_token(self) -> str:
        return self._tok

    control_backend = None
    _control = TaskManagerMixin._control
    _control_cypher = TaskManagerMixin._control_cypher
    _work_item_engine = TaskManagerMixin._work_item_engine
    _checkpoint_db = TaskManagerMixin._checkpoint_db

    _claim_next_task = TaskManagerMixin._claim_next_task
    _update_task_status = TaskManagerMixin._update_task_status
    _fail_or_retry_task = TaskManagerMixin._fail_or_retry_task
    _tick_task_reaper = TaskManagerMixin._tick_task_reaper


def _add_task(b: EpistemicGraphBackend, tid: str, **meta) -> None:
    b.add_node(tid, node_type="Task", status="pending", metadata=_encode_metadata(meta))


def _task(b: EpistemicGraphBackend, tid: str) -> dict:
    rows = b.execute(
        "MATCH (t:Task {id: $id}) RETURN t.status as status, t.metadata as meta",
        {"id": tid},
    )
    assert rows, f"no :Task node for {tid}"
    return {"status": rows[0]["status"], "meta": _decode_metadata(rows[0].get("meta"))}


def _claim(h: _LifecycleHarness, job_id: str) -> tuple[str, dict]:
    h._candidates = [{"id": job_id, "meta": None}]
    result = h._claim_next_task()
    assert result is not None, f"claim of {job_id} unexpectedly lost"
    return result


def test_claim_then_complete_commits_workitem_succeeded():
    b = EpistemicGraphBackend()
    _add_task(b, "job-ok", target="/x")
    h = _LifecycleHarness(b)

    job_id, meta = _claim(h, "job-ok")
    h._update_task_status(job_id, "completed", {"chunks_added": 3})

    row = _task(b, "job-ok")
    assert row["status"] == "completed"
    assert row["meta"]["chunks_added"] == 3

    item = wi.get_work_item(h._work_item_engine, meta["work_item_id"])
    assert item["status"] == "succeeded"
    assert item["result_ref"] == "outcome:ingest_task:job-ok"


def test_direct_failed_status_is_non_retryable():
    """A direct ``_update_task_status(status='failed')`` (e.g. missing target
    — never routed through ``_fail_or_retry_task``) commits the WorkItem
    non-retryable: terminal 'failed', no backoff/retry."""
    b = EpistemicGraphBackend()
    _add_task(b, "job-bad", target="/x")
    h = _LifecycleHarness(b)

    job_id, meta = _claim(h, "job-bad")
    h._update_task_status(job_id, "failed", {"error": "missing target"})

    assert _task(b, "job-bad")["status"] == "failed"
    item = wi.get_work_item(h._work_item_engine, meta["work_item_id"])
    assert item["status"] == "failed"


def test_fail_or_retry_schedules_backoff_matching_workitem_math():
    """First app-level failure: retried with backoff computed by
    ``work_item.commit_result`` (base * 2**(attempt-1), attempt=1 => base)."""
    b = EpistemicGraphBackend()
    _add_task(b, "job-retry", target="/x")
    h = _LifecycleHarness(b)

    job_id, meta = _claim(h, "job-retry")
    before = _task(b, "job-retry")
    assert before["status"] == "running"

    h._fail_or_retry_task(job_id, "boom")

    after = _task(b, "job-retry")
    assert after["status"] == "scheduled"
    assert after["meta"]["attempts"] == 1
    assert after["meta"]["max_attempts"] == 3
    assert "eta_unix" in after["meta"]
    assert "claimed_by" not in after["meta"]

    item = wi.get_work_item(h._work_item_engine, meta["work_item_id"])
    assert item["status"] == "ready"  # requeued, not terminal
    assert item["attempt"] == 1
    # backoff_base_s=30 * 2**(attempt-1=0) == 30
    assert item["next_retry_at"] - item["updated_at"] == item["backoff_base_s"]


def test_fail_or_retry_dead_letters_after_max_attempts():
    """Repeated app-level failures exhaust max_attempts (3) and dead-letter —
    the SAME attempt/DLQ machinery :AgentTask already uses, reused here."""
    b = EpistemicGraphBackend()
    _add_task(b, "job-dlq", target="/x", max_attempts=3)
    h = _LifecycleHarness(b)

    work_item_id = None
    for _ in range(2):
        job_id, meta = _claim(h, "job-dlq")
        work_item_id = meta["work_item_id"]
        h._fail_or_retry_task(job_id, "boom again")
        # Bypass the real backoff window for the test (a claim's next attempt
        # would otherwise wait out the scheduled backoff) — the retry
        # DECISION/backoff MATH is what's under test, not wall-clock timing.
        b.execute(
            "MATCH (w:WorkItem {id: $id}) SET w.next_retry_at = $t",
            {"id": work_item_id, "t": 0.0},
        )

    # Third claim + failure exhausts max_attempts (attempt reaches 3).
    job_id, meta = _claim(h, "job-dlq")
    item_before = wi.get_work_item(h._work_item_engine, work_item_id)
    assert item_before["attempt"] == 3
    h._fail_or_retry_task(job_id, "final blow")

    after = _task(b, "job-dlq")
    assert after["status"] == "dead_letter"
    item = wi.get_work_item(h._work_item_engine, work_item_id)
    assert item["status"] == "dead_letter"


def test_reaper_requeues_shadowed_task_via_lease_expiry():
    """A shadowed (migrated) task whose lease expired is reaped by
    ``work_item.reap_expired_leases`` — NOT the legacy host-token heuristic
    (which now skips any row carrying ``work_item_id``)."""
    b = EpistemicGraphBackend()
    _add_task(b, "job-stuck", target="/x")
    h = _LifecycleHarness(b)

    job_id, meta = _claim(h, "job-stuck")
    assert _task(b, "job-stuck")["status"] == "running"

    # Simulate the lease having expired (worker died / exceeded the runtime
    # cap) without waiting out the real TTL.
    b.execute(
        "MATCH (w:WorkItem {id: $id}) SET w.lease_expires_at = $t",
        {"id": meta["work_item_id"], "t": 0.0},
    )

    with patch(
        "agent_utilities.knowledge_graph.core.host_lock.effective_daemon_role",
        return_value="host",
    ):
        h._tick_task_reaper()

    after = _task(b, "job-stuck")
    assert after["status"] == "pending"
    assert "claimed_by" not in after["meta"]
    item = wi.get_work_item(h._work_item_engine, meta["work_item_id"])
    assert item["status"] == "ready"


def test_reaper_dead_letters_shadowed_task_past_max_attempts():
    """A shadowed task whose lease expired AND has exhausted max_attempts is
    dead-lettered by the reap pass, not endlessly requeued."""
    b = EpistemicGraphBackend()
    _add_task(b, "job-poison", target="/x")
    h = _LifecycleHarness(b)

    job_id, meta = _claim(h, "job-poison")
    work_item_id = meta["work_item_id"]
    # Push attempt to the cap directly (equivalent to 2 prior crash-reclaims).
    b.execute(
        "MATCH (w:WorkItem {id: $id}) SET w.attempt = $a, w.lease_expires_at = $t",
        {"id": work_item_id, "a": 3, "t": 0.0},
    )

    with patch(
        "agent_utilities.knowledge_graph.core.host_lock.effective_daemon_role",
        return_value="host",
    ):
        h._tick_task_reaper()

    assert _task(b, "job-poison")["status"] == "dead_letter"
    item = wi.get_work_item(h._work_item_engine, work_item_id)
    assert item["status"] == "dead_letter"


def test_reaper_skips_legacy_sweep_for_shadowed_running_task():
    """A shadowed task with a LIVE (unexpired) lease must not be touched by
    either reap pass (the WorkItem pass sees a live lease; the legacy sweep
    skips it outright because it carries work_item_id)."""
    b = EpistemicGraphBackend()
    _add_task(b, "job-live", target="/x")
    h = _LifecycleHarness(b)

    _claim(h, "job-live")

    with patch(
        "agent_utilities.knowledge_graph.core.host_lock.effective_daemon_role",
        return_value="host",
    ):
        h._tick_task_reaper()

    assert _task(b, "job-live")["status"] == "running"
