"""Unified scheduling/queue model: bucketed priority claim, delayed/blocked
promotion, and app-level retry→backoff→dead-letter (CONCEPT:KG-2.113).

Exercises the new ``TaskManagerMixin`` queue primitives against a real
``EpistemicGraphBackend`` — the L1 graph path (no Postgres), which is the
zero-infra default and the one that must work everywhere. Ordering, delayed
visibility, dependencies, and retry are all enforced WITHOUT ORDER BY / range
predicates (the L1 interpreter supports only equality), so these tests pin that
the equality-bucket + per-minute-sweep design behaves correctly.
"""

import time
from unittest.mock import patch

from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
    EpistemicGraphBackend,
)
from agent_utilities.knowledge_graph.core.engine_tasks import (
    TaskManagerMixin,
    _decode_metadata,
    _encode_metadata,
)

HOST = "livehost:1:1700000000"


class _QHarness:
    """Minimal object exposing exactly what the queue primitives touch."""

    def __init__(self, backend: EpistemicGraphBackend, token: str = HOST):
        self.backend = backend
        self._tok = token

    def query_cypher(self, q, params=None):
        return self.backend.execute(q, params or {})

    @property
    def _control(self):
        # KG-2.148: control-plane ops (CAS claim, queue/:Task reads) route here;
        # the in-memory harness uses ONE backend for content + control (no
        # isolated __control__ graph in tests), mirroring production fallback.
        return self.backend

    def _control_cypher(self, cypher, params=None):
        return self.backend.execute(cypher, params or {})

    def _get_host_token(self) -> str:
        return self._tok

    def _checkpoint_db(self) -> None:  # no-op for the in-memory backend
        return None

    _select_pending_task = TaskManagerMixin._select_pending_task
    _claim_next_task = TaskManagerMixin._claim_next_task
    _update_task_status = TaskManagerMixin._update_task_status
    _fail_or_retry_task = TaskManagerMixin._fail_or_retry_task
    _deps_state = TaskManagerMixin._deps_state
    _tick_promotion_sweep = TaskManagerMixin._tick_promotion_sweep
    prioritize_task = TaskManagerMixin.prioritize_task


def _add(b, tid, status="pending", **props):
    meta = props.pop("meta", {})
    b.add_node(
        tid,
        node_type="Task",
        status=status,
        metadata=_encode_metadata(meta),
        **props,
    )


def _status(b, tid):
    rows = b.execute("MATCH (t:Task {id: $id}) RETURN t.status as s", {"id": tid})
    return rows[0]["s"] if rows else None


def _meta(b, tid):
    rows = b.execute("MATCH (t:Task {id: $id}) RETURN t.metadata as m", {"id": tid})
    return _decode_metadata(rows[0]["m"]) if rows else {}


def _sweep(h):
    with patch(
        "agent_utilities.knowledge_graph.core.host_lock.effective_daemon_role",
        return_value="host",
    ):
        h._tick_promotion_sweep()


# ---- bucketed priority claim --------------------------------------------------


def test_claim_picks_lowest_bucket_first():
    b = EpistemicGraphBackend()
    _add(b, "bg", prio_bucket=3, meta={"target": "/bg", "type": "document"})
    _add(b, "crit", prio_bucket=0, meta={"target": "/crit", "type": "document"})
    _add(b, "norm", prio_bucket=2, meta={"target": "/norm", "type": "document"})
    h = _QHarness(b)
    claimed = [h._claim_next_task()[0] for _ in range(3)]
    assert claimed == ["crit", "norm", "bg"]
    assert h._claim_next_task() is None  # queue drained


def test_claim_stamps_ownership_and_running():
    b = EpistemicGraphBackend()
    _add(b, "j", prio_bucket=2, meta={"target": "/j", "type": "document"})
    h = _QHarness(b)
    job_id, meta = h._claim_next_task()
    assert job_id == "j"
    assert meta["claimed_by"] == HOST and "claim_unix" in meta
    assert _status(b, "j") == "running"


def test_legacy_priority_string_still_claimable():
    """A pending node predating prio_bucket (only the legacy string) is claimed."""
    b = EpistemicGraphBackend()
    _add(b, "old", priority="high", meta={"target": "/old", "type": "document"})
    h = _QHarness(b)
    assert h._claim_next_task()[0] == "old"


# ---- delayed visibility (scheduled → pending) ---------------------------------


def test_scheduled_future_not_promoted():
    b = EpistemicGraphBackend()
    eta = time.time() + 3600
    _add(b, "s", status="scheduled", due_bucket=int(eta // 60), meta={"eta_unix": eta})
    h = _QHarness(b)
    _sweep(h)
    assert _status(b, "s") == "scheduled"  # not due yet


def test_scheduled_due_is_promoted():
    b = EpistemicGraphBackend()
    eta = time.time() - 5
    _add(b, "s", status="scheduled", due_bucket=int(eta // 60), meta={"eta_unix": eta})
    h = _QHarness(b)
    _sweep(h)
    assert _status(b, "s") == "pending"  # due → claimable


# ---- dependency gating (blocked → pending / cancelled) ------------------------


def test_blocked_waits_then_promotes_on_dep_complete():
    b = EpistemicGraphBackend()
    _add(b, "dep", status="running", meta={"target": "/dep"})
    _add(b, "child", status="blocked", meta={"depends_on": ["dep"]})
    h = _QHarness(b)
    _sweep(h)
    assert _status(b, "child") == "blocked"  # dep not done
    b.execute("MATCH (t:Task {id: 'dep'}) SET t.status = 'completed'")
    _sweep(h)
    assert _status(b, "child") == "pending"  # dep done → unblocked


def test_blocked_cancelled_on_broken_dep():
    b = EpistemicGraphBackend()
    _add(b, "dep", status="dead_letter", meta={"target": "/dep"})
    _add(b, "child", status="blocked", meta={"depends_on": ["dep"]})
    h = _QHarness(b)
    _sweep(h)
    assert _status(b, "child") == "cancelled"  # never run on a broken precondition


def test_deps_state_tri():
    b = EpistemicGraphBackend()
    _add(b, "a", status="completed")
    _add(b, "bbad", status="failed")
    _add(b, "cwait", status="pending")
    h = _QHarness(b)
    assert h._deps_state([]) == "ready"
    assert h._deps_state(["a"]) == "ready"
    assert h._deps_state(["a", "bbad"]) == "broken"
    assert h._deps_state(["a", "cwait"]) == "waiting"


# ---- app-level retry → backoff → dead-letter ---------------------------------


def test_failure_reschedules_with_backoff():
    b = EpistemicGraphBackend()
    _add(
        b,
        "f",
        status="running",
        meta={"target": "/f", "type": "document", "max_attempts": 3, "attempts": 0},
    )
    h = _QHarness(b)
    h._fail_or_retry_task("f", "boom")
    assert _status(b, "f") == "scheduled"
    m = _meta(b, "f")
    assert m["attempts"] == 1 and m["eta_unix"] > time.time() and m["error"] == "boom"
    assert "claimed_by" not in m  # lease cleared so it can be re-claimed


def test_failure_dead_letters_at_cap():
    b = EpistemicGraphBackend()
    _add(
        b,
        "f",
        status="running",
        meta={"target": "/f", "type": "document", "max_attempts": 2, "attempts": 1},
    )
    h = _QHarness(b)
    h._fail_or_retry_task("f", "boom")  # attempts -> 2 == cap
    assert _status(b, "f") == "dead_letter"
    assert _meta(b, "f")["attempts"] == 2


# ---- prioritize_task numeric buckets -----------------------------------------


def test_prioritize_accepts_numeric_bucket():
    b = EpistemicGraphBackend()
    _add(b, "p", prio_bucket=2, meta={"target": "/p"})
    h = _QHarness(b)
    res = h.prioritize_task("p", 0)
    assert res["status"] == "success" and res["prio_bucket"] == 0
    rows = b.execute("MATCH (t:Task {id: 'p'}) RETURN t.prio_bucket as pb")
    assert rows[0]["pb"] == 0
