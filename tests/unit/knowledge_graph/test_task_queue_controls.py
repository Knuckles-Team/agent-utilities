"""Job-queue controls: cancel / clear(by status incl. all/zombie) / prioritize.

CONCEPT:KG-2.8 — operator control over the ingestion queue. Exercises the
``TaskManagerMixin`` methods against a real ``EpistemicGraphBackend``.
"""

import time

from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
    EpistemicGraphBackend,
)
from agent_utilities.knowledge_graph.core.engine_tasks import (
    TaskManagerMixin,
    _encode_metadata,
)

LIVE = "livehost:999:1700000003"


class _QHarness:
    def __init__(self, backend, token=LIVE):
        self.backend = backend
        self._tok = token

    def query_cypher(self, q, params=None):
        return self.backend.execute(q, params or {})

    @property
    def _control(self):
        # KG-2.148: control-plane ops route here; the in-memory harness uses ONE
        # backend for content + control (no isolated __control__ graph in tests).
        return self.backend

    def _control_cypher(self, cypher, params=None):
        return self.backend.execute(cypher, params or {})

    def _get_host_token(self):
        return self._tok

    cancel_task = TaskManagerMixin.cancel_task
    clear_tasks = TaskManagerMixin.clear_tasks
    prioritize_task = TaskManagerMixin.prioritize_task


def _add(b, tid, status="pending", **meta):
    b.add_node(tid, node_type="Task", status=status, metadata=_encode_metadata(meta))


def _status(b, tid):
    r = b.execute("MATCH (t:Task {id: $id}) RETURN t.status as s", {"id": tid})
    return r[0]["s"] if r else None


def _count(b):
    r = b.execute("MATCH (t:Task) RETURN count(t) as c")
    return r[0]["c"] if r else 0


def test_cancel_marks_cancelled():
    b = EpistemicGraphBackend()
    _add(b, "job-1", status="pending", target="/x")
    res = _QHarness(b).cancel_task("job-1")
    assert res["status"] == "success" and res["prev_status"] == "pending"
    assert _status(b, "job-1") == "cancelled"


def test_cancel_unknown_job_errors():
    b = EpistemicGraphBackend()
    res = _QHarness(b).cancel_task("nope")
    assert res["status"] == "error"


def test_clear_by_status_deletes_only_matching():
    b = EpistemicGraphBackend()
    _add(b, "c-done", status="completed", target="/a")
    _add(b, "c-pend", status="pending", target="/b")
    res = _QHarness(b).clear_tasks("completed")
    assert res["cleared"] == 1
    assert _status(b, "c-done") is None  # deleted
    assert _status(b, "c-pend") == "pending"  # kept


def test_clear_all_empties_queue():
    b = EpistemicGraphBackend()
    _add(b, "a1", status="pending")
    _add(b, "a2", status="running")
    _add(b, "a3", status="failed")
    res = _QHarness(b).clear_tasks("all")
    assert res["cleared"] == 3 and res["remaining"] == 0
    assert _count(b) == 0


def test_clear_zombie_targets_only_unowned_running():
    b = EpistemicGraphBackend()
    # Orphan: running, foreign/absent token.
    _add(b, "z1", status="running", claimed_by="deadhost:1:1", claim_unix=time.time())
    # Live: running, owned by THIS host — must survive a zombie clear.
    _add(b, "z2", status="running", claimed_by=LIVE, claim_unix=time.time())
    # Pending — not a zombie.
    _add(b, "z3", status="pending")
    res = _QHarness(b).clear_tasks("zombie")
    assert res["cleared"] == 1
    assert _status(b, "z1") is None  # orphan cleared
    assert _status(b, "z2") == "running"  # live task untouched
    assert _status(b, "z3") == "pending"  # pending untouched


def test_clear_invalid_status_errors():
    b = EpistemicGraphBackend()
    res = _QHarness(b).clear_tasks("bogus")
    assert res["status"] == "error"


def test_prioritize_sets_high_priority():
    b = EpistemicGraphBackend()
    _add(b, "p1", status="pending", target="/x")
    res = _QHarness(b).prioritize_task("p1", "high")
    assert res["status"] == "success" and res["priority"] == "high"
    r = b.execute("MATCH (t:Task {id: $id}) RETURN t.priority as p", {"id": "p1"})
    assert r[0]["p"] == "high"
    # And it is now selectable by the priority-aware poll's first tier.
    hot = b.execute(
        "MATCH (t:Task {status: 'pending', priority: 'high'}) RETURN t.id as id"
    )
    assert {row["id"] for row in hot} == {"p1"}


def test_prioritize_rejects_bad_level():
    b = EpistemicGraphBackend()
    _add(b, "p2", status="pending")
    res = _QHarness(b).prioritize_task("p2", "urgent")
    assert res["status"] == "error"
