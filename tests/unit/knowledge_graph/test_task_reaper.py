"""Zombie/stuck task reaper (CONCEPT:EG-KG.storage.nonblocking-checkpoint ingestion durability).

When a worker/host process dies mid-task, the Task is stranded in ``running``
forever. ``TaskManagerMixin._tick_task_reaper`` requeues such orphans to
``pending`` using the singleton host token as ground truth: any ``running`` task
not claimed by the *live* host token (past a grace) had its worker die.

These exercise the reaper logic against a real ``EpistemicGraphBackend``.
"""

import time
from unittest.mock import patch

from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
    EpistemicGraphBackend,
)
from agent_utilities.knowledge_graph.core.engine_tasks import (
    TaskManagerMixin,
    _encode_metadata,
)

LIVE = "livehost:222:1700000002"
DEAD = "deadhost:111:1700000001"


class _ReaperHarness:
    """Minimal object exposing exactly what _tick_task_reaper touches."""

    def __init__(self, backend: EpistemicGraphBackend, token: str):
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

    def _get_host_token(self) -> str:
        return self._tok

    # Bind the real method under test.
    _tick_task_reaper = TaskManagerMixin._tick_task_reaper
    _get_host_token_real = TaskManagerMixin._get_host_token


def _add_task(b, tid, status="running", **meta):
    b.add_node(tid, node_type="Task", status=status, metadata=_encode_metadata(meta))


def _status(b, tid):
    rows = b.execute("MATCH (t:Task {id: $id}) RETURN t.status as s", {"id": tid})
    return rows[0]["s"] if rows else None


def _run_reaper(h, role="host"):
    with patch(
        "agent_utilities.knowledge_graph.core.host_lock.effective_daemon_role",
        return_value=role,
    ):
        h._tick_task_reaper()


def test_requeues_orphan_from_dead_host():
    b = EpistemicGraphBackend()
    # Orphan: running, claimed by a now-dead host token, claimed long ago.
    _add_task(b, "job-z", claimed_by=DEAD, claim_unix=time.time() - 1000, target="/x")
    # Live: running, claimed by THIS host, fresh — must be left alone.
    _add_task(b, "job-live", claimed_by=LIVE, claim_unix=time.time(), target="/y")

    _run_reaper(_ReaperHarness(b, LIVE))

    assert _status(b, "job-z") == "pending"  # orphan requeued
    assert _status(b, "job-live") == "running"  # live task untouched


def test_requeues_unstamped_legacy_orphan():
    b = EpistemicGraphBackend()
    # First-deploy case: a 'running' task claimed BEFORE the reaper existed has no
    # claimed_by token, only a started_at. On a fresh host it's provably an orphan
    # (singleton → nobody else runs workers) and must be requeued past the grace,
    # not left until the 2h absolute cap.
    import datetime as _dt

    old = (_dt.datetime.now(_dt.UTC) - _dt.timedelta(minutes=30)).isoformat()
    _add_task(b, "job-legacy", started_at=old, target="/x")  # no claimed_by

    _run_reaper(_ReaperHarness(b, LIVE))

    assert _status(b, "job-legacy") == "pending"


def test_respects_orphan_grace_window():
    b = EpistemicGraphBackend()
    # Cross-token but only just claimed (within the 90s grace) — e.g. a brief
    # election hand-off. Must NOT be requeued yet.
    _add_task(b, "job-fresh", claimed_by=DEAD, claim_unix=time.time() - 5, target="/x")

    _run_reaper(_ReaperHarness(b, LIVE))

    assert _status(b, "job-fresh") == "running"


def test_poison_pill_task_is_failed_not_looped():
    b = EpistemicGraphBackend()
    # Already requeued up to the cap: the next reap must FAIL it, not loop.
    _add_task(
        b,
        "job-poison",
        claimed_by=DEAD,
        claim_unix=time.time() - 1000,
        reaper_resets=3,
        target="/x",
    )

    _run_reaper(_ReaperHarness(b, LIVE))

    assert _status(b, "job-poison") == "failed"


def test_noop_on_client_role():
    b = EpistemicGraphBackend()
    _add_task(b, "job-c", claimed_by=DEAD, claim_unix=time.time() - 1000, target="/x")

    _run_reaper(_ReaperHarness(b, LIVE), role="client")

    assert _status(b, "job-c") == "running"  # clients never reap


def test_host_token_is_stable_and_unique_per_process():
    h = _ReaperHarness(EpistemicGraphBackend(), LIVE)
    t1 = h._get_host_token_real()
    t2 = h._get_host_token_real()
    assert t1 == t2  # cached/stable within a process
    assert t1.count(":") >= 2  # hostname:pid:boot-second
