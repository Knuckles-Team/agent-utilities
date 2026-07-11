"""Backend-agnostic :Task claim via the WorkItem state machine (AU-P1-CL).

The worker claim used to be a single raw ``compare_and_set_node_fields`` on
the ``:Task`` node's own ``status`` field. AU-P1-CL made the ingestion
queue's claim/commit/reap arbitration authoritative on a deterministic
shadow ``:WorkItem`` (:mod:`agent_utilities.orchestration.work_item`) instead
— the SAME engine-native CAS/lease/attempt machinery ``:AgentTask`` dispatch
already uses — so these tests were rewritten (from mocking a single
``backend.compare_and_set_node_fields`` call) to exercise the REAL WorkItem
transitions against a real in-memory ``EpistemicGraphBackend`` (no isolated
``__control__`` graph in these tests — the same pattern
``test_task_reaper.py``/``test_task_queue_controls.py`` already use).

These exercise ``TaskManagerMixin._claim_next_task`` with ``_select_pending_task``
stubbed to hand back a controlled candidate queue (the bucket-ascending
selection itself is covered elsewhere: ``test_select_pending_admission.py``):
a winning claim must create+claim the shadow WorkItem, mirror
``:Task.status: running`` + the shadow's id/epoch, and skip a candidate whose
shadow is already claimed elsewhere; two sequential claims of the same row
must produce one winner and one loser.
"""

from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
    EpistemicGraphBackend,
)
from agent_utilities.knowledge_graph.core.engine_tasks import (
    TaskManagerMixin,
    _decode_metadata,
    _encode_metadata,
)
from agent_utilities.orchestration import work_item as wi

TOKEN = "claimhost:333:1700000003"


class _ClaimHarness:
    """Minimal object exposing exactly what _claim_next_task touches.

    ``_select_pending_task`` is stubbed to hand back a controlled queue of
    candidate rows (the bucket-ascending selection is covered elsewhere); the
    win/lose arbitration is the REAL ``work_item`` state machine against a
    real ``EpistemicGraphBackend`` bound as BOTH ``backend`` and the control
    plane (``_control`` falls back to ``self.backend`` when no isolated
    ``control_backend`` is set — CONCEPT:AU-KG.backend.schedule-on-control-graph).
    """

    def __init__(self, candidates, backend=None):
        self.backend = backend if backend is not None else EpistemicGraphBackend()
        self._candidates = list(candidates)
        self._tok = TOKEN

    def _select_pending_task(self, admit=None):
        # ORCH-1.81 added an admission predicate; this harness hands back a
        # controlled candidate queue regardless (admission is tested separately).
        return self._candidates.pop(0) if self._candidates else None

    def _make_admission(self):
        # ORCH-1.81: disable the admission gate for these pure claim tests.
        return None

    def _get_host_token(self) -> str:
        return self._tok

    control_backend = None
    _control = TaskManagerMixin._control
    _control_cypher = TaskManagerMixin._control_cypher
    _work_item_engine = TaskManagerMixin._work_item_engine

    # Bind the real method under test.
    _claim_next_task = TaskManagerMixin._claim_next_task


def _add_task(b: EpistemicGraphBackend, tid: str, **meta) -> None:
    b.add_node(tid, node_type="Task", status="pending", metadata=_encode_metadata(meta))


def _task_row(tid: str) -> dict:
    return {"id": tid, "meta": None}


def _task_status(b: EpistemicGraphBackend, tid: str) -> str | None:
    rows = b.execute("MATCH (t:Task {id: $id}) RETURN t.status as s", {"id": tid})
    return rows[0]["s"] if rows else None


def test_claim_wins_creates_running_shadow_and_stamps_task():
    """A winning claim creates+claims the shadow WorkItem and mirrors the win
    onto the legacy :Task node (status + work_item_id/epoch stamped)."""
    b = EpistemicGraphBackend()
    _add_task(b, "job-1", target="/x")
    h = _ClaimHarness(candidates=[_task_row("job-1")], backend=b)

    result = h._claim_next_task()

    assert result is not None
    job_id, meta = result
    assert job_id == "job-1"
    assert meta["claimed_by"] == TOKEN
    assert "claim_unix" in meta and "started_at" in meta
    work_item_id = meta["work_item_id"]
    assert work_item_id == wi.ingest_task_work_item_id("job-1")
    assert meta["work_item_epoch"] == 1

    # Legacy :Task mirror reflects the win (unchanged shape + the new stamps).
    assert _task_status(b, "job-1") == "running"
    rows = b.execute("MATCH (t:Task {id: $id}) RETURN t.metadata as m", {"id": "job-1"})
    stamped = _decode_metadata(rows[0]["m"])
    assert stamped["work_item_id"] == work_item_id
    assert stamped["claimed_by"] == TOKEN

    # The shadow WorkItem is the REAL authority: running, attempt=1, leased
    # by this host's token.
    item = wi.get_work_item(h._work_item_engine, work_item_id)
    assert item is not None
    assert item["status"] == "running"
    assert item["attempt"] == 1
    assert item["lease_owner"] == TOKEN


def test_claim_skips_candidate_whose_shadow_already_claimed():
    """A candidate whose shadow WorkItem a peer already claimed (still within
    its lease) is skipped — the claimer moves on to the next candidate."""
    b = EpistemicGraphBackend()
    _add_task(b, "job-lost", target="/x")
    _add_task(b, "job-won", target="/y")
    h = _ClaimHarness(
        candidates=[_task_row("job-lost"), _task_row("job-won")], backend=b
    )

    # A peer already won job-lost's shadow (fresh lease — not stale).
    peer_item_id = wi.ingest_task_work_item_id("job-lost")
    wi.submit_work_item(
        h._work_item_engine,
        kind="ingest_task",
        payload_ref="job-lost",
        work_item_id=peer_item_id,
    )
    peer_claim = wi.claim_specific(
        h._work_item_engine, peer_item_id, token="peerhost:1:1"
    )
    assert peer_claim is not None

    result = h._claim_next_task()

    assert result is not None
    job_id, _meta = result
    assert job_id == "job-won"  # the already-claimed candidate was skipped
    # The peer's claim on job-lost is untouched (this claimer never mirrored
    # a win onto it).
    assert _task_status(b, "job-lost") == "pending"


def test_claim_returns_none_when_idle():
    """No pending candidates → no claim attempt, returns None."""
    h = _ClaimHarness(candidates=[])

    assert h._claim_next_task() is None


def test_claim_returns_none_when_all_candidates_already_claimed():
    """Every candidate's shadow is already claimed elsewhere → idle (None),
    never a phantom claim."""
    b = EpistemicGraphBackend()
    _add_task(b, "a", target="/a")
    _add_task(b, "b", target="/b")
    h = _ClaimHarness(candidates=[_task_row("a"), _task_row("b")], backend=b)

    for job_id in ("a", "b"):
        item_id = wi.ingest_task_work_item_id(job_id)
        wi.submit_work_item(
            h._work_item_engine,
            kind="ingest_task",
            payload_ref=job_id,
            work_item_id=item_id,
        )
        assert (
            wi.claim_specific(h._work_item_engine, item_id, token="peerhost:1:1")
            is not None
        )

    assert h._claim_next_task() is None
    assert _task_status(b, "a") == "pending"
    assert _task_status(b, "b") == "pending"


def test_two_sequential_claims_of_same_task_first_wins_second_loses():
    """First claimer wins; a second claimer of the SAME row loses.

    Models the cross-host race the WorkItem CAS now arbitrates: only one
    ``ready -> leased`` transition succeeds.
    """
    b = EpistemicGraphBackend()
    _add_task(b, "job-shared", target="/x")

    def make_harness() -> _ClaimHarness:
        return _ClaimHarness(candidates=[_task_row("job-shared")], backend=b)

    first = make_harness()._claim_next_task()
    second = make_harness()._claim_next_task()

    assert first is not None and first[0] == "job-shared"  # winner
    assert second is None  # loser got no claim, no other candidate
    assert _task_status(b, "job-shared") == "running"
