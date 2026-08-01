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

These exercise ``TaskManagerMixin._claim_next_task``. A later migration
(ORCH-1.8x) retired the Python-side ``_select_pending_task`` candidate-queue
cursor entirely — ``_claim_next_task`` now claims exclusively through the
native ``work_item.claim_next`` queue scan ("native selection owns ordering,
quota, dependency release, and lease recovery" — work_item.py), so a claimable
job needs a REAL ``:WorkItem`` submitted into the ``ingest_task`` queue, not
just a candidate handed back by a stub (``_add_task`` below does both: the
legacy ``:Task`` node AND its native WorkItem shadow). A winning claim must
create+claim the shadow WorkItem, mirror ``:Task.status: running`` + the
shadow's id/epoch, and skip a candidate whose shadow is already claimed
elsewhere; two sequential claims of the same row must produce one winner and
one loser.
"""

import threading

from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
    EpistemicGraphBackend,
)
from agent_utilities.knowledge_graph.core.engine_tasks import (
    TaskManagerMixin,
    _encode_metadata,
)
from agent_utilities.orchestration import work_item as wi

TOKEN = "claimhost:333:1700000003"


class _ClaimHarness:
    """Minimal object exposing exactly what _claim_next_task touches.

    ``_claim_next_task`` claims through the native WorkItem queue directly
    (see the module docstring) with ``worker_id`` left ``None`` here, so the
    ``hydration_reserved``/admission-policy branch (which needs
    ``worker_id``) is never reached — these tests exercise only the win/lose
    WorkItem CAS arbitration. That arbitration is the REAL ``work_item``
    state machine against a real ``EpistemicGraphBackend`` bound as BOTH
    ``backend`` and the control plane. ``_control`` (engine_tasks.py) now
    fails closed when ``control_backend`` is unset
    (CONCEPT:AU-KG.backend.schedule-on-control-graph hardened it — no more
    falling back to ``self.backend``), so this harness binds both explicitly
    to the same backend, matching how a real engine's ``control_backend`` is
    always configured in production.
    """

    def __init__(self, backend=None):
        self.backend = backend if backend is not None else EpistemicGraphBackend()
        self.control_backend = self.backend
        self._tok = TOKEN
        # _claim_next_task remembers/reads the in-flight native claim through
        # these two (engine_tasks.py's real __init__ sets the same pair).
        self._active_work_item_claims: dict = {}
        self._active_work_item_claims_lock = threading.Lock()

    def _get_host_token(self) -> str:
        return self._tok

    _control = TaskManagerMixin._control
    _control_cypher = TaskManagerMixin._control_cypher
    _work_item_engine = TaskManagerMixin._work_item_engine
    _remember_work_item_claim = TaskManagerMixin._remember_work_item_claim
    _active_work_item_claim = TaskManagerMixin._active_work_item_claim
    _ingest_task_metadata = TaskManagerMixin._ingest_task_metadata

    # Bind the real method under test.
    _claim_next_task = TaskManagerMixin._claim_next_task


def _add_task(b: EpistemicGraphBackend, tid: str, **meta) -> None:
    """Create the legacy ``:Task`` node AND its native ``ingest_task`` WorkItem
    shadow — ``_claim_next_task`` claims exclusively from the native queue (see
    the module docstring), so a job is only claimable once both exist."""
    b.add_node(tid, node_type="Task", status="pending", metadata=_encode_metadata(meta))
    wi.submit_work_item(
        _ClaimHarness(backend=b)._work_item_engine,
        kind="ingest_task",
        payload_ref=tid,
        work_item_id=wi.ingest_task_work_item_id(tid),
    )


def _task_status(b: EpistemicGraphBackend, tid: str) -> str | None:
    rows = b.execute("MATCH (t:Task {id: $id}) RETURN t.status as s", {"id": tid})
    return rows[0]["s"] if rows else None


def test_claim_wins_creates_running_shadow_and_stamps_task():
    """A winning claim creates+claims the shadow WorkItem; the returned meta
    is the RAW WorkItem metadata plus the winning claim's own lease identity,
    and the legacy :Task node is left untouched by design.

    Was originally written expecting ``_claim_next_task`` to stamp
    ``claimed_by``/``claim_unix``/``started_at``/``work_item_id``/
    ``work_item_epoch`` onto the returned meta and mirror ``status:
    "running"`` onto the legacy ``:Task`` node. Only the FIRST half of that
    was ever built: D-W2-12 (``fix/w2-w2-tail``) added ``claimed_by``/
    ``work_item_epoch``/``work_item_id`` stamping (the winning claim's own
    lease fields — see ``test_claim_next_task_stamps_claimed_by_and_
    work_item_epoch_onto_meta`` below) because the caller had no way to say
    WHO holds a task or WHICH fencing generation it runs under; the legacy
    ``:Task`` node CAS-update never landed and still isn't done by any call
    site — AU-P1-CL made the native WorkItem lease the SOLE win/lose
    authority and the legacy ``:Task`` node stays a read-only historical
    mirror nothing here updates post-migration.
    """
    b = EpistemicGraphBackend()
    _add_task(b, "job-1", target="/x")
    h = _ClaimHarness(backend=b)

    result = h._claim_next_task()

    assert result is not None
    job_id, meta = result
    assert job_id == "job-1"
    # _add_task never passes metadata= to submit_work_item, so the only keys
    # present are the D-W2-12 lease-identity stamp added on top of the (here,
    # empty) raw WorkItem metadata.
    assert meta == {
        "claimed_by": TOKEN,
        "work_item_epoch": 1,
        "work_item_id": wi.ingest_task_work_item_id("job-1"),
    }

    # Legacy :Task node is untouched by the native claim (no post-migration
    # mirror step exists).
    assert _task_status(b, "job-1") == "pending"

    # The shadow WorkItem is the REAL authority: attempt=1, leased by this
    # host's token. Native ClaimWorkItem never promotes an ingest_task item
    # past "leased" — work_item.mark_running's own docstring documents
    # "leased"/"running" as ONE engine-native ownership decision (no separate
    # native transition exists for this WorkItem kind; only the unrelated
    # AgentTask kind ever stores the literal "running" string) — so "leased"
    # is the correct post-claim status here, not "running".
    work_item_id = wi.ingest_task_work_item_id("job-1")
    item = wi.get_work_item(h._work_item_engine, work_item_id)
    assert item is not None
    assert item["status"] == "leased"
    assert item["attempt"] == 1
    assert item["lease_owner"] == TOKEN


def test_claim_skips_candidate_whose_shadow_already_claimed():
    """A candidate whose shadow WorkItem a peer already claimed (still within
    its lease) is skipped — the claimer moves on to the next candidate."""
    b = EpistemicGraphBackend()
    _add_task(b, "job-lost", target="/x")
    _add_task(b, "job-won", target="/y")
    h = _ClaimHarness(backend=b)

    # A peer already won job-lost's shadow (fresh lease — not stale). Its
    # WorkItem was already submitted by _add_task above; just claim it.
    peer_item_id = wi.ingest_task_work_item_id("job-lost")
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
    """No pending WorkItems → no claim attempt, returns None."""
    h = _ClaimHarness()

    assert h._claim_next_task() is None


def test_claim_returns_none_when_all_candidates_already_claimed():
    """Every candidate's shadow is already claimed elsewhere → idle (None),
    never a phantom claim."""
    b = EpistemicGraphBackend()
    _add_task(b, "a", target="/a")
    _add_task(b, "b", target="/b")
    h = _ClaimHarness(backend=b)

    for job_id in ("a", "b"):
        item_id = wi.ingest_task_work_item_id(job_id)
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
        return _ClaimHarness(backend=b)

    first = make_harness()._claim_next_task()
    second = make_harness()._claim_next_task()

    assert first is not None and first[0] == "job-shared"  # winner
    assert second is None  # loser got no claim, no other candidate
    # The legacy :Task node is never mirrored on claim (see
    # test_claim_wins_creates_running_shadow_and_stamps_task's docstring) —
    # the native WorkItem lease above is the sole win/lose authority.
    assert _task_status(b, "job-shared") == "pending"


class _MetaStampHarness:
    """Minimal ``TaskManagerMixin`` host that mocks the ingestion-queue
    ``IngestionMixin._claim_next_task`` (``engine_tasks.py``) plumbing rather
    than exercising a real ``EpistemicGraphBackend`` — this proves the D-W2-12
    fix (the returned ``meta`` dict is stamped with the winning claim's own
    lease identity) without depending on the live engine's control-plane
    wiring, which the other tests in this file need for their real
    ``:Task``/``:WorkItem`` state machine coverage.
    """

    control_backend = object()
    backend = object()
    # Never dereferenced -- claim_next/mark_running are mocked below, so this
    # only needs to exist as an opaque attribute the real _claim_next_task
    # passes through unexamined.
    _work_item_engine = object()

    def __init__(self) -> None:
        self._active_work_item_claims: dict = {}
        self._active_work_item_claims_lock = threading.Lock()

    def _get_host_token(self) -> str:
        return "claimhost:9:1700000009"

    def _ingest_task_metadata(self, job_id: str) -> dict:
        return {"type": "document", "target": "/x"}

    _remember_work_item_claim = TaskManagerMixin._remember_work_item_claim
    _active_work_item_claim = TaskManagerMixin._active_work_item_claim
    _claim_next_task = TaskManagerMixin._claim_next_task


def test_claim_next_task_stamps_claimed_by_and_work_item_epoch_onto_meta(monkeypatch):
    """D-W2-12: ``_claim_next_task`` must surface WHO claimed a task and under
    which lease epoch on the ``meta`` dict it hands back — previously it
    silently dropped both, even though the winning ``claim`` already carries
    them (``lease_owner``/``lease_epoch``, CONCEPT:AU-KG native WorkItem
    lease fields)."""
    h = _MetaStampHarness()
    fake_claim = {
        "work_item_id": "wi:ingest_task:job-1",
        "kind": "ingest_task",
        "payload_ref": "job-1",
        "lease_owner": h._get_host_token(),
        "lease_epoch": 1,
        "fence_token": 1,
        "fencing_token": 1,
        "attempt": 1,
        "max_attempts": 3,
        "_native": True,
    }
    monkeypatch.setattr(wi, "claim_next", lambda *a, **kw: fake_claim)
    monkeypatch.setattr(wi, "mark_running", lambda *a, **kw: True)

    result = h._claim_next_task()

    assert result is not None
    job_id, meta = result
    assert job_id == "job-1"
    assert meta["claimed_by"] == h._get_host_token()
    assert meta["work_item_epoch"] == 1
    assert meta["work_item_id"] == "wi:ingest_task:job-1"
