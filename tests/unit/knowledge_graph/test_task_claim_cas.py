"""Backend-agnostic :Task claim via the engine compare-and-set (CONCEPT:AU-KG.compute.user-override-prompt-library).

The worker claim used to be a single-host ``threading.Lock`` + Postgres advisory
lock guarding a read-then-``SET``. It is now arbitrated by the engine's atomic
``compare_and_set`` (held under the graph write lock), so the claim is exactly-
once *across hosts* regardless of the configured mirror backend.

These exercise ``TaskManagerMixin._claim_next_task`` with the backend CAS mocked
(no live engine): the claim must CAS ``status: pending → running`` and stamp
ownership, must skip a candidate whose CAS lost, and two sequential claims of the
same row must produce one winner and one loser.
"""

from unittest.mock import MagicMock

from agent_utilities.knowledge_graph.core.engine_tasks import (
    TaskManagerMixin,
    _decode_metadata,
)

TOKEN = "claimhost:333:1700000003"


class _ClaimHarness:
    """Minimal object exposing exactly what _claim_next_task touches.

    ``_select_pending_task`` is stubbed to hand back a controlled queue of
    candidate rows (the bucket-ascending selection is covered elsewhere); the
    backend's ``compare_and_set_node_fields`` is a mock so no engine is needed.
    """

    def __init__(self, candidates, cas_results):
        self.backend = MagicMock()
        # cas_results: list[bool] consumed per CAS attempt (winner/loser).
        self.backend.compare_and_set_node_fields.side_effect = list(cas_results)
        self._candidates = list(candidates)
        self._tok = TOKEN

    def _select_pending_task(self, admit=None):
        # ORCH-1.81 added an admission predicate; this harness hands back a
        # controlled candidate queue regardless (admission is tested separately).
        return self._candidates.pop(0) if self._candidates else None

    def _make_admission(self):
        # ORCH-1.81: disable the admission gate for these pure CAS tests.
        return None

    def _get_host_token(self) -> str:
        return self._tok

    # The claim CAS now routes through the control-plane accessor
    # (CONCEPT:AU-KG.backend.schedule-on-control-graph): ``_control`` falls back to ``self.backend`` when no
    # isolated ``control_backend`` is set, so binding the real property keeps the
    # CAS pointed at the mocked backend in these tests.
    control_backend = None
    _control = TaskManagerMixin._control
    _control_cypher = TaskManagerMixin._control_cypher

    # Bind the real method under test.
    _claim_next_task = TaskManagerMixin._claim_next_task


def test_claim_uses_cas_with_pending_condition_and_running_update():
    """A winning claim CASes status pending→running and stamps ownership."""
    h = _ClaimHarness(
        candidates=[{"id": "job-1", "meta": None}],
        cas_results=[True],
    )

    result = h._claim_next_task()

    assert result is not None
    job_id, meta = result
    assert job_id == "job-1"

    # Exactly one CAS, with the right condition + updates.
    h.backend.compare_and_set_node_fields.assert_called_once()
    args, kwargs = h.backend.compare_and_set_node_fields.call_args
    called_id, conditions, updates = args
    assert called_id == "job-1"
    assert conditions == {"status": "pending"}
    assert updates["status"] == "running"

    # The metadata update is the encoded ownership stamp the reaper decodes.
    stamped = _decode_metadata(updates["metadata"])
    assert stamped["claimed_by"] == TOKEN
    assert "claim_unix" in stamped
    assert "started_at" in stamped
    # Returned meta matches what was written.
    assert meta["claimed_by"] == TOKEN


def test_claim_skips_candidate_whose_cas_lost():
    """A CAS that returns False (peer won the row) is not treated as claimed;
    the claimer moves on to the next candidate."""
    h = _ClaimHarness(
        candidates=[{"id": "job-lost", "meta": None}, {"id": "job-won", "meta": None}],
        cas_results=[False, True],
    )

    result = h._claim_next_task()

    assert result is not None
    job_id, _meta = result
    assert job_id == "job-won"  # the lost candidate was skipped
    assert h.backend.compare_and_set_node_fields.call_count == 2
    claimed_ids = [
        c.args[0] for c in h.backend.compare_and_set_node_fields.call_args_list
    ]
    assert claimed_ids == ["job-lost", "job-won"]


def test_claim_returns_none_when_idle():
    """No pending candidates → no CAS, returns None."""
    h = _ClaimHarness(candidates=[], cas_results=[])

    assert h._claim_next_task() is None
    h.backend.compare_and_set_node_fields.assert_not_called()


def test_claim_returns_none_when_all_candidates_lost():
    """Every candidate lost its CAS → returns None (idle), never a phantom claim."""
    h = _ClaimHarness(
        candidates=[{"id": "a", "meta": None}, {"id": "b", "meta": None}],
        cas_results=[False, False],
    )

    assert h._claim_next_task() is None
    assert h.backend.compare_and_set_node_fields.call_count == 2


def test_two_sequential_claims_of_same_task_first_wins_second_loses():
    """First claimer wins (True); a second claimer of the SAME row loses (False).

    Models the cross-host race the engine CAS now arbitrates: only one flip of
    pending→running succeeds.
    """
    backend = MagicMock()
    # The row, in a shared store. After the first CAS wins, it's 'running'; a
    # second CAS guarded on status=='pending' must fail.
    state = {"status": "pending"}

    def fake_cas(node_id, conditions, updates):
        for k, v in conditions.items():
            if state.get(k) != v:
                return False
        state.update(updates)
        return True

    backend.compare_and_set_node_fields.side_effect = fake_cas

    def make_harness():
        h = _ClaimHarness(candidates=[], cas_results=[])
        h.backend = backend
        # Both claimers see the same single pending row on their first select.
        h._candidates = [{"id": "job-shared", "meta": None}]
        return h

    first = make_harness()._claim_next_task()
    second = make_harness()._claim_next_task()

    assert first is not None and first[0] == "job-shared"  # winner
    assert second is None  # loser got False → no claim, no other candidate
    assert state["status"] == "running"
