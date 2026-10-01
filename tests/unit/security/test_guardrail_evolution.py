"""AU-SEC-R007: bounded tightening auto-applies; loosening always waits for a
human approval, filed on the fleet action-approval queue."""

from __future__ import annotations

from typing import Any

from agent_utilities.security.guardrail_evolution import (
    EvolutionOutcome,
    GuardrailEvolution,
    ThrottleEvidence,
    fallback_move,
)
from agent_utilities.security.guardrail_profile import ProfileMove
from tests.unit.security.test_guardrail_profile import bounds as _bounds
from tests.unit.security.test_guardrail_profile import policy_at as _policy_at


def _history(*steps: tuple[str, str]) -> list[dict[str, str]]:
    return [{"action": action, "reason": reason} for action, reason in steps]


def _cell(
    level: int,
    *,
    epoch: int = 7,
    history: list[dict[str, str]] | None = None,
    at_capacity: bool = False,
) -> dict[str, Any]:
    return {
        "cell_id": "fleet/child/github",
        "capacity": 8,
        "epoch": epoch,
        "throttle": {
            "policy": _policy_at(level),
            "ceiling": 8 if at_capacity else 4,
            "history": history or [],
        },
    }


class _FakeStore:
    """An in-memory :class:`ProfileStore`."""

    def __init__(self, initial: dict[str, Any] | None) -> None:
        self._cell = initial
        self.applied: list[Any] = []

    async def cell(self, cell_id: str) -> dict[str, Any] | None:
        return self._cell

    async def apply(self, plan: Any, cell: dict[str, Any]) -> None:
        self.applied.append(plan)
        self._cell = {
            **cell,
            "throttle": {**cell["throttle"], "policy": plan.policy.model_dump()},
            "epoch": plan.epoch + 1,
        }


class _FakeApprovals:
    """An in-memory :class:`LoosenApprovals`: files at most one pending request,
    and only reports a plan granted once a test marks it so."""

    def __init__(self) -> None:
        self.requested: list[Any] = []
        self._granted: Any | None = None

    def grant_next(self) -> None:
        """Mark whatever plan gets requested next as already approved."""
        self._granted = "pending"

    async def granted(self, plan: Any) -> str | None:
        if self._granted == plan.approval_params() or self._granted == "approved":
            return "approval-1"
        return None

    async def request(self, plan: Any) -> str | None:
        self.requested.append(plan)
        if self._granted == "pending":
            self._granted = plan.approval_params()
        return "approval-pending-1"


class TestFallbackMove:
    def test_repeated_narrowings_tighten(self) -> None:
        evidence = ThrottleEvidence(
            level=1, levels=5, windows=5, narrowed=3, healthy=0, at_capacity=False
        )
        assert fallback_move(evidence) is ProfileMove.TIGHTEN

    def test_a_long_healthy_run_at_capacity_loosens(self) -> None:
        evidence = ThrottleEvidence(
            level=2, levels=5, windows=16, narrowed=0, healthy=16, at_capacity=True
        )
        assert fallback_move(evidence) is ProfileMove.LOOSEN

    def test_mixed_evidence_holds(self) -> None:
        evidence = ThrottleEvidence(
            level=2, levels=5, windows=4, narrowed=1, healthy=1, at_capacity=False
        )
        assert fallback_move(evidence) is ProfileMove.HOLD

    def test_a_move_past_the_ladder_edge_never_escapes_as_hold(self) -> None:
        """Tightest level: the rule wants TIGHTEN but it is not a legal move,
        so the evolution step holds rather than leaving the declared bounds."""
        evidence = ThrottleEvidence(
            level=4, levels=5, windows=5, narrowed=3, healthy=0, at_capacity=False
        )
        assert fallback_move(evidence) is ProfileMove.HOLD


class TestGuardrailEvolution:
    async def test_an_unmanaged_cell_is_reported_and_untouched(self) -> None:
        store = _FakeStore({"cell_id": "x", "epoch": 1})
        evolution = GuardrailEvolution(store, _FakeApprovals())
        result = await evolution.evolve("x", _bounds())
        assert result.outcome is EvolutionOutcome.UNMANAGED
        assert store.applied == []

    async def test_a_missing_cell_is_reported_and_untouched(self) -> None:
        store = _FakeStore(None)
        evolution = GuardrailEvolution(store, _FakeApprovals())
        result = await evolution.evolve("missing", _bounds())
        assert result.outcome is EvolutionOutcome.UNMANAGED

    async def test_bounded_tightening_applies_automatically_no_approval_needed(
        self,
    ) -> None:
        cell = _cell(1, history=_history(*[("narrowed", "over_budget")] * 3))
        store = _FakeStore(cell)
        approvals = _FakeApprovals()
        evolution = GuardrailEvolution(store, approvals)

        result = await evolution.evolve(cell["cell_id"], _bounds())

        assert result.outcome is EvolutionOutcome.TIGHTENED
        assert (result.from_level, result.to_level) == (1, 2)
        assert len(store.applied) == 1
        assert approvals.requested == [], "a tightening never touches the approval queue"

    async def test_a_proposed_loosening_is_filed_not_applied(self) -> None:
        cell = _cell(
            2,
            at_capacity=True,
            history=_history(*[("held", "healthy_window")] * 16),
        )
        store = _FakeStore(cell)
        approvals = _FakeApprovals()
        evolution = GuardrailEvolution(store, approvals)

        result = await evolution.evolve(cell["cell_id"], _bounds())

        assert result.outcome is EvolutionOutcome.AWAITING_APPROVAL
        assert store.applied == [], "a loosening is never applied on a proposal"
        assert len(approvals.requested) == 1
        assert approvals.requested[0].move is ProfileMove.LOOSEN

    async def test_a_loosening_applies_only_once_a_human_already_approved_it(
        self,
    ) -> None:
        cell = _cell(2, at_capacity=True)
        store = _FakeStore(cell)
        approvals = _FakeApprovals()
        approvals._granted = "approved"
        evolution = GuardrailEvolution(store, approvals)

        result = await evolution.evolve(cell["cell_id"], _bounds())

        assert result.outcome is EvolutionOutcome.LOOSENED
        assert result.approval_id == "approval-1"
        assert len(store.applied) == 1
        assert store.applied[0].move is ProfileMove.LOOSEN

    async def test_an_off_ladder_hand_set_policy_is_unmanaged(self) -> None:
        cell = _cell(1)
        cell["throttle"]["policy"]["recovery_ppm"] = 1
        store = _FakeStore(cell)
        evolution = GuardrailEvolution(store, _FakeApprovals())

        result = await evolution.evolve(cell["cell_id"], _bounds())

        assert result.outcome is EvolutionOutcome.UNMANAGED
        assert store.applied == []

    async def test_quiet_evidence_holds_without_touching_anything(self) -> None:
        cell = _cell(2, history=_history(("held", "healthy_window")))
        store = _FakeStore(cell)
        approvals = _FakeApprovals()
        evolution = GuardrailEvolution(store, approvals)

        result = await evolution.evolve(cell["cell_id"], _bounds())

        assert result.outcome is EvolutionOutcome.HELD
        assert store.applied == []
        assert approvals.requested == []
