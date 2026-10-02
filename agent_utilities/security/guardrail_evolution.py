"""Guardrail profile evolution: evidence proposes, bounds and approvals dispose
(AU-SEC-R007).

One evolution step for one throttled EG ``CapacityCell``:

1. A LOOSENING a human already approved for the cell's current epoch is
   applied first (the approval binds the exact plan and epoch, so it applies at
   most once and never to a later, identical-looking move).
2. Otherwise the cell's recorded AIMD history (:class:`ThrottleEvidence`)
   proposes hold / tighten / loosen through a deterministic, auditable rule
   (:func:`fallback_move`): tighten after repeated error-budget narrowings,
   loosen only after a long run of healthy windows at the full declared
   capacity, otherwise hold. Only the moves the operator's declared ladder
   allows are ever offered, so no answer can leave the declared bounds.
3. A tightening inside the operator's declared ladder is applied at once, as
   a compare-and-swap on the cell epoch.
4. A loosening is NEVER applied on a proposal: it is filed as an
   ``action.approval`` for a human, whatever tier the action policy file gives
   the kind, and waits for step 1 on a later tick.

Learned signals only tighten; anything that relaxes a guardrail needs a
person.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from agent_utilities.security.guardrail_profile import (
    GuardrailBounds,
    ProfileMove,
    ProfilePlan,
    ThrottlePolicy,
    level_of,
    plan_move,
)

__all__ = [
    "ActionPolicyLoosenApprovals",
    "EgProfileStore",
    "EvolutionOutcome",
    "EvolutionResult",
    "GuardrailEvolution",
    "GuardrailEvolutionError",
    "LOOSEN_ACTION_KIND",
    "LOOSEN_AFTER_HEALTHY_WINDOWS",
    "LoosenApprovals",
    "ProfileStore",
    "TIGHTEN_AFTER_NARROWINGS",
    "ThrottleEvidence",
    "fallback_move",
]

#: The ``action.approval`` kind a loosening waits on.
LOOSEN_ACTION_KIND = "guardrail.loosen"
_APPLIED = frozenset({"accepted", "replayed"})
#: Narrowings in the recorded history that make the deterministic rule tighten.
TIGHTEN_AFTER_NARROWINGS = 3
#: Counted healthy windows (and no narrowing) before the rule proposes loosening.
LOOSEN_AFTER_HEALTHY_WINDOWS = 16
_HEALTHY = frozenset({"healthy_window", "within_budget", "at_declared_ceiling"})
_UNCOUNTED = frozenset({"insufficient_samples"})


@dataclass(frozen=True, slots=True)
class ThrottleEvidence:
    """What one cell's recorded throttle history says (AU-SEC-R007)."""

    level: int
    levels: int
    windows: int
    narrowed: int
    healthy: int
    at_capacity: bool

    @classmethod
    def from_cell(
        cls, cell: Mapping[str, Any], level: int, levels: int
    ) -> ThrottleEvidence:
        throttle = cell.get("throttle") or {}
        history: Sequence[Mapping[str, Any]] = throttle.get("history") or ()
        counted = [step for step in history if step.get("reason") not in _UNCOUNTED]
        return cls(
            level=level,
            levels=levels,
            windows=len(counted),
            narrowed=sum(1 for step in counted if step.get("action") == "narrowed"),
            healthy=sum(1 for step in counted if step.get("reason") in _HEALTHY),
            at_capacity=int(throttle.get("ceiling", -1))
            == int(cell.get("capacity", 0)),
        )

    def legal_moves(self) -> list[ProfileMove]:
        moves = [ProfileMove.HOLD]
        if self.level < self.levels - 1:
            moves.append(ProfileMove.TIGHTEN)
        if self.level > 0:
            moves.append(ProfileMove.LOOSEN)
        return moves


def fallback_move(evidence: ThrottleEvidence) -> ProfileMove:
    """The deterministic, auditable rule that proposes hold/tighten/loosen.

    Learned signals may only ever narrow what this rule would already do
    (never loosen more readily); today it IS the proposer, so every move it
    returns is already the most conservative legal choice.
    """
    legal = evidence.legal_moves()
    if evidence.narrowed >= TIGHTEN_AFTER_NARROWINGS:
        wanted = ProfileMove.TIGHTEN
    elif (
        evidence.narrowed == 0
        and evidence.windows >= LOOSEN_AFTER_HEALTHY_WINDOWS
        and evidence.healthy == evidence.windows
        and evidence.at_capacity
    ):
        wanted = ProfileMove.LOOSEN
    else:
        wanted = ProfileMove.HOLD
    return wanted if wanted in legal else ProfileMove.HOLD


class GuardrailEvolutionError(RuntimeError):
    """A refused or failed application, with a stable machine code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


class EvolutionOutcome(StrEnum):
    """What one evolution step did."""

    UNMANAGED = "unmanaged"
    HELD = "held"
    TIGHTENED = "tightened"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVAL_UNAVAILABLE = "approval_unavailable"
    LOOSENED = "loosened"


@dataclass(frozen=True, slots=True)
class EvolutionResult:
    """One step's outcome, with the ids that make it auditable."""

    cell_id: str
    outcome: EvolutionOutcome
    move: ProfileMove = ProfileMove.HOLD
    from_level: int | None = None
    to_level: int | None = None
    approval_id: str | None = None


class ProfileStore(Protocol):
    """Where a cell's live policy is read and replaced."""

    async def cell(self, cell_id: str) -> Mapping[str, Any] | None: ...

    async def apply(self, plan: ProfilePlan, cell: Mapping[str, Any]) -> None: ...


class LoosenApprovals(Protocol):
    """The human approval a loosening waits on."""

    async def granted(self, plan: ProfilePlan) -> str | None:
        """The durable approval bound to exactly ``plan``; files nothing."""
        ...

    async def request(self, plan: ProfilePlan) -> str | None:
        """File (or find) the pending approval for ``plan``."""
        ...


def _managed_level(
    bounds: GuardrailBounds, cell: Mapping[str, Any] | None
) -> int | None:
    throttle = None if cell is None else cell.get("throttle")
    if not isinstance(throttle, Mapping):
        return None
    return level_of(bounds, ThrottlePolicy.model_validate(throttle["policy"]))


def _result(
    plan: ProfilePlan, outcome: EvolutionOutcome, **ids: Any
) -> EvolutionResult:
    return EvolutionResult(
        cell_id=plan.cell_id,
        outcome=outcome,
        move=plan.move,
        from_level=plan.from_level,
        to_level=plan.to_level,
        **ids,
    )


class GuardrailEvolution:
    """Runs one bounded evolution step per call; holds no state between calls."""

    def __init__(self, store: ProfileStore, approvals: LoosenApprovals) -> None:
        self._store = store
        self._approvals = approvals

    async def evolve(self, cell_id: str, bounds: GuardrailBounds) -> EvolutionResult:
        cell = await self._store.cell(cell_id)
        level = _managed_level(bounds, cell)
        if cell is None or level is None:
            return EvolutionResult(cell_id=cell_id, outcome=EvolutionOutcome.UNMANAGED)
        approved = await self._approved_loosening(bounds, cell)
        if approved is not None:
            return approved
        evidence = ThrottleEvidence.from_cell(cell, level, bounds.levels)
        move = fallback_move(evidence)
        plan = plan_move(bounds, cell, move)
        if plan is None:
            return EvolutionResult(cell_id=cell_id, outcome=EvolutionOutcome.HELD)
        if move is ProfileMove.TIGHTEN:
            await self._store.apply(plan, cell)
            return _result(plan, EvolutionOutcome.TIGHTENED)
        return await self._file_loosening(plan)

    async def _approved_loosening(
        self, bounds: GuardrailBounds, cell: Mapping[str, Any]
    ) -> EvolutionResult | None:
        plan = plan_move(bounds, cell, ProfileMove.LOOSEN)
        if plan is None:
            return None
        granted = await self._approvals.granted(plan)
        if granted is None:
            return None
        await self._store.apply(plan, cell)
        return _result(plan, EvolutionOutcome.LOOSENED, approval_id=granted)

    async def _file_loosening(self, plan: ProfilePlan) -> EvolutionResult:
        pending = await self._approvals.request(plan)
        outcome = (
            EvolutionOutcome.AWAITING_APPROVAL
            if pending
            else EvolutionOutcome.APPROVAL_UNAVAILABLE
        )
        return _result(plan, outcome, approval_id=pending)


def _clock_ms() -> int:
    return time.time_ns() // 1_000_000


class EgProfileStore:
    """:class:`ProfileStore` over one tenant's EG capacity ledger.

    ``client`` is the tenant's async EG client bound to a verified context
    that holds ``capacity:read`` and ``capacity:admin`` (a policy is an
    operator-owned declaration; only its ladder-bounded tightening and an
    approved loosening are written here).
    """

    def __init__(
        self, client: Any, *, tenant_ref: str, clock_ms: Callable[[], int] = _clock_ms
    ) -> None:
        self._capacity = client.capacity_leases
        self._tenant = tenant_ref
        self._clock_ms = clock_ms

    async def cell(self, cell_id: str) -> Mapping[str, Any] | None:
        answer = await self._capacity.status(
            {
                "schema_version": "1",
                "tenant_ref": self._tenant,
                "cell_id": cell_id,
                "lease_id": None,
                "max_count": 1,
                "cursor": None,
            }
        )
        cells = [row for row in answer["cells"] if row.get("cell_id") == cell_id]
        return cells[0] if cells else None

    async def apply(self, plan: ProfilePlan, cell: Mapping[str, Any]) -> None:
        now_ms = self._clock_ms()
        throttle = {**cell["throttle"], "policy": plan.policy.model_dump()}
        answer = await self._capacity.update_cell(
            {
                "schema_version": "1",
                "cell": {
                    **cell,
                    "throttle": throttle,
                    "epoch": plan.epoch + 1,
                    "updated_at_ms": now_ms,
                },
                "expected_epoch": plan.epoch,
                "now_ms": now_ms,
            }
        )
        if answer["decision"] not in _APPLIED:
            raise GuardrailEvolutionError(
                "GUARDRAIL_APPLY_REFUSED", str(answer["decision"])
            )


class ActionPolicyLoosenApprovals:
    """:class:`LoosenApprovals` on the fleet ``action.approval`` queue.

    ``policy`` is an :class:`~agent_utilities.orchestration.action_policy.ActionPolicy`
    bound to the verified engine session. Only a DURABLE approval bound to the
    plan's exact digest counts as granted: a kind the policy file tiers as
    ``auto`` is still filed for a human, because a loosening is never
    automatic.
    """

    def __init__(self, policy: Any) -> None:
        self._policy = policy

    def _request(self, plan: ProfilePlan) -> Any:
        from agent_utilities.orchestration.action_policy import ActionRequest

        return ActionRequest(
            kind=LOOSEN_ACTION_KIND,
            target=plan.cell_id,
            params=plan.approval_params(),
            source="guardrail_evolution",
            reason="AU-SEC-R007: loosen one declared guardrail level after sustained health",
        )

    def _granted(self, plan: ProfilePlan) -> str | None:
        return self._policy.granted_approval_id(self._request(plan))

    def _file(self, plan: ProfilePlan) -> str | None:
        from agent_utilities.orchestration.action_policy import DECISION_QUEUE

        request = self._request(plan)
        decision = self._policy.decide(request)
        if decision.decision == DECISION_QUEUE:
            return decision.approval_id
        if not decision.allowed:
            return None
        return self._policy.queue_approval(request)

    async def granted(self, plan: ProfilePlan) -> str | None:
        return await asyncio.to_thread(self._granted, plan)

    async def request(self, plan: ProfilePlan) -> str | None:
        return await asyncio.to_thread(self._file, plan)
