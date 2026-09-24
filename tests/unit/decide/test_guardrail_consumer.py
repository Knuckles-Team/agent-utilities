"""EH-407: guardrail profiles evolve through Decide -- tighten in bounds, loosen by approval."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any

import pytest

from agent_utilities.orchestration.action_policy import (
    DECISION_ALLOW,
    DECISION_DENY,
    DECISION_QUEUE,
)
from agent_utilities.security.guardrail_evolution import (
    ActionPolicyLoosenApprovals,
    EgProfileStore,
    EvolutionOutcome,
    GuardrailEvolution,
    GuardrailEvolutionError,
)
from agent_utilities.security.guardrail_profile import (
    ProfileMove,
    ProfilePlan,
    plan_move,
)
from tests.unit.decide.fakes import FakeTransport, abstained, acted
from tests.unit.security.test_guardrail_profile import bounds, cell, policy_at

CELL = "fleet/child/github"


def _history(narrowed: int, healthy: int) -> list[dict[str, Any]]:
    steps = [{"action": "narrowed", "reason": "error_budget_exceeded"}] * narrowed
    return steps + [{"action": "held", "reason": "within_budget"}] * healthy


@dataclass
class FakeStore:
    cells: dict[str, dict[str, Any]] = field(default_factory=dict)
    applied: list[ProfilePlan] = field(default_factory=list)

    async def cell(self, cell_id: str) -> dict[str, Any] | None:
        return copy.deepcopy(self.cells.get(cell_id))

    async def apply(self, plan: ProfilePlan, current: dict[str, Any]) -> None:
        live = self.cells[plan.cell_id]
        assert live["epoch"] == plan.epoch, "an apply is a CAS on the epoch"
        live["throttle"]["policy"] = plan.policy.model_dump()
        live["epoch"] += 1
        self.applied.append(plan)


@dataclass
class FakeApprovals:
    granted_plans: set[str] = field(default_factory=set)
    filed: list[ProfilePlan] = field(default_factory=list)
    available: bool = True

    @staticmethod
    def key(plan: ProfilePlan) -> str:
        return json.dumps(plan.approval_params(), sort_keys=True)

    async def granted(self, plan: ProfilePlan) -> str | None:
        return "approval:ok" if self.key(plan) in self.granted_plans else None

    async def request(self, plan: ProfilePlan) -> str | None:
        self.filed.append(plan)
        return "approval:pending" if self.available else None


def _world(level: int, narrowed: int = 0, healthy: int = 0) -> tuple[Any, ...]:
    live = cell(level)
    live["throttle"]["history"] = _history(narrowed, healthy)
    store = FakeStore({CELL: live})
    approvals = FakeApprovals()
    return store, approvals, GuardrailEvolution(store, approvals)


async def test_repeated_narrowings_tighten_one_level_without_a_human(
    eg: FakeTransport,
) -> None:
    eg.answer = abstained()
    store, approvals, evolution = _world(1, narrowed=3, healthy=5)
    result = await evolution.evolve(CELL, bounds())
    assert result.outcome is EvolutionOutcome.TIGHTENED
    assert (result.from_level, result.to_level) == (1, 2)
    assert store.cells[CELL]["throttle"]["policy"] == policy_at(2)
    assert approvals.filed == []
    assert eg.requests[0]["question"]["safety"] == "policy", "never explored"


async def test_eg_may_propose_a_tightening_the_rule_would_not_make(
    eg: FakeTransport,
) -> None:
    eg.answer = acted("tighten")
    store, _approvals, evolution = _world(0, healthy=4)
    assert (
        await evolution.evolve(CELL, bounds())
    ).outcome is EvolutionOutcome.TIGHTENED
    assert store.cells[CELL]["epoch"] == 8


async def test_a_proposed_loosening_waits_for_a_human(eg: FakeTransport) -> None:
    eg.answer = acted("loosen")
    store, approvals, evolution = _world(2, healthy=20)
    result = await evolution.evolve(CELL, bounds())
    assert result.outcome is EvolutionOutcome.AWAITING_APPROVAL
    assert result.approval_id == "approval:pending"
    assert store.applied == [], "a loosening is never applied on a proposal"
    assert approvals.filed[0].to_level == 1


async def test_an_approved_loosening_applies_once_and_cannot_be_replayed(
    eg: FakeTransport,
) -> None:
    eg.answer = abstained()
    store, approvals, evolution = _world(2)
    approvals.granted_plans.add(FakeApprovals.key(approvals_plan(store)))
    first = await evolution.evolve(CELL, bounds())
    assert first.outcome is EvolutionOutcome.LOOSENED
    assert first.approval_id == "approval:ok" and eg.requests == []
    # Back at level 2 by a later tightening: the old approval named epoch 7.
    store.cells[CELL]["throttle"]["policy"] = policy_at(2)
    again = await evolution.evolve(CELL, bounds())
    assert again.outcome is EvolutionOutcome.HELD
    assert len(store.applied) == 1


def approvals_plan(store: FakeStore) -> ProfilePlan:
    plan = plan_move(bounds(), store.cells[CELL], ProfileMove.LOOSEN)
    assert plan is not None
    return plan


async def test_no_move_past_the_tightest_level_is_even_offered(
    eg: FakeTransport,
) -> None:
    eg.answer = acted("tighten")
    store, _approvals, evolution = _world(4, narrowed=6)
    result = await evolution.evolve(CELL, bounds())
    assert result.outcome is EvolutionOutcome.HELD
    offered = {
        option["option_id"] for option in eg.requests[0]["candidates"]["options"]
    }
    assert "tighten" not in offered and store.applied == []


async def test_a_hand_set_policy_is_left_alone(eg: FakeTransport) -> None:
    store, _approvals, evolution = _world(1, narrowed=9)
    store.cells[CELL]["throttle"]["policy"]["error_budget_ppm"] = 12_345
    result = await evolution.evolve(CELL, bounds())
    assert result.outcome is EvolutionOutcome.UNMANAGED
    assert eg.requests == [] and store.applied == []


async def test_a_loosening_with_no_approval_queue_reports_it(eg: FakeTransport) -> None:
    eg.answer = acted("loosen")
    _store, approvals, evolution = _world(3, healthy=20)
    approvals.available = False
    result = await evolution.evolve(CELL, bounds())
    assert result.outcome is EvolutionOutcome.APPROVAL_UNAVAILABLE


class _Policy:
    def __init__(self, decision: str, approval_id: str | None = None) -> None:
        self.answer = SimpleNamespace(
            decision=decision,
            approval_id=approval_id,
            allowed=decision == DECISION_ALLOW,
        )
        self.queued: list[Any] = []
        self.granted_ids: dict[str, str] = {}

    def decide(self, request: Any) -> Any:
        return self.answer

    def queue_approval(self, request: Any) -> str:
        self.queued.append(request)
        return "approval:queued"

    def granted_approval(self, request: Any) -> str | None:
        return self.granted_ids.get(request.digest())


def _plan() -> ProfilePlan:
    plan = plan_move(bounds(), cell(2), ProfileMove.LOOSEN)
    assert plan is not None
    return plan


@pytest.mark.parametrize(
    ("decision", "approval_id", "expected", "queued"),
    [
        (DECISION_ALLOW, None, "approval:queued", 1),
        (DECISION_QUEUE, "approval:held", "approval:held", 0),
        (DECISION_DENY, None, None, 0),
    ],
)
async def test_a_loosening_is_filed_for_a_human_whatever_the_policy_tier(
    decision: str, approval_id: str | None, expected: str | None, queued: int
) -> None:
    policy = _Policy(decision, approval_id)
    approvals = ActionPolicyLoosenApprovals(policy)
    assert await approvals.request(_plan()) == expected
    assert len(policy.queued) == queued


async def test_only_an_approval_bound_to_the_exact_plan_counts() -> None:
    policy = _Policy(DECISION_ALLOW)
    approvals = ActionPolicyLoosenApprovals(policy)
    plan = _plan()
    assert await approvals.granted(plan) is None
    policy.granted_ids[approvals._request(plan).digest()] = "approval:1"
    assert await approvals.granted(plan) == "approval:1"
    assert await approvals.granted(replace(plan, epoch=plan.epoch + 1)) is None


class _Capacity:
    def __init__(self, decision: str = "accepted") -> None:
        self.decision = decision
        self.updates: list[dict[str, Any]] = []
        self.cells = [cell(1)]

    async def status(self, request: dict[str, Any]) -> dict[str, Any]:
        assert request["cell_id"] == CELL and request["tenant_ref"] == "tenant-a"
        return {
            "schema_version": "1",
            "cells": self.cells,
            "leases": [],
            "next_cursor": None,
        }

    async def update_cell(self, request: dict[str, Any]) -> dict[str, Any]:
        self.updates.append(request)
        return {
            "schema_version": "1",
            "decision": self.decision,
            "cell": None,
            "message": None,
        }


async def test_the_eg_store_replaces_only_the_policy_by_epoch_cas() -> None:
    capacity = _Capacity()
    store = EgProfileStore(
        SimpleNamespace(capacity_leases=capacity),
        tenant_ref="tenant-a",
        clock_ms=lambda: 99,
    )
    current = await store.cell(CELL)
    assert current is not None
    plan = plan_move(bounds(), current, ProfileMove.TIGHTEN)
    assert plan is not None
    await store.apply(plan, current)
    sent = capacity.updates[0]
    assert (sent["expected_epoch"], sent["cell"]["epoch"]) == (7, 8)
    assert sent["cell"]["throttle"]["policy"] == policy_at(2)
    assert sent["cell"]["throttle"]["ceiling"] == 8 and sent["cell"]["capacity"] == 8
    capacity.decision = "stale_epoch"
    with pytest.raises(GuardrailEvolutionError) as refused:
        await store.apply(plan, current)
    assert refused.value.code == "GUARDRAIL_APPLY_REFUSED"
