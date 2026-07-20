"""Phase-2/3 autonomy + economics: adaptive model router, autonomy ramp, goal SLA.

CONCEPT:AU-ORCH.routing.adaptive-role-routing (router), OS-5.49 (autonomy ramp), ORCH-1.78 (goals-as-contracts).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from agent_utilities.core import model_router
from agent_utilities.core.goal_sla import assess_goal_sla, evaluate_goal_slas
from agent_utilities.orchestration.autonomy_ramp import (
    assess_trust,
    clears_ramp,
    record_trust,
    reset_trust,
)

_NOW = datetime(2026, 6, 21, 12, 0, 0, tzinfo=UTC)


# ── ORCH-1.79 adaptive model router ───────────────────────────────────────────
@dataclass
class _RoleSpec:
    tier: str = "medium"
    tags: list | None = None


class _FakeRegistry:
    models = [object()]  # non-empty

    def __init__(self):
        self.seen_confidence = None

    def resolve_role(self, role):
        return _RoleSpec(tier="medium", tags=[])

    def pick_for_task_adaptive(
        self, *, complexity, confidence_signal, routing_percentile, required_tags
    ):
        self.seen_confidence = confidence_signal
        return f"model@{complexity}:{confidence_signal:.2f}"


@pytest.mark.concept("AU-ORCH.routing.adaptive-role-routing")
def test_router_confidence_starts_neutral_and_learns():
    reset = model_router.reset_routes
    reset()
    key = model_router.route_key(role="planner")
    assert model_router.route_confidence(key) == 0.5
    # repeated successes raise confidence; failures lower it
    for _ in range(5):
        model_router.record_model_outcome(key, success=True)
    assert model_router.route_confidence(key) > 0.5
    model_router.reset_routes()
    for _ in range(5):
        model_router.record_model_outcome(key, success=False)
    assert model_router.route_confidence(key) < 0.5
    reset()


@pytest.mark.concept("AU-ORCH.routing.adaptive-role-routing")
def test_pick_adaptive_feeds_learned_confidence():
    model_router.reset_routes()
    reg = _FakeRegistry()
    # seed high confidence for planner → router passes it to pick_for_task_adaptive
    for _ in range(8):
        model_router.record_model_outcome(
            model_router.route_key(role="planner"), success=True
        )
    out = model_router.pick_adaptive(reg, "planner")
    assert out is not None
    assert reg.seen_confidence > 0.5  # learned signal threaded through
    model_router.reset_routes()


@pytest.mark.concept("AU-ORCH.routing.adaptive-role-routing")
def test_pick_adaptive_none_without_registry():
    assert model_router.pick_adaptive(None, "planner") is None

    class _Empty:
        models = []

    assert model_router.pick_adaptive(_Empty(), "planner") is None


# ── OS-5.49 autonomy ramp ─────────────────────────────────────────────────────
@pytest.mark.concept("AU-OS.governance.autonomy-change-proposer")
def test_assess_trust_predicate():
    assert assess_trust(20, 19, min_samples=20, threshold=0.9) is True
    assert assess_trust(20, 17, min_samples=20, threshold=0.9) is False  # 85% < 90%
    assert assess_trust(5, 5, min_samples=20, threshold=0.9) is False  # too few samples


@pytest.mark.concept("AU-OS.governance.autonomy-change-proposer")
def test_record_trust_and_clears_ramp():
    reset_trust()
    for _ in range(19):
        record_trust(None, "claude", "ticket.close", success=True)
    record_trust(None, "claude", "ticket.close", success=False)  # 19/20 = 95%
    assert clears_ramp(None, "claude", "ticket.close", min_samples=20, threshold=0.9)
    assert not clears_ramp(None, "claude", "deploy.restart")  # no history
    reset_trust()


@pytest.mark.concept("AU-OS.governance.autonomy-change-proposer")
def test_action_policy_graduates_only_allowlisted_and_trusted(monkeypatch):
    from agent_utilities.orchestration.action_policy import (
        TIER_APPROVAL,
        TIER_AUTO_NOTIFY,
        ActionPolicy,
        ActionRequest,
    )

    reset_trust()
    pol = ActionPolicy(engine=None)
    # a rule that asks for approval
    monkeypatch.setattr(
        pol, "_match", lambda req: (type("R", (), {"tier": TIER_APPROVAL})(), {})
    )
    monkeypatch.setattr(
        pol, "_load_file_policy", lambda: {"ramp_eligible": ["ticket.*"]}
    )

    req = ActionRequest(kind="ticket.close", target="SN-1", actor_id="claude")
    # no trust yet -> stays ask
    assert pol.classify(req) == TIER_APPROVAL
    for _ in range(20):
        record_trust(None, "claude", "ticket.close", success=True)
    # now graduates to auto_notify
    assert pol.classify(req) == TIER_AUTO_NOTIFY
    # a NON-allowlisted kind never graduates even when trusted
    for _ in range(20):
        record_trust(None, "claude", "deploy.restart", success=True)
    req2 = ActionRequest(kind="deploy.restart", target="x", actor_id="claude")
    assert pol.classify(req2) == TIER_APPROVAL
    reset_trust()


# ── ORCH-1.78 goals-as-contracts SLA ──────────────────────────────────────────
@pytest.mark.concept("AU-ORCH.session.escalate-breached-goals")
def test_assess_goal_sla_states():
    base = _NOW.timestamp()
    # open 30s with a 3600s SLA -> on_track
    assert (
        assess_goal_sla(created_at=base - 30, sla_seconds=3600, now=_NOW)["state"]
        == "on_track"
    )
    # open 3000s of 3600 (>80%) -> at_risk
    assert (
        assess_goal_sla(created_at=base - 3000, sla_seconds=3600, now=_NOW)["state"]
        == "at_risk"
    )
    # open 4000s of 3600 -> breached
    v = assess_goal_sla(created_at=base - 4000, sla_seconds=3600, now=_NOW)
    assert v["state"] == "breached" and v["breach"] is True
    # no sla -> never escalated
    assert (
        assess_goal_sla(created_at=base - 99999, sla_seconds=None, now=_NOW)["state"]
        == "no_sla"
    )


@pytest.mark.concept("AU-ORCH.session.escalate-breached-goals")
def test_evaluate_goal_slas_collects_breaches():
    base = _NOW.timestamp()

    class _GoalEngine:
        def query_cypher(self, cypher, params):
            if "loop_kind = 'develop'" in cypher:
                return [
                    {
                        "id": "goal:a",
                        "objective": "triage P1s",
                        "status": "running",
                        "created_at": base - 5000,
                        "sla_seconds": 3600,
                        "escalate_to": "",
                    },
                    {
                        "id": "goal:b",
                        "objective": "slow",
                        "status": "running",
                        "created_at": base - 10,
                        "sla_seconds": 3600,
                        "escalate_to": "",
                    },
                ]
            return []

    rep = evaluate_goal_slas(_GoalEngine(), now=_NOW)
    assert rep["checked"] == 2
    assert [b["goal"] for b in rep["breached"]] == ["goal:a"]
