"""The unified artifact-promotion gate (CONCEPT:AU-AHE.evolution.unified-promotion-gate)
— ``evaluate_promotion``/``promote`` generalizing ``skill_gate.evaluate_promotion`` and
``program_optimization.should_promote`` into one comparison rule, then the SAME
``action_policy.decide()`` veto every reserved-kind call site uses.
"""

from __future__ import annotations

from typing import Any

from agent_utilities.harness.reward_signal import RewardSignal
from agent_utilities.orchestration.action_policy import ActionDecision, ActionRequest
from agent_utilities.orchestration.artifact_promotion import (
    PromotionCandidate,
    evaluate_promotion,
    promote,
)


def _candidate(
    *, candidate_value: float, incumbent_value: float | None, **kw: Any
) -> PromotionCandidate:
    return PromotionCandidate(
        artifact_kind=kw.pop("artifact_kind", "skill"),
        artifact_id=kw.pop("artifact_id", "skill:demo"),
        candidate_ref=kw.pop("candidate_ref", "skill_version:demo:abc123"),
        candidate_reward=RewardSignal(value=candidate_value, source="internal_corpus"),
        incumbent_reward=(
            RewardSignal(value=incumbent_value, source="internal_corpus")
            if incumbent_value is not None
            else None
        ),
        **kw,
    )


class _FakePolicy:
    """Recording ActionPolicy double returning a canned decision — mirrors
    ``test_auto_merge_action_policy.py``'s ``_FakePolicy``."""

    def __init__(self, decision: str, *, reason: str = "r", approval_id=None):
        self._decision = decision
        self._reason = reason
        self._approval_id = approval_id
        self.requests: list[ActionRequest] = []

    def decide(self, request: ActionRequest) -> ActionDecision:
        self.requests.append(request)
        return ActionDecision(
            decision=self._decision,
            tier="approval_required",
            request=request,
            reason=self._reason,
            approval_id=self._approval_id,
        )


class _BoomPolicy:
    def decide(self, request: ActionRequest) -> ActionDecision:
        raise RuntimeError("policy backend down")


# ---------------------------------------------------------------------------
# evaluate_promotion — generalizes skill_gate (strict) and should_promote (>=)
# ---------------------------------------------------------------------------


class TestEvaluatePromotion:
    def test_strict_true_requires_strictly_greater(self):
        # skill_gate.evaluate_promotion parity: a tie never promotes.
        assert evaluate_promotion(_candidate(candidate_value=0.7, incumbent_value=0.7)) is False
        assert evaluate_promotion(_candidate(candidate_value=0.71, incumbent_value=0.7)) is True

    def test_strict_false_allows_tie_at_min_delta(self):
        # program_optimization.should_promote parity: candidate >= baseline + min_delta.
        candidate = _candidate(candidate_value=0.7, incumbent_value=0.7)
        assert evaluate_promotion(candidate, strict=False) is True

    def test_min_delta_raises_the_bar(self):
        candidate = _candidate(candidate_value=0.72, incumbent_value=0.7)
        assert evaluate_promotion(candidate, strict=False, min_delta=0.05) is False
        assert evaluate_promotion(candidate, strict=False, min_delta=0.01) is True

    def test_no_incumbent_reward_always_eligible(self):
        # Comparison-less vector (spec/claim shape) — gated elsewhere (quality +
        # governance), not by a held-out score comparison.
        candidate = _candidate(candidate_value=0.1, incumbent_value=None)
        assert evaluate_promotion(candidate) is True


# ---------------------------------------------------------------------------
# promote — the full decision boundary
# ---------------------------------------------------------------------------


class TestPromote:
    def test_losing_candidate_never_consults_action_policy(self):
        policy = _FakePolicy("allow")
        verdict = promote(
            None, _candidate(candidate_value=0.5, incumbent_value=0.7), policy=policy
        )
        assert verdict.eligible is False
        assert verdict.approved is False
        assert verdict.decision == ""
        assert policy.requests == [], "a losing candidate has nothing to decide"

    def test_winning_candidate_consults_action_policy_with_synthesized_kind(self):
        policy = _FakePolicy("queue_approval", approval_id="action_approval:x")
        verdict = promote(
            None, _candidate(candidate_value=0.9, incumbent_value=0.7), policy=policy
        )
        assert verdict.eligible is True
        assert verdict.decision == "queue_approval"
        assert verdict.approved is False
        assert verdict.approval_id == "action_approval:x"
        (request,) = policy.requests
        assert request.kind == "promote_skill_version"
        assert request.target == "skill:demo"

    def test_policy_kind_override_is_used_verbatim(self):
        policy = _FakePolicy("allow")
        candidate = _candidate(
            candidate_value=0.9,
            incumbent_value=None,
            artifact_kind="spec",
            artifact_id="proposal:1",
            policy_kind="merge_promotion",
        )
        verdict = promote(None, candidate, policy=policy)
        assert verdict.approved is True
        (request,) = policy.requests
        assert request.kind == "merge_promotion"

    def test_allow_and_allow_notify_are_approved(self):
        for decision in ("allow", "allow_notify"):
            policy = _FakePolicy(decision)
            verdict = promote(
                None,
                _candidate(candidate_value=0.9, incumbent_value=0.7),
                policy=policy,
            )
            assert verdict.approved is True, decision
            assert verdict.decision == decision

    def test_deny_and_queue_approval_are_not_approved(self):
        for decision in ("deny", "queue_approval"):
            policy = _FakePolicy(decision)
            verdict = promote(
                None,
                _candidate(candidate_value=0.9, incumbent_value=0.7),
                policy=policy,
            )
            assert verdict.approved is False, decision
            assert verdict.decision == decision

    def test_policy_failure_fails_closed(self):
        verdict = promote(
            None,
            _candidate(candidate_value=0.9, incumbent_value=0.7),
            policy=_BoomPolicy(),
        )
        assert verdict.eligible is True
        assert verdict.decision == "deny"
        assert verdict.approved is False
        assert "fail closed" in verdict.reason

    def test_never_raises_on_policy_failure(self):
        # The whole point of a governance gate: a broken policy backend must
        # deny, not crash the caller's promotion cycle.
        promote(
            None,
            _candidate(candidate_value=0.9, incumbent_value=0.7),
            policy=_BoomPolicy(),
        )
