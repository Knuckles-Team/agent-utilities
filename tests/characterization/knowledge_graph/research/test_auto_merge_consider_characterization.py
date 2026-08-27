"""Characterization tests for ``GovernedAutoMerger.consider`` (CX-AU-09).

CCN 11 at time of writing
(``agent_utilities/knowledge_graph/research/auto_merge.py``). These tests
pin the OBSERVED, black-box behaviour before any decomposition: the
disabled/ineligible not-promoted reason strings, the action-policy deny path
(exact reason text, publication stays None, action_decision recorded), the
promotion-exception path (degrades to a reason string, never raises, merged
stays False, publish is NEVER called), and the promotion-returns-False path
(distinct "promotion failed" reason, publish still never called).

Per the two-commit discipline, this file must be added and pass GREEN
against the UNMODIFIED ``auto_merge.py`` before any refactor commit, and
must not change during the refactor commit that follows.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.enrichment.orchestration import TeamSpec
from agent_utilities.knowledge_graph.research.auto_merge import (
    GovernedAutoMerger,
    MergePolicy,
)
from agent_utilities.orchestration.action_policy import ActionDecision, ActionRequest


def _strong_team() -> TeamSpec:
    return TeamSpec(
        name="Resolver Team",
        goal="Address open KG topics about retrieval quality",
        lead="Lead",
        members=["Researcher", "Validator"],
        description="A complete, well-formed team proposal.",
    )


def _weak_team() -> TeamSpec:
    return TeamSpec(name="bare", goal="", lead="", members=[])


class _FakeActionPolicy:
    def __init__(self, decision: str, *, reason: str = "r", approval_id=None):
        self._decision = decision
        self._reason = reason
        self._approval_id = approval_id

    def decide(self, request: ActionRequest) -> ActionDecision:
        return ActionDecision(
            decision=self._decision,
            tier="approval_required",
            request=request,
            reason=self._reason,
            approval_id=self._approval_id,
        )


def test_disabled_policy_reason_is_exact() -> None:
    merger = GovernedAutoMerger(
        engine=None, policy=MergePolicy(enabled=False, require_governance_valid=False)
    )
    ev = merger.consider(_strong_team())
    assert ev.merged is False
    assert ev.reason == "proposal-only (auto-merge disabled)"
    assert ev.action_decision is None
    assert ev.publication is None


def test_ineligible_proposal_reason_joins_failures() -> None:
    merger = GovernedAutoMerger(
        engine=None, policy=MergePolicy(enabled=True, require_governance_valid=False)
    )
    ev = merger.consider(_weak_team())
    assert ev.merged is False
    assert ev.reason == "proposal-only: " + "; ".join(ev.failures)
    assert ev.failures  # weak team has at least the quality-threshold failure


def test_action_policy_deny_reason_text_and_no_publication() -> None:
    called: list = []
    merger = GovernedAutoMerger(
        engine=None,
        policy=MergePolicy(enabled=True, require_governance_valid=False),
        promoter=lambda spec: called.append(spec) or True,
        action_policy=_FakeActionPolicy("deny", reason="policy says no"),
    )
    ev = merger.consider(_strong_team())
    assert ev.merged is False
    assert called == []
    assert ev.publication is None
    assert ev.action_decision == {
        "decision": "deny",
        "reason": "policy says no",
        "approval_id": None,
    }
    assert ev.reason == "blocked by action policy (merge_promotion): policy says no"


def test_promotion_exception_degrades_to_reason_never_raises_never_publishes() -> None:
    published: list = []

    def _boom(spec):
        raise RuntimeError("promoter exploded")

    merger = GovernedAutoMerger(
        engine=None,
        policy=MergePolicy(enabled=True, require_governance_valid=False),
        promoter=_boom,
        action_policy=_FakeActionPolicy("allow"),
        publisher=type(
            "P",
            (),
            {"publish": staticmethod(lambda spec, engine=None: published.append(1))},
        )(),
    )
    ev = merger.consider(_strong_team())
    assert ev.merged is False
    assert ev.reason == "promotion error: promoter exploded"
    assert ev.publication is None
    assert published == []


def test_promotion_returns_false_gets_distinct_reason_never_publishes() -> None:
    published: list = []
    merger = GovernedAutoMerger(
        engine=None,
        policy=MergePolicy(enabled=True, require_governance_valid=False),
        promoter=lambda spec: False,
        action_policy=_FakeActionPolicy("allow"),
        publisher=type(
            "P",
            (),
            {"publish": staticmethod(lambda spec, engine=None: published.append(1))},
        )(),
    )
    ev = merger.consider(_strong_team())
    assert ev.merged is False
    assert ev.reason == "promotion failed"
    assert ev.publication is None
    assert published == []


def test_successful_promotion_calls_publish_and_sets_reason() -> None:
    merger = GovernedAutoMerger(
        engine=None,
        policy=MergePolicy(enabled=True, require_governance_valid=False),
        promoter=lambda spec: True,
        action_policy=_FakeActionPolicy("allow"),
    )
    ev = merger.consider(_strong_team())
    assert ev.merged is True
    assert ev.reason == "auto-merged"
    assert ev.publication is not None


def test_every_path_is_audited_exactly_once() -> None:
    merger = GovernedAutoMerger(
        engine=None, policy=MergePolicy(enabled=False, require_governance_valid=False)
    )
    merger.consider(_strong_team())
    records = merger.audit.query(action="loop_engine.auto_merge")
    assert len(records) == 1
