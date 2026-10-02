"""Tests for governed golden-loop auto-merge (CONCEPT:AU-AHE.assimilation.research-auto-merge).

Covers the conservative default (propose-only unless enabled), the quality +
governance + regression gates, AND the live wiring into the golden-loop cycle:
a high-score governed proposal auto-merges; a low-score one stays proposal-only.

@pytest.mark.concept("AU-AHE.assimilation.research-auto-merge")
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.enrichment.orchestration import TeamSpec
from agent_utilities.knowledge_graph.research.auto_merge import (
    GovernedAutoMerger,
    MergePolicy,
)
from tests.unit.fleet_autonomy_fakes import FakeEngine


def _patch_governed_publish(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Patch ``change_publisher.governed_publish`` and return its call log."""
    called: list[bool] = []
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.research.change_publisher.governed_publish",
        lambda *a, **k: called.append(True) or {"status": "published"},
    )
    return called


pytestmark = pytest.mark.concept("AU-AHE.assimilation.research-auto-merge")


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


def _fake_engine_with_auto_tier() -> FakeEngine:
    """A ``FakeEngine`` with a KG-stored ``governance_rule`` override
    relaxing ``merge_promotion`` to the ``auto`` tier -- the same real path
    production uses to grant a receipt-backed ``approve`` disposition. The
    shipped DEFAULT tier (unconfigured actor) resolves to ``hold``, which
    the shared promotion contract (``PromotionOutcome.approved``) correctly
    does NOT activate (see test_auto_merge_action_policy.py's
    ``test_shipped_default_holds_and_blocks_promotion``); tests here that
    mean to exercise an ACTIVATED merge need a genuinely approved decision
    through this real path, not a loosened gate.
    """
    engine = FakeEngine()
    engine.add_node(
        "rule:promo-auto",
        "governance_rule",
        properties={
            "scope": "action_policy",
            "kind": "merge_promotion",
            "target": "*",
            "tier": "auto",
        },
    )
    return engine


# ---------------------------------------------------------------------------
# Policy + scoring
# ---------------------------------------------------------------------------


class TestPolicy:
    def test_default_policy_disabled(self):
        assert MergePolicy.from_env(None).enabled is False

    def test_default_threshold_conservative(self):
        assert MergePolicy().quality_threshold >= 0.85

    def test_env_enables(self, monkeypatch):
        monkeypatch.setenv("KG_GOLDEN_AUTO_MERGE", "1")
        assert MergePolicy.from_env().enabled is True


class TestScoring:
    def test_strong_proposal_scores_high(self):
        assert GovernedAutoMerger.score_proposal(_strong_team()) >= 0.85

    def test_weak_proposal_scores_low(self):
        assert GovernedAutoMerger.score_proposal(_weak_team()) < 0.85

    def test_explicit_score_wins(self):
        class S:
            quality_score = 0.99
            name = "x"

        assert GovernedAutoMerger.score_proposal(S()) == 0.99


# ---------------------------------------------------------------------------
# Evaluation + governed promotion
# ---------------------------------------------------------------------------


def _consider_strong_team_governed(engine) -> tuple[list, object]:
    """Consider ``_strong_team()`` with governance required (validator
    stubbed to always pass), recording promotions. Shared by the
    tier-relaxed/default-tier pair below, which differ only in whether
    ``engine`` carries a tier-relaxing ``governance_rule`` override."""
    promoted: list = []
    merger = GovernedAutoMerger(
        engine=engine,
        policy=MergePolicy(enabled=True, require_governance_valid=True),
        governance_validator=lambda spec: True,
        promoter=lambda spec: promoted.append(spec) or True,
    )
    return promoted, merger.consider(_strong_team())


class TestGovernedMerge:
    def test_disabled_never_promotes_even_if_eligible(self):
        merger = GovernedAutoMerger(
            engine=None,
            policy=MergePolicy(enabled=False, require_governance_valid=False),
        )
        ev = merger.consider(_strong_team())
        assert ev.eligible is True  # would qualify
        assert ev.merged is False  # but stays proposal-only (safe default)
        assert "proposal-only" in ev.reason

    def test_low_score_stays_proposal_only(self):
        promoted = []
        merger = GovernedAutoMerger(
            engine=None,
            policy=MergePolicy(enabled=True, require_governance_valid=False),
            promoter=lambda spec: promoted.append(spec) or True,
        )
        ev = merger.consider(_weak_team())
        assert ev.merged is False
        assert promoted == []
        assert any("quality" in f for f in ev.failures)

    def test_high_score_governed_auto_merges(self):
        """``engine=_fake_engine_with_auto_tier()``: the shipped DEFAULT tier
        (approval_required) resolves to a ``hold`` disposition, which the
        shared promotion contract (``PromotionOutcome.approved``) correctly
        does NOT activate — a bare ``engine=None``/default ``FakeEngine()``
        could never auto-merge under the real default tier. Relax the tier
        via the same KG-stored ``governance_rule`` override production uses,
        so this proves the merge lifecycle itself through a genuinely
        approved decision, not the approval-queue mechanics.
        """
        promoted, ev = _consider_strong_team_governed(_fake_engine_with_auto_tier())
        assert ev.merged is True
        assert len(promoted) == 1
        assert ev.reason == "auto-merged"

    def test_default_tier_holds_and_blocks_governed_merge(self):
        """Control for the test above: WITHOUT the tier-relaxing
        ``governance_rule`` override, the same otherwise-clean, strong,
        governance-valid proposal does NOT merge -- ``hold`` cannot
        activate a promotion, full stop."""
        promoted, ev = _consider_strong_team_governed(FakeEngine())
        assert ev.merged is False
        assert promoted == []
        assert ev.action_decision["decision"] == "hold"

    def test_governance_invalid_blocks_merge(self):
        merger = GovernedAutoMerger(
            engine=None,
            policy=MergePolicy(enabled=True, require_governance_valid=True),
            governance_validator=lambda spec: False,
            promoter=lambda spec: True,
        )
        ev = merger.consider(_strong_team())
        assert ev.merged is False
        assert "governance/SHACL invalid" in ev.failures

    def test_bare_claim_skips_governed_publish(self, monkeypatch):
        """D14: a bare Claim (C4's mined-finding artifact) is not git-publishable —
        ``_publish`` must skip ``governed_publish`` cleanly, never call it."""
        called = _patch_governed_publish(monkeypatch)
        merger = GovernedAutoMerger(
            engine=_fake_engine_with_auto_tier(),
            policy=MergePolicy(
                enabled=True, require_governance_valid=False, quality_threshold=0.0
            ),
            promoter=lambda spec: True,
        )
        spec = {
            "id": "claim:insight:abc123",
            "name": "a mined finding",
            "goal": "x implies y",
            "type": "Claim",
            "quality_score": 0.9,
        }
        ev = merger.consider(spec)
        assert ev.merged is True
        assert called == []  # governed_publish was NEVER invoked for a Claim
        assert ev.publication == {
            "status": "skipped",
            "reason": (
                "a bare Claim is not a git-publishable artifact — "
                "governed_publish is skipped (D14)"
            ),
        }

    def test_non_claim_spec_still_calls_governed_publish(self, monkeypatch):
        """Control: a normal TeamSpec-shaped merge still goes through governed_publish."""
        called = _patch_governed_publish(monkeypatch)
        merger = GovernedAutoMerger(
            engine=_fake_engine_with_auto_tier(),
            policy=MergePolicy(enabled=True, require_governance_valid=False),
            promoter=lambda spec: True,
        )
        ev = merger.consider(_strong_team())
        assert ev.merged is True
        assert called == [True]
        assert ev.publication == {"status": "published"}

    def test_regression_blocks_merge(self):
        merger = GovernedAutoMerger(
            engine=None,
            policy=MergePolicy(enabled=True, require_governance_valid=False),
            regression_check=lambda spec: False,
            promoter=lambda spec: True,
        )
        ev = merger.consider(_strong_team())
        assert ev.merged is False
        assert "regression detected" in ev.failures

    def test_every_consideration_is_audited(self):
        merger = GovernedAutoMerger(
            engine=_fake_engine_with_auto_tier(),
            policy=MergePolicy(enabled=True, require_governance_valid=False),
            promoter=lambda spec: True,
        )
        merger.consider(_strong_team())
        records = merger.audit.query(action="loop_engine.auto_merge")
        assert records and records[0].details["merged"] is True


# ---------------------------------------------------------------------------
# LIVE-PATH: golden-loop cycle drives the auto-merger
# ---------------------------------------------------------------------------


class TestLoopAutoMergeLivePath:
    """Wire-first: LoopController._synthesize_team consults the merger."""

    def _controller(self, monkeypatch, *, auto_merge, team):
        from agent_utilities.knowledge_graph.research.loop_controller import (
            LoopController,
        )

        # The tier-relaxed fake engine: the shipped DEFAULT action-policy
        # tier (approval_required) resolves to "hold", which the shared
        # promotion contract (PromotionOutcome.approved) correctly does NOT
        # activate. The auto_merge=True live-path test below means to prove
        # an activated merge, so it needs a genuinely approved decision
        # through the real KG-stored governance_rule override path.
        ctrl = LoopController(_fake_engine_with_auto_tier(), auto_merge=auto_merge)

        # Replace the synthesis primitives at their SOURCE modules (the controller
        # re-imports them at call time) so the cycle yields our team proposal
        # deterministically — no LLM / embeddings required.
        ctrl._capability_search = lambda: lambda q, top_k=5: []  # type: ignore[assignment]

        import agent_utilities.knowledge_graph.enrichment.cards as cards_mod
        import agent_utilities.knowledge_graph.enrichment.synthesize as synth_mod

        monkeypatch.setattr(synth_mod, "synthesize_team", lambda *a, **k: (team, []))
        monkeypatch.setattr(synth_mod, "persist_synthesis", lambda *a, **k: (1, 0))
        monkeypatch.setattr(
            cards_mod, "make_lite_llm_fn", lambda *a, **k: lambda prompt: "{}"
        )
        # governance is required by default; allow it so a high-score can merge.
        ctrl._merger.policy.require_governance_valid = False
        return ctrl

    def test_high_score_proposal_auto_merges_on_cycle_live_path(self, monkeypatch):
        monkeypatch.setattr(
            GovernedAutoMerger, "_default_promote", lambda self, spec: True
        )
        ctrl = self._controller(monkeypatch, auto_merge=True, team=_strong_team())
        topics = [{"id": "topic:1", "name": "retrieval quality"}]
        out = ctrl._synthesize_team(topics)
        assert out is not None
        assert out["auto_merge"]["merged"] is True

    def test_low_score_proposal_stays_proposal_only_live_path(self, monkeypatch):
        ctrl = self._controller(monkeypatch, auto_merge=True, team=_weak_team())
        topics = [{"id": "topic:1", "name": "x"}]
        out = ctrl._synthesize_team(topics)
        assert out is not None
        assert out["auto_merge"]["merged"] is False

    def test_default_cycle_is_propose_only_live_path(self, monkeypatch):
        # auto_merge omitted → conservative default: never merges.
        ctrl = self._controller(monkeypatch, auto_merge=None, team=_strong_team())
        topics = [{"id": "topic:1", "name": "x"}]
        out = ctrl._synthesize_team(topics)
        assert out is not None
        assert out["auto_merge"]["merged"] is False
        assert "disabled" in out["auto_merge"]["reason"]
