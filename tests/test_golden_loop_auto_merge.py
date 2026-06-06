"""Tests for governed golden-loop auto-merge (CONCEPT:AHE-3.14).

Covers the conservative default (propose-only unless enabled), the quality +
governance + regression gates, AND the live wiring into the golden-loop cycle:
a high-score governed proposal auto-merges; a low-score one stays proposal-only.

@pytest.mark.concept("AHE-3.14")
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.enrichment.orchestration import TeamSpec
from agent_utilities.knowledge_graph.research.auto_merge import (
    GovernedAutoMerger,
    MergePolicy,
)

pytestmark = pytest.mark.concept("AHE-3.14")


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
        promoted = []
        merger = GovernedAutoMerger(
            engine=None,
            policy=MergePolicy(enabled=True, require_governance_valid=True),
            governance_validator=lambda spec: True,
            promoter=lambda spec: promoted.append(spec) or True,
        )
        ev = merger.consider(_strong_team())
        assert ev.merged is True
        assert len(promoted) == 1
        assert ev.reason == "auto-merged"

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
            engine=None,
            policy=MergePolicy(enabled=True, require_governance_valid=False),
            promoter=lambda spec: True,
        )
        merger.consider(_strong_team())
        records = merger.audit.query(action="golden_loop.auto_merge")
        assert records and records[0].details["merged"] is True


# ---------------------------------------------------------------------------
# LIVE-PATH: golden-loop cycle drives the auto-merger
# ---------------------------------------------------------------------------


class _FakeBackend:
    def __init__(self):
        self.writes = []


class _FakeEngine:
    def __init__(self):
        self.backend = _FakeBackend()


class TestGoldenLoopAutoMergeLivePath:
    """Wire-first: GoldenLoopController._synthesize_team consults the merger."""

    def _controller(self, monkeypatch, *, auto_merge, team):
        from agent_utilities.knowledge_graph.research.golden_loop import (
            GoldenLoopController,
        )

        ctrl = GoldenLoopController(_FakeEngine(), auto_merge=auto_merge)

        # Stub the synthesis primitives at their SOURCE modules (the controller
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
