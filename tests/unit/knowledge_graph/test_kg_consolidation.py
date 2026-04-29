#!/usr/bin/python
"""Unit tests for the ``consolidation.py`` minimum-viable skeleton.

Covers:

* ``ConsolidationProposal`` construction and confidence bounds.
* ``ConsolidationProposal.compute_signature`` is stable and
  order-independent over ``evidence_node_ids``.
* ``EpisodeToPreferenceRule.detect`` produces the expected number of
  proposals on a synthetic in-memory graph.
* Rule respects ``min_evidence_count`` (below-threshold scenarios emit 0
  proposals).
* Rule skips episodes with outcome below ``reward_threshold``.
* ``ConsolidationEngine.run`` isolates broken rules (per §4.2).
* ``ConsolidationEngine.dedup_by_signature`` removes duplicates.
"""

from __future__ import annotations

import networkx as nx
import pytest
from pydantic import ValidationError

from agent_utilities.knowledge_graph.consolidation import (
    ConsolidationEngine,
    ConsolidationProposal,
    EpisodeToPreferenceRule,
)
from agent_utilities.knowledge_graph.engine import IntelligenceGraphEngine


def _make_episode(
    g: nx.MultiDiGraph,
    episode_id: str,
    tool_name: str,
    reward: float,
) -> None:
    """Wire up one (Episode)-[:USED_TOOL]->(ToolCall) +
    (Episode)-[:PRODUCED_OUTCOME]->(OutcomeEvaluation) triplet.
    """
    g.add_node(episode_id, type="episode", timestamp="2026-01-01T00:00:00Z")
    tool_id = f"tc:{episode_id}"
    outcome_id = f"oc:{episode_id}"
    g.add_node(tool_id, type="tool_call", tool_name=tool_name)
    g.add_node(outcome_id, type="outcome_evaluation", reward=reward)
    g.add_edge(episode_id, tool_id, type="used_tool")
    g.add_edge(episode_id, outcome_id, type="produced_outcome")


@pytest.fixture
def synthetic_engine() -> IntelligenceGraphEngine:
    g = nx.MultiDiGraph()
    # 5 successful terraform episodes
    for i in range(5):
        _make_episode(g, f"ep:tf-{i}", "terraform", reward=0.9)
    # 2 successful ansible episodes (below min_evidence_count=5)
    for i in range(2):
        _make_episode(g, f"ep:ansible-{i}", "ansible", reward=0.95)
    # 1 failed terraform episode — must be ignored
    _make_episode(g, "ep:tf-fail", "terraform", reward=0.2)
    return IntelligenceGraphEngine(g)


# ---------------------------------------------------------------------------
# ConsolidationProposal
# ---------------------------------------------------------------------------


def test_proposal_confidence_upper_bound_rejected() -> None:
    with pytest.raises(ValidationError):
        ConsolidationProposal(
            proposal_id="p1",
            rule_name="r",
            proposed_node_type="PreferenceNode",
            proposed_payload={"x": 1},
            confidence=1.5,
            created_at="2026-01-01T00:00:00Z",
        )


def test_proposal_confidence_lower_bound_rejected() -> None:
    with pytest.raises(ValidationError):
        ConsolidationProposal(
            proposal_id="p1",
            rule_name="r",
            proposed_node_type="PreferenceNode",
            proposed_payload={"x": 1},
            confidence=-0.01,
            created_at="2026-01-01T00:00:00Z",
        )


def test_proposal_invalid_node_type_rejected() -> None:
    with pytest.raises(ValidationError):
        ConsolidationProposal(
            proposal_id="p1",
            rule_name="r",
            proposed_node_type="UnknownNode",  # type: ignore[arg-type]
            proposed_payload={},
            confidence=0.5,
            created_at="2026-01-01T00:00:00Z",
        )


def test_proposal_signature_is_order_independent() -> None:
    p1 = ConsolidationProposal(
        proposal_id="a",
        rule_name="r1",
        proposed_node_type="PreferenceNode",
        proposed_payload={},
        evidence_node_ids=["x", "y", "z"],
        confidence=0.5,
        created_at="2026-01-01T00:00:00Z",
    )
    p2 = ConsolidationProposal(
        proposal_id="a",
        rule_name="r1",
        proposed_node_type="PreferenceNode",
        proposed_payload={},
        evidence_node_ids=["z", "y", "x"],  # reordered
        confidence=0.5,
        created_at="2026-01-01T00:00:00Z",
    )
    assert p1.compute_signature() == p2.compute_signature()


def test_proposal_signature_differs_across_rules() -> None:
    p1 = ConsolidationProposal(
        proposal_id="a",
        rule_name="r1",
        proposed_node_type="PreferenceNode",
        proposed_payload={},
        evidence_node_ids=["x"],
        confidence=0.5,
        created_at="2026-01-01T00:00:00Z",
    )
    p2 = ConsolidationProposal(
        proposal_id="a",
        rule_name="r2",  # different rule
        proposed_node_type="PreferenceNode",
        proposed_payload={},
        evidence_node_ids=["x"],
        confidence=0.5,
        created_at="2026-01-01T00:00:00Z",
    )
    assert p1.compute_signature() != p2.compute_signature()


# ---------------------------------------------------------------------------
# EpisodeToPreferenceRule
# ---------------------------------------------------------------------------


def test_rule_emits_one_proposal_for_terraform(
    synthetic_engine: IntelligenceGraphEngine,
) -> None:
    rule = EpisodeToPreferenceRule(min_evidence_count=5)
    proposals = rule.detect(synthetic_engine)
    assert len(proposals) == 1
    p = proposals[0]
    assert p.proposed_node_type == "PreferenceNode"
    assert p.proposed_payload["value"] == "terraform"
    assert len(p.evidence_node_ids) == 5
    # Confidence = min(1.0, 0.6 + 0.05*5) = 0.85
    assert p.confidence == pytest.approx(0.85)


def test_rule_skips_below_threshold(
    synthetic_engine: IntelligenceGraphEngine,
) -> None:
    """Ansible has only 2 successful episodes (< min_evidence_count=5)."""
    rule = EpisodeToPreferenceRule(min_evidence_count=5)
    proposals = rule.detect(synthetic_engine)
    tools = {p.proposed_payload["value"] for p in proposals}
    assert "ansible" not in tools


def test_rule_ignores_failed_episodes() -> None:
    g = nx.MultiDiGraph()
    # 5 failed terraform episodes — no proposal should emerge
    for i in range(5):
        _make_episode(g, f"ep:fail-{i}", "terraform", reward=0.1)
    eng = IntelligenceGraphEngine(g)
    rule = EpisodeToPreferenceRule(min_evidence_count=5)
    proposals = rule.detect(eng)
    assert proposals == []


def test_rule_honours_custom_min_evidence_count(
    synthetic_engine: IntelligenceGraphEngine,
) -> None:
    """With min_evidence_count=2 ansible now also fires."""
    rule = EpisodeToPreferenceRule(min_evidence_count=2)
    proposals = rule.detect(synthetic_engine)
    tools = {p.proposed_payload["value"] for p in proposals}
    assert "terraform" in tools
    assert "ansible" in tools


# ---------------------------------------------------------------------------
# ConsolidationEngine
# ---------------------------------------------------------------------------


def test_engine_runs_registered_rules(
    synthetic_engine: IntelligenceGraphEngine,
) -> None:
    ce = ConsolidationEngine(synthetic_engine)
    ce.register(EpisodeToPreferenceRule(min_evidence_count=5))
    proposals = ce.run(dry_run=True)
    assert len(proposals) == 1


def test_engine_isolates_broken_rules(
    synthetic_engine: IntelligenceGraphEngine,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A rule that raises an exception should not crash the run."""

    class BrokenRule:
        name = "broken"
        min_evidence_count = 1
        min_confidence = 0.0

        def detect(
            self, engine: IntelligenceGraphEngine
        ) -> list[ConsolidationProposal]:
            raise RuntimeError("intentional failure for the test")

    ce = ConsolidationEngine(synthetic_engine)
    ce.register(BrokenRule())
    ce.register(EpisodeToPreferenceRule(min_evidence_count=5))

    with caplog.at_level("WARNING"):
        proposals = ce.run(dry_run=True)

    # Terraform proposal still emerges from the good rule
    assert len(proposals) == 1
    # Warning logged for the broken rule
    assert any(
        "broken" in rec.message for rec in caplog.records
    ), "expected a warning for the broken rule"


def test_engine_dedup_by_signature() -> None:
    p1 = ConsolidationProposal(
        proposal_id="a",
        rule_name="r",
        proposed_node_type="PreferenceNode",
        proposed_payload={},
        evidence_node_ids=["x"],
        confidence=0.5,
        created_at="2026-01-01T00:00:00Z",
    )
    p1.signature = p1.compute_signature()
    p2 = ConsolidationProposal(
        proposal_id="b",  # different id but same signature
        rule_name="r",
        proposed_node_type="PreferenceNode",
        proposed_payload={},
        evidence_node_ids=["x"],
        confidence=0.5,
        created_at="2026-01-01T00:00:00Z",
    )
    p2.signature = p2.compute_signature()
    assert p1.signature == p2.signature

    g = nx.MultiDiGraph()
    ce = ConsolidationEngine(IntelligenceGraphEngine(g))
    deduped = ce.dedup_by_signature([p1, p2])
    assert len(deduped) == 1


def test_engine_empty_graph_yields_no_proposals() -> None:
    g = nx.MultiDiGraph()
    ce = ConsolidationEngine(IntelligenceGraphEngine(g))
    ce.register(EpisodeToPreferenceRule(min_evidence_count=5))
    assert ce.run(dry_run=True) == []
