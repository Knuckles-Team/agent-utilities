"""Tests for the Memory Consolidation Engine.

Concept: memory-consolidation
"""

from __future__ import annotations

import networkx as nx
import pytest
from typing import cast

from agent_utilities.knowledge_graph.consolidation import (
    ConsolidationEngine,
    DecisionToPrincipleRule,
    EpisodeToPreferenceRule,
)
from agent_utilities.knowledge_graph.engine import IntelligenceGraphEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeEngine:
    """Minimal engine stub with an in-memory NetworkX graph."""

    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()


def _populate_episodes(engine: FakeEngine, count: int = 6, reward: float = 0.9) -> None:
    """Add episode nodes with high reward to the graph.

    The EpisodeToPreferenceRule detects via outgoing edges:
    - episode -[used_tool]-> tool_node
    - episode -[produced_outcome]-> outcome_node (with reward attr)
    """
    # Create a shared tool node
    engine.graph.add_node("tool_code_search", type="tool", tool_name="code_search", name="code_search")
    for i in range(count):
        ep_id = f"ep_{i}"
        outcome_id = f"outcome_{i}"
        engine.graph.add_node(ep_id, type="episode", importance_score=0.5)
        engine.graph.add_node(outcome_id, type="outcome_evaluation", reward=reward)
        engine.graph.add_edge(ep_id, "tool_code_search", type="used_tool")
        engine.graph.add_edge(ep_id, outcome_id, type="produced_outcome")


def _populate_decisions(engine: FakeEngine, approach: str = "structured output", count: int = 4) -> None:
    """Add decision nodes with a shared approach to the graph."""
    for i in range(count):
        engine.graph.add_node(
            f"dec_{i}",
            type="decision",
            approach=approach,
            outcome="success",
            importance_score=0.6,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.concept("memory-consolidation")
def test_persist_proposals_creates_nodes() -> None:
    """Running consolidation in non-dry_run mode should create proposal nodes."""
    engine = FakeEngine()
    _populate_episodes(engine, count=6)

    consolidation = ConsolidationEngine(engine=cast(IntelligenceGraphEngine, engine))
    consolidation.register(EpisodeToPreferenceRule(min_evidence_count=5))

    proposals = consolidation.run(dry_run=False)

    assert len(proposals) >= 1
    # Verify proposal node exists in graph
    proposal_nodes = [
        nid for nid, d in engine.graph.nodes(data=True)
        if d.get("type") == "proposal"
    ]
    assert len(proposal_nodes) >= 1
    assert engine.graph.nodes[proposal_nodes[0]]["status"] == "pending"


@pytest.mark.concept("memory-consolidation")
def test_decision_to_principle_rule_detects() -> None:
    """DecisionToPrincipleRule should detect repeated approach patterns."""
    engine = FakeEngine()
    _populate_decisions(engine, approach="use structured output", count=5)

    rule = DecisionToPrincipleRule(min_evidence_count=3)
    proposals = rule.detect(cast(IntelligenceGraphEngine, engine))

    assert len(proposals) >= 1
    assert proposals[0].rule_name == "decision_to_principle"
    assert proposals[0].proposed_node_type == "PrincipleNode"
    assert len(proposals[0].evidence_node_ids) >= 3


@pytest.mark.concept("memory-consolidation")
def test_approve_proposal_creates_real_node() -> None:
    """Approving a proposal should create the real target node."""
    engine = FakeEngine()
    _populate_decisions(engine, count=4)

    consolidation = ConsolidationEngine(engine=cast(IntelligenceGraphEngine, engine))
    consolidation.register(DecisionToPrincipleRule(min_evidence_count=3))
    proposals = consolidation.run(dry_run=False)

    assert len(proposals) >= 1
    pid = proposals[0].proposal_id

    # Approve it
    result = consolidation.approve_proposal(pid)
    assert result is True
    assert engine.graph.nodes[pid]["status"] == "approved"

    # Check that a real node was created
    promoted_edges = [
        (u, v) for u, v, d in engine.graph.edges(data=True)
        if d.get("type") == "PROMOTED_TO"
    ]
    assert len(promoted_edges) >= 1


@pytest.mark.concept("memory-consolidation")
def test_reject_proposal_updates_status() -> None:
    """Rejecting a proposal should set status to 'rejected'."""
    engine = FakeEngine()
    _populate_decisions(engine, count=5)

    consolidation = ConsolidationEngine(engine=cast(IntelligenceGraphEngine, engine))
    consolidation.register(DecisionToPrincipleRule(min_evidence_count=3))
    proposals = consolidation.run(dry_run=False)

    assert len(proposals) >= 1
    pid = proposals[0].proposal_id
    result = consolidation.reject_proposal(pid, reason="Not convincing")

    assert result is True
    assert engine.graph.nodes[pid]["status"] == "rejected"
    assert engine.graph.nodes[pid]["rejection_reason"] == "Not convincing"


@pytest.mark.concept("memory-consolidation")
def test_get_pending_proposals() -> None:
    """get_pending_proposals should return only pending proposals."""
    engine = FakeEngine()
    _populate_decisions(engine, count=5)

    consolidation = ConsolidationEngine(engine=cast(IntelligenceGraphEngine, engine))
    consolidation.register(DecisionToPrincipleRule(min_evidence_count=3))
    proposals = consolidation.run(dry_run=False)

    pending = consolidation.get_pending_proposals()
    assert len(pending) >= 1

    # Approve one, then check pending decreases
    consolidation.approve_proposal(proposals[0].proposal_id)
    pending_after = consolidation.get_pending_proposals()
    assert len(pending_after) < len(pending)


@pytest.mark.concept("memory-consolidation")
def test_dedup_prevents_re_proposal() -> None:
    """Same evidence pattern should not produce duplicate proposals."""
    engine = FakeEngine()
    _populate_decisions(engine, count=5)

    rule = DecisionToPrincipleRule(min_evidence_count=3)
    proposals_1 = rule.detect(cast(IntelligenceGraphEngine, engine))
    proposals_2 = rule.detect(cast(IntelligenceGraphEngine, engine))

    consolidation = ConsolidationEngine(engine=cast(IntelligenceGraphEngine, engine))
    all_proposals = proposals_1 + proposals_2
    deduped = consolidation.dedup_by_signature(all_proposals)

    assert len(deduped) < len(all_proposals)
    assert len(deduped) == len(proposals_1)
