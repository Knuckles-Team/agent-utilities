#!/usr/bin/python
"""Plan synthesis from KG neighborhood (VU-8).

CONCEPT:AU-KG.query.vendor-agnostic-traversal
"""

import pytest

from agent_utilities.knowledge_graph.assimilation import (
    hydrate_feature,
    synthesize_plan_for_feature,
    synthesize_plans,
)
from agent_utilities.knowledge_graph.assimilation.plan_synthesis import _default_synth
from tests.unit.assimilation_graph_fakes import market_engine as _Engine
from tests.unit.assimilation_graph_fakes import two_open_features as _nodes
from tests.unit.fleet_autonomy_fakes import verified_fleet_session

pytestmark = pytest.mark.concept("AU-KG.query.vendor-agnostic-traversal")


@pytest.fixture(autouse=True)
def _verified_session():
    """Every Gap call binds the ambient verified tenant (EG's rule)."""
    with verified_fleet_session():
        yield


def test_hydrate_feature_pulls_neighborhood():
    engine = _Engine(_nodes())
    engine.link_nodes(
        "f1", "f2", "HAS_SYNERGY_WITH", properties={"_rel": "HAS_SYNERGY_WITH"}
    )
    nb = hydrate_feature(engine, "f1")
    assert nb["name"] == "exec-rag planner"
    assert nb["pillar"] == "KG"
    assert nb["sources"] == ["arxiv:pyrag"]
    assert nb["synergies"] == ["f2"]


def test_default_template_is_grounded():
    # The deterministic fallback (no LLM) is grounded in the feature's neighborhood.
    engine = _Engine(_nodes())
    plan = _default_synth(hydrate_feature(engine, "f1"))
    assert "exec-rag planner" in plan["title"]
    assert "arxiv:pyrag" in plan["body"]  # grounded in the source
    assert "AU-KG.retrieval.memory-first-retrieval" in plan["body"]


def test_synthesize_folds_into_canonical_gap_and_spec():
    # Wave-6 D1/WP#5: the research/OSS track now folds into the ONE canonical :Gap +
    # develop-able :SpecProposal lifecycle, not the old dead-end sdd_plan node.
    engine = _Engine(_nodes())
    # inject a synth_fn so the test does not depend on LLM availability
    proposal = synthesize_plan_for_feature(
        engine, "f1", synth_fn=lambda nb: {"title": "T", "body": "B"}
    )
    # plan_id is now the persisted :SpecProposal id (title-derived), not plan:f1.
    assert proposal.plan_id == "spec_proposal:t"
    data = dict(engine.graph.nodes(data=True))
    # ONE canonical Gap was upserted through EG's typed Gap surface (the
    # harness-evolution work-market requirement), a :SpecProposal was
    # persisted, and NO sdd_plan node was written.
    assert {gap_id for _, gap_id in engine.market.gap_rows} == {"gap:research:f1"}
    assert "gap:research:f1" not in data
    assert data["spec_proposal:t"]["type"] == "SpecProposal"
    assert "plan:f1" not in data
    assert data["f1"]["status"] == "proposed"
    # The feature is ADDRESSED_BY the spec proposal (not a dead-end plan node).
    addressed = [
        e for e in engine.graph._out["f1"] if e[2].get("_rel") == "ADDRESSED_BY"
    ]
    assert addressed and addressed[0][1] == "spec_proposal:t"


def test_injected_synth_fn_is_used():
    engine = _Engine(_nodes())
    proposal = synthesize_plan_for_feature(
        engine, "f1", synth_fn=lambda nb: {"title": "Custom", "body": "custom body"}
    )
    assert proposal.title == "Custom" and proposal.body == "custom body"


def test_synthesize_plans_top_n_and_idempotent():
    engine = _Engine(_nodes())
    first = synthesize_plans(engine, top_n=5)
    assert {p.feature_id for p in first} == {"f1", "f2"}
    # second pass: both are now 'proposed' → skipped (idempotent)
    second = synthesize_plans(engine, top_n=5)
    assert second == []


def test_synthesize_plans_respects_top_n():
    engine = _Engine(_nodes())
    out = synthesize_plans(engine, top_n=1)
    assert len(out) == 1
