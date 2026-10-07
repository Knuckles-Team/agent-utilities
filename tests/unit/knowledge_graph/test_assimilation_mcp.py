#!/usr/bin/python
"""Assimilation MCP action + public pass entrypoint (VU-9).

CONCEPT:AU-KG.query.vendor-agnostic-traversal
"""

from pathlib import Path

import pytest

from agent_utilities.knowledge_graph.research.loop_controller import (
    run_assimilation_pass,
)
from tests.unit.assimilation_graph_fakes import market_engine as _Engine
from tests.unit.assimilation_graph_fakes import two_open_features as _nodes
from tests.unit.fleet_autonomy_fakes import verified_fleet_session

pytestmark = pytest.mark.concept("AU-KG.query.vendor-agnostic-traversal")


def test_run_assimilation_pass_without_synthesis():
    rep = run_assimilation_pass(_Engine(_nodes()))
    assert rep["skipped"] is False
    assert rep["open_gaps"] == 2
    assert "proposed_plans" not in rep


def test_run_assimilation_pass_with_synthesis():
    engine = _Engine(_nodes())
    market = engine.market
    with verified_fleet_session():
        rep = run_assimilation_pass(engine, synthesize=True, top_n=5)
    assert {p["feature_id"] for p in rep["proposed_plans"]} == {"f1", "f2"}
    # each feature became ONE canonical research Gap through EG's typed upsert
    assert {gap_id for _, gap_id in market.gap_rows} == {
        "gap:research:f1",
        "gap:research:f2",
    }
    # features flipped to proposed (idempotent next pass)
    assert dict(engine.graph.nodes(data=True))["f1"]["status"] == "proposed"


def test_run_assimilation_pass_idempotent_and_force():
    engine = _Engine(_nodes())
    run_assimilation_pass(engine)
    assert run_assimilation_pass(engine)["skipped"] is True
    assert run_assimilation_pass(engine, force=True)["skipped"] is False


def test_mcp_assimilate_action_is_wired():
    # Regression guard: graph_evolution routes the assimilate action.
    src = Path("agent_utilities/mcp/tools/evolution_tools.py").read_text(
        encoding="utf-8"
    )
    assert 'action == "assimilate"' in src
    assert "run_assimilation_pass" in src
