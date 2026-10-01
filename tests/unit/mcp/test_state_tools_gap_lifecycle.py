"""Tests for the graph_loops MCP tool's Gap-lifecycle actions (CONCEPT:AU-AHE.
harness.canonical-gap-lifecycle, Wave 6): 'gaps' (list), 'submit_gap' (create),
'gap' (get one + provenance chain).

Operator-facing twin of the canonical ``research/gaps.py`` functions every
discovery track (failure/research/skill/audit) already calls internally —
these tests prove the SAME functions are now reachable, over MCP + REST, for a
human/operator to submit and inspect a gap by hand.

Mirrors the ``_CollectingMCP`` + stub-engine pattern used across the other MCP
tool-surface tests (e.g. ``test_skill_evolution.py``'s ``_SkillEvoStubEngine``)
— no live engine required.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from agent_utilities.knowledge_graph.research import gaps
from agent_utilities.mcp import kg_server
from agent_utilities.mcp.tools.state_tools import register_state_tools
from tests.unit.fleet_autonomy_fakes import verified_fleet_session
from tests.unit.work_market_fakes import attach_market


@pytest.fixture(autouse=True)
def _verified_session():
    """Every Gap call binds the ambient verified tenant (EG's rule)."""
    with verified_fleet_session():
        yield


class _CollectingMCP:
    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, *, name, description="", tags=None):  # noqa: ANN001
        def _deco(fn):
            self.tools[name] = fn
            return fn

        return _deco


class _GapStubEngine:
    """Minimal engine double: the typed EG Gap surfaces research/gaps.py calls."""

    def __init__(self) -> None:
        self.backend = object()
        self.market = attach_market(self)


def _register(monkeypatch) -> object:
    mcp = _CollectingMCP()
    register_state_tools(mcp)
    return mcp.tools["graph_loops"]


async def _call(tool, **kwargs: Any) -> dict[str, Any]:
    defaults = dict(
        action="list",
        objective="",
        kind="research",
        loop_id="",
        validation_cmd="",
        end_state="",
        skill_ref="",
        max_topics=5,
        limit=10,
        priority_bucket=2,
        spec_id="",
        decision="",
        status="",
        mine_discovery=None,
        placement_scan_limit=200,
        placement_canary_tolerance=0.10,
        data_json="{}",
    )
    defaults.update(kwargs)
    return json.loads(await tool(**defaults))


def test_registered_and_routed():
    mcp = _CollectingMCP()
    register_state_tools(mcp)
    assert kg_server.REGISTERED_TOOLS.get("graph_loops") is not None
    assert kg_server.ACTION_TOOL_ROUTES.get("graph_loops") == "/graph/loops"


@pytest.mark.asyncio
async def test_submit_gap_requires_source_signature_statement(monkeypatch):
    eng = _GapStubEngine()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: eng)
    tool = _register(monkeypatch)

    out = await _call(tool, action="submit_gap", data_json=json.dumps({}))
    assert "error" in out
    assert eng.market.gap_rows == {}


@pytest.mark.asyncio
async def test_submit_gap_then_list_then_get_with_provenance(monkeypatch):
    eng = _GapStubEngine()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: eng)
    tool = _register(monkeypatch)

    submitted = await _call(
        tool,
        action="submit_gap",
        data_json=json.dumps(
            {
                "source": "manual",
                "signature": "widget-fails-under-load",
                "statement": "The widget endpoint 500s under concurrent load.",
                "domain": "reliability",
                "severity": 0.9,
            }
        ),
    )
    assert submitted["action"] == "submit_gap"
    gap_id = submitted["gap"]["id"]
    assert gap_id == "gap:manual:widget-fails-under-load"
    assert submitted["gap"]["status"] == "open"
    assert submitted["gap"]["priority_bucket"] == 0  # severity 0.9 -> bucket 0

    # Idempotent on the canonical id -- resubmitting the same source+signature
    # upserts the SAME Gap (EG folds the new evidence in), never a duplicate.
    resubmitted = await _call(
        tool,
        action="submit_gap",
        data_json=json.dumps(
            {
                "source": "manual",
                "signature": "widget-fails-under-load",
                "statement": "Updated statement.",
            }
        ),
    )
    assert resubmitted["gap"]["id"] == gap_id
    assert len(eng.market.gap_rows) == 1

    listed = await _call(tool, action="gaps", limit=10)
    assert listed["action"] == "gaps"
    assert [g["id"] for g in listed["gaps"]] == [gap_id]

    fetched = await _call(tool, action="gap", loop_id=gap_id)
    assert fetched["action"] == "gap"
    assert fetched["gap"]["id"] == gap_id
    # No provenance yet -- neither hop of the D6 chain has been written.
    assert fetched["provenance"]["specified_by_spec_id"] is None
    assert fetched["provenance"]["resolved_by"] is None

    # Walk the provenance chain through the SAME typed transitions the real pipeline uses
    # (link_gap_to_spec, then the develop-Loop's publish) and confirm 'gap'
    # surfaces both hops from EG's own Gap record.
    assert gaps.link_gap_to_spec(eng, gap_id, "spec:widget-fix")
    assert gaps.mark_gap_resolved(eng, gap_id, reference="loop:develop:widget-fix")

    fetched_with_provenance = await _call(tool, action="gap", loop_id=gap_id)
    assert fetched_with_provenance["provenance"]["specified_by_spec_id"] == (
        "spec:widget-fix"
    )
    assert fetched_with_provenance["provenance"]["resolved_by"] == (
        "loop:develop:widget-fix"
    )


@pytest.mark.asyncio
async def test_gap_action_requires_an_id(monkeypatch):
    eng = _GapStubEngine()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: eng)
    tool = _register(monkeypatch)

    out = await _call(tool, action="gap", loop_id="")
    assert "error" in out


@pytest.mark.asyncio
async def test_gap_action_not_found(monkeypatch):
    eng = _GapStubEngine()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: eng)
    tool = _register(monkeypatch)

    out = await _call(tool, action="gap", loop_id="gap:missing:nope")
    assert out["error"] == "gap not found"


@pytest.mark.asyncio
async def test_gaps_list_is_empty_with_no_gaps(monkeypatch):
    eng = _GapStubEngine()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: eng)
    tool = _register(monkeypatch)

    out = await _call(tool, action="gaps")
    assert out["gaps"] == []
