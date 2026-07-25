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

from agent_utilities.mcp import kg_server
from agent_utilities.mcp.tools.state_tools import register_state_tools


class _CollectingMCP:
    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, *, name, description="", tags=None):  # noqa: ANN001
        def _deco(fn):
            self.tools[name] = fn
            return fn

        return _deco


class _GapStubEngine:
    """Minimal engine double covering exactly what research/gaps.py needs."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[tuple[str, str, str]] = []
        self.backend = object()

    def add_node(
        self, node_id: str, node_type: str, properties: dict[str, Any] | None = None
    ) -> None:
        self.nodes[node_id] = {"id": node_id, "type": node_type, **(properties or {})}

    def add_edge(
        self, source: str, target: str, rel_type: str = "", **properties: Any
    ) -> None:
        self.edges.append((source, target, rel_type))

    def query_cypher(
        self, q: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        params = params or {}
        if "WHERE n.id = $id" in q:
            node = self.nodes.get(params.get("id"))
            return [{"n": node}] if node else []
        if "MATCH (n:Gap) RETURN n LIMIT 1000" in q:
            return [{"n": n} for n in self.nodes.values() if n.get("type") == "Gap"]
        if "SPECIFIED_BY" in q and "RESOLVES" in q:
            gid = params.get("id")
            spec_id = next(
                (t for s, t, r in self.edges if s == gid and r == "SPECIFIED_BY"), None
            )
            loop_id = next(
                (s for s, t, r in self.edges if t == gid and r == "RESOLVES"), None
            )
            return [{"spec_id": spec_id, "loop_id": loop_id}]
        return []


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
    assert eng.nodes == {}


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
    # upserts the SAME node, never a duplicate.
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
    assert len([n for n in eng.nodes.values() if n.get("type") == "Gap"]) == 1

    listed = await _call(tool, action="gaps", limit=10)
    assert listed["action"] == "gaps"
    assert [g["id"] for g in listed["gaps"]] == [gap_id]

    fetched = await _call(tool, action="gap", loop_id=gap_id)
    assert fetched["action"] == "gap"
    assert fetched["gap"]["id"] == gap_id
    # No provenance yet -- neither hop of the D6 chain has been written.
    assert fetched["provenance"]["specified_by_spec_id"] is None
    assert fetched["provenance"]["resolved_by_loop_id"] is None

    # Wire the D6 provenance chain by hand (what link_gap_to_spec /
    # spec_proposals._bind_develop_loop do in the real pipeline) and confirm
    # 'gap' surfaces both hops.
    eng.add_edge(gap_id, "spec:widget-fix", "SPECIFIED_BY")
    eng.add_edge("loop:develop:widget-fix", gap_id, "RESOLVES")

    fetched_with_provenance = await _call(tool, action="gap", loop_id=gap_id)
    assert fetched_with_provenance["provenance"]["specified_by_spec_id"] == (
        "spec:widget-fix"
    )
    assert fetched_with_provenance["provenance"]["resolved_by_loop_id"] == (
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
