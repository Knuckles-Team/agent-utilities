"""Plan 05 — SHACL ingestion gate.

A node missing a required property must be rejected (quarantined with a
violation report); a valid node passes through untouched.  Also verifies
the gate is wired into the pipeline phase graph ahead of the commit phase.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pyshacl")
pytest.importorskip("rdflib")

from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
from agent_utilities.knowledge_graph.pipeline.phases.shacl_gate import (
    _DEFAULT_SHAPES,
    execute_shacl_gate,
    shacl_gate_phase,
    validate_graph,
)
from agent_utilities.knowledge_graph.pipeline.types import PipelineContext
from agent_utilities.models.knowledge_graph import PipelineConfig


def _ctx(graph: GraphComputeEngine) -> PipelineContext:
    cfg = PipelineConfig(workspace_path="/tmp/shacl-gate-test")
    return PipelineContext(config=cfg, graph=graph)


def test_invalid_node_is_quarantined_with_report() -> None:
    """An Agent missing its required ``name`` is quarantined with a report."""
    g = GraphComputeEngine()
    g.add_node("bad_agent", {"type": "agent"})  # missing required :name

    conforms, violations, report_text = validate_graph(g, _DEFAULT_SHAPES)
    assert conforms is False
    assert "bad_agent" in violations
    assert any("name" in m.lower() for m in violations["bad_agent"])


@pytest.mark.asyncio
async def test_gate_phase_routes_invalid_node_to_quarantine() -> None:
    """The phase reroutes the violating node's type to the quarantine marker
    and attaches the violation report; the valid node is untouched."""
    g = GraphComputeEngine()
    g.add_node("good_agent", {"type": "agent", "name": "Planner"})
    g.add_node("bad_agent", {"type": "agent"})  # missing :name

    ctx = _ctx(g)
    out = await execute_shacl_gate(ctx, {})

    assert out["status"] == "completed"
    assert out["conforms"] is False
    assert "bad_agent" in out["quarantined_nodes"]
    assert "good_agent" not in out["quarantined_nodes"]
    assert out["report"]  # human-readable report attached on rejection

    bad = dict(g.nodes.get("bad_agent"))
    assert bad["type"] == ctx.config.shacl_quarantine_marker  # -> Invalid
    assert bad["shacl_valid"] is False
    assert bad["shacl_original_type"] == "agent"
    assert "name" in bad["shacl_report"].lower()

    good = dict(g.nodes.get("good_agent"))
    assert good["type"] == "agent"  # untouched
    assert "shacl_valid" not in good


@pytest.mark.asyncio
async def test_valid_tool_passes_invalid_tool_rejected() -> None:
    """A Tool with name + capabilityCategory passes; one missing the
    category is rejected."""
    g = GraphComputeEngine()
    g.add_node(
        "good_tool",
        {"type": "tool", "name": "GitLab", "capabilityCategory": "source_control"},
    )
    g.add_node("bad_tool", {"type": "tool", "name": "OnlyName"})  # no category

    ctx = _ctx(g)
    out = await execute_shacl_gate(ctx, {})

    assert "bad_tool" in out["quarantined_nodes"]
    assert "good_tool" not in out["quarantined_nodes"]

    assert dict(g.nodes.get("good_tool"))["type"] == "tool"
    assert dict(g.nodes.get("bad_tool"))["type"] == ctx.config.shacl_quarantine_marker


@pytest.mark.asyncio
async def test_all_valid_nodes_conform() -> None:
    """When every node satisfies its shape, nothing is quarantined."""
    g = GraphComputeEngine()
    g.add_node("a", {"type": "agent", "name": "A"})
    g.add_node(
        "t",
        {"type": "tool", "name": "T", "capabilityCategory": "itsm"},
    )

    out = await execute_shacl_gate(_ctx(g), {})
    assert out["quarantined_count"] == 0
    assert out["conforms"] is True


def test_gate_wired_before_commit() -> None:
    """The shacl_gate phase is registered and the sync (commit) phase
    depends on it, so the gate runs before persistence."""
    from agent_utilities.knowledge_graph.pipeline.phases import PHASES

    by_name = {p.name: p for p in PHASES}
    assert "shacl_gate" in by_name
    assert shacl_gate_phase.name == "shacl_gate"
    assert "shacl_gate" in by_name["sync"].deps
