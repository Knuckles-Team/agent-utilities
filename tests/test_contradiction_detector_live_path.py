"""Live-path: graph_analyze action='contradictions' surfaces node↔node friction (KG-2.83).

Wire-First proof — the detector is reachable through the real graph_analyze tool
(MCP + REST twin /graph/analyze) and runs against retrieved graph neighbours.
"""

import json

import pytest

from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
from agent_utilities.mcp import kg_server


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.engine.get_active_backend",
        lambda: None,
    )
    g = GraphComputeEngine(backend_type="rust")
    for node in g.node_ids():
        g.remove_node(node)
    eng = IntelligenceGraphEngine(db_path=":memory:")
    eng.graph.add_node(
        "belief1",
        name="lithium bottleneck",
        description="lithium cost is the binding constraint on EV adoption",
    )
    eng.graph.add_node(
        "belief2",
        name="solar",
        description="solar panel efficiency keeps improving",
    )
    return eng


@pytest.mark.asyncio
async def test_contradictions_action_surfaces_friction(engine, monkeypatch):
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)
    kg_server.ensure_tools_registered()
    res = await kg_server._execute_tool(
        "graph_analyze",
        action="contradictions",
        query="sodium-ion batteries now undercut lithium on cost, so lithium is not the binding constraint",
        top_k=10,
    )
    findings = json.loads(res)
    assert isinstance(findings, list)
    # The new claim opposes belief1 (lithium is the binding constraint).
    assert any(f["conflict_id"] == "belief1" for f in findings)
    for f in findings:
        assert f["severity"] in {"high", "medium", "low"}
        assert "reason" in f
