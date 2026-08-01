"""Live-path: graph_research action='contradictions' surfaces node↔node friction (KG-2.83).

Wire-First proof — the detector is reachable through the real graph_research tool
and runs against retrieved graph neighbours.

NOTE: this was originally wired to ``graph_analyze``. The ``graph_analyze`` ->
focused-tool split (analyze_suite.py's module docstring: "graph_analyze had
grown to ~30 actions spanning six unrelated domains... this splits that
surface into a few COHESIVE, intent-scoped tools") left 'contradictions'
unreachable from ANY registered tool — graph_analyze's own allowed-action set
narrowed to {inspect, enrichment_coverage, process_writeback, placement_plan,
infra_sweep, security_scan} and none of the five new focused tools
(graph_code/graph_research/graph_evaluate/graph_explain/graph_observe) picked
it up, even though its implementation
(agent_utilities/mcp/tools/analysis_tools.py's `_run_analysis_action`,
CONCEPT:AU-KG.research.explicit-node-node-contradiction) was untouched. Its
own CONCEPT id and "night-shift Critic" framing are research-domain, matching
graph_research's assimilation/research-craft actions (KG-2.83), so
'contradictions' was added to graph_research's action set
(GraphResearchAction) rather than reopened on graph_analyze — restoring a
genuinely orphaned capability instead of relaxing the intentional six-action
graph_analyze surface the split was for.
"""

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
        "graph_research",
        action="contradictions",
        query="sodium-ion batteries now undercut lithium on cost, so lithium is not the binding constraint",
        top_k=10,
    )
    # graph_research returns the sole typed EvidenceBundle (not a raw JSON
    # string): execute_focused_analysis wraps the action's `json.dumps(results)`
    # via EvidenceBundle.from_payload, which — for a list payload — lands each
    # row in `.claims` (models/evidence_bundle.py). `.contradictions` is a
    # SEPARATE derived heuristic scan over claim text, not this action's own
    # findings list, so `.claims` is the right field here.
    findings = res.claims
    assert isinstance(findings, list)
    # The new claim opposes belief1 (lithium is the binding constraint).
    assert any(f["conflict_id"] == "belief1" for f in findings)
    for f in findings:
        assert f["severity"] in {"high", "medium", "low"}
        assert "reason" in f
