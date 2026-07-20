"""Strict-current ownership contract for the focused Graph-OS execution tools."""

from __future__ import annotations

import inspect
import json

from agent_utilities.mcp import kg_server
from agent_utilities.mcp.tools import (
    register_agent_execution_tools,
    register_analysis_tools,
    register_domain_ops_tools,
    register_evolution_tools,
    register_governance_tools,
    register_job_tools,
    register_rlm_tools,
    register_workflow_tools,
)
from scripts.gen_graphos_manifest import harvest_actions


class _FakeMCP:
    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, *, name: str, **_metadata):
        def _capture(function):
            self.tools[name] = function
            return function

        return _capture


def _register_all() -> _FakeMCP:
    mcp = _FakeMCP()
    for registrar in (
        register_analysis_tools,
        register_agent_execution_tools,
        register_domain_ops_tools,
        register_evolution_tools,
        register_governance_tools,
        register_job_tools,
        register_rlm_tools,
        register_workflow_tools,
    ):
        registrar(mcp)
    return mcp


def test_orchestration_capabilities_have_one_current_owner() -> None:
    mcp = _register_all()

    expected_actions = {
        "graph_agents": {"swarm", "computer_use", "synthesize_org", "run_org"},
        "graph_domain_ops": {
            "allocate_budget",
            "fit_markov_regime",
            "register_rlm_actor",
        },
        "graph_evolution": {
            "assimilate",
            "distill_skills",
            "standardize",
            "failure_ingest",
            "optimize_component",
            "publish_proposal",
        },
        "graph_governance": {
            "grant_approval",
            "submit_risk_veto",
            "verify_action",
        },
        "graph_jobs": {"dispatch", "status"},
        "graph_rlm": {"run", "benchmark"},
        "graph_workflows": {
            "compile",
            "compile_process",
            "list",
            "execute",
            "dispatch",
            "status",
            "export",
        },
    }
    assert sum(map(len, expected_actions.values())) == 27
    for tool, actions in expected_actions.items():
        assert harvest_actions(mcp.tools[tool]) == actions

    orchestrate = mcp.tools["graph_orchestrate"]
    assert "action" not in inspect.signature(orchestrate).parameters
    assert harvest_actions(orchestrate) == set()


def test_focused_tools_publish_collapsed_rest_routes() -> None:
    _register_all()
    expected_routes = {
        "graph_agents": "/graph/agents",
        "graph_domain_ops": "/graph/domain-ops",
        "graph_evolution": "/graph/evolution",
        "graph_governance": "/graph/governance",
        "graph_jobs": "/graph/jobs",
        "graph_rlm": "/graph/rlm",
        "graph_workflows": "/graph/workflows",
    }
    assert {
        tool: kg_server.ACTION_TOOL_ROUTES.get(tool) for tool in expected_routes
    } == expected_routes


def test_budget_domain_action_uses_composed_engine_capability(monkeypatch) -> None:
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
    from agent_utilities.knowledge_graph.orchestration.engine_enterprise import (
        EnterpriseEngineMixin,
    )

    class _RecordingGraph:
        def __init__(self) -> None:
            self.nodes: dict[str, dict[str, object]] = {}

        def add_node(self, node_id: str, **properties: object) -> None:
            self.nodes[node_id] = properties

    assert issubclass(IntelligenceGraphEngine, EnterpriseEngineMixin)
    engine = object.__new__(IntelligenceGraphEngine)
    engine.graph = _RecordingGraph()
    engine.backend = None
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)

    tool = _register_all().tools["graph_domain_ops"]
    result = json.loads(
        tool(
            action="allocate_budget",
            target_id="business-unit:test",
            amount=2500.0,
            currency="USD",
        )
    )

    assert result["business_unit_id"] == "business-unit:test"
    assert result["amount"] == 2500.0
    assert result["budget_id"] in engine.graph.nodes
