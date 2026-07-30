"""Strict-current ownership contract for the focused Graph-OS execution tools."""

from __future__ import annotations

import asyncio
import inspect
import json
import threading
import time
from collections.abc import Callable
from typing import Any, cast

import pytest

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
        self.tools: dict[str, Callable[..., Any]] = {}

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
            "audit_scan",
            "distill_skills",
            "standardize",
            "failure_ingest",
            "optimize_component",
            "publish_proposal",
        },
        "graph_governance": {
            "grant_approval",
            "ownership_apply",
            "ownership_report",
            "policy_status",
            "submit_risk_veto",
            "verify_action",
        },
        "graph_jobs": {"dispatch", "status", "cancel"},
        "graph_rlm": {"run", "benchmark", "evolve_prompt"},
        "graph_workflows": {
            "compile",
            "compile_process",
            "list",
            "execute",
            "execute_dynamic",
            "dispatch",
            "status",
            "export",
        },
    }
    assert sum(map(len, expected_actions.values())) == 34
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


@pytest.mark.asyncio
async def test_execute_dynamic_live_route_reaches_governed_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_utilities.mcp.tools import workflow_tools
    from agent_utilities.orchestration.manager import Orchestrator

    engine = object()
    calls: list[dict[str, object]] = []

    async def execute_dynamic(_self, workflow_id: str, task: str, **kwargs):
        calls.append({"workflow_id": workflow_id, "task": task, **kwargs})
        return {
            "backend": "pydantic-ai-harness.dynamic_workflow.DynamicWorkflow",
            "workflow_run_id": "run:00000000000000000000000000000000",
        }

    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)
    monkeypatch.setattr(
        workflow_tools,
        "_workflow_gate",
        lambda _engine, _name: {"allowed": True},
    )
    monkeypatch.setattr(Orchestrator, "execute_dynamic_workflow", execute_dynamic)
    tool = _register_all().tools["graph_workflows"]

    payload = json.loads(
        await tool(
            action="execute_dynamic",
            workflow="review",
            task="review the change",
            name="",
            export_format="json",
            max_steps=12,
            limit=50,
            max_agent_calls=6,
            max_concurrency=3,
            budget_tokens=4000,
            model_class="economy",
            dynamic_fallback="error",
        )
    )

    assert payload["result"]["backend"].endswith("DynamicWorkflow")
    assert calls == [
        {
            "workflow_id": "review",
            "task": "review the change",
            "max_steps": 12,
            "max_agent_calls": 6,
            "max_concurrency": 3,
            "budget_tokens": 4000,
            "model_class": "economy",
            "unavailable_fallback": "error",
        }
    ]


@pytest.mark.asyncio
async def test_workflow_catalog_read_does_not_block_the_server_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow native graph read must not starve health/MCP request handling."""

    from agent_utilities.knowledge_graph.workflow_store import WorkflowStore

    engine = object()
    entered = threading.Event()
    fallback_release = threading.Event()

    def slow_list(_self, limit: int):
        entered.set()
        # The fallback prevents a broken implementation from hanging pytest,
        # while elapsed-time below distinguishes event-loop blocking.
        fallback_release.wait(0.5)
        return [{"name": "slow", "step_count": limit}]

    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)
    monkeypatch.setattr(WorkflowStore, "list_workflows", slow_list)
    tool = _register_all().tools["graph_workflows"]

    started = time.monotonic()
    request = asyncio.create_task(
        tool(
            action="list",
            workflow="",
            task="",
            name="",
            export_format="json",
            max_steps=30,
            limit=1,
            max_agent_calls=50,
            max_concurrency=8,
            budget_tokens=None,
            model_class="standard",
            dynamic_fallback="error",
        )
    )
    assert await asyncio.to_thread(entered.wait, 0.2)
    # Reaching this line before releasing the graph read proves the shared
    # event loop remained schedulable.
    fallback_release.set()
    payload = json.loads(await asyncio.wait_for(request, timeout=1.0))

    assert payload["workflows"][0]["name"] == "slow"
    assert time.monotonic() - started < 0.3


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
    engine = cast(Any, object.__new__(IntelligenceGraphEngine))
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
