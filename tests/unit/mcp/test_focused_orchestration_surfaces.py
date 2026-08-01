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
        "graph_agents": {
            "swarm",
            "computer_use",
            "synthesize_org",
            "run_org",
            "reason",
        },
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
            "evidence_lineage",
        },
        "graph_governance": {
            "grant_approval",
            "ownership_apply",
            "ownership_report",
            "policy_status",
            "submit_risk_veto",
            "verify_action",
        },
        "graph_jobs": {"dispatch", "status", "cancel", "input"},
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
    # 37, computed from `expected_actions` above, not guessed. Reconciliation
    # gate 2 merged TWO lanes that each add exactly one action:
    #   feat/wave7-followups-evolution -> graph_evolution "evidence_lineage" (D-71-4)
    #   feat/wave25-followups-mcpapps  -> graph_jobs      "input"
    # Each lane updated the action SET but not this total (they touched different
    # lines, so git merged both sets cleanly and left the stale count). The
    # `harvest_actions(...) == actions` assertion below is what actually pins the
    # surface against the real tools; this total is the redundant ratchet that
    # catches a silently added action.
    assert sum(map(len, expected_actions.values())) == 37
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
            workflow_run_id="",
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
            # No resume requested -> a brand-new run, explicitly.
            "workflow_run_id": None,
        }
    ]


@pytest.mark.asyncio
async def test_execute_dynamic_forwards_workflow_run_id_so_resume_is_reachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: the governed resume cache is keyed strictly on
    ``workflow_run_id``, but neither ``graph_workflows`` nor
    ``Orchestrator.execute_dynamic_workflow`` accepted or forwarded one -- so
    every live invocation minted a fresh id and the resume path, though fully
    implemented and unit-tested, was unreachable from the only entry point that
    exists.
    """
    from agent_utilities.mcp.tools import workflow_tools
    from agent_utilities.orchestration.manager import Orchestrator

    engine = object()
    calls: list[dict[str, object]] = []

    async def execute_dynamic(_self, workflow_id: str, task: str, **kwargs):
        calls.append({"workflow_id": workflow_id, "task": task, **kwargs})
        return {"backend": "x", "workflow_run_id": kwargs.get("workflow_run_id")}

    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)
    monkeypatch.setattr(
        workflow_tools,
        "_workflow_gate",
        lambda _engine, _name: {"allowed": True},
    )
    monkeypatch.setattr(Orchestrator, "execute_dynamic_workflow", execute_dynamic)
    tool = _register_all().tools["graph_workflows"]

    await tool(
        action="execute_dynamic",
        workflow="review",
        task="review the change",
        workflow_run_id="run:deadbeef",
    )
    assert calls[0]["workflow_run_id"] == "run:deadbeef"


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


@pytest.mark.asyncio
async def test_workflow_compile_live_route_isolates_graph_phases_and_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real GraphOS compile route keeps lookup and persistence off-loop."""

    from agent_utilities.knowledge_graph.core.session import (
        GraphSession,
        current_session,
        use_session,
    )
    from agent_utilities.knowledge_graph.workflow_compiler import WorkflowCompiler
    from agent_utilities.knowledge_graph.workflow_store import WorkflowStore
    from agent_utilities.mcp.tools import workflow_tools
    from agent_utilities.models.company_brain import ActorType
    from agent_utilities.security.brain_context import (
        ActorContext,
        current_actor,
        use_actor,
    )

    engine = object()
    main_thread = threading.current_thread()
    phase_order: list[str] = []
    phase_threads: list[threading.Thread] = []
    authority: list[tuple[object, object]] = []
    entered = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]

    def _phase(index: int, name: str) -> None:
        phase_order.append(name)
        phase_threads.append(threading.current_thread())
        authority.append((current_actor(), current_session()))
        entered[index].set()
        release[index].wait(0.5)

    def slow_match(_self, _text: str, _domain: str):
        _phase(0, "match")
        return "executor", []

    def slow_save(_self, **_kwargs):
        _phase(1, "save")
        return "workflow:stable"

    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)
    monkeypatch.setattr(WorkflowCompiler, "_match_agent", slow_match)
    monkeypatch.setattr(WorkflowStore, "save_workflow", slow_save)
    monkeypatch.setattr(
        workflow_tools,
        "_workflow_mermaid",
        lambda _engine, _name: "graph TD; executor",
    )
    tool = _register_all().tools["graph_workflows"]

    actor = ActorContext(
        actor_id="agent:compile-test",
        actor_type=ActorType.AI_AGENT,
        roles=("kg:write",),
        tenant_id="tenant:test",
        authenticated=True,
    )
    session = GraphSession(
        actor=actor,
        tenant="tenant:test",
        scopes=frozenset({"kg:read", "kg:write"}),
        graph="tenant:test",
        policy_version="test",
        audience="graph-os",
    )

    heartbeats = 0
    stop_heartbeat = False

    async def heartbeat() -> None:
        nonlocal heartbeats
        while not stop_heartbeat:
            heartbeats += 1
            await asyncio.sleep(0.002)

    with use_actor(actor), use_session(session):
        request = asyncio.create_task(
            tool(
                action="compile",
                workflow="",
                task="inspect repository health",
                name="stable",
                export_format="json",
                max_steps=30,
                limit=50,
                max_agent_calls=50,
                max_concurrency=8,
                budget_tokens=None,
                model_class="standard",
                dynamic_fallback="error",
            )
        )

    pulse = asyncio.create_task(heartbeat())
    try:
        for index in range(2):
            assert await asyncio.to_thread(entered[index].wait, 0.2)
            before = heartbeats
            await asyncio.sleep(0.02)
            assert heartbeats > before
            assert not request.done()
            release[index].set()
        payload = json.loads(await asyncio.wait_for(request, timeout=1.0))
    finally:
        for event in release:
            event.set()
        stop_heartbeat = True
        await pulse

    assert payload["workflow_id"] == "workflow:stable"
    assert phase_order == ["match", "save"]
    assert all(thread is not main_thread for thread in phase_threads)
    assert authority == [(actor, session), (actor, session)]


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


@pytest.mark.asyncio
async def test_graph_agents_reason_live_path_drives_real_cot_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``graph_agents action="reason"`` must actually drive the real
    ``agent_utilities.graph.reasoning`` package (CONCEPT:AU-ORCH.planning.
    reasoning-graph-topologies) -- not merely import it. Before this test's
    wiring, NOTHING outside ``tests/`` and the ``graph/reasoning`` package
    itself referenced ``run_cot``/``register_topology``/
    ``record_topology_outcome`` -- a fully built, unit-tested capability with
    zero live callers. This drives the tool end to end with a scripted LLM
    step function and asserts the OUTPUT is derived from the real
    ``run_cot``/``TopologySpec``/``register_topology``/
    ``record_topology_outcome`` machinery, not a standalone unit test of the
    topology module.
    """
    from agent_utilities.mcp.tools import agent_execution_tools

    class _StubBackend:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        def execute(self, query, params):
            self.calls.append((query, params))

    class _StubEngine:
        def __init__(self) -> None:
            self.nodes: dict[str, tuple[str, dict]] = {}
            self.backend = _StubBackend()

        def add_node(self, node_id, node_type, props):
            self.nodes[node_id] = (node_type, props)

    engine = _StubEngine()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)

    scripted = [
        agent_execution_tools.ReasoningStepOutput(
            summary="restated the question",
            content="6 times 7 is a multiplication of two small integers",
            is_final=False,
        ),
        agent_execution_tools.ReasoningStepOutput(
            summary="computed the product", content="42", is_final=True
        ),
    ]

    class _FakeRunResult:
        def __init__(self, output) -> None:
            self.output = output

    class _FakeAgent:
        def __init__(self, steps) -> None:
            self._steps = list(steps)

        def run_sync(self, _prompt: str):
            step = self._steps.pop(0) if self._steps else scripted[-1]
            return _FakeRunResult(step)

    fake_agent = _FakeAgent(scripted)
    monkeypatch.setattr(
        "agent_utilities.core.model_factory.create_model", lambda **_kw: object()
    )
    monkeypatch.setattr(
        "agent_utilities.core.contextual_model.create_context_agent",
        lambda **_kw: fake_agent,
    )

    tool = _register_all().tools["graph_agents"]
    payload = json.loads(
        await tool(
            action="reason",
            task="What is 6*7?",
            context="",
            context_ref="",
            max_fan_out=5,
            max_steps=10,
            host="",
            container_id="",
            options_json="{}",
            topology="cot",
            num_samples=3,
        )
    )

    # The answer/node-count are only producible by actually running run_cot
    # over the scripted steps -- not a fabricated/static response.
    assert payload["topology"] == "cot"
    assert payload["answer"] == "42"
    assert payload["node_count"] == 3  # root + the two scripted steps
    assert payload["rationale_summary"] == [
        "restated the question",
        "computed the product",
    ]
    assert payload["termination"]["success"] is True
    assert payload["termination"]["degraded"] is False

    # register_topology/record_topology_outcome (topology.py's own API) must
    # have actually written through to the engine, keyed by the REAL
    # content-addressed COT_SPEC.topology_id.
    from agent_utilities.graph.reasoning import COT_SPEC

    assert payload["topology_id"] == COT_SPEC.topology_id
    node_type, props = engine.nodes[COT_SPEC.topology_id]
    assert node_type == "reasoning_topology_version"
    assert props["artifact_id"] == "cot"
    assert props["version_hash"] == COT_SPEC.digest

    assert engine.backend.calls, "record_topology_outcome never reached the backend"
    _query, params = engine.backend.calls[-1]
    assert params["tid"] == COT_SPEC.topology_id
    assert params["score"] == 0.8
def test_graph_evolution_evidence_lineage_action_reaches_the_shared_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(D-71-4) ``evidence_lineage`` was import-only; this proves the MCP action
    dispatches into the SAME ``evidence.evidence_lineage()`` core."""
    from agent_utilities.knowledge_graph.research import evidence as evidence_module

    calls: list[tuple[Any, str]] = []

    def _fake_lineage(engine: Any, evidence_id: str) -> dict[str, Any]:
        calls.append((engine, evidence_id))
        return {"evidence_id": evidence_id, "found": True, "chain": [{"stage": "evidence"}]}

    engine = object()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)
    monkeypatch.setattr(evidence_module, "evidence_lineage", _fake_lineage)

    tool = _register_all().tools["graph_evolution"]
    payload = json.loads(
        tool(action="evidence_lineage", target="evolution_evidence:abc")
    )

    assert calls == [(engine, "evolution_evidence:abc")]
    assert payload == {
        "evidence_id": "evolution_evidence:abc",
        "found": True,
        "chain": [{"stage": "evidence"}],
    }
