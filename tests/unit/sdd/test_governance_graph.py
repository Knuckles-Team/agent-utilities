"""Tests for SDD governance graph nodes, policies, and process flows.

CONCEPT:AU-AHE.harness.harness-evolution — Spec-Driven Development
"""

import pytest

from agent_utilities.graph.client import create_or_merge_node
from agent_utilities.graph.models import (
    Policy,
    ProcessFlow,
    ProcessStep,
)
from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine


def _real_engine() -> tuple[GraphComputeEngine, IntelligenceGraphEngine]:
    """A real, fixture-isolated (graph, engine) pair, actually bound together.

    A ``MagicMock()`` backend can't satisfy engine_query.query_cypher's
    governance layer (tenant scope + row ACL, including ACL
    *hydration* against the durable store via
    ``active.graph_compute.._client.nodes.properties_batch`` when a node's
    ACL isn't already cached) — a mocked ``properties_batch`` call returns a
    MagicMock, not a dict, which secured_reads._durable_access_rows
    correctly rejects ("Durable ACL hydration response is invalid"), so
    every governed read through a mock backend fails closed. Use a real
    isolated backend instead.
    """
    from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
        EpistemicGraphBackend,
    )

    graph = GraphComputeEngine(backend_type="rust")
    backend = EpistemicGraphBackend()
    backend._graph = graph
    engine = IntelligenceGraphEngine(backend=backend)
    IntelligenceGraphEngine.set_active(engine)
    return graph, engine


def _grant_read(node_id: str) -> None:
    """Grant this suite's ambient test actor read access to ``node_id``.

    engine_query.query_cypher's row-level ACL check denies any node with no
    explicit ACL outright, privileged or not (secured_reads.permit: "Nodes
    without an ACL are denied").
    """
    from _test_engine import TEST_AGENT_ID

    from agent_utilities.knowledge_graph.ontology.permissioning import build_acl

    build_acl(node_id, data_owner=TEST_AGENT_ID)


def test_policy_model_validation():
    """Test that Policy model validates correctly."""
    policy = Policy(
        id="pol:tdd_01",
        name="TDD Policy",
        description="Always write tests before code.",
        policy_id="pol:tdd_01",
        condition="new_feature == True",
        action="enforce_tdd",
    )
    assert policy.name == "TDD Policy"
    assert policy.policy_id == "pol:tdd_01"


def test_process_flow_model_validation():
    """Test that ProcessFlow model validates correctly."""
    flow = ProcessFlow(
        id="flow:feat_01",
        name="Feature Implementation",
        goal="Implement a new feature with tests",
        flow_id="flow:feat_01",
        start_step="step:01",
    )
    assert flow.name == "Feature Implementation"
    assert flow.flow_id == "flow:feat_01"


@pytest.mark.asyncio
async def test_create_or_merge_node_idempotency():
    """Test that create_or_merge_node adds nodes to the graph."""
    graph, _engine = _real_engine()

    policy = Policy(
        id="pol:safety_01",
        name="Safety First",
        description="Check safety before execution",
        policy_id="pol:safety_01",
        condition="always",
        action="check_safety",
    )

    result = await create_or_merge_node(policy)
    assert result is not None
    assert "pol:safety_01" in graph
    assert graph.nodes["pol:safety_01"]["name"] == "Safety First"


@pytest.mark.asyncio
async def test_process_step_sequence():
    """Test process step relationship logic."""
    step1 = ProcessStep(
        id="step:plan", name="Plan", step_id="step:plan", step_type="tool_call"
    )
    step2 = ProcessStep(
        id="step:exec", name="Execute", step_id="step:exec", step_type="tool_call"
    )

    graph, _engine = _real_engine()

    await create_or_merge_node(step1)
    await create_or_merge_node(step2)

    graph.add_edge(step1.step_id, step2.step_id, relationship="NEXT")

    assert "NEXT" in [
        d.get("relationship") for u, v, d in graph.out_edges(step1.step_id, data=True)
    ]


from unittest.mock import MagicMock, patch


def test_engine_policy_discovery():
    """Test that the engine can discover policies via Cypher."""
    from _test_engine import TEST_TENANT

    _graph, engine = _real_engine()
    engine.add_node(
        "pol:01",
        "Policy",
        {
            "name": "TDD Policy",
            "description": "Always write tests",
            # query_cypher's mandatory tenant scope (KG-2.6) filters on this;
            # a plain EpistemicGraphBackend (unlike BrainGuardedBackend) does
            # not stamp it automatically.
            "tenant_id": TEST_TENANT,
        },
    )
    _grant_read("pol:01")

    policies = engine.find_relevant_policies("TDD")

    assert len(policies) == 1
    assert policies[0]["name"] == "TDD Policy"


def test_engine_process_discovery():
    """Test that the engine can discover process flows via Cypher."""
    from _test_engine import TEST_TENANT

    _graph, engine = _real_engine()
    engine.add_node(
        "flow:01",
        "ProcessFlow",
        {
            "name": "Feature Flow",
            "goal": "Implement feature",
            "tenant_id": TEST_TENANT,
        },
    )
    _grant_read("flow:01")

    processes = engine.find_relevant_processes("feature")

    assert len(processes) == 1
    assert processes[0]["name"] == "Feature Flow"


@pytest.mark.asyncio
async def test_maintenance_model_validation(caplog):
    """Test the automated model validation routine in maintenance."""
    from agent_utilities.knowledge_graph.core.maintainer import GraphMaintainer

    GraphComputeEngine(backend_type="rust")
    mock_backend = MagicMock()
    # Return a node missing required fields
    mock_backend.execute.return_value = [{"n": {"id": "bad_pol", "name": "Invalid"}}]

    engine = IntelligenceGraphEngine(backend=mock_backend)
    maintainer = GraphMaintainer(engine=engine)

    with caplog.at_level("WARNING"):
        count = maintainer.validate_all_graph_models()
        assert "Invalid Policy node" in caplog.text
        # Should be 0 since the only policy was invalid
        assert count == 0


@pytest.mark.asyncio
async def test_process_executor_node_logic():
    """Test the load_and_execute_process_flow builder step execution."""
    from agent_utilities.graph.nodes import load_and_execute_process_flow
    from agent_utilities.graph.state import GraphDeps, GraphState

    with patch(
        "agent_utilities.graph.nodes.get_process_graph_backend"
    ) as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Mock policy retrieval sequence
        mock_client.execute.side_effect = [
            [
                {
                    "f": {
                        "name": "SOP 1",
                        "goal": "Fix bug",
                        "flow_id": "flow:bug",
                        "id": "flow:bug",
                        "start_step": "step:1",
                    },
                    "steps": [],
                }
            ],
            [
                {
                    "p": {
                        "name": "Guardrail",
                        "description": "Check logs",
                        "id": "pol:1",
                        "policy_id": "pol:1",
                        "condition": "any",
                        "action": "log",
                    }
                }
            ],
        ]

        state = GraphState(query="Fix bug")
        state.current_flow_id = "flow:bug"
        deps = GraphDeps(
            tag_prompts={}, tag_env_vars={}, mcp_toolsets=[], event_queue=MagicMock()
        )

        ctx = MagicMock()
        ctx.state = state
        ctx.deps = deps

        result = await load_and_execute_process_flow(ctx)
        assert result == "dispatcher"
        assert state.current_process_flow is not None
        assert state.current_process_flow.name == "SOP 1"
        assert len(state.active_policies) == 1
