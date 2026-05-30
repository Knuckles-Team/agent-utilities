"""Tests for Dynamic Subgraph Orchestration."""

# CONCEPT:ORCH-1.1

from unittest.mock import Mock

from agent_utilities.models.knowledge_graph import TeamComposition
from agent_utilities.orchestration.engine import AgentOrchestrationEngine


def test_dynamic_subgraph_synthesize_team():
    """Test synthesizing a team dynamically."""
    mock_engine = Mock()
    mock_backend = Mock()
    mock_engine.backend = mock_backend

    # Mock agents return
    mock_backend.execute.side_effect = [
        # Query 1: Retrieve candidate agents
        [
            {
                "agent_id": "agent_a",
                "role": "planner",
                "name": "Planner",
                "model_id": "m1",
            },
            {
                "agent_id": "agent_b",
                "role": "researcher",
                "name": "Researcher",
                "model_id": "m2",
            },
            {
                "agent_id": "agent_c",
                "role": "writer",
                "name": "Writer",
                "model_id": "m3",
            },
        ],
        # Query 2: Tools for planner
        [{"tool_name": "plan_task"}],
        # Query 3: Tools for researcher
        [{"tool_name": "search_web"}, {"tool_name": "read_doc"}],
        # Query 4: Tools for writer
        [{"tool_name": "write_report"}],
    ]

    orchestrator = AgentOrchestrationEngine(engine=mock_engine)
    team = orchestrator.synthesize_team(query="Research and write a report on AI")

    assert isinstance(team, TeamComposition)
    assert team.execution_mode in ["sequential", "parallel", "mixed"]
    assert len(team.adaptive_agent_router) == 3
    assert team.adaptive_agent_router[0]["role"] == "planner"
    assert team.adaptive_agent_router[1]["tools"] == ["search_web", "read_doc"]


def test_dynamic_subgraph_with_delegated_authority():
    """Test synthesizing a team dynamically with delegated authority constraints."""
    mock_engine = Mock()
    mock_backend = Mock()
    mock_engine.backend = mock_backend

    # Mock agents return (simulating the graph returning authorized agents)
    mock_backend.execute.side_effect = [
        # Query 1: Retrieve candidate agents with authority
        [
            {
                "agent_id": "agent_secure",
                "role": "financial_auditor",
                "name": "Secure Auditor",
            },
        ],
        # Query 2: Tools for financial_auditor
        [{"tool_name": "read_ledger"}],
    ]

    orchestrator = AgentOrchestrationEngine(engine=mock_engine)
    team = orchestrator.synthesize_team(
        query="Audit the Q3 ledger",
        domain="finance",
        delegated_authority="human_cf_officer_01",
    )

    assert isinstance(team, TeamComposition)
    assert len(team.adaptive_agent_router) == 1
    assert team.adaptive_agent_router[0]["role"] == "financial_auditor"

    # Check that the execute call contained the delegated_authority parameter
    called_args, called_kwargs = mock_backend.execute.call_args_list[0]
    assert called_args[1]["delegated_authority"] == "human_cf_officer_01"
