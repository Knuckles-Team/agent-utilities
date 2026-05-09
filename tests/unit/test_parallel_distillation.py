import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from agent_utilities.graph.state import GraphDeps
from agent_utilities.graph.verification import parallel_trajectory_distiller


@pytest.fixture
def mock_deps():
    deps = MagicMock(spec=GraphDeps)
    deps.knowledge_engine = MagicMock()
    deps.knowledge_engine.add_node = MagicMock()
    deps.agent_model = MagicMock()
    return deps


@pytest.mark.asyncio
async def test_parallel_trajectory_distiller(mock_deps, monkeypatch):
    trajectories = [
        {"candidate_id": 0, "success": False, "output": "Failed due to timeout."},
        {
            "candidate_id": 1,
            "success": True,
            "output": "Succeeded by decomposing the query.",
        },
        {"candidate_id": 2, "success": True, "output": "Succeeded via decomposition."},
    ]
    query = "Extract data from a large repository."

    # Mock distillation_agent.run
    mock_run = AsyncMock()

    # Create a mock data object that resembles what the agent would return
    mock_res = MagicMock()
    mock_res.data = MagicMock()
    mock_res.data.tactical_condition = "Large repository data extraction"
    mock_res.data.tactical_action = "Decompose the query into smaller chunks"
    mock_res.data.confidence = 0.95

    mock_run.return_value = mock_res

    # Patch pydantic_ai.Agent.run directly because Agent is imported locally inside the function
    monkeypatch.setattr("pydantic_ai.Agent.run", mock_run)

    # Use a real string for agent_model so pydantic_ai doesn't try to JSON serialize a MagicMock in its __init__
    from pydantic_ai.models.test import TestModel

    mock_deps.agent_model = TestModel()

    mock_upsert = MagicMock()
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.ogm.KGMapper.upsert", mock_upsert
    )

    await parallel_trajectory_distiller(mock_deps, trajectories, query=query)

    # Assert upsert was called
    mock_upsert.assert_called_once()
    added_node = mock_upsert.call_args[0][0]

    assert added_node.type == "experience"
    assert "Parallel Tactic" in added_node.name
    assert added_node.condition == "Large repository data extraction"
    assert added_node.action == "Decompose the query into smaller chunks"

    # Check that CONCEPT:KG-2.4 PositionalInteractionEncoder mapping occurred
    assert "enc_pi" in added_node.metadata
    assert added_node.metadata["source"] == "parallel_scaling_distillation"
    assert len(added_node.metadata["enc_pi"]) > 0
