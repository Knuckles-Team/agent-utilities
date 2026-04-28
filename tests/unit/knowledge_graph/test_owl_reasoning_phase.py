#!/usr/bin/python
"""Unit tests for OWL Reasoning Pipeline Phase."""

import pytest
from unittest.mock import MagicMock, patch
import networkx as nx
from agent_utilities.knowledge_graph.pipeline.phases.owl_reasoning import execute_owl_reasoning
from agent_utilities.models.knowledge_graph import PipelineConfig

@pytest.mark.asyncio
async def test_owl_reasoning_phase_execution():
    # Setup context and config
    context = MagicMock()
    context.graph = nx.MultiDiGraph()
    context.engine = MagicMock()
    context.config = PipelineConfig(
        workspace_path=".",
        enable_owl_reasoning=True
    )

    # Mock OWLBridge and create_owl_backend
    with patch("agent_utilities.knowledge_graph.pipeline.phases.owl_reasoning.create_owl_backend") as mock_create:
        with patch("agent_utilities.knowledge_graph.pipeline.phases.owl_reasoning.OWLBridge") as mock_bridge_cls:
            mock_backend = MagicMock()
            mock_create.return_value = mock_backend

            mock_bridge = MagicMock()
            mock_bridge_cls.return_value = mock_bridge
            mock_bridge.run_cycle.return_value = {"inferred": 5}

            result = await execute_owl_reasoning(context, {})

            assert result["status"] == "completed"
            assert result["inferred"] == 5
            mock_bridge.run_cycle.assert_called_once()
            mock_backend.close.assert_called_once()

@pytest.mark.asyncio
async def test_owl_reasoning_phase_disabled():
    context = MagicMock()
    context.config = PipelineConfig(
        workspace_path=".",
        enable_owl_reasoning=False
    )

    result = await execute_owl_reasoning(context, {})

    assert result["status"] == "skipped"
    assert "disabled" in result["reason"]
