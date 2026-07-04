#!/usr/bin/python
"""CONCEPT:AU-KG.ingest.engineering-rules"""

"""Unit tests for OWL Reasoning Pipeline Phase."""

from unittest.mock import MagicMock, patch

import pytest

from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
from agent_utilities.knowledge_graph.pipeline.phases.owl_reasoning import (
    bootstrap_ontology_path,
    execute_owl_reasoning,
)
from agent_utilities.models.knowledge_graph import PipelineConfig


class _FakeGraph:
    """Minimal graph exposing ``nodes(data=True)`` like networkx/GraphComputeEngine."""

    def __init__(self, records):
        self._records = records  # list of (node_id, data)

    def nodes(self, data=False):
        if data:
            yield from self._records
        else:
            yield from (n for n, _ in self._records)


@pytest.mark.asyncio
async def test_owl_reasoning_phase_execution():
    # Setup context and config
    context = MagicMock()
    context.graph = GraphComputeEngine(backend_type="rust")
    context.engine = MagicMock()
    context.config = PipelineConfig(workspace_path=".", enable_owl_reasoning=True)

    # Mock OWLBridge and create_owl_backend
    with patch(
        "agent_utilities.knowledge_graph.pipeline.phases.owl_reasoning.create_owl_backend"
    ) as mock_create:
        with patch(
            "agent_utilities.knowledge_graph.pipeline.phases.owl_reasoning.OWLBridge"
        ) as mock_bridge_cls:
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


def test_bootstrap_ontology_path_derives_from_graph(tmp_path):
    records = [
        ("p1", {"type": "Product", "price": 9.99, "name": "Widget"}),
        ("p2", {"type": "Product", "price": 19.0, "name": "Gadget"}),
    ]
    ctx = MagicMock()
    ctx.graph = _FakeGraph(records)
    ctx.config = PipelineConfig(workspace_path=".")  # bootstrap defaults

    path = bootstrap_ontology_path(ctx)
    assert path and path.endswith(".ttl")
    ttl = open(path, encoding="utf-8").read()
    assert ":Product a rdfs:Class ." in ttl
    assert "rdfs:range xsd:decimal" in ttl  # price typed


def test_bootstrap_ontology_path_returns_none_when_empty():
    ctx = MagicMock()
    ctx.graph = _FakeGraph([])
    ctx.config = PipelineConfig(workspace_path=".")
    assert bootstrap_ontology_path(ctx) is None


@pytest.mark.asyncio
async def test_owl_reasoning_uses_bootstrapped_ontology():
    records = [("p1", {"type": "Product", "price": 9.99})]
    context = MagicMock()
    context.graph = _FakeGraph(records)
    context.config = PipelineConfig(
        workspace_path=".",
        enable_owl_reasoning=True,
        enable_ontology_bootstrap=True,
        owl_ontology_path=None,
    )
    with (
        patch(
            "agent_utilities.knowledge_graph.pipeline.phases.owl_reasoning.create_owl_backend"
        ) as mock_create,
        patch(
            "agent_utilities.knowledge_graph.pipeline.phases.owl_reasoning.OWLBridge"
        ) as mock_bridge_cls,
    ):
        mock_create.return_value = MagicMock()
        mock_bridge_cls.return_value.run_cycle.return_value = {"inferred": 1}
        result = await execute_owl_reasoning(context, {})
        assert result["status"] == "completed"
        used_path = mock_create.call_args.kwargs["ontology_path"]
        assert (
            "bootstrap_ontology_" in used_path
        )  # the derived ontology, not the bundle


@pytest.mark.asyncio
async def test_owl_reasoning_phase_disabled():
    context = MagicMock()
    context.config = PipelineConfig(workspace_path=".", enable_owl_reasoning=False)

    result = await execute_owl_reasoning(context, {})

    assert result["status"] == "skipped"
    assert "disabled" in result["reason"]
