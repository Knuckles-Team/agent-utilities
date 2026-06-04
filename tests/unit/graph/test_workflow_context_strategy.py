"""Plan 03 Step 3: workflow-context shielding migrated from the deleted
graph/workflow_context_router.py into the routing strategy package.
"""

from __future__ import annotations

import pytest

from agent_utilities.graph.routing import ShieldedResult, WorkflowContextRouter
from agent_utilities.graph.routing.strategies.workflow_context import (
    ShieldedResult as DirectShielded,
)


def test_old_module_is_gone():
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("agent_utilities.graph.workflow_context_router")


def test_single_source_identity():
    # Package re-export and the strategy module are the same class object.
    assert ShieldedResult is DirectShielded


def test_shielded_result_prompt_string():
    sr = ShieldedResult(workflow_id="wf-1", summary="bounded ctx")
    assert sr.to_prompt_string() == "bounded ctx"
    empty = ShieldedResult(workflow_id="wf-2")
    assert "No contextual summary" in empty.to_prompt_string()


@pytest.mark.asyncio
async def test_route_context_without_engine():
    router = WorkflowContextRouter(engine=None)
    result = await router.route_context("what is the weather")
    assert isinstance(result, ShieldedResult)
    assert result.workflow_id.startswith("wf-")
    assert result.allowed_namespaces == []


@pytest.mark.asyncio
async def test_route_context_with_engine():
    class FakeEngine:
        def search_hybrid(self, query, top_k=3):
            return [{"id": "a"}, {"id": "b"}]

    router = WorkflowContextRouter(engine=FakeEngine())
    result = await router.route_context("find workflow")
    assert result.allowed_namespaces == ["ephemeral-a", "ephemeral-b"]
    assert "workflow contexts" in result.summary
