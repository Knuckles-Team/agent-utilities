"""Tests for SDD tools (setup, spec save, status).

CONCEPT:AU-009 — Spec-Driven Development
"""

import pytest
from unittest.mock import MagicMock, patch
from pydantic_ai import RunContext
from agent_utilities.tools.sdd_tools import (
    get_project_context,
    setup_sdd,
    save_spec,
    get_sdd_status
)
from agent_utilities.models import AgentDeps

@pytest.fixture
def mock_ctx():
    deps = MagicMock(spec=AgentDeps)
    deps.workspace_path = "/tmp/workspace"
    ctx = MagicMock(spec=RunContext)
    ctx.deps = deps
    return ctx

@pytest.mark.asyncio
async def test_setup_sdd(mock_ctx):
    with patch("agent_utilities.tools.sdd_tools.SDDManager") as MockManager:
        result = await setup_sdd(mock_ctx, "test_project")
        assert "initialized" in result
        MockManager.return_value.initialize.assert_called_once_with("test_project")

@pytest.mark.asyncio
async def test_get_project_context(mock_ctx):
    with patch("agent_utilities.tools.sdd_tools.SDDManager") as MockManager:
        MockManager.return_value.get_constitution.return_value = {
            "metadata": {"project_name": "Test"},
            "tech_stack": "Python",
            "vision": "World domination"
        }
        result = await get_project_context(mock_ctx)
        assert "Project: Test" in result
        assert "Vision: World domination" in result

@pytest.mark.asyncio
async def test_save_spec(mock_ctx):
    with patch("agent_utilities.tools.sdd_tools.SDDManager") as MockManager:
        result = await save_spec(mock_ctx, "feat1", "Detailed specification content")
        assert "saved successfully" in result
        MockManager.return_value.save.assert_called_once()

@pytest.mark.asyncio
async def test_get_sdd_status(mock_ctx):
    with patch("agent_utilities.tools.sdd_tools.SDDManager") as MockManager:
        MockManager.return_value.load.side_effect = [
            MagicMock(title="Spec"), # load Spec
            MagicMock(tasks=[MagicMock(), MagicMock()]) # load Tasks
        ]
        result = await get_sdd_status(mock_ctx, "feat1")
        assert "Spec: Found" in result
        assert "2 tasks found" in result
