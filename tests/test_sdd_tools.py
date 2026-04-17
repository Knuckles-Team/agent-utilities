import pytest
import os
from unittest.mock import MagicMock
from pathlib import Path
from pydantic_ai import RunContext

from agent_utilities.tools.sdd_tools import (
    save_constitution,
    load_constitution,
    save_feature_spec,
    load_feature_spec,
    save_task_list,
    load_task_list
)
from agent_utilities.models import (
    ProjectConstitution,
    FeatureSpec,
    TaskList,
    Task,
    TaskPhase,
    AgentDeps
)

@pytest.fixture
def mock_ctx(tmp_path):
    """Provide a mock RunContext with a temporary workspace path."""
    deps = AgentDeps(workspace_path=str(tmp_path))
    ctx = MagicMock(spec=RunContext)
    ctx.deps = deps
    return ctx

@pytest.mark.asyncio
async def test_constitution_tools(mock_ctx):
    """Test save_constitution and load_constitution."""
    constitution = ProjectConstitution(vision="Test Vision")

    # Test Save
    msg = await save_constitution(mock_ctx, constitution)
    assert "Constitution saved" in msg

    # Test Load
    loaded = await load_constitution(mock_ctx)
    assert loaded.vision == "Test Vision"

@pytest.mark.asyncio
async def test_feature_spec_tools(mock_ctx):
    """Test save_feature_spec and load_feature_spec."""
    spec = FeatureSpec(name="Test Feature", description="Desc")
    feature_id = "feat-1"

    # Test Save
    msg = await save_feature_spec(mock_ctx, spec, feature_id)
    assert f"Feature spec '{feature_id}' saved" in msg

    # Test Load
    loaded = await load_feature_spec(mock_ctx, feature_id)
    assert loaded.name == "Test Feature"

@pytest.mark.asyncio
async def test_task_list_tools(mock_ctx):
    """Test save_task_list and load_task_list."""
    task_list = TaskList(phases=[TaskPhase(name="Phase 1", tasks=[])])
    feature_id = "feat-1"

    # Test Save
    msg = await save_task_list(mock_ctx, task_list, feature_id)
    assert f"Task list for '{feature_id}' saved" in msg

    # Test Load
    loaded = await load_task_list(mock_ctx, feature_id)
    assert len(loaded.phases) == 1
    assert loaded.phases[0].name == "Phase 1"

@pytest.mark.asyncio
async def test_load_nonexistent(mock_ctx):
    """Verify loading nonexistent artifacts handles errors gracefully."""
    # load_constitution should return empty model
    loaded_c = await load_constitution(mock_ctx)
    assert loaded_c.vision == ""

    # others should return None or handle gracefully if designed so
    # load_feature_spec returns None if not found
    loaded_s = await load_feature_spec(mock_ctx, "ghost")
    assert loaded_s is None
