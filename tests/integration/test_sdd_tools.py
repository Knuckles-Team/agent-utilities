import pytest
from unittest.mock import MagicMock
from pydantic_ai import RunContext

from agent_utilities.tools.sdd_tools import (
    save_constitution,
    load_constitution,
    save_spec,
    load_spec,
    save_tasks,
    load_tasks
)
from agent_utilities.models import (
    ProjectConstitution,
    Spec,
    Tasks,
    Task,
    AgentDeps,
    UserStory
)

@pytest.fixture
def mock_ctx(tmp_path):
    """Provide a mock RunContext with a temporary workspace path."""
    deps = AgentDeps(workspace_path=tmp_path)
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
async def test_spec_tools(mock_ctx):
    """Test save_spec and load_spec."""
    spec = Spec(
        feature_id="feat-1",
        title="Test Feature",
        user_stories=[
            UserStory(id="US1", title="US1", description="D1", acceptance_criteria=[])
        ]
    )
    feature_id = "feat-1"

    # Test Save
    msg = await save_spec(mock_ctx, spec, feature_id)
    assert f"Spec '{feature_id}' saved" in msg

    # Test Load
    loaded = await load_spec(mock_ctx, feature_id)
    assert loaded is not None
    assert loaded.title == "Test Feature"

@pytest.mark.asyncio
async def test_tasks_tools(mock_ctx):
    """Test save_tasks and load_tasks."""
    tasks = Tasks(feature_id="feat-1", tasks=[])
    feature_id = "feat-1"

    # Test Save
    msg = await save_tasks(mock_ctx, tasks, feature_id)
    assert f"Tasks for '{feature_id}' saved" in msg

    # Test Load
    loaded = await load_tasks(mock_ctx, feature_id)
    assert loaded is not None
    assert len(loaded.tasks) == 0

@pytest.mark.asyncio
async def test_load_nonexistent(mock_ctx):
    """Verify loading nonexistent artifacts handles errors gracefully."""
    # load_constitution should return empty model
    loaded_c = await load_constitution(mock_ctx)
    assert loaded_c.vision == ""

    # load_spec returns None if not found
    loaded_s = await load_spec(mock_ctx, "ghost")
    assert loaded_s is None
