"""CONCEPT:AU-ECO.messaging.native-backend-abstraction"""

from unittest.mock import MagicMock

import pytest

from agent_utilities.tools.developer_tools import project_search


@pytest.fixture
def mock_ctx(tmp_path):
    ctx = MagicMock()
    ctx.deps.workspace_path = tmp_path
    return ctx


@pytest.mark.asyncio
async def test_project_search(mock_ctx, tmp_path):
    (tmp_path / "file.txt").write_text("foo\n", encoding="utf-8")

    res = await project_search(mock_ctx, "foo", ".")

    assert "file.txt" in res
    assert str(tmp_path) not in res


@pytest.mark.asyncio
async def test_project_search_rejects_unbounded_query(mock_ctx):
    result = await project_search(mock_ctx, "x" * 4_097, ".")

    assert "exceeds the input limit" in result


@pytest.mark.asyncio
async def test_project_search_rejects_workspace_escape(mock_ctx):
    result = await project_search(mock_ctx, "secret", "../")
    assert "outside the assigned workspace" in result
