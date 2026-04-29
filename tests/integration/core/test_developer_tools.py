from unittest.mock import MagicMock, patch

import pytest

from agent_utilities.tools.developer_tools import (
    create_file,
    delete_file,
    project_search,
    replace_in_file,
)


@pytest.fixture
def mock_ctx():
    return MagicMock()


@pytest.mark.asyncio
async def test_project_search(mock_ctx):
    with patch("subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "file.txt:1:foo"
        mock_run.return_value = mock_result
        res = await project_search(mock_ctx, "foo", ".")
        assert "file.txt" in res


@pytest.mark.asyncio
async def test_create_file(mock_ctx, tmp_path):
    f = tmp_path / "test.txt"
    res = await create_file(mock_ctx, str(f), "hello world")
    assert f.read_text() == "hello world"
    assert "Created file" in res


@pytest.mark.asyncio
async def test_delete_file(mock_ctx, tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("hello")
    res = await delete_file(mock_ctx, str(f))
    assert not f.exists()
    assert "Deleted" in res


@pytest.mark.asyncio
async def test_replace_in_file(mock_ctx, tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("hello world")
    res = await replace_in_file(mock_ctx, str(f), "hello", "hi")
    assert f.read_text() == "hi world"
    assert "Successfully updated" in res
