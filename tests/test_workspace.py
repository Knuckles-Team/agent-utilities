import pytest
import os
from unittest.mock import patch, mock_open
from agent_utilities.workspace import (
    initialize_workspace,
    get_workspace_path,
    write_workspace_file,
    load_workspace_file
)

def test_get_workspace_path():
    path = get_workspace_path("test.md")
    assert path is not None

def test_initialize_workspace(tmp_path):
    with patch("agent_utilities.workspace.get_agent_workspace", return_value=tmp_path):
        initialize_workspace()
        assert (tmp_path / "main_agent.md").exists()

def test_write_and_load_file(tmp_path):
    with patch("agent_utilities.workspace.get_agent_workspace", return_value=tmp_path):
        write_workspace_file("test.md", "content")
        content = load_workspace_file("test.md")
        assert content == "content"
