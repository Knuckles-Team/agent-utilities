from unittest.mock import patch

from agent_utilities.core.workspace import (
    get_workspace_path,
    initialize_workspace,
    load_workspace_file,
    write_workspace_file,
)


def test_get_workspace_path():
    path = get_workspace_path("test.md")
    assert path is not None


def test_initialize_workspace(tmp_path):
    import json

    with patch("agent_utilities.core.workspace.get_agent_workspace", return_value=tmp_path):
        initialize_workspace()
        main_agent = tmp_path / "main_agent.json"
        assert main_agent.exists()
        data = json.loads(main_agent.read_text(encoding="utf-8"))
        assert data["name"] == "main-agent"
        assert data["type"] == "prompt"
        assert data["content"].startswith("# Main Agent")


def test_write_and_load_file(tmp_path):
    with patch("agent_utilities.core.workspace.get_agent_workspace", return_value=tmp_path):
        write_workspace_file("test.md", "content")
        content = load_workspace_file("test.md")
        assert content == "content"
