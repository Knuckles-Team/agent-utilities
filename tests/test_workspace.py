import os
import pytest
from unittest.mock import patch
from agent_utilities.workspace import (
    get_agent_workspace,
    initialize_workspace,
    load_workspace_file,
    write_workspace_file,
    parse_identity,
    serialize_identity,
)


@pytest.fixture
def temp_workspace(tmp_path):
    with patch("agent_utilities.workspace.get_agent_workspace", return_value=tmp_path):
        yield tmp_path


def test_get_agent_workspace_env(tmp_path):
    os.environ["AGENT_WORKSPACE"] = str(tmp_path)
    try:
        # Reset the global variable if it was set
        import agent_utilities.workspace as ws

        ws.WORKSPACE_DIR = None

        path = get_agent_workspace()
        assert path == tmp_path.resolve()
    finally:
        os.environ.pop("AGENT_WORKSPACE", None)


def test_initialize_workspace(temp_workspace):
    initialize_workspace()
    assert (temp_workspace / "IDENTITY.md").exists()
    assert (temp_workspace / "USER.md").exists()
    assert (temp_workspace / "MEMORY.md").exists()
    assert (temp_workspace / "MCP_CONFIG.json").exists() or (
        temp_workspace / "mcp_config.json"
    ).exists()


def test_load_write_workspace_file(temp_workspace):
    filename = "test.txt"
    content = "hello workspace"
    write_workspace_file(filename, content)

    loaded = load_workspace_file(filename)
    assert loaded == content


def test_parse_serialize_identity():
    content = """# IDENTITY.md
## [default]
 * **Name:** Test Bot
 * **Role:** Tester
 * **Emoji:** 🧪
 * **Vibe:** Technical

 ### System Prompt
 You are a test bot.
"""
    model = parse_identity(content)
    assert model.name == "Test Bot"
    assert model.role == "Tester"
    assert "You are a test bot." in model.system_prompt

    serialized = serialize_identity(model)
    assert "**Name:** Test Bot" in serialized
    assert "tester" in serialized.lower()


def test_append_to_md_file(temp_workspace):
    from agent_utilities.workspace import append_to_md_file

    filename = "log.md"
    (temp_workspace / filename).write_text("# Log\n")

    append_to_md_file(filename, "Entry 1")
    content = (temp_workspace / filename).read_text()
    assert "Entry 1" in content


def test_parse_node_registry():
    """Test parsing the NODE_AGENTS.md registry."""
    from agent_utilities.workspace import parse_node_registry

    content = """# NODE_AGENTS.md
## Agent Mapping Table

| Name | Description | System Prompt | Tools | Tag / ID | Source MCP / Skill |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Repo Manager | Manages local git repositories | You are repo manager | git_action | repository_manager | repository_mcp |
| DevOps | Infrastructure expert | You are devops | deploy | devops_engineer | devops_mcp |
"""
    registry = parse_node_registry(content)
    assert len(registry.agents) == 2
    assert registry.agents[0].name == "Repo Manager"
    assert registry.agents[0].tag == "repository_manager"


def test_get_agent_icon_path(temp_workspace):
    """Test icon path resolution."""
    from agent_utilities.workspace import get_agent_icon_path

    # Default behavior: icon.png in workspace
    # Must exist for the function to return it
    (temp_workspace / "icon.png").write_text("fake icon")
    icon_path = get_agent_icon_path()
    assert icon_path == str(temp_workspace / "icon.png")

    # Custom icon
    custom_icon = temp_workspace / "custom.png"
    custom_icon.write_text("fake image")
    with patch("agent_utilities.workspace.CORE_FILES", {"ICON": "custom.png"}):
        icon_path = get_agent_icon_path()
        assert icon_path == str(custom_icon)
