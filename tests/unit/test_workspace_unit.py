import pytest
import os
from pathlib import Path
from agent_utilities import workspace

@pytest.fixture
def temp_workspace(tmp_path):
    """Fixture to provide a clean temporary workspace."""
    # Set the global WORKSPACE_DIR for testing
    original_workspace = workspace.WORKSPACE_DIR
    workspace.WORKSPACE_DIR = str(tmp_path)
    yield tmp_path
    workspace.WORKSPACE_DIR = original_workspace

def test_md_table_escape():
    assert workspace.md_table_escape("hello|world") == "hello\\|world"
    assert workspace.md_table_escape("hello\nworld") == "hello<br/>world"
    assert workspace.md_table_escape("hello\\nworld") == "hello<br/>world"
    assert workspace.md_table_escape(None) == ""

def test_smart_truncate():
    assert workspace.smart_truncate("hello world", 100) == "hello world"
    assert workspace.smart_truncate("hello world", 5) == "hello..."
    assert workspace.smart_truncate(None, 10) == "-"

def test_get_agent_workspace_env(monkeypatch, tmp_path):
    # Test Tier 2: AGENT_WORKSPACE env var
    workspace.WORKSPACE_DIR = None
    env_dir = tmp_path / "env_workspace"
    env_dir.mkdir()
    monkeypatch.setenv("AGENT_WORKSPACE", str(env_dir))
    assert workspace.get_agent_workspace() == env_dir

def test_get_workspace_path(temp_workspace):
    path = workspace.get_workspace_path("test.txt")
    assert path == temp_workspace / "test.txt"

def test_initialize_workspace(temp_workspace):
    workspace.initialize_workspace()
    assert (temp_workspace / "main_agent.md").exists()
    assert (temp_workspace / "mcp_config.json").exists()

def test_load_workspace_file(temp_workspace):
    test_file = temp_workspace / "test.txt"
    test_file.write_text("hello content")
    assert workspace.load_workspace_file("test.txt") == "hello content"
    assert workspace.load_workspace_file("missing.txt", default="fallback") == "fallback"

def test_write_workspace_file(temp_workspace):
    workspace.write_workspace_file("new.txt", "new content")
    assert (temp_workspace / "new.txt").read_text() == "new content"

def test_list_workspace_files(temp_workspace):
    (temp_workspace / "a.txt").touch()
    (temp_workspace / "b.txt").touch()
    files = workspace.list_workspace_files()
    assert "a.txt" in files
    assert "b.txt" in files

def test_md_file_operations(temp_workspace):
    workspace.write_md_file("test.md", "hello md")
    assert workspace.read_md_file("test.md") == "hello md"
    
    workspace.append_to_md_file("test.md", "appended")
    content = workspace.read_md_file("test.md")
    assert "hello md" in content
    assert "appended" in content
    
    with pytest.raises(ValueError):
        workspace.write_md_file("not_md.txt", "content")

def test_skill_lifecycle(temp_workspace):
    # Create
    msg = workspace.create_new_skill("Test Skill", "A test skill")
    assert "✅ Created" in msg
    skill_dir = temp_workspace / "skills" / "test-skill"
    assert skill_dir.exists()
    assert (skill_dir / "SKILL.md").exists()
    
    # Read
    content = workspace.read_skill_md("Test Skill")
    assert "test-skill" in content
    
    # Update
    msg = workspace.write_skill_md("Test Skill", "updated content")
    assert "✅ Updated" in msg
    assert workspace.read_skill_md("Test Skill").strip() == "updated content"
    
    # Delete
    msg = workspace.delete_skill_from_disk("Test Skill")
    assert "✅ Deleted" in msg
    assert not skill_dir.exists()

def test_delete_missing_skill(temp_workspace):
    msg = workspace.delete_skill_from_disk("Missing Skill")
    assert "❌ Skill" in msg

def test_read_missing_skill(temp_workspace):
    msg = workspace.read_skill_md("Missing Skill")
    assert "❌ SKILL.md not found" in msg

def test_write_missing_skill(temp_workspace):
    msg = workspace.write_skill_md("Missing Skill", "content")
    assert "❌ Skill" in msg
