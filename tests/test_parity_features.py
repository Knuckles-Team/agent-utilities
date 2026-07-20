import tempfile
from pathlib import Path

from agent_utilities.file_safety import backup_file_pre_edit
from agent_utilities.nested_context import get_nested_context


def test_backup_file_pre_edit():
    # Create a temporary file to back up
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(b"Hello world safety test")
        temp_file_path = Path(temp_file.name)

    try:
        # Perform the backup
        backup_path = backup_file_pre_edit(temp_file_path)

        # Verify backup was created
        assert backup_path is not None
        backup_path_obj = Path(backup_path)
        assert backup_path_obj.exists()
        assert backup_path_obj.is_file()

        # Verify content is identical
        with open(backup_path_obj, encoding="utf-8") as f:
            content = f.read()
        assert content == "Hello world safety test"

        # Cleanup backup
        backup_path_obj.unlink()
    finally:
        # Cleanup original
        if temp_file_path.exists():
            temp_file_path.unlink()


def test_backup_file_pre_edit_nonexistent():
    # Test backing up a non-existent file returns None
    result = backup_file_pre_edit(Path("/path/to/nonexistent/file/safety/test.txt"))
    assert result is None


def test_get_nested_context():
    # Setup temporary directory structure resembling a workspace
    with tempfile.TemporaryDirectory() as temp_workspace:
        workspace_path = Path(temp_workspace)

        # Workspace root instructions
        global_agents = workspace_path / "AGENTS.md"
        global_agents.write_text("Global Agent Instructions", encoding="utf-8")

        global_instructions = workspace_path / "INSTRUCTIONS.md"
        global_instructions.write_text("Global Core Instructions", encoding="utf-8")

        # Nested subdirectory
        nested_dir = workspace_path / "src" / "frontend"
        nested_dir.mkdir(parents=True, exist_ok=True)

        # Subfolder-specific instructions
        sub_agents = nested_dir / "AGENTS.md"
        sub_agents.write_text("Frontend Specific Instructions", encoding="utf-8")

        # Retrieve nested context
        aggregated = get_nested_context(nested_dir, workspace_root=workspace_path)

        # Assertions
        assert "Global Agent Instructions" in aggregated
        assert "Global Core Instructions" in aggregated
        assert "Frontend Specific Instructions" in aggregated

        # Check ordering: global should appear before nested subfolder
        global_agent_index = aggregated.find("Global Agent Instructions")
        sub_agent_index = aggregated.find("Frontend Specific Instructions")
        assert global_agent_index < sub_agent_index
