from unittest.mock import patch
from agent_utilities.prompt_builder import (
    extract_section_from_md,
    build_system_prompt_from_workspace,
    resolve_prompt,
    extract_agent_metadata
)

def test_extract_section_from_md():
    """Test extracting sections between markdown headers."""
    content = """# Title
Some intro.

## Section 1
Content of section 1.
Multi-line.

## Section 2
Content of section 2.

### Subsection
Sub content.

## Section 3
Last content.
"""
    assert extract_section_from_md(content, "Section 1") == "Content of section 1.\nMulti-line."
    assert extract_section_from_md(content, "Section 2") == "Content of section 2."
    assert extract_section_from_md(content, "Section 3") == "Last content."
    assert extract_section_from_md(content, "Missing") is None

def test_build_system_prompt_from_workspace():
    """Test aggregation of core workspace files."""
    mock_files = {
        "main_agent.md": "I am an agent.",
        "USER.md": "You are a user.",
        "MEMORY.md": ""
    }

    with patch("agent_utilities.prompt_builder.load_workspace_file", side_effect=lambda x: mock_files.get(x, "")):
        prompt = build_system_prompt_from_workspace(fallback_prompt="Special info.")
        assert "main_agent.md" in prompt
        assert "I am an agent." in prompt
        assert "USER.md" in prompt
        assert "You are a user." in prompt
        assert "MEMORY.md" not in prompt
        assert "Special info." in prompt

def test_resolve_prompt():
    """Test resolving @ references in prompts."""
    with patch("agent_utilities.prompt_builder.load_workspace_file", return_value="file content"):
        assert resolve_prompt("@test.md") == "file content"
        assert resolve_prompt("raw prompt") == "raw prompt"

    with patch("agent_utilities.prompt_builder.load_workspace_file", return_value=""):
        assert resolve_prompt("@missing.md") == "@missing.md"

def test_extract_agent_metadata():
    """Test extraction of metadata from identity content."""
    content = """# main_agent.md
## [default]
 * **Name:** TestBot
 * **Role:** Tester
 * **Emoji:** 🤖
 * **Vibe:** Technical

 ### System Prompt
 You are a tester.
"""
    meta = extract_agent_metadata(content)
    assert meta["name"] == "TestBot"
    assert meta["description"] == "Tester"
    assert meta["emoji"] == "🤖"
    assert meta["content"] == "You are a tester."
