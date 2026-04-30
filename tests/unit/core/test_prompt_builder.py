import json
from unittest.mock import patch

import pytest

from agent_utilities.prompting.builder import (
    _extract_prompt_content,
    build_system_prompt_from_workspace,
    extract_agent_metadata,
    extract_section_from_md,
    resolve_prompt,
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
    assert (
        extract_section_from_md(content, "Section 1")
        == "Content of section 1.\nMulti-line."
    )
    assert extract_section_from_md(content, "Section 2") == "Content of section 2."
    assert extract_section_from_md(content, "Section 3") == "Last content."
    assert extract_section_from_md(content, "Missing") is None


def test_build_system_prompt_from_workspace():
    """Test aggregation of core workspace files (JSON blueprint format)."""
    main_agent_json = json.dumps(
        {
            "name": "main-agent",
            "type": "prompt",
            "description": "Primary orchestrator.",
            "capabilities": ["workspace-manager"],
            "content": "I am an agent.",
        },
        indent=2,
    )
    mock_files = {
        "main_agent.json": main_agent_json,
    }

    with patch(
        "agent_utilities.prompting.builder.load_workspace_file",
        side_effect=lambda x: mock_files.get(x, ""),
    ):
        prompt = build_system_prompt_from_workspace(fallback_prompt="Special info.")
        assert "main_agent.json" in prompt
        assert "I am an agent." in prompt
        assert "Special info." in prompt


def test_build_system_prompt_from_workspace_invalid_main_agent_json():
    """Malformed main_agent.json is logged as a warning, not raised.

    The function should still return a usable prompt (falling back to the
    packaged default or an empty prompt body) without crashing agent startup.
    """
    mock_files = {"main_agent.json": "I am an agent."}
    with patch(
        "agent_utilities.prompting.builder.load_workspace_file",
        side_effect=lambda x: mock_files.get(x, ""),
    ):
        prompt = build_system_prompt_from_workspace(fallback_prompt="fallback body")
        # Fallback body should still be appended even when main_agent.json
        # is malformed.
        assert "fallback body" in prompt


def test_extract_prompt_content_raises_on_non_json():
    """_extract_prompt_content raises ValueError for non-JSON payloads."""
    with pytest.raises(ValueError):
        _extract_prompt_content("I am a bare markdown prompt.")
    with pytest.raises(ValueError):
        _extract_prompt_content("")
    with pytest.raises(ValueError):
        _extract_prompt_content('{"name": "no-content-key"}')


def test_extract_prompt_content_happy_path():
    """_extract_prompt_content returns the content body for JSON blueprints."""
    payload = json.dumps({"name": "n", "content": "hello"})
    assert _extract_prompt_content(payload) == "hello"

    # Falls back to `input` when `content` is missing
    payload_legacy = json.dumps({"task": "t", "input": "# Plan"})
    assert _extract_prompt_content(payload_legacy) == "# Plan"


def test_resolve_prompt():
    """Test resolving @ references in prompts."""
    with patch(
        "agent_utilities.prompting.builder.load_workspace_file",
        return_value="file content",
    ):
        assert resolve_prompt("@test.md") == "file content"
        assert resolve_prompt("raw prompt") == "raw prompt"

    with patch("agent_utilities.prompting.builder.load_workspace_file", return_value=""):
        assert resolve_prompt("@missing.md") == "@missing.md"


def test_extract_agent_metadata_json_blueprint():
    """Modern JSON blueprint: ``name``/``description``/``content`` keys."""
    blueprint = json.dumps(
        {
            "name": "TestBot",
            "description": "Tester",
            "capabilities": ["qa"],
            "content": "You are a tester.",
            "emoji": "🤖",
            "vibe": "Technical",
        },
        indent=2,
    )
    meta = extract_agent_metadata(blueprint)
    assert meta["name"] == "TestBot"
    assert meta["description"] == "Tester"
    assert meta["emoji"] == "🤖"
    assert meta["vibe"] == "Technical"
    assert meta["content"] == "You are a tester."


def test_extract_agent_metadata_non_json_returns_default():
    """Non-JSON input returns generic default metadata with a warning.

    The legacy YAML-frontmatter and star-based markdown branches have been
    removed; any content that is not a JSON object yields the default meta
    dict rather than attempting to parse markdown.
    """
    content = """# main_agent
## [default]
 * **Name:** TestBot
 * **Role:** Tester
"""
    meta = extract_agent_metadata(content)
    # Defaults are preserved — no markdown parsing takes place.
    assert meta["name"] == "Agent"
    assert meta["description"] == "AI Agent"
    assert meta["emoji"] == "🤖"
