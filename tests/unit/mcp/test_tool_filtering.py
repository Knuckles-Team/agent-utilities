"""Tests for tool filtering, skill matching, and tag-based discovery.

CONCEPT:AU-ECO.messaging.native-backend-abstraction — Agent Tool System
"""

from unittest.mock import MagicMock

from agent_utilities.tools.tool_filtering import (
    extract_skill_tags,
    extract_tool_tags,
    filter_tools_by_tag,
    skill_matches_tags,
)


def test_extract_skill_tags(tmp_path):
    skill_dir = tmp_path / "my_skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("""---
tags: [alpha, beta]
---
""")
    tags = extract_skill_tags(str(skill_dir))
    assert tags == ["alpha", "beta"]


def test_extract_tool_tags():
    # Mock tool with host-normalized meta tags
    tool = MagicMock()
    tool.meta = {"tags": ["mcp_tag"]}
    assert extract_tool_tags(tool) == ["mcp_tag"]

    # Mock tool with direct tags
    tool2 = MagicMock()
    tool2.tags = ["direct_tag"]
    tool2.meta = None
    tool2.metadata = None
    assert extract_tool_tags(tool2) == ["direct_tag"]


def test_filter_tools_by_tag():
    tool1 = MagicMock()
    tool1.tags = ["tag1"]
    tool2 = MagicMock()
    tool2.tags = ["tag2"]

    filtered = filter_tools_by_tag([tool1, tool2], "tag1")
    assert len(filtered) == 1
    assert filtered[0] == tool1


def test_skill_matches_tags(tmp_path):
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("""---
tags: [tag1]
categories: [cat1]
---
""")
    assert skill_matches_tags(str(skill_dir), ["tag1"]) is True
    assert skill_matches_tags(str(skill_dir), ["cat1"]) is True
    assert skill_matches_tags(str(skill_dir), ["skill"]) is True  # matches basename
    assert skill_matches_tags(str(skill_dir), ["other"]) is False
