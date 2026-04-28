from unittest.mock import MagicMock
from agent_utilities.tool_filtering import (
    _parse_skill_from_md,
    load_skills_from_directory,
    extract_skill_tags,
    extract_tool_tags,
    filter_tools_by_tag,
    skill_matches_tags
)

def test_parse_skill_from_md(tmp_path):
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text("""---
name: Test Skill
description: A test skill
version: 1.0.0
tags: [tag1, tag2]
---
# Content
""")
    skill = _parse_skill_from_md(skill_file, "test_id")
    assert skill is not None
    assert skill["name"] == "Test Skill"
    assert "tag1" in skill["tags"]
    assert skill["version"] == "1.0.0"

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
    # Mock tool with fastmcp tags
    tool = MagicMock()
    tool.meta = {"fastmcp": {"tags": ["mcp_tag"]}}
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
    assert skill_matches_tags(str(skill_dir), ["skill"]) is True # matches basename
    assert skill_matches_tags(str(skill_dir), ["other"]) is False

def test_load_skills_from_directory(tmp_path):
    s1 = tmp_path / "s1"
    s1.mkdir()
    (s1 / "SKILL.md").write_text("---\nname: S1\n---")

    s2 = tmp_path / "s2"
    s2.mkdir()
    (s2 / "SKILL.md").write_text("---\nname: S2\n---")

    skills = load_skills_from_directory(str(tmp_path))
    assert len(skills) == 2
    names = [s["name"] for s in skills]
    assert "S1" in names
    assert "S2" in names
