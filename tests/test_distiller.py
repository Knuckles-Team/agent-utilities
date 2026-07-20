#!/usr/bin/python
from __future__ import annotations

import ast
import os
import shutil
import tempfile

import pytest
import yaml

from agent_utilities.knowledge_graph.distillation.physical_distiller import (
    PhysicalDistillationEngine,
)


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for distillation tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_distill_skill(temp_workspace):
    """Test that physical skill files are correctly parsed and updated."""
    engine = PhysicalDistillationEngine(workspace_root=temp_workspace)

    # 1. Create a dummy SKILL.md
    skill_dir = os.path.join(temp_workspace, "skills", "test_skill")
    os.makedirs(skill_dir, exist_ok=True)
    skill_file = os.path.join(skill_dir, "SKILL.md")

    initial_content = """---
name: old_skill_name
description: This is a legacy skill description.
domain: infra
tags: ['legacy', 'dns']
requires: ['adguard-home-agent']
---

# Test Skill Body
This is the workflow description.
"""
    with open(skill_file, "w", encoding="utf-8") as f:
        f.write(initial_content)

    # 2. Distill updated properties
    success = engine.distill_skill(
        skill_id="test_skill",
        new_name="evolved_skill_name",
        new_description="This is a highly advanced evolved skill description.",
        artifact_path=skill_dir,
        tags=["evolved", "dns", "auto"],
        requires=["adguard-home-agent", "new-dependency"],
    )

    assert success is True

    # 3. Read back and assert values
    with open(skill_file, encoding="utf-8") as f:
        updated_content = f.read()

    frontmatter = yaml.safe_load(updated_content.split("---", 2)[1])
    assert frontmatter["name"] == "evolved_skill_name"
    assert (
        frontmatter["description"]
        == "This is a highly advanced evolved skill description."
    )
    assert frontmatter["tags"] == ["evolved", "dns", "auto"]
    assert frontmatter["requires"] == ["adguard-home-agent", "new-dependency"]
    assert "# Test Skill Body" in updated_content
    assert "This is the workflow description." in updated_content


def test_distill_mcp_tool_with_existing_docstring(temp_workspace):
    """Test updating an existing docstring in a python file."""
    engine = PhysicalDistillationEngine(workspace_root=temp_workspace)

    py_file = os.path.join(temp_workspace, "tool.py")
    initial_code = """
@mcp.tool()
def test_tool(param1: str) -> str:
    \"\"\"This is an old docstring.

    Multi-line explanation here.
    \"\"\"
    return "done"
"""
    with open(py_file, "w", encoding="utf-8") as f:
        f.write(initial_code)

    success = engine.distill_mcp_tool(
        tool_name="test_tool",
        new_description="This is a brand new description of the tool.",
        file_path=py_file,
        function_name="test_tool",
    )

    assert success is True

    with open(py_file, encoding="utf-8") as f:
        updated_code = f.read()

    function = next(
        node
        for node in ast.walk(ast.parse(updated_code))
        if isinstance(node, ast.FunctionDef)
    )
    assert ast.get_docstring(function) == "This is a brand new description of the tool."


def test_distill_mcp_tool_without_docstring(temp_workspace):
    """Test injecting a docstring where none exists."""
    engine = PhysicalDistillationEngine(workspace_root=temp_workspace)

    py_file = os.path.join(temp_workspace, "tool_no_doc.py")
    initial_code = """
@mcp.tool()
def test_tool_no_doc(param1: str) -> str:
    x = 10
    return "done"
"""
    with open(py_file, "w", encoding="utf-8") as f:
        f.write(initial_code)

    success = engine.distill_mcp_tool(
        tool_name="test_tool_no_doc",
        new_description="Injected tool docstring.",
        file_path=py_file,
        function_name="test_tool_no_doc",
    )

    assert success is True

    with open(py_file, encoding="utf-8") as f:
        updated_code = f.read()

    function = next(
        node
        for node in ast.walk(ast.parse(updated_code))
        if isinstance(node, ast.FunctionDef)
    )
    assert ast.get_docstring(function) == "Injected tool docstring."
    assert "    x = 10" in updated_code


def test_distill_system_prompt(temp_workspace):
    """Test system prompt file overwrite."""
    engine = PhysicalDistillationEngine(workspace_root=temp_workspace)
    prompt_file = os.path.join(temp_workspace, "prompts", "IDENTITY.md")

    success = engine.distill_system_prompt(
        file_path=prompt_file,
        new_content="You are Antigravity, a self-evolving system prompt.",
    )

    assert success is True

    with open(prompt_file, encoding="utf-8") as f:
        content = f.read()

    assert content == "You are Antigravity, a self-evolving system prompt."


def test_distiller_rejects_path_escape_and_identifying_content(temp_workspace):
    engine = PhysicalDistillationEngine(workspace_root=temp_workspace)
    outside = os.path.join(os.path.dirname(temp_workspace), "outside.md")

    assert engine.distill_system_prompt(outside, "safe") is False
    assert (
        engine.distill_system_prompt(
            "prompts/private.md", "Contact person@example.test for access"
        )
        is False
    )
    assert not os.path.exists(outside)


def test_legacy_direct_commit_is_permanently_retired(temp_workspace):
    engine = PhysicalDistillationEngine(workspace_root=temp_workspace)
    assert engine.commit_distilled_changes(["SKILL.md"]) is False
