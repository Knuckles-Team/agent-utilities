#!/usr/bin/python
from __future__ import annotations

import os
import shutil
import tempfile
import pytest
from typing import Any

from agent_utilities.knowledge_graph.distillation.physical_distiller import PhysicalDistillationEngine
from agent_utilities.harness.evolve_agent import EvolveAgent


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
        skill_code_path=skill_dir,
        tags=["evolved", "dns", "auto"],
        requires=["adguard-home-agent", "new-dependency"],
    )

    assert success is True

    # 3. Read back and assert values
    with open(skill_file, "r", encoding="utf-8") as f:
        updated_content = f.read()

    assert "name: evolved_skill_name" in updated_content
    assert "description: This is a highly advanced evolved skill description." in updated_content
    assert "tags: ['evolved', 'dns', 'auto']" in updated_content
    assert "requires: ['adguard-home-agent', 'new-dependency']" in updated_content
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

    with open(py_file, "r", encoding="utf-8") as f:
        updated_code = f.read()

    assert '"""This is a brand new description of the tool."""' in updated_code
    assert "Multi-line explanation here." not in updated_code


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

    with open(py_file, "r", encoding="utf-8") as f:
        updated_code = f.read()

    assert '"""Injected tool docstring."""' in updated_code
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

    with open(prompt_file, "r", encoding="utf-8") as f:
        content = f.read()

    assert content == "You are Antigravity, a self-evolving system prompt."


def test_dspy_dynamic_optimizers_selection(temp_workspace):
    """Verify that EvolveAgent properly instantiates and configures alternate optimizers."""
    # Test MIPROv2 configuration
    agent_mipro = EvolveAgent(
        workspace_path=temp_workspace,
        dspy_optimizer_type="MIPROv2"
    )
    assert agent_mipro.dspy_optimizer_type == "MIPROv2"

    # Test BootstrapFewShotWithRandomSearch configuration
    agent_search = EvolveAgent(
        workspace_path=temp_workspace,
        dspy_optimizer_type="BootstrapFewShotWithRandomSearch"
    )
    assert agent_search.dspy_optimizer_type == "BootstrapFewShotWithRandomSearch"

    # Test default
    agent_default = EvolveAgent(workspace_path=temp_workspace)
    assert agent_default.dspy_optimizer_type == "BootstrapFewShot"
