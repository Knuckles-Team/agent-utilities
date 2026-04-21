import pytest
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from agent_utilities.agent_registry_builder import parse_frontmatter, ingest_prompts_to_graph
from agent_utilities.graph.config_helpers import get_discovery_registry, load_specialized_prompts
from agent_utilities.models import MCPAgent, MCPAgentRegistryModel
from agent_utilities.workspace import CORE_FILES

def test_parse_frontmatter():
    content = """---
name: test_agent
description: "A test agent"
skills: [test-skill]
---
# Actual Content
"""
    metadata = parse_frontmatter(content)
    assert metadata is not None
    assert metadata["name"] == "test_agent"
    assert metadata["description"] == "A test agent"
    assert metadata["skills"] == ["test-skill"]

def test_parse_frontmatter_no_metadata():
    content = "# No Frontmatter"
    assert parse_frontmatter(content) is None

def test_load_specialized_prompts(tmp_path):
    mock_agent = MCPAgent(name="test", description="desc", agent_type="prompt")
    mock_registry = MCPAgentRegistryModel(agents=[mock_agent])

    prompt_content = "---\nname: test\n---\n# Real Prompt"

    with patch("agent_utilities.graph.config_helpers.get_discovery_registry", return_value=mock_registry):
        # Mock file system
        with patch("agent_utilities.graph.config_helpers.Path.exists", return_value=True):
            with patch("agent_utilities.graph.config_helpers.Path.read_text", return_value=prompt_content):
                res = load_specialized_prompts("test")
                assert "# Real Prompt" in res
                assert "---" not in res
