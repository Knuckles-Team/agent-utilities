import pytest
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from agent_utilities.agent_registry_builder import parse_frontmatter, rebuild_node_agents_md
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
    assert metadata["name"] == "test_agent"
    assert metadata["description"] == "A test agent"
    assert metadata["skills"] == ["test-skill"]

def test_parse_frontmatter_no_metadata():
    content = "# No Frontmatter"
    assert parse_frontmatter(content) is None

@pytest.mark.asyncio
async def test_rebuild_node_agents_md(tmp_path):
    # Setup mock workspace
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    agent_data = workspace / "agent_data"
    agent_data.mkdir()

    prompts_dir = Path(__file__).parent.parent / "agent_utilities" / "prompts"
    # Ensure prompts_dir exists for the rebuilder to scan,
    # but we will mock its path to use our tmp_path

    mcp_config = {
        "mcpServers": {
            "test-server": {"command": "echo"}
        }
    }
    mcp_config_file = agent_data / "mcp_config.json"
    mcp_config_file.write_text(json.dumps(mcp_config))

    registry_file = agent_data / "NODE_AGENTS.md"

    with patch("agent_utilities.agent_registry_builder.get_workspace_path") as mock_ws_path:
        def side_effect(key):
            if key == CORE_FILES["NODE_AGENTS"]: return registry_file
            if key == CORE_FILES["MCP_CONFIG"]: return mcp_config_file
            return workspace / key
        mock_ws_path.side_effect = side_effect

        # Mock prompts directory to a controlled set of files
        fake_prompts = tmp_path / "fake_prompts"
        fake_prompts.mkdir()
        test_prompt = fake_prompts / "test_agent.md"
        test_prompt.write_text("---\nname: test_agent\nskills: [skill1]\n---\n# Test Agent")

        with patch("agent_utilities.agent_registry_builder.Path") as mock_path:
            # This is tricky because Path is used for many things.
            # We'll mock the specific glob/exists calls instead if possible,
            # but let's try patching the prompts_dir logic directly if we can't.
            pass

        # Instead of deep mocking Path, let's just run it and see if it picks up real prompts,
        # but ensure our MCP config is handled.
        registry = await rebuild_node_agents_md()

        assert any(a.name == "test-server" for a in registry.agents)
        assert registry_file.exists()
        assert "| test-server |" in registry_file.read_text()

def test_get_discovery_registry(tmp_path):
    registry_file = tmp_path / "NODE_AGENTS.md"
    content = """# NODE_AGENTS.md
## Agent Mapping Table
| Name | Description | System Prompt | Tag | Skills | Tools | Skill Count | Tool Count | Avg Score |
|------|-------------|---------------|-----|--------|-------|-------------|------------|-----------|
| test_agent | desc | prompt.md | expert | skill1 | - | 1 | 0 | 100 |
"""
    registry_file.write_text(content)

    with patch("agent_utilities.graph.config_helpers.load_workspace_file", return_value=content):
        registry = get_discovery_registry()
        assert len(registry.agents) == 1
        assert registry.agents[0].name == "test_agent"

def test_load_specialized_prompts(tmp_path):
    mock_agent = MCPAgent(name="test", prompt_file="prompts/test.md")
    mock_registry = MCPAgentRegistryModel(agents=[mock_agent])

    prompt_content = "---\nname: test\n---\n# Real Prompt"

    with patch("agent_utilities.graph.config_helpers.get_discovery_registry", return_value=mock_registry):
        # Mock file system
        with patch("agent_utilities.graph.config_helpers.Path.exists", return_value=True):
            with patch("agent_utilities.graph.config_helpers.Path.read_text", return_value=prompt_content):
                res = load_specialized_prompts("test")
                assert "# Real Prompt" in res
                assert "---" not in res
