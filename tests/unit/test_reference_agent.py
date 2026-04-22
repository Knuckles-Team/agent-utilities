#!/usr/bin/env python3
"""
Tests for Reference Agent Examples

Tests the example agents in examples/reference_agent/ to ensure they work correctly.
"""

from pathlib import Path


class TestBasicAgent:
    """Test basic agent creation and execution."""

    def test_basic_agent_creation(self):
        """Test that a basic agent can be created."""
        from agent_utilities import create_agent

        agent, _ = create_agent(name="TestBasicAgent")
        assert agent is not None
        assert agent.name == "TestBasicAgent"


class TestGraphAgent:
    """Test graph orchestration agent."""

    def test_graph_agent_creation(self):
        """Test that a graph agent can be created with universal skills."""
        from agent_utilities import create_agent

        agent, _ = create_agent(
            name="TestGraphAgent", skill_types=["universal", "graphs"]
        )
        assert agent is not None
        assert agent.name == "TestGraphAgent"


class TestMCPAgent:
    """Test MCP integration agent."""

    def test_mcp_agent_creation_without_config(self):
        """Test that an MCP agent can be created (even without config)."""
        from agent_utilities import create_agent

        # Should create agent even if config doesn't exist
        agent, _ = create_agent(name="TestMCPAgent", skill_types=["universal"])
        assert agent is not None
        assert agent.name == "TestMCPAgent"


class TestKnowledgeGraphAgent:
    """Test knowledge graph integration agent."""

    def test_knowledge_graph_agent_creation(self):
        """Test that a knowledge graph agent can be created."""
        from agent_utilities import create_agent

        agent, _ = create_agent(
            name="TestKnowledgeGraphAgent", skill_types=["universal", "graphs"]
        )
        assert agent is not None
        assert agent.name == "TestKnowledgeGraphAgent"


class TestProtocolAgent:
    """Test protocol adapter agent."""

    def test_protocol_agent_creation(self):
        """Test that a protocol agent can be created with ACP enabled."""
        from agent_utilities import create_agent

        agent, _ = create_agent(name="TestProtocolAgent", skill_types=["universal"])
        assert agent is not None
        assert agent.name == "TestProtocolAgent"


class TestMemoryAgent:
    """Test memory primitives agent."""

    def test_memory_agent_creation(self):
        """Test that a memory agent can be created with knowledge base."""
        from agent_utilities import create_agent

        agent, _ = create_agent(name="TestMemoryAgent", skill_types=["universal"])
        assert agent is not None
        assert agent.name == "TestMemoryAgent"


class TestReferenceAgentFiles:
    """Test that reference agent example files exist and are importable."""

    def test_basic_agent_file_exists(self):
        """Test that basic_agent.py exists."""
        path = Path("examples/reference_agent/basic_agent.py")
        assert path.exists()

    def test_graph_agent_file_exists(self):
        """Test that graph_agent.py exists."""
        path = Path("examples/reference_agent/graph_agent.py")
        assert path.exists()

    def test_mcp_agent_file_exists(self):
        """Test that mcp_agent.py exists."""
        path = Path("examples/reference_agent/mcp_agent.py")
        assert path.exists()

    def test_knowledge_graph_agent_file_exists(self):
        """Test that knowledge_graph_agent.py exists."""
        path = Path("examples/reference_agent/knowledge_graph_agent.py")
        assert path.exists()

    def test_protocol_agent_file_exists(self):
        """Test that protocol_agent.py exists."""
        path = Path("examples/reference_agent/protocol_agent.py")
        assert path.exists()

    def test_memory_agent_file_exists(self):
        """Test that memory_agent.py exists."""
        path = Path("examples/reference_agent/memory_agent.py")
        assert path.exists()

    def test_readme_file_exists(self):
        """Test that README.md exists."""
        path = Path("examples/reference_agent/README.md")
        assert path.exists()
