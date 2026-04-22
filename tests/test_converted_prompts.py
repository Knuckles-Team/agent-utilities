import json

from agent_utilities.graph.config_helpers import load_specialized_prompts


def test_load_converted_prompt():
    """Test that a prompt converted from Markdown loads correctly as JSON."""
    content = load_specialized_prompts("planner")

    # Should be valid JSON
    data = json.loads(content)
    assert data["task"] == "decompose user requests into high-fidelity tasks"
    assert "Project Planner System Prompt" in data["input"]
    assert "tools" in data
    assert "constitution-generator" in data["tools"]


def test_load_base_agent():
    """Test that the large base_agent prompt loads correctly."""
    content = load_specialized_prompts("base_agent")
    data = json.loads(content)
    assert data["task"] == "base_agent"
    assert len(data["input"]) > 1000
