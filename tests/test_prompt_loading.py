import json

from agent_utilities.graph.config_helpers import load_specialized_prompts


def test_load_specialized_prompts_structured():
    """Test that load_specialized_prompts loads from structured JSON files."""
    # The file we created earlier: content_generation.json
    prompt_content = load_specialized_prompts("content_generation")

    # It should be a valid JSON string
    data = json.loads(prompt_content)
    assert data["task"] == "write content"
    assert data["platform"] == "twitter"
    assert "structure" in data
    assert data["structure"]["hook"] == "curiosity-driven"


def test_load_specialized_prompts_fallback():
    """Test fallback behavior for missing prompts."""
    content = load_specialized_prompts("non_existent_prompt")
    assert "helpful assistant specialized in non_existent_prompt" in content
