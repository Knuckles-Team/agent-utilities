
from agent_utilities.graph.config_helpers import load_specialized_prompts


def test_load_specialized_prompts_structured():
    """Test that load_specialized_prompts loads from structured JSON files."""
    prompt_content = load_specialized_prompts("content_generation")

    # It should return a rendered string
    assert isinstance(prompt_content, str)
    assert len(prompt_content) > 5


def test_load_specialized_prompts_fallback():
    """Test fallback behavior for missing prompts."""
    content = load_specialized_prompts("non_existent_prompt")
    assert "helpful assistant specialized in non_existent_prompt" in content
