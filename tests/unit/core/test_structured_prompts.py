import json
from typing import Any

from agent_utilities.prompting.structured import NestedStructure, StructuredPrompt


def test_structured_prompt_validation():
    """Test that StructuredPrompt validates correctly with various fields."""
    data: dict[str, Any] = {
        "task": "write a tweet",
        "topic": "dopamine detox",
        "style": "viral",
        "length": "under 280 characters",
        "tone": "punchy and contrarian",
    }
    prompt = StructuredPrompt(**data)
    assert prompt.task == "write a tweet"
    assert prompt.topic == "dopamine detox"

    # Test rendering
    rendered = prompt.render()
    parsed = json.loads(rendered)
    assert parsed["task"] == "write a tweet"
    assert "platform" not in parsed


def test_nested_structure():
    """Test that NestedStructure works within StructuredPrompt."""
    data: dict[str, Any] = {
        "task": "write a thread",
        "platform": "twitter",
        "structure": {
            "hook": "curiosity-driven, under 10 words",
            "body": "3 insights with examples",
            "cta": "question that sparks replies",
        },
        "topic": "founder productivity",
    }
    prompt = StructuredPrompt(**data)
    assert isinstance(prompt.structure, NestedStructure)
    assert prompt.structure.hook == "curiosity-driven, under 10 words"

    rendered = prompt.render()
    parsed = json.loads(rendered)
    assert parsed["structure"]["hook"] == "curiosity-driven, under 10 words"


def test_extra_fields():
    """Test that extra fields are allowed and included in rendering."""
    data: dict[str, Any] = {"task": "custom task", "custom_field": "custom value", "another_field": 123}
    prompt = StructuredPrompt(**data)
    assert prompt.custom_field == "custom value"  # type: ignore[attr-defined]

    rendered = prompt.render()
    parsed = json.loads(rendered)
    assert parsed["custom_field"] == "custom value"
    assert parsed["another_field"] == 123


def test_from_kg():
    """Test hydration from KG-like dictionary."""
    kg_data: dict[str, Any] = {
        "name": "tweet_expert",
        "topic": "AI",
        "tone": "professional",
        "extra_info": "important",
    }
    prompt = StructuredPrompt.from_kg(kg_data)
    assert prompt.task == "tweet_expert"
    assert prompt.topic == "AI"
    assert prompt.tone == "professional"
    assert prompt.extra_info == "important"  # type: ignore[attr-defined]


def test_from_kg_with_blueprint():
    """Test hydration from KG with an embedded JSON blueprint."""
    blueprint = {"task": "special task", "goal": "be awesome"}
    kg_data: dict[str, Any] = {"json_blueprint": json.dumps(blueprint)}
    prompt = StructuredPrompt.from_kg(kg_data)
    assert prompt.task == "special task"
    assert prompt.goal == "be awesome"
