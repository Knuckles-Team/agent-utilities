"""Tests for CONCEPT:AU-ECO.messaging.native-backend-abstraction — A2A-Native PlannerAgent (Graph-Backed Skill).

Validates:
    - ``PlannerGraphSkill`` instantiation and configuration
    - ``_extract_query()`` message parsing (standard and fallback)
    - Skill metadata (id, name, tags)
"""

import pytest

from agent_utilities.protocols.a2a_graph_skill import PlannerGraphSkill


@pytest.mark.concept("CONCEPT:AU-ECO.messaging.native-backend-abstraction")
class TestPlannerGraphSkill:
    """Test suite for the A2A graph-backed skill."""

    def test_instantiation(self):
        """Skill should instantiate with correct defaults."""
        skill = PlannerGraphSkill(
            graph=None,
            graph_config=None,
        )
        assert skill.id == "planner"
        assert skill.name == "Planner"
        assert "planner" in skill.tags
        assert skill.input_modes == ["text"]
        assert skill.output_modes == ["text"]

    def test_custom_metadata(self):
        """Should accept custom metadata."""
        skill = PlannerGraphSkill(
            graph=None,
            graph_config=None,
            skill_id="custom_planner",
            name="Custom Planner",
            description="Custom graph planner",
            tags=["custom", "graph"],
        )
        assert skill.id == "custom_planner"
        assert skill.name == "Custom Planner"
        assert skill.tags == ["custom", "graph"]


@pytest.mark.concept("CONCEPT:AU-ECO.messaging.native-backend-abstraction")
class TestExtractQuery:
    """Test suite for message parsing."""

    def test_extract_user_text_part(self):
        """Should extract text from standard A2A user message."""
        messages = [
            {
                "role": "user",
                "parts": [{"kind": "text", "text": "Hello world"}],
            }
        ]
        result = PlannerGraphSkill._extract_query(messages)
        assert result == "Hello world"

    def test_extract_latest_user_message(self):
        """Should extract the LATEST user message, not the first."""
        messages = [
            {
                "role": "user",
                "parts": [{"kind": "text", "text": "First message"}],
            },
            {
                "role": "assistant",
                "parts": [{"kind": "text", "text": "Reply"}],
            },
            {
                "role": "user",
                "parts": [{"kind": "text", "text": "Second message"}],
            },
        ]
        result = PlannerGraphSkill._extract_query(messages)
        assert result == "Second message"

    def test_fallback_to_content_field(self):
        """Should fallback to content field if no parts."""
        messages = [
            {"role": "user", "content": "Fallback content"},
        ]
        result = PlannerGraphSkill._extract_query(messages)
        assert result == "Fallback content"

    def test_empty_messages(self):
        """Should return empty string for empty messages."""
        result = PlannerGraphSkill._extract_query([])
        assert result == ""

    def test_no_user_messages(self):
        """Should return empty string when no user messages exist."""
        messages = [
            {
                "role": "assistant",
                "parts": [{"kind": "text", "text": "Only assistant"}],
            },
        ]
        result = PlannerGraphSkill._extract_query(messages)
        assert result == ""

    def test_skips_non_text_parts(self):
        """Should skip non-text parts (e.g., image)."""
        messages = [
            {
                "role": "user",
                "parts": [
                    {"kind": "image", "data": "base64..."},
                    {"kind": "text", "text": "Describe this"},
                ],
            },
        ]
        result = PlannerGraphSkill._extract_query(messages)
        assert result == "Describe this"
