"""Tests for the AU-DEV-R001.1 development-skill topic inventory model."""

import pytest
from pydantic import ValidationError

from agent_utilities.skills.dev_skill_topics import (
    REQUIRED_TOPICS,
    DevelopmentSkillTopicInventory,
)


def test_accepts_skill_covering_all_required_topics() -> None:
    inventory = DevelopmentSkillTopicInventory(
        skill_name="agent-utilities-development",
        topics=REQUIRED_TOPICS,
    )
    assert inventory.skill_name == "agent-utilities-development"


def test_refuses_skill_missing_a_required_topic() -> None:
    """AU-DEV-R001: a skill missing a required topic is refused."""
    with pytest.raises(ValidationError, match="AU-DEV-R001"):
        DevelopmentSkillTopicInventory(
            skill_name="incomplete-development-skill",
            topics=tuple(t for t in REQUIRED_TOPICS if t != "contract regeneration"),
        )
