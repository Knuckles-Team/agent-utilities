"""AU-DEV-R001.2: audit the real agent-utilities-development skill.

AU-DEV-R001.1 added a typed ``DevelopmentSkillTopicInventory`` model that
refuses a skill body missing a required topic, proven against a synthetic
fixture. This slice applies that model to the real, checked-in
``agent-utilities-development`` skill: AU's own development skill must
actually declare every required topic, not just a fixture that claims to.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_utilities.skills.dev_skill_topics import (
    REQUIRED_TOPICS,
    DevelopmentSkillTopicInventory,
)

_SKILL_MD = (
    Path(__file__).resolve().parents[3]
    / "agent_utilities"
    / "skills"
    / "agent-utilities-development"
    / "SKILL.md"
)


def _declared_topics(skill_md: Path) -> tuple[str, ...]:
    body = skill_md.read_text(encoding="utf-8").lower()
    return tuple(topic for topic in REQUIRED_TOPICS if topic in body)


@pytest.mark.spec("AU-DEV-R001.2")
def test_real_development_skill_declares_every_required_topic() -> None:
    assert _SKILL_MD.is_file(), f"missing real development skill at {_SKILL_MD}"
    inventory = DevelopmentSkillTopicInventory(
        skill_name="agent-utilities-development",
        topics=_declared_topics(_SKILL_MD),
    )
    assert inventory.skill_name == "agent-utilities-development"
    assert set(REQUIRED_TOPICS).issubset({t.lower() for t in inventory.topics})
