"""Typed model for development-skill required-topic coverage.

CONCEPT:AU-DEV.skills.development-workflow-documentation

AU-DEV-R001 requires that agent-utilities (and the equivalent per-repository
development skills for epistemic-graph, graph-os, and the connector SDK)
document architecture boundaries, supported build hosts, quality gates,
contract regeneration, and identity setup.

This is the ``.1`` slice for that row: a typed inventory model that refuses
a skill body missing one of the required topics. Trimming each repository's
AGENTS.md down to navigation-plus-pointer is the remaining behavior
(AU-DEV-R001.2+); see ``specs/au-developer-environment/tasks.md`` for the
recorded split.
"""

from __future__ import annotations

from pydantic import BaseModel, field_validator

REQUIRED_TOPICS: tuple[str, ...] = (
    "architecture boundaries",
    "build hosts",
    "quality gates",
    "contract regeneration",
    "identity setup",
)


class DevelopmentSkillTopicInventory(BaseModel):
    """A development skill's declared topic coverage.

    Construction refuses a skill whose ``topics`` do not cover every entry
    in ``REQUIRED_TOPICS``, per AU-DEV-R001.
    """

    skill_name: str
    topics: tuple[str, ...]

    @field_validator("topics")
    @classmethod
    def _refuse_missing_required_topics(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
        covered = {topic.lower() for topic in value}
        missing = [topic for topic in REQUIRED_TOPICS if topic not in covered]
        if missing:
            raise ValueError(
                "development skill is missing required topics "
                f"{missing} (AU-DEV-R001)"
            )
        return value
