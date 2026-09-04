"""Dependency-free actor classifications for verified security authority."""

from __future__ import annotations

from enum import StrEnum


class ActorType(StrEnum):
    """Provenance classification for a verified actor.

    Actor type is descriptive and never grants authorization. It lives at the
    lower security boundary so identity handling does not import the aggregate
    application-model package.
    """

    HUMAN = "human"
    AI_AGENT = "ai_agent"
    AUTOMATED_SERVICE = "automated_service"
    HYBRID_TEAM = "hybrid_team"
    SYSTEM = "system"


__all__ = ["ActorType"]
