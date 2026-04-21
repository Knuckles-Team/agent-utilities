from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ProgressEntry(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat() + "Z")
    message: str
    git_commit: str | None = None
    logs: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProgressLog(BaseModel):
    entries: list[ProgressEntry] = Field(default_factory=list)


class SprintContract(BaseModel):
    goals: list[str] = Field(default_factory=list)
    definition_of_done: list[str] = Field(default_factory=list)
    test_criteria: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
