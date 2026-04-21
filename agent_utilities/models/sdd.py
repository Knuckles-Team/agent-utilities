from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class TaskStatus(StrEnum):
    """Enumeration of possible task execution states."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class UserStory(BaseModel):
    id: str
    title: str
    description: str
    acceptance_criteria: list[str] = Field(default_factory=list)


class Spec(BaseModel):
    feature_id: str
    title: str
    user_stories: list[UserStory]
    non_functional_requirements: list[str] = Field(default_factory=list)
    success_metrics: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Task(BaseModel):
    id: str
    title: str
    description: str
    file_paths: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    parallel: bool = False
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: str | None = None
    result: str | None = None
    git_commit: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Tasks(BaseModel):
    feature_id: str = "default"
    tasks: list[Task] = Field(default_factory=list)
    parallel_waves: list[list[str]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ImplementationPlan(BaseModel):
    feature_id: str
    title: str
    approach: str = ""
    technical_context: str = ""
    proposed_changes: list[str] = Field(default_factory=list)
    tradeoffs: dict[str, str] = Field(default_factory=dict)
    risks: list[str] = Field(default_factory=list)
    risk_assessment: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProjectConstitution(BaseModel):
    vision: str = ""
    mission: str = ""
    core_principles: list[str] = Field(default_factory=list)
    tech_stack: dict[str, Any] = Field(default_factory=dict)
    quality_gates: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
