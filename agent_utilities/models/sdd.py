from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Enumeration of possible task execution states."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class UserStory(BaseModel):
    id: str
    title: str
    description: str
    acceptance_criteria: List[str] = Field(default_factory=list)


class Spec(BaseModel):
    feature_id: str
    title: str
    user_stories: List[UserStory]
    non_functional_requirements: List[str] = Field(default_factory=list)
    success_metrics: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Task(BaseModel):
    id: str
    title: str
    description: str
    file_paths: List[str] = Field(default_factory=list)
    depends_on: List[str] = Field(default_factory=list)
    parallel: bool = False
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: Optional[str] = None
    result: Optional[str] = None
    git_commit: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Tasks(BaseModel):
    feature_id: str = "default"
    tasks: List[Task] = Field(default_factory=list)
    parallel_waves: List[List[str]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ImplementationPlan(BaseModel):
    feature_id: str
    title: str
    technical_context: str = ""
    proposed_changes: List[str] = Field(default_factory=list)
    tradeoffs: Dict[str, str] = Field(default_factory=dict)
    risk_assessment: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProjectConstitution(BaseModel):
    vision: str = ""
    mission: str = ""
    core_principles: List[str] = Field(default_factory=list)
    tech_stack: Dict[str, Any] = Field(default_factory=dict)
    quality_gates: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
