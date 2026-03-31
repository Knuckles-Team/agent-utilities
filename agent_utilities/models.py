from .config import *
from .workspace import *
import os
import datetime
from enum import Enum
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

__version__ = "0.2.39"


class PeriodicTask(BaseModel):
    id: str
    name: str
    interval_minutes: int
    prompt: str
    last_run: datetime = Field(default_factory=datetime.now)
    active: bool = True


from dataclasses import dataclass
from pathlib import Path
import asyncio


@dataclass
class AgentDeps:
    """Standard dependencies for agent tools."""

    workspace_path: Path
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ssl_verify: bool = True
    auth_token: Optional[str] = None

    elicitation_queue: Optional[asyncio.Queue] = None
    graph_event_queue: Optional[asyncio.Queue] = None
    request_id: str = ""
    """Unique identifier for the current request context (correlation ID)."""
    approval_timeout: float = 0.0
    """Timeout in seconds for human approval. 0 = infinite."""
    provider: Optional[str] = None
    model_id: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class Task(BaseModel):
    id: str = Field(default_factory=lambda: "task_" + os.urandom(4).hex())
    title: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: Optional[str] = None
    dependencies: List[str] = Field(default_factory=list)
    result: Optional[str] = None
    git_commit: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TaskPhase(BaseModel):
    name: str
    tasks: List[Task] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING


class TaskList(BaseModel):
    phases: List[TaskPhase] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    current_phase_index: int = 0
    version: str = "1.0.0"


class ProgressEntry(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat() + "Z")
    message: str
    git_commit: Optional[str] = None
    logs: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProgressLog(BaseModel):
    entries: List[ProgressEntry] = Field(default_factory=list)


class SprintContract(BaseModel):
    goals: List[str] = Field(default_factory=list)
    definition_of_done: List[str] = Field(default_factory=list)
    test_criteria: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
