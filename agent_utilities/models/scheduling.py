from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class PeriodicTask(BaseModel):
    id: str
    name: str
    interval_minutes: int
    prompt: str
    last_run: datetime = Field(default_factory=datetime.now)
    active: bool = True


class CronTaskModel(BaseModel):
    id: str
    name: str
    interval_minutes: int
    prompt: str
    last_run: str = "—"
    next_approx: str = "—"


class CronRegistryModel(BaseModel):
    tasks: List[CronTaskModel] = Field(default_factory=list)


class CronLogEntryModel(BaseModel):
    timestamp: str
    task_id: str
    task_name: str = ""
    status: str = "success"
    message: str = ""
    chat_id: Optional[str] = None


class CronLogModel(BaseModel):
    entries: List[CronLogEntryModel] = Field(default_factory=list)
