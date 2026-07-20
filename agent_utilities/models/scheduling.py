from datetime import datetime

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
    tasks: list[CronTaskModel] = Field(default_factory=list)


class CronLogEntryModel(BaseModel):
    timestamp: str
    task_id: str
    task_name: str = ""
    status: str = "success"
    message: str = ""
    chat_id: str | None = None


class CronLogModel(BaseModel):
    entries: list[CronLogEntryModel] = Field(default_factory=list)
