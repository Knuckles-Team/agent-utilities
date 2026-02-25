from datetime import datetime
from pydantic import BaseModel, Field

__version__ = "0.1.9"


class PeriodicTask(BaseModel):
    id: str
    name: str
    interval_minutes: int
    prompt: str
    last_run: datetime = Field(default_factory=datetime.now)
    active: bool = True
