import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel


@dataclass
class AgentDeps:
    workspace_path: Path = field(default_factory=Path)
    knowledge_engine: Any | None = None
    user_id: str | None = None
    session_id: str | None = None
    ssl_verify: bool = True
    auth_token: str | None = None
    elicitation_queue: asyncio.Queue | None = None
    graph_event_queue: asyncio.Queue | None = None
    request_id: str = ""
    approval_timeout: float = 0.0
    provider: str | None = None
    model_id: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    mcp_toolsets: list[Any] = field(default_factory=list)
    patterns: Any | None = None


class IdentityModel(BaseModel):
    name: str = "AI Agent"
    role: str = ""
    emoji: str = "🤖"
    vibe: str = ""
    system_prompt: str = ""


class UserModel(BaseModel):
    name: str = "User"
    emoji: str = "👤"


class A2APeerModel(BaseModel):
    name: str
    url: str
    description: str = ""
    capabilities: str = ""
    auth: str = "none"
    notes: str = ""


class A2ARegistryModel(BaseModel):
    peers: list[A2APeerModel] = field(default_factory=list)
