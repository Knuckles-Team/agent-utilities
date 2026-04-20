import asyncio
from pathlib import Path
from typing import Optional, List, Any
from dataclasses import dataclass, field
from pydantic import BaseModel


@dataclass
class AgentDeps:
    workspace_path: Path = field(default_factory=Path)
    knowledge_engine: Optional[Any] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ssl_verify: bool = True
    auth_token: Optional[str] = None
    elicitation_queue: Optional[asyncio.Queue] = None
    graph_event_queue: Optional[asyncio.Queue] = None
    request_id: str = ""
    approval_timeout: float = 0.0
    provider: Optional[str] = None
    model_id: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    mcp_toolsets: List[Any] = field(default_factory=list)


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
    peers: List[A2APeerModel] = field(default_factory=list)
