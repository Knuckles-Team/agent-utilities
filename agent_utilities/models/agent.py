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
    external_ontologies: list[str] = field(default_factory=list)
    # CONCEPT:ORCH-1.40 — the invoker↔spawned message channel this agent may talk back on.
    message_channel_id: str | None = None

    def __post_init__(self):
        # Auto-register external ontologies with the active graph engine
        if self.external_ontologies:
            try:
                from agent_utilities.knowledge_graph.core.engine import (
                    IntelligenceGraphEngine,
                )

                engine = IntelligenceGraphEngine.get_active()
                if engine and hasattr(engine, "register_external_ontology"):
                    for ext in self.external_ontologies:
                        parts = ext.split("|", 1)
                        uri = parts[0].strip()
                        endpoint = parts[1].strip() if len(parts) > 1 else None
                        engine.register_external_ontology(uri, endpoint)
            except ImportError:
                pass


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
