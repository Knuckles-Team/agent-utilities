#!/usr/bin/python
"""Multi-agent team coordination capability with ACP integration.

Manages team membership, shared tasks, and message routing via the
Agent Communication Protocol (ACP). Persists state to the Knowledge Graph.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability

from ..models.knowledge_graph import RegistryNodeType, TaskNode, TeamNode

logger = logging.getLogger(__name__)


@dataclass
class SharedTodoItem:
    id: str
    content: str
    status: str = "pending"
    assigned_to: str | None = None
    created_by: str | None = None


@dataclass
class TeamCapability(AbstractCapability[Any]):
    """Capability that enables multi-agent team coordination.

    Integrates with ACP for messaging and persists team state in the graph.
    """

    team_id: str | None = None
    members: list[str] = field(default_factory=list)

    async def for_run(self, ctx: RunContext[Any]) -> TeamCapability:
        return replace(self)

    async def create_team(
        self, ctx: RunContext[Any], name: str, member_ids: list[str]
    ) -> str:
        """Create a new team and record it in the graph."""
        self.team_id = f"team_{uuid.uuid4().hex[:8]}"
        self.members = member_ids

        engine = getattr(ctx.deps, "graph_engine", None)
        if engine:
            node = TeamNode(
                id=self.team_id,
                type=RegistryNodeType.TEAM,
                name=name,
                status="active",
                member_count=len(member_ids),
                importance_score=0.8,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                metadata={"members": member_ids},
            )
            try:
                engine.graph.add_node(node.id, **node.model_dump())
                for member in member_ids:
                    # Link members to team
                    engine.graph.add_edge(member, self.team_id, type="BELONGS_TO_TEAM")
            except Exception:
                pass
        return self.team_id

    async def add_task(
        self, ctx: RunContext[Any], content: str, assigned_to: str | None = None
    ) -> str:
        """Add a shared task to the team and graph."""
        task_id = f"task_{uuid.uuid4().hex[:8]}"

        engine = getattr(ctx.deps, "graph_engine", None)
        if engine:
            node = TaskNode(
                id=task_id,
                type=RegistryNodeType.TASK,
                name=f"Task: {content[:30]}",
                content=content,
                status="pending",
                assigned_to=assigned_to,
                created_by=getattr(ctx.deps, "agent_id", "orchestrator"),
                importance_score=0.6,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
            try:
                engine.graph.add_node(node.id, **node.model_dump())
                if self.team_id:
                    engine.graph.add_edge(task_id, self.team_id, type="BELONGS_TO_TEAM")
                if assigned_to:
                    engine.graph.add_edge(
                        task_id, assigned_to, type="ASSIGNED_TO_AGENT"
                    )
            except Exception:
                pass
        return task_id

    async def message_member(
        self, ctx: RunContext[Any], member_id: str, message: str
    ) -> bool:
        """Send a message to a team member via ACP."""
        # ACP integration: Use pydantic-acp bridge if available
        acp_session = getattr(ctx.deps, "acp_session", None)
        if acp_session:
            # Send via ACP message bus
            try:
                await acp_session.send_p2p(
                    member_id, {"type": "team_message", "content": message}
                )
                return True
            except Exception as e:
                logger.error(f"Failed to send ACP P2P message: {e}")

        # Fallback to a2a tools if ACP session not active
        return False
