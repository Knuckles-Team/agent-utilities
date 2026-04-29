#!/usr/bin/python
"""Multi-agent team coordination capability with ACP integration.

Manages team membership, shared tasks, and message routing via the
Agent Communication Protocol (ACP). Persists state to the Knowledge Graph.

Falls back to A2A when ACP is unavailable.

Concept: team-coordination
"""

from __future__ import annotations

import contextlib
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
    Falls back to A2A tools when ACP session is unavailable.
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
            with contextlib.suppress(Exception):
                engine.graph.add_node(node.id, **node.model_dump())
                for member in member_ids:
                    # Link members to team
                    engine.graph.add_edge(member, self.team_id, type="BELONGS_TO_TEAM")
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
            with contextlib.suppress(Exception):
                engine.graph.add_node(node.id, **node.model_dump())
                if self.team_id:
                    engine.graph.add_edge(task_id, self.team_id, type="BELONGS_TO_TEAM")
                if assigned_to:
                    engine.graph.add_edge(
                        task_id, assigned_to, type="ASSIGNED_TO_AGENT"
                    )
        return task_id

    async def message_member(
        self, ctx: RunContext[Any], member_id: str, message: str
    ) -> bool:
        """Send a message to a team member via ACP, with A2A fallback."""
        # Strategy 1: ACP integration
        acp_session = getattr(ctx.deps, "acp_session", None)
        if acp_session:
            try:
                await acp_session.send_p2p(
                    member_id, {"type": "team_message", "content": message}
                )
                return True
            except Exception as e:
                logger.error(f"Failed to send ACP P2P message: {e}")

        # Strategy 2: A2A fallback
        a2a_client = getattr(ctx.deps, "a2a_client", None)
        if a2a_client:
            try:
                await a2a_client.send(
                    target_agent=member_id,
                    payload={"type": "team_message", "content": message},
                )
                logger.debug("Message sent to %s via A2A fallback", member_id)
                return True
            except Exception as e:
                logger.error(f"A2A fallback also failed for {member_id}: {e}")

        logger.warning(
            "No transport available to reach %s (neither ACP nor A2A)", member_id
        )
        return False

    async def discover_teams(self, ctx: RunContext[Any]) -> list[dict[str, Any]]:
        """Discover all active teams from the Knowledge Graph.

        Returns a list of dicts with team_id, name, status, and member_count.
        """
        engine = getattr(ctx.deps, "graph_engine", None)
        if not engine:
            return []

        teams: list[dict[str, Any]] = []
        for node_id, data in engine.graph.nodes(data=True):
            if (
                data.get("type") == RegistryNodeType.TEAM
                and data.get("status") == "active"
            ):
                teams.append(
                    {
                        "team_id": node_id,
                        "name": data.get("name", ""),
                        "status": data.get("status", ""),
                        "member_count": data.get("member_count", 0),
                    }
                )
        return teams

    async def update_task_status(
        self, ctx: RunContext[Any], task_id: str, status: str
    ) -> bool:
        """Update the status of a task in the Knowledge Graph.

        Args:
            ctx: Run context.
            task_id: The ID of the task node.
            status: New status (e.g., 'pending', 'in_progress', 'done').

        Returns:
            True if the update was successful.
        """
        engine = getattr(ctx.deps, "graph_engine", None)
        if not engine:
            return False

        if task_id not in engine.graph:
            logger.warning("Task %s not found in graph", task_id)
            return False

        engine.graph.nodes[task_id]["status"] = status
        engine.graph.nodes[task_id]["updated_at"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
        logger.info("Task %s status updated to '%s'", task_id, status)
        return True

    async def get_team_members(self, ctx: RunContext[Any]) -> list[str]:
        """Get the member IDs for the current team from the Knowledge Graph.

        Walks BELONGS_TO_TEAM edges in reverse to find connected agent nodes.
        """
        engine = getattr(ctx.deps, "graph_engine", None)
        if not engine or not self.team_id:
            return list(self.members)

        members: list[str] = []
        for src, tgt, data in engine.graph.in_edges(self.team_id, data=True):
            if data.get("type") == "BELONGS_TO_TEAM":
                members.append(src)
        return members or list(self.members)
