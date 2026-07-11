#!/usr/bin/python
from __future__ import annotations

"""Multi-agent team coordination capability with ACP integration.

CONCEPT:AU-AHE.evaluation.interpretability-tests

Manages team membership, shared tasks, and message routing via the
Agent Communication Protocol (ACP). Persists state to the Knowledge Graph.

Falls back to A2A when ACP is unavailable.

Concept: team-coordination
"""


import contextlib
import logging
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability

# Multi-agent team coordination capability

logger = logging.getLogger(__name__)


@dataclass
class SharedTodoItem:
    id: str
    content: str
    status: str = "pending"
    assigned_to: str | None = None
    created_by: str | None = None


class _GraphComputeWorkItemView:
    """Adapts a ``GraphComputeEngine`` (``ctx.deps.graph_engine.graph`` — the
    exact object :class:`TeamCapability` already reads/writes) to the
    ``add_node``/``query_cypher``/``compare_and_set_node_fields`` protocol
    :mod:`agent_utilities.orchestration.work_item` expects, so team
    ``:TaskNode`` states can be migrated onto the SAME WorkItem state machine
    as the ``:AgentTask``/ingestion-queue paths WITHOUT introducing a second
    storage location — the shadow WorkItem lives in the exact graph object
    team tasks already live in today (CONCEPT:AU-AHE.evaluation.interpretability-tests / AU-P1-CL).

    Two adaptations are required because ``GraphComputeEngine``'s own
    ``add_node``/``query_cypher`` signatures differ from the protocol:

    * ``add_node`` stamps an explicit ``id`` property — the native Cypher
      engine's ``{id: $id}`` node-identity match requires it as a real
      stored property (unlike the backend-routed engines, which id-fast-path
      around this).
    * ``query_cypher`` has no separate params map (the wire protocol carries
      only literal query text) — params are inlined via
      ``EpistemicGraphBackend._inline_cypher_params``, the SAME primitive
      that backend already uses for this exact engine.
    """

    def __init__(self, graph: Any) -> None:
        self._graph = graph

    def add_node(
        self,
        node_id: str,
        node_type: str,
        properties: dict[str, Any] | None = None,
        ephemeral: bool = False,
    ) -> None:
        props = dict(properties or {})
        props["type"] = node_type
        props["id"] = node_id
        self._graph.add_node(node_id, props)

    def link_nodes(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: dict | None = None,
        ephemeral: bool = False,
    ) -> None:
        with contextlib.suppress(Exception):
            self._graph.add_edge(source_id, target_id, rel_type=str(rel_type))

    def query_cypher(
        self, cypher: str, params: dict | None = None
    ) -> list[dict[str, Any]]:
        from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
            EpistemicGraphBackend,
        )

        inlined = EpistemicGraphBackend._inline_cypher_params(cypher, params or {})
        try:
            return self._graph.query_cypher(inlined) or []
        except Exception as e:  # noqa: BLE001 — read is best-effort
            logger.debug("team-task work-item view: query_cypher failed: %s", e)
            return []

    def compare_and_set_node_fields(
        self, node_id: str, conditions: dict[str, Any], updates: dict[str, Any]
    ) -> bool:
        return bool(
            self._graph.compare_and_set_node_fields(node_id, conditions, updates)
        )


# Free-form team status word -> the WorkItem transition it drives. Anything
# NOT listed here (the team-collab API is intentionally lenient — arbitrary
# caller strings are accepted) simply skips the WorkItem transition and
# mirrors the literal string onto the legacy field, unchanged.
_TEAM_STATUS_TRANSITIONS: dict[str, str] = {
    "in_progress": "start",
    "running": "start",
    "started": "start",
    "completed": "succeed",
    "done": "succeed",
    "succeeded": "succeed",
    "cancelled": "cancel",
    "canceled": "cancel",
    "failed": "fail",
    "error": "fail",
}


def _team_task_claim_for_commit(
    view: _GraphComputeWorkItemView, task_id: str, tenant: str
) -> dict[str, Any]:
    """Ensure the shadow exists, claim it if still ``ready``, and return a
    claim dict usable for :func:`~agent_utilities.orchestration.work_item.commit_result`.

    Team tasks are single-writer (no competing worker pool), so when the
    shadow is already ``running`` (a prior ``in_progress`` transition already
    claimed it) this reads its CURRENT fencing epoch rather than re-claiming
    — safe here precisely because there is no concurrent claimant to fence
    against, unlike the ingestion queue / AgentTask dispatch.
    """
    from ..orchestration import work_item as _wi

    item_id = _wi.ensure_team_task_work_item(view, task_id, tenant=tenant)
    item = _wi.get_work_item(view, item_id)
    if item is not None and item.get("status") == _wi.WorkItemStatus.READY.value:
        claim = _wi.claim_specific(view, item_id)
        if claim is not None:
            _wi.mark_running(view, item_id, claim)
            return claim
        item = _wi.get_work_item(view, item_id)
    return {"work_item_id": item_id, "fence_token": (item or {}).get("lease_epoch")}


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
            from ..models.knowledge_graph import RegistryNodeType, TeamNode

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
            from ..models.knowledge_graph import RegistryNodeType, TaskNode

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
            # AU-P1-CL: create the ``ready`` shadow WorkItem alongside the
            # legacy TaskNode — best-effort, and independent of the legacy
            # write above so a WorkItem hiccup never blocks task creation.
            with contextlib.suppress(Exception):
                from ..orchestration import work_item as _wi

                _wi.ensure_team_task_work_item(
                    _GraphComputeWorkItemView(engine.graph),
                    task_id,
                    tenant=self.team_id or "",
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
        from ..knowledge_graph.core.bounded_read import iter_nodes_by_types
        from ..models.knowledge_graph import RegistryNodeType

        # Bounded per-label fetch (CONCEPT:AU-KG.ingest.never-scan-whole-graph) — never a whole-graph node pull.
        for node_id, data in iter_nodes_by_types(engine.graph, RegistryNodeType.TEAM):
            if data.get("status") == "active":
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

        AU-P1-CL: drives the REAL transition through the WorkItem state
        machine first (:func:`~agent_utilities.orchestration.work_item.start_team_task_work_item` /
        ``commit_result``/``cancel_work_item``) — best-effort, so a WorkItem
        plumbing hiccup never blocks the legacy mirror write below, which is
        what ``list_team_tasks``/existing callers read. The legacy field
        always mirrors the CALLER'S literal ``status`` string (this API is
        intentionally lenient, e.g. 'done' is accepted though not one of the
        canonical WorkItem-mapped words) — WorkItem is the authoritative
        decider of WHETHER/HOW a transition lands, not what string ends up in
        the field.

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

        transition = _TEAM_STATUS_TRANSITIONS.get(status.strip().lower())
        if transition is not None:
            with contextlib.suppress(Exception):
                from ..orchestration import work_item as _wi

                view = _GraphComputeWorkItemView(engine.graph)
                tenant = self.team_id or ""
                if transition == "start":
                    _wi.start_team_task_work_item(view, task_id, tenant=tenant)
                elif transition == "cancel":
                    _wi.cancel_work_item(view, _wi.team_task_work_item_id(task_id))
                else:  # "succeed" / "fail"
                    claim = _team_task_claim_for_commit(view, task_id, tenant)
                    _wi.commit_result(
                        view,
                        claim["work_item_id"],
                        claim,
                        outcome="succeeded" if transition == "succeed" else "failed",
                        retryable=False,
                    )

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
