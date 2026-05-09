#!/usr/bin/python
"""Persistent Background Agents (CONCEPT:ORCH-1.19).

Manages long-running background agents coordinated via the Knowledge Graph.
Unlike ephemeral request agents (which are spawned per-request and destroyed),
persistent agents maintain state across sessions, react to KG events, and
coordinate with ephemeral agents.

Architecture::

    ┌─────────────────────────────────────────────┐
    │          CognitiveScheduler (OS-5.2)        │
    │   Unified scheduler for all agent types     │
    ├─────────────────────────────────────────────┤
    │                                             │
    │  ┌──────────────┐   ┌──────────────┐       │
    │  │  Ephemeral   │   │  Persistent  │       │
    │  │   Agents     │   │   Agents     │       │
    │  │ (per-request)│   │ (background) │       │
    │  └──────┬───────┘   └──────┬───────┘       │
    │         │                  │                │
    │         └──────┬───────────┘                │
    │                │                            │
    │         ┌──────┴───────┐                    │
    │         │ Knowledge    │                    │
    │         │ Graph (KG)   │                    │
    │         │ Coordination │                    │
    │         └──────────────┘                    │
    └─────────────────────────────────────────────┘

Lifecycle: registered → idle → running → idle → ... → terminated

Agent types:
    - **Monitor**: Watches KG for specific conditions (e.g., anomaly detection)
    - **Scheduler**: Runs periodic tasks (e.g., daily research ingestion)
    - **Rebalancer**: Continuously adjusts configurations (e.g., portfolio)
    - **Background**: General-purpose long-running agent
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from ..models.knowledge_graph import (
    PersistentAgentNode,
    RegistryNodeType,
)

if TYPE_CHECKING:
    from ..knowledge_graph.core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)


class PersistentAgentManager:
    """Manages persistent background agents coordinated via the KG.

    CONCEPT:ORCH-1.19 — Persistent Background Agents

    Each persistent agent is a ``PersistentAgentNode`` in the KG with:
        - ``subscriptions``: Event types to react to
        - ``schedule_cron``: Periodic execution schedule
        - ``state_snapshot``: Serialized context for continuity
        - ``heartbeat_ts``: Liveness tracking

    The manager integrates with the unified ``CognitiveScheduler`` (OS-5.2)
    for task scheduling and execution.

    Args:
        engine: The IntelligenceGraphEngine for KG coordination.
    """

    def __init__(self, engine: IntelligenceGraphEngine | None = None):
        self.engine = engine
        self._registered_agents: dict[str, PersistentAgentNode] = {}

    def register_agent(
        self,
        agent_id: str,
        name: str,
        agent_type: str = "background",
        subscriptions: list[str] | None = None,
        schedule_cron: str = "",
        specialist_ids: list[str] | None = None,
        tool_ids: list[str] | None = None,
        model_id: str = "",
        max_concurrent: int = 1,
    ) -> PersistentAgentNode:
        """Register a new persistent background agent.

        Args:
            agent_id: Unique agent identifier.
            name: Human-readable agent name.
            agent_type: Classification (background, monitor, scheduler, rebalancer).
            subscriptions: Event types this agent reacts to.
            schedule_cron: Cron expression for periodic execution.
            specialist_ids: Agent IDs this agent can spawn.
            tool_ids: Tools available to this agent.
            model_id: Preferred model for inference.
            max_concurrent: Maximum concurrent executions.

        Returns:
            The registered PersistentAgentNode.
        """
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        node = PersistentAgentNode(
            id=agent_id,
            name=name,
            agent_type=agent_type,
            subscriptions=subscriptions or [],
            schedule_cron=schedule_cron,
            heartbeat_ts=timestamp,
            status="idle",
            specialist_ids=specialist_ids or [],
            tool_ids=tool_ids or [],
            model_id=model_id,
            max_concurrent=max_concurrent,
        )

        # Store in local registry
        self._registered_agents[agent_id] = node

        # Persist to KG
        if self.engine:
            try:
                import json

                data = {
                    "id": agent_id,
                    "name": name,
                    "type": RegistryNodeType.PERSISTENT_AGENT.value,
                    "agent_type": agent_type,
                    "subscriptions": subscriptions or [],
                    "schedule_cron": schedule_cron,
                    "heartbeat_ts": timestamp,
                    "status": "idle",
                    "specialist_ids": specialist_ids or [],
                    "tool_ids": tool_ids or [],
                    "model_id": model_id,
                    "max_concurrent": max_concurrent,
                    "state_snapshot": json.dumps({}),
                }
                self.engine._upsert_node("PersistentAgent", agent_id, data)
                logger.info(
                    "[CONCEPT:ORCH-1.19] Registered persistent agent '%s' (type=%s)",
                    agent_id,
                    agent_type,
                )
            except Exception as e:
                logger.warning("Failed to persist agent registration: %s", e)

        return node

    def heartbeat(self, agent_id: str) -> None:
        """Update an agent's heartbeat timestamp.

        Should be called periodically by running agents to indicate liveness.

        Args:
            agent_id: The agent to update.
        """
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        if agent_id in self._registered_agents:
            self._registered_agents[agent_id].heartbeat_ts = timestamp

        if self.engine and self.engine.backend:
            try:
                self.engine.backend.execute(
                    "MATCH (a:PersistentAgent) WHERE a.id = $aid "
                    "SET a.heartbeat_ts = $ts",
                    {"aid": agent_id, "ts": timestamp},
                )
            except Exception as e:
                logger.debug("Heartbeat update failed: %s", e)

    def update_status(self, agent_id: str, status: str) -> None:
        """Update an agent's lifecycle status.

        Args:
            agent_id: The agent to update.
            status: New status (idle, running, paused, terminated).
        """
        if agent_id in self._registered_agents:
            self._registered_agents[agent_id].status = status

        if self.engine and self.engine.backend:
            try:
                self.engine.backend.execute(
                    "MATCH (a:PersistentAgent) WHERE a.id = $aid "
                    "SET a.status = $status",
                    {"aid": agent_id, "status": status},
                )
                logger.info(
                    "[CONCEPT:ORCH-1.19] Agent '%s' status → %s",
                    agent_id,
                    status,
                )
            except Exception as e:
                logger.debug("Status update failed: %s", e)

    def save_state(self, agent_id: str, state: dict[str, Any]) -> None:
        """Save an agent's state snapshot for session continuity.

        Args:
            agent_id: The agent whose state to save.
            state: Arbitrary state dict to persist.
        """
        if agent_id in self._registered_agents:
            self._registered_agents[agent_id].state_snapshot = state

        if self.engine and self.engine.backend:
            try:
                import json

                self.engine.backend.execute(
                    "MATCH (a:PersistentAgent) WHERE a.id = $aid "
                    "SET a.state_snapshot = $state",
                    {"aid": agent_id, "state": json.dumps(state)},
                )
            except Exception as e:
                logger.debug("State save failed: %s", e)

    def load_state(self, agent_id: str) -> dict[str, Any]:
        """Load an agent's saved state snapshot.

        Args:
            agent_id: The agent whose state to load.

        Returns:
            The state dict, or empty dict if not found.
        """
        # Try local first
        if agent_id in self._registered_agents:
            return self._registered_agents[agent_id].state_snapshot

        # Try KG
        if self.engine and self.engine.backend:
            try:
                import json

                results = self.engine.backend.execute(
                    "MATCH (a:PersistentAgent) WHERE a.id = $aid "
                    "RETURN a.state_snapshot AS state",
                    {"aid": agent_id},
                )
                if results:
                    raw = results[0].get("state", "{}")
                    if isinstance(raw, str):
                        return json.loads(raw)
                    return raw if isinstance(raw, dict) else {}
            except Exception as e:
                logger.debug("State load failed: %s", e)

        return {}

    def find_subscribers(self, event_type: str) -> list[str]:
        """Find all persistent agents subscribed to a given event type.

        Used by the EventStreamIngester (Company Brain) to route events
        to the correct persistent agents.

        Args:
            event_type: The event type to match (e.g., 'data.new').

        Returns:
            List of agent IDs subscribed to this event type.
        """
        subscribers: list[str] = []

        # Check local registry
        for aid, agent in self._registered_agents.items():
            if event_type in agent.subscriptions and agent.status != "terminated":
                subscribers.append(aid)

        # Check KG for agents not in local registry
        if self.engine and self.engine.backend:
            try:
                results = self.engine.backend.execute(
                    "MATCH (a:PersistentAgent) "
                    "WHERE $event IN a.subscriptions AND a.status <> 'terminated' "
                    "RETURN a.id AS aid",
                    {"event": event_type},
                )
                for r in results:
                    aid = r.get("aid", "")
                    if aid and aid not in subscribers:
                        subscribers.append(aid)
            except Exception as e:
                logger.debug("Subscriber lookup failed: %s", e)

        return subscribers

    def list_agents(
        self,
        status: str | None = None,
        agent_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """List registered persistent agents.

        Args:
            status: Optional filter by status.
            agent_type: Optional filter by agent type.

        Returns:
            List of agent summaries.
        """
        agents: list[dict[str, Any]] = []

        # Combine local and KG agents
        seen_ids: set[str] = set()

        for aid, agent in self._registered_agents.items():
            if status and agent.status != status:
                continue
            if agent_type and agent.agent_type != agent_type:
                continue
            agents.append(
                {
                    "id": aid,
                    "name": agent.name,
                    "agent_type": agent.agent_type,
                    "status": agent.status,
                    "heartbeat_ts": agent.heartbeat_ts,
                    "subscriptions": agent.subscriptions,
                }
            )
            seen_ids.add(aid)

        if self.engine and self.engine.backend:
            try:
                where_clauses = ["a.type = 'persistent_agent'"]
                params: dict[str, Any] = {}
                if status:
                    where_clauses.append("a.status = $status")
                    params["status"] = status
                if agent_type:
                    where_clauses.append("a.agent_type = $atype")
                    params["atype"] = agent_type

                where = " AND ".join(where_clauses)
                results = self.engine.backend.execute(
                    f"MATCH (a:PersistentAgent) WHERE {where} "
                    "RETURN a.id AS id, a.name AS name, "
                    "a.agent_type AS atype, a.status AS status, "
                    "a.heartbeat_ts AS hb",
                    params,
                )
                for r in results:
                    aid = r.get("id", "")
                    if aid and aid not in seen_ids:
                        agents.append(
                            {
                                "id": aid,
                                "name": r.get("name", ""),
                                "agent_type": r.get("atype", ""),
                                "status": r.get("status", ""),
                                "heartbeat_ts": r.get("hb", ""),
                            }
                        )
            except Exception as e:
                logger.debug("Agent listing failed: %s", e)

        return agents

    def terminate_agent(self, agent_id: str) -> None:
        """Terminate a persistent agent.

        Args:
            agent_id: The agent to terminate.
        """
        self.update_status(agent_id, "terminated")

        if agent_id in self._registered_agents:
            del self._registered_agents[agent_id]

        logger.info(
            "[CONCEPT:ORCH-1.19] Terminated persistent agent '%s'",
            agent_id,
        )

    def prune_stale_agents(self, max_age_seconds: int = 3600) -> int:
        """Terminate agents that haven't sent a heartbeat recently.

        Args:
            max_age_seconds: Maximum age of heartbeat before considered stale.

        Returns:
            Number of agents terminated.
        """
        now = time.time()
        pruned = 0

        for aid in list(self._registered_agents.keys()):
            agent = self._registered_agents[aid]
            if agent.status == "terminated":
                continue
            if agent.heartbeat_ts:
                try:
                    hb_time = time.mktime(
                        time.strptime(agent.heartbeat_ts, "%Y-%m-%dT%H:%M:%SZ")
                    )
                    if now - hb_time > max_age_seconds:
                        self.terminate_agent(aid)
                        pruned += 1
                except (ValueError, OverflowError):
                    pass

        if pruned:
            logger.info("[CONCEPT:ORCH-1.19] Pruned %d stale persistent agents", pruned)
        return pruned
