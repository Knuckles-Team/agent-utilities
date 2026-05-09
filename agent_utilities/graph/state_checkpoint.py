#!/usr/bin/python
"""Execution State Checkpointing (CONCEPT:ORCH-1.16).

Bridges the ephemeral ``GraphState`` and the persistent Knowledge Graph.
Checkpoints are created at HSM transition boundaries and on session
completion, enabling:

    - **Session resume**: Recover from crashes or context limits.
    - **Cross-session learning**: The KG can query active/past states.
    - **Multi-agent coordination**: Other agents can see execution state.

Data flow::

    GraphState  ──checkpoint()──→  SessionCheckpointNode (in KG)
    GraphState  ←──restore()───←  SessionCheckpointNode (from KG)

The checkpointer is lightweight — it serializes only the essential
GraphState fields, not the full agent context or model weights.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

from ..models.knowledge_graph import (
    RegistryNodeType,
)

if TYPE_CHECKING:
    from ..knowledge_graph.core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)


class StateCheckpointer:
    """Checkpoints GraphState to/from the Knowledge Graph.

    CONCEPT:ORCH-1.16 — Execution State Persistence

    Converts key GraphState fields to a ``SessionCheckpointNode`` in the KG,
    enabling session resume, cross-session learning, and active execution
    state queries from other agents.

    Usage::

        checkpointer = StateCheckpointer(engine)

        # At transition boundaries:
        checkpoint_id = checkpointer.checkpoint(state)

        # To resume:
        restored_state = checkpointer.restore(session_id)

    Args:
        engine: The IntelligenceGraphEngine for KG persistence.
    """

    def __init__(self, engine: IntelligenceGraphEngine | None = None):
        self.engine = engine

    def checkpoint(
        self,
        state: Any,
        session_id: str | None = None,
        status: str = "active",
    ) -> str:
        """Persist current execution state to the Knowledge Graph.

        Args:
            state: The GraphState object (or any state-like object with
                   attributes: query, plan, specialist_results, etc.).
            session_id: Optional explicit session ID (auto-generated if None).
            status: Checkpoint status (active, completed, failed, suspended).

        Returns:
            The session checkpoint ID.
        """
        if session_id is None:
            session_id = f"sess:{uuid.uuid4().hex[:12]}"

        checkpoint_id = f"ckpt:{session_id}:{int(time.time())}"
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # Extract state fields defensively
        query = getattr(state, "query", "") or ""
        plan = getattr(state, "plan", "") or ""
        node_history = list(getattr(state, "node_history", []) or [])
        current_node = str(node_history[-1]) if node_history else ""

        # Specialist results
        specialist_results: dict[str, str] = {}
        raw_results = getattr(state, "specialist_results", None)
        if isinstance(raw_results, dict):
            specialist_results = {k: str(v)[:500] for k, v in raw_results.items()}
        elif isinstance(raw_results, list):
            for i, r in enumerate(raw_results):
                specialist_results[f"result_{i}"] = str(r)[:500]

        # Token usage
        total_tokens = 0
        usage = getattr(state, "usage", None)
        if usage:
            total_tokens = getattr(usage, "total_tokens", 0) or 0

        # State data for full reconstruction
        state_data: dict[str, Any] = {}
        for attr in ("routed_domain", "routed_specialist", "active_topology"):
            val = getattr(state, attr, None)
            if val is not None:
                try:
                    state_data[attr] = str(val)
                except Exception:
                    pass  # nosec

        # Topology template ID if available
        topo_id = ""
        active_topo = getattr(state, "active_topology", None)
        if active_topo:
            topo_id = getattr(active_topo, "id", str(active_topo))

        node_data = {
            "id": checkpoint_id,
            "name": f"Checkpoint: {query[:50]}..."
            if len(query) > 50
            else f"Checkpoint: {query}",
            "type": RegistryNodeType.SESSION_CHECKPOINT.value,
            "session_id": session_id,
            "query": query[:1000],
            "plan": plan[:2000],
            "specialist_results": json.dumps(specialist_results),
            "node_history": node_history,
            "current_node": current_node,
            "total_usage_tokens": total_tokens,
            "state_data": json.dumps(state_data),
            "status": status,
            "topology_template_id": topo_id,
            "timestamp": timestamp,
        }

        if self.engine:
            if self.engine.backend:
                try:
                    self.engine._upsert_node(
                        "SessionCheckpoint", checkpoint_id, node_data
                    )
                    logger.info(
                        "[CONCEPT:ORCH-1.16] Checkpointed session '%s' (status=%s, nodes=%d)",
                        session_id,
                        status,
                        len(node_history),
                    )
                except Exception as e:
                    logger.warning("Failed to checkpoint to backend: %s", e)
                    self.engine.graph.add_node(checkpoint_id, **node_data)
            else:
                # Memory-only mode: store directly in NX graph
                self.engine.graph.add_node(checkpoint_id, **node_data)
                logger.info(
                    "[CONCEPT:ORCH-1.16] Checkpointed session '%s' to NX (memory-only)",
                    session_id,
                )
        else:
            logger.debug("No engine available — checkpoint stored in memory only")

        return checkpoint_id

    def restore(self, session_id: str) -> dict[str, Any] | None:
        """Restore execution state from the Knowledge Graph.

        Args:
            session_id: The session ID to restore.

        Returns:
            A dict with the reconstructed state fields, or None if not found.
        """
        if not self.engine:
            return None

        checkpoint = None

        # Try backend (Tier 1) first
        if self.engine.backend:
            try:
                results = self.engine.backend.execute(
                    "MATCH (c:SessionCheckpoint) "
                    "WHERE c.session_id = $sid "
                    "RETURN c "
                    "ORDER BY c.timestamp DESC LIMIT 1",
                    {"sid": session_id},
                )
                if results:
                    checkpoint = results[0]
                    if isinstance(checkpoint, dict) and "c" in checkpoint:
                        checkpoint = checkpoint["c"]
            except Exception as e:
                logger.debug("Backend checkpoint restore failed: %s", e)

        # Fallback to NX graph
        if checkpoint is None:
            for nid, data in self.engine.graph.nodes(data=True):
                if (
                    data.get("type") == RegistryNodeType.SESSION_CHECKPOINT.value
                    and data.get("session_id") == session_id
                ):
                    checkpoint = dict(data)
                    break

        if checkpoint is None:
            logger.info("No checkpoint found for session '%s'", session_id)
            return None

        # Deserialize fields
        state_dict: dict[str, Any] = {
            "session_id": checkpoint.get("session_id", session_id),
            "query": checkpoint.get("query", ""),
            "plan": checkpoint.get("plan", ""),
            "node_history": checkpoint.get("node_history", []),
            "current_node": checkpoint.get("current_node", ""),
            "total_usage_tokens": checkpoint.get("total_usage_tokens", 0),
            "status": checkpoint.get("status", "active"),
            "topology_template_id": checkpoint.get("topology_template_id", ""),
        }

        # Parse specialist results
        sr = checkpoint.get("specialist_results", "{}")
        if isinstance(sr, str):
            try:
                state_dict["specialist_results"] = json.loads(sr)
            except (json.JSONDecodeError, TypeError):
                state_dict["specialist_results"] = {}
        else:
            state_dict["specialist_results"] = sr

        # Parse state data
        sd = checkpoint.get("state_data", "{}")
        if isinstance(sd, str):
            try:
                state_dict["state_data"] = json.loads(sd)
            except (json.JSONDecodeError, TypeError):
                state_dict["state_data"] = {}
        else:
            state_dict["state_data"] = sd

        logger.info(
            "[CONCEPT:ORCH-1.16] Restored session '%s' (status=%s)",
            session_id,
            state_dict.get("status"),
        )
        return state_dict

    def list_sessions(
        self,
        status: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """List recent session checkpoints.

        Args:
            status: Optional filter by status (active, completed, etc.).
            limit: Maximum number of results.

        Returns:
            List of session checkpoint summaries.
        """
        sessions: list[dict[str, Any]] = []

        if self.engine and self.engine.backend:
            try:
                where = "WHERE c.type = 'session_checkpoint'"
                if status:
                    where += " AND c.status = $status"
                results = self.engine.backend.execute(
                    f"MATCH (c:SessionCheckpoint) {where} "
                    f"RETURN c.session_id AS sid, c.query AS query, "
                    f"c.status AS status, c.timestamp AS ts "
                    f"ORDER BY c.timestamp DESC LIMIT $limit",
                    {"status": status or "", "limit": limit},
                )
                for r in results:
                    sessions.append(
                        {
                            "session_id": r.get("sid", ""),
                            "query": (r.get("query", "") or "")[:100],
                            "status": r.get("status", ""),
                            "timestamp": r.get("ts", ""),
                        }
                    )
            except Exception as e:
                logger.debug("Session listing failed: %s", e)

        return sessions

    def mark_completed(self, session_id: str, success: bool = True) -> None:
        """Mark a session checkpoint as completed or failed.

        Args:
            session_id: The session to update.
            success: Whether the session completed successfully.
        """
        status = "completed" if success else "failed"

        if self.engine and self.engine.backend:
            try:
                self.engine.backend.execute(
                    "MATCH (c:SessionCheckpoint) "
                    "WHERE c.session_id = $sid "
                    "SET c.status = $status",
                    {"sid": session_id, "status": status},
                )
                logger.info(
                    "[CONCEPT:ORCH-1.16] Marked session '%s' as %s",
                    session_id,
                    status,
                )
            except Exception as e:
                logger.debug("Failed to mark session status: %s", e)
