#!/usr/bin/python
"""Ontological State Checkpointing (CONCEPT:KG-2.53).

Persist Pydantic Graph active states as ExecutionStateNodes within the Knowledge Graph.
Deprecates local execution states, enabling background agents (AHE-3.17) to seamlessly
pick up, monitor, and resume paused executions across a distributed environment.
"""

import json
import logging
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from .engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)


class ExecutionStateNode(BaseModel):
    """Pydantic model representing a paused or active execution state."""

    node_id: str
    run_id: str
    agent_id: str
    state_data: str  # JSON serialized state
    status: str
    timestamp: str


class StateCheckpointer:
    """Manages the persistence of execution states to the KG.

    CONCEPT:KG-2.53 — Ontological State Checkpointing
    """

    def __init__(self, engine: IntelligenceGraphEngine):
        self.engine = engine

    def save_state(
        self,
        run_id: str,
        agent_id: str,
        current_node: str,
        state_data: dict[str, Any],
        status: str = "paused",
    ) -> str:
        """Save an execution state into the Knowledge Graph."""
        node_id = f"state:{run_id}:{current_node}"

        state_node = ExecutionStateNode(
            node_id=node_id,
            run_id=run_id,
            agent_id=agent_id,
            state_data=json.dumps(state_data),
            status=status,
            timestamp=datetime.utcnow().isoformat(),
        )

        if self.engine.backend:
            # Upsert the state node
            data = state_node.model_dump()
            self.engine.backend.execute(
                """
                MERGE (s:ExecutionStateNode {node_id: $node_id})
                SET s += $props
                """,
                {"node_id": node_id, "props": data},
            )

            # Link it to the agent
            self.engine.backend.execute(
                """
                MATCH (a {id: $agent_id}), (s:ExecutionStateNode {node_id: $node_id})
                MERGE (a)-[:HAS_STATE]->(s)
                """,
                {"agent_id": agent_id, "node_id": node_id},
            )
            logger.debug(f"[KG-2.53] Saved state {node_id} for agent {agent_id}")
        else:
            self.engine.graph.add_node(node_id, **state_node.model_dump())
            self.engine.graph.add_edge(agent_id, node_id, type="HAS_STATE")

        return node_id

    def load_state(self, run_id: str) -> dict[str, Any] | None:
        """Retrieve the latest execution state for a run."""
        if self.engine.backend:
            res = self.engine.backend.execute(
                """
                MATCH (s:ExecutionStateNode {run_id: $run_id})
                RETURN s
                ORDER BY s.timestamp DESC LIMIT 1
                """,
                {"run_id": run_id},
            )
            if res:
                s_data = res[0].get("s", {})
                if "state_data" in s_data:
                    return json.loads(s_data["state_data"])
        else:
            # Fallback to NX
            states = []
            for n, d in self.engine.graph.nodes(data=True):
                if d.get("run_id") == run_id and d.get("status") in (
                    "paused",
                    "active",
                ):
                    states.append(d)
            if states:
                states.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
                latest = states[0]
                if "state_data" in latest:
                    return json.loads(latest["state_data"])
        return None
