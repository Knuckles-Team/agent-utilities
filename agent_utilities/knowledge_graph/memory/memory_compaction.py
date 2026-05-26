#!/usr/bin/python
"""Semantic Compaction (CONCEPT:KG-2.20).

Compacts low-level trace/episodic memory nodes into consolidated,
high-level declarative knowledge representations to prevent graph explosion.
"""

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class SemanticCompactor:
    """Semantic Compactor for Knowledge Graph trace nodes (CONCEPT:KG-2.20)."""

    def __init__(self, engine: Any, compute_engine: Any = None) -> None:
        self.engine = engine
        self._compute_engine = compute_engine

    def compact_traces(self, agent_id: str, threshold: int = 10) -> int:
        """Find trace/process nodes for a given agent and compact them.

        Traces represent highly verbose execution steps. When they exceed
        the threshold, we compress them into high-level summary edges/nodes
        and prune the verbose details.

        If a Rust-backed GraphComputeEngine was provided, uses the compiled
        ``compact_nodes_by_type`` FFI for zero round-trip compaction.
        """
        if not self.engine:
            return 0

        # Fast path: compiled Rust compaction (avoids N+2 Cypher round-trips)
        if self._compute_engine is not None:
            rust_graph = getattr(self._compute_engine, "_rust_graph", None)
            if rust_graph is not None and hasattr(rust_graph, "compact_nodes_by_type"):
                try:
                    removed = rust_graph.compact_nodes_by_type(
                        "AgentProcess", threshold
                    )
                    if removed:
                        logger.info(
                            f"SemanticCompactor (Rust): Compacted {len(removed)} "
                            f"AgentProcess nodes via compiled FFI"
                        )
                        return len(removed)
                except Exception as e:
                    logger.warning(
                        f"Rust compaction failed, falling back to Cypher: {e}"
                    )

        # Try to find all trace nodes for the agent from the database
        try:
            # We fetch all AgentProcess/Trace nodes linked to this agent
            query = (
                "MATCH (a:Agent {id: $agent_id})-[:HAS_PROCESS]->(p:AgentProcess) "
                "RETURN p.id, p.state, p.tokens_used"
            )
            res = self.engine.backend.execute(query, {"agent_id": agent_id})

            trace_nodes = []
            if res and hasattr(res, "rows"):
                trace_nodes = res.rows
            elif isinstance(res, list):
                trace_nodes = res

            if len(trace_nodes) < threshold:
                return 0

            # Aggregate stats
            total_tokens = 0
            states_summary: dict[str, int] = {}
            for row in trace_nodes:
                # row can be a dict, a list/tuple, or an object
                state = "unknown"
                tokens = 0
                pid = None
                if isinstance(row, dict):
                    pid = row.get("p.id") or row.get("id")
                    state = row.get("p.state") or row.get("state") or "unknown"
                    tokens = row.get("p.tokens_used") or row.get("tokens_used") or 0
                elif isinstance(row, list | tuple) and len(row) >= 3:
                    pid, state, tokens = row[0], row[1], row[2]

                if pid:
                    total_tokens += int(tokens or 0)
                    states_summary[state] = states_summary.get(state, 0) + 1

            summary_node_id = f"summary:agent:{agent_id}:{len(trace_nodes)}_compacted"

            # 1. Create consolidated summary node
            query_summary = (
                "MERGE (s:SemanticSummary {id: $summary_id}) "
                "SET s.name = $name, "
                "    s.compacted_count = $compacted_count, "
                "    s.total_tokens_consumed = $total_tokens, "
                "    s.agent_id = $agent_id"
            )
            self.engine.backend.execute(
                query_summary,
                {
                    "summary_id": summary_node_id,
                    "name": f"Compacted Trace Summary for Agent {agent_id}",
                    "compacted_count": len(trace_nodes),
                    "total_tokens": total_tokens,
                    "agent_id": agent_id,
                },
            )

            # 2. Link summary to Agent
            query_link = (
                "MATCH (a:Agent {id: $agent_id}) "
                "MATCH (s:SemanticSummary {id: $summary_id}) "
                "MERGE (a)-[:HAS_COMPACTED_HISTORY]->(s)"
            )
            self.engine.backend.execute(
                query_link,
                {"agent_id": agent_id, "summary_id": summary_node_id},
            )

            # 3. Delete verbose process/trace nodes to prune database
            deleted = 0
            for row in trace_nodes:
                pid = None
                if isinstance(row, dict):
                    pid = row.get("p.id") or row.get("id")
                elif isinstance(row, list | tuple):
                    pid = row[0]

                if pid:
                    query_delete = "MATCH (p:AgentProcess {id: $pid}) DETACH DELETE p"
                    self.engine.backend.execute(query_delete, {"pid": pid})
                    deleted += 1

            logger.info(
                f"SemanticCompactor: Compacted {deleted} traces for agent '{agent_id}' "
                f"into summary node '{summary_node_id}'"
            )
            return deleted

        except Exception as e:
            logger.error(f"SemanticCompactor failed during compaction: {e}")
            return 0
