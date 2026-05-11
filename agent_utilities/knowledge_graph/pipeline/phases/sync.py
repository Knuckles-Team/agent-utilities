"""CONCEPT:KG-2.0"""

import logging
from typing import Any

from ...backends import create_backend
from ..types import (
    PhaseResult,
    PipelineContext,
    PipelinePhase,
)

logger = logging.getLogger(__name__)

# Mapping from RegistryNodeType enum values to DDL table names
# This ensures sync uses the exact table names defined in schema_definition.py
_TYPE_TO_TABLE = {
    "agent": "Agent",
    "tool": "Tool",
    "skill": "Skill",
    "prompt": "Prompt",
    "memory": "Memory",
    "file": "Code",
    "symbol": "Code",
    "module": "Code",
    "client": "Client",
    "user": "User",
    "preference": "Preference",
    "job": "Job",
    "log": "Log",
    "message": "Message",
    "chat_summary": "ChatSummary",
    "thread": "Thread",
    "heartbeat": "Heartbeat",
    "reasoning_trace": "ReasoningTrace",
    "tool_call": "ToolCall",
    "entity": "Entity",
    "event": "Event",
    "reflection": "Reflection",
    "goal": "Goal",
    "episode": "Episode",
    "fact": "Fact",
    "concept": "Concept",
    "capability": "Capability",
    "callable_resource": "CallableResource",
    "tool_metadata": "ToolMetadata",
    "spawned_agent": "SpawnedAgent",
    "system_prompt": "SystemPrompt",
    "outcome_evaluation": "OutcomeEvaluation",
    "critique": "Critique",
    "self_evaluation": "SelfEvaluation",
    "experiment": "Experiment",
    "proposed_skill": "ProposedSkill",
    "server": "Server",
    "observation": "Observation",
    "action": "Action",
}


async def execute_sync(
    ctx: PipelineContext, deps: dict[str, PhaseResult]
) -> dict[str, Any]:
    """Phase 12: Persist to the configured graph backend."""

    if not ctx.config.persist_to_ladybug:
        return {"status": "skipped", "reason": "persistence disabled"}

    # Use the shared backend from context, or create one via factory
    db = ctx.backend
    if db is None:
        db_path = ctx.config.ladybug_path or "knowledge_graph.db"
        db = create_backend(db_path=db_path)
    if db is None:
        return {"status": "skipped", "reason": "graph backend not available"}
    graph = ctx.nx_graph

    nodes_synced = 0
    edges_synced = 0

    # Sync Nodes
    for node_id, data in graph.nodes(data=True):
        raw_type = str(data.get("type", "")).lower()
        # Use the mapping to get the correct DDL table name
        label = _TYPE_TO_TABLE.get(raw_type)
        if not label:
            # Fallback: capitalize first letter of each word
            label = "".join(
                word.capitalize() for word in raw_type.replace("_", " ").split()
            )
        if not label:
            continue

        props = {k: v for k, v in data.items() if v is not None}
        if "ingestion_timestamp" in ctx.metadata:
            props["last_seen_timestamp"] = ctx.metadata["ingestion_timestamp"]

        try:
            valid_keys = None
            is_ladybug = db.__class__.__name__ == "LadybugBackend"
            if is_ladybug:
                from agent_utilities.models.schema_definition import SCHEMA

                for node in SCHEMA.nodes:
                    if node.name == label:
                        valid_keys = set(node.columns.keys())
                        break

            keys = [k for k in props.keys() if k != "id"]
            if valid_keys is not None:
                keys = [k for k in keys if k in valid_keys]

            set_clause = (
                " SET " + ", ".join([f"n.{k} = $props_{k}" for k in keys])
                if keys
                else ""
            )
            params: dict[str, Any] = {"id": node_id}
            for k in keys:
                params[f"props_{k}"] = props[k]

            db.execute(
                f"MERGE (n:{label} {{id: $id}}){set_clause}",
                params,
            )
            nodes_synced += 1
        except Exception as e:
            logger.debug(f"Failed to sync node {node_id}: {e}")

    # Sync Edges
    for u, v, data in graph.edges(data=True):
        etype = str(data.get("type", "rel")).upper()
        etype = "".join(c for c in etype if c.isalnum() or c == "_")
        if not etype:
            continue
        try:
            db.execute(
                f"MATCH (a {{id: $uid}}), (b {{id: $vid}}) MERGE (a)-[r:{etype}]->(b)",
                {"uid": u, "vid": v},
            )
            edges_synced += 1
        except Exception as e:
            logger.debug(f"Failed to sync edge {u}->{v}: {e}")

    # Sweep stale codebase nodes
    if "ingestion_timestamp" in ctx.metadata:
        ts = ctx.metadata["ingestion_timestamp"]
        workspace_path = ctx.config.workspace_path
        try:
            db.execute(
                "MATCH (n:Code) WHERE n.file_path STARTS WITH $workspace_path AND (n.last_seen_timestamp < $ts OR n.last_seen_timestamp IS NULL) DETACH DELETE n",
                {"workspace_path": workspace_path, "ts": ts},
            )
            logger.info("Sweep complete: deleted stale codebase nodes.")
        except Exception as e:
            logger.debug(f"Failed to sweep stale nodes: {e}")

    return {"nodes_synced": nodes_synced, "edges_synced": edges_synced}


sync_phase = PipelinePhase(
    name="sync", deps=["centrality", "embedding", "registry"], execute_fn=execute_sync
)
