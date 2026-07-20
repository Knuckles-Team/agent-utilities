"""CONCEPT:AU-KG.query.object-graph-mapper"""

import logging
import re
from typing import Any

from ...backends import create_backend
from ..types import (
    PhaseResult,
    PipelineContext,
    PipelinePhase,
)

logger = logging.getLogger(__name__)
_CYPHER_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")


def _safe_graph_identifier(value: object, *, default: str = "") -> str:
    rendered = re.sub(r"\W+", "_", str(value or default)).strip("_")
    return rendered if _CYPHER_IDENTIFIER.fullmatch(rendered) else default


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
    "database_table": "DatabaseTable",
    "database_column": "DatabaseColumn",
    "database_view": "DatabaseView",
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
    "relationship": "Relationship",
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
    graph = ctx.graph

    authority_backend = getattr(db, "_authority", db)
    if authority_backend.__class__.__name__ == "EpistemicGraphBackend":
        # The operational graph authority accepts one governed native graph
        # slice. Do not compile the in-memory graph into client-side Cypher or
        # emulate UNWIND batches in Python.
        from ...core.materialization import write_entities

        entities: list[dict[str, Any]] = []
        for node_id, data in graph.nodes(data=True):
            raw_type = str(data.get("node_type", "")).strip()
            if not raw_type:
                continue
            props = {key: value for key, value in data.items() if value is not None}
            props.update({"id": str(node_id), "node_type": raw_type})
            if "ingestion_timestamp" in ctx.metadata:
                props["last_seen_timestamp"] = ctx.metadata["ingestion_timestamp"]
            entities.append(props)

        relationships = [
            {
                **{
                    key: value
                    for key, value in data.items()
                    if key not in {"type", "rel_type", "relationship_type", "relation"}
                    and value is not None
                },
                "source": str(source),
                "target": str(target),
                "relationship": str(data.get("relationship") or "RELATED"),
            }
            for source, target, data in graph.edges(data=True)
        ]
        return write_entities(
            db,
            "code-graph",
            entities,
            relationships,
            delta=True,
        )

    nodes_synced = 0
    edges_synced = 0

    # Sync Nodes
    nodes_by_group: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]] = {}
    for node_id, data in graph.nodes(data=True):
        raw_type = str(data.get("node_type", "")).lower()
        label = _TYPE_TO_TABLE.get(raw_type) or "".join(
            word.capitalize() for word in raw_type.replace("_", " ").split()
        )
        label = _safe_graph_identifier(label)
        if not label:
            continue

        props = {k: v for k, v in data.items() if v is not None}
        if "ingestion_timestamp" in ctx.metadata:
            props["last_seen_timestamp"] = ctx.metadata["ingestion_timestamp"]

        # JSON serialize dict/list properties for database compatibility
        import json

        valid_keys = None
        if db.__class__.__name__ == "LadybugBackend":
            from agent_utilities.models.schema_definition import SCHEMA

            for node in SCHEMA.nodes:
                if node.name == label:
                    valid_keys = set(node.columns.keys())
                    break

        if valid_keys is not None and "metadata" in valid_keys:
            extra_props = {}
            for k in list(props.keys()):
                if k != "id" and k not in valid_keys:
                    extra_props[k] = props.pop(k)
            if extra_props:
                curr_meta = props.get("metadata", {})
                if isinstance(curr_meta, str):
                    try:
                        curr_meta = json.loads(curr_meta)
                    except Exception:
                        curr_meta = {}
                if not isinstance(curr_meta, dict):
                    curr_meta = {}
                curr_meta.update(extra_props)
                props["metadata"] = curr_meta

        for k, v in list(props.items()):
            if isinstance(v, dict | list):
                props[k] = json.dumps(v)

        keys = sorted(
            [
                k
                for k in props.keys()
                if k != "id"
                and isinstance(k, str)
                and _CYPHER_IDENTIFIER.fullmatch(k)
                and (valid_keys is None or k in valid_keys)
            ]
        )
        group_key = (label, tuple(keys))
        if group_key not in nodes_by_group:
            nodes_by_group[group_key] = []

        params = {"id": node_id}
        for k in keys:
            params[f"props_{k}"] = props[k]
        nodes_by_group[group_key].append(params)

    for (label, group_keys), batch in nodes_by_group.items():
        set_clause = (
            " SET " + ", ".join([f"n.{k} = $props_{k}" for k in group_keys])
            if group_keys
            else ""
        )
        query = f"MERGE (n:{label} {{id: $id}}){set_clause}"
        batch_size = getattr(ctx.config, "ingest_batch_size", 500)
        for i in range(0, len(batch), batch_size):
            chunk = batch[i : i + batch_size]
            try:
                db.execute_batch(query, chunk)
                nodes_synced += len(chunk)
            except Exception as exc:
                logger.error(
                    "Failed to sync node chunk: error_type=%s", type(exc).__name__
                )

    # Sync Edges
    edges_by_type: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for u, v, data in graph.edges(data=True):
        etype = _safe_graph_identifier(str(data.get("relationship", "rel")).upper())
        if not etype:
            continue

        u_type = str(graph.nodes[u].get("node_type", "")).lower()
        v_type = str(graph.nodes[v].get("node_type", "")).lower()
        u_label = _TYPE_TO_TABLE.get(u_type) or "".join(
            word.capitalize() for word in u_type.replace("_", " ").split()
        )
        v_label = _TYPE_TO_TABLE.get(v_type) or "".join(
            word.capitalize() for word in v_type.replace("_", " ").split()
        )
        u_label = _safe_graph_identifier(u_label, default="Code")
        v_label = _safe_graph_identifier(v_label, default="Code")

        u_label_str = f":{u_label}" if u_label else ":Code"
        v_label_str = f":{v_label}" if v_label else ":Code"

        edge_key = (etype, u_label_str, v_label_str)
        if edge_key not in edges_by_type:
            edges_by_type[edge_key] = []
        edges_by_type[edge_key].append({"uid": u, "vid": v})

    for (etype, u_label_str, v_label_str), batch in edges_by_type.items():
        query = f"MATCH (a{u_label_str} {{id: $uid}}), (b{v_label_str} {{id: $vid}}) MERGE (a)-[r:{etype}]->(b)"
        batch_size = getattr(ctx.config, "ingest_batch_size", 500)
        for i in range(0, len(batch), batch_size):
            chunk = batch[i : i + batch_size]
            try:
                db.execute_batch(query, chunk)
                edges_synced += len(chunk)
            except Exception as exc:
                logger.error(
                    "Failed to sync edge chunk: error_type=%s", type(exc).__name__
                )

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
        except Exception as exc:
            logger.debug(
                "Failed to sweep stale nodes: error_type=%s", type(exc).__name__
            )

    return {"nodes_synced": nodes_synced, "edges_synced": edges_synced}


sync_phase = PipelinePhase(
    name="sync",
    # shacl_gate runs the SHACL ingestion gate (quarantine) before commit.
    deps=["centrality", "embedding", "registry", "shacl_gate"],
    execute_fn=execute_sync,
)
