"""CONCEPT:AU-KG.query.object-graph-mapper"""

import logging
import re
from typing import Any

from agent_utilities.security.identifiers import CYPHER_IDENTIFIER_RE

from ...backends import create_backend
from ..types import (
    PhaseResult,
    PipelineContext,
    PipelinePhase,
)

logger = logging.getLogger(__name__)


def _safe_graph_identifier(value: object, *, default: str = "") -> str:
    """Sanitize then validate against the shared Cypher identifier grammar.

    Coerces to a safe identifier or falls back to ``default`` rather than
    raising — this phase intentionally skips a single malformed node/edge
    instead of aborting the whole sync (CONCEPT:AU-KG.query.object-graph-mapper).
    """
    rendered = re.sub(r"\W+", "_", str(value or default)).strip("_")
    return rendered if CYPHER_IDENTIFIER_RE.fullmatch(rendered) else default


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


def _resolve_sync_backend(ctx: PipelineContext) -> Any:
    """Use the shared backend from context, or create one via factory.

    Extracted verbatim from ``execute_sync`` (pure extract-method, no
    behaviour change).
    """
    db = ctx.backend
    if db is None:
        db_path = ctx.config.ladybug_path or "knowledge_graph.db"
        db = create_backend(db_path=db_path)
    return db


def _prepare_one_epistemic_entity(
    ctx: PipelineContext, node_id: Any, data: dict
) -> "dict[str, Any] | None":
    """Build one entity dict for the epistemic-backend fast path, or ``None`` to skip.

    Extracted verbatim from ``_sync_via_epistemic_backend``'s entity-building loop.
    """
    raw_type = str(data.get("node_type", "")).strip()
    if not raw_type:
        return None
    props = {key: value for key, value in data.items() if value is not None}
    props.update({"id": str(node_id), "node_type": raw_type})
    if "ingestion_timestamp" in ctx.metadata:
        props["last_seen_timestamp"] = ctx.metadata["ingestion_timestamp"]
    return props


def _build_epistemic_entities(ctx: PipelineContext, graph: Any) -> list[dict[str, Any]]:
    """Extracted verbatim: the entity-building loop of ``_sync_via_epistemic_backend``."""
    entities: list[dict[str, Any]] = []
    for node_id, data in graph.nodes(data=True):
        entity = _prepare_one_epistemic_entity(ctx, node_id, data)
        if entity is not None:
            entities.append(entity)
    return entities


def _build_epistemic_relationship(
    source: Any, target: Any, data: dict
) -> dict[str, Any]:
    """Build one relationship dict for the epistemic-backend fast path.

    Extracted verbatim from ``_sync_via_epistemic_backend``'s relationship
    list comprehension.
    """
    return {
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


def _sync_via_epistemic_backend(
    ctx: PipelineContext, db: Any, graph: Any
) -> dict[str, Any]:
    """Native ``EpistemicGraphBackend`` fast path.

    Extracted verbatim from ``execute_sync`` (pure extract-method, no
    behaviour change). The operational graph authority accepts one governed
    native graph slice -- do not compile the in-memory graph into
    client-side Cypher or emulate UNWIND batches in Python.
    """
    from ...core.materialization import write_entities

    entities = _build_epistemic_entities(ctx, graph)
    relationships = [
        _build_epistemic_relationship(source, target, data)
        for source, target, data in graph.edges(data=True)
    ]
    return write_entities(
        db,
        "code-graph",
        entities,
        relationships,
        delta=True,
    )


def _resolve_sync_node_label(raw_type: str) -> str:
    """Extracted verbatim from ``execute_sync``'s node-grouping loop."""
    label = _TYPE_TO_TABLE.get(raw_type) or "".join(
        word.capitalize() for word in raw_type.replace("_", " ").split()
    )
    return _safe_graph_identifier(label)


def _resolve_sync_node_schema_keys(db: Any, label: str) -> "set[str] | None":
    """LadybugBackend schema-column lookup for one node label.

    Extracted verbatim from ``execute_sync``'s node-grouping loop.
    """
    if db.__class__.__name__ != "LadybugBackend":
        return None
    from agent_utilities.models.schema_definition import SCHEMA

    for node in SCHEMA.nodes:
        if node.name == label:
            return set(node.columns.keys())
    return None


def _fold_sync_node_metadata(props: dict, valid_keys: "set[str] | None") -> None:
    """Fold unrecognized keys into ``props["metadata"]``.

    Extracted verbatim from ``execute_sync``'s node-grouping loop. Mutates
    ``props`` in place.
    """
    if valid_keys is None or "metadata" not in valid_keys:
        return

    import json

    extra_props = {}
    for k in list(props.keys()):
        if k != "id" and k not in valid_keys:
            extra_props[k] = props.pop(k)
    if not extra_props:
        return

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


def _serialize_sync_node_props(props: dict) -> None:
    """JSON-serialize dict/list property values in place.

    Extracted verbatim from ``execute_sync``'s node-grouping loop.
    """
    import json

    for k, v in list(props.items()):
        if isinstance(v, dict | list):
            props[k] = json.dumps(v)


def _sync_node_batch_keys(props: dict, valid_keys: "set[str] | None") -> list[str]:
    """Sorted list of safe, schema-valid property keys for one node.

    Extracted verbatim from ``execute_sync``'s node-grouping loop.
    """
    return sorted(
        [
            k
            for k in props.keys()
            if k != "id"
            and isinstance(k, str)
            and CYPHER_IDENTIFIER_RE.fullmatch(k)
            and (valid_keys is None or k in valid_keys)
        ]
    )


def _prepare_one_sync_node(
    ctx: PipelineContext, db: Any, node_id: Any, data: dict
) -> "tuple[str, list[str], dict] | None":
    """Compute ``(label, sorted_keys, params)`` for one node, or ``None`` to skip.

    Extracted verbatim from ``execute_sync``'s node-grouping loop body.
    """
    raw_type = str(data.get("node_type", "")).lower()
    label = _resolve_sync_node_label(raw_type)
    if not label:
        return None

    props = {k: v for k, v in data.items() if v is not None}
    if "ingestion_timestamp" in ctx.metadata:
        props["last_seen_timestamp"] = ctx.metadata["ingestion_timestamp"]

    valid_keys = _resolve_sync_node_schema_keys(db, label)

    _fold_sync_node_metadata(props, valid_keys)
    _serialize_sync_node_props(props)

    keys = _sync_node_batch_keys(props, valid_keys)
    params = {"id": node_id}
    for k in keys:
        params[f"props_{k}"] = props[k]
    return label, keys, params


def _group_sync_nodes(
    ctx: PipelineContext, db: Any, graph: Any
) -> dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]]:
    """Extracted verbatim: the node-grouping loop of ``execute_sync``."""
    nodes_by_group: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]] = {}
    for node_id, data in graph.nodes(data=True):
        prepared = _prepare_one_sync_node(ctx, db, node_id, data)
        if prepared is None:
            continue
        label, keys, params = prepared
        group_key = (label, tuple(keys))
        if group_key not in nodes_by_group:
            nodes_by_group[group_key] = []
        nodes_by_group[group_key].append(params)
    return nodes_by_group


def _write_synced_node_batches(
    ctx: PipelineContext,
    db: Any,
    nodes_by_group: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]],
) -> int:
    """Extracted verbatim: the node MERGE batch-write loop of ``execute_sync``."""
    nodes_synced = 0
    for (label, group_keys), batch in nodes_by_group.items():
        # Re-validated HERE, at the point of interpolation. Extracting this
        # writer out of ``execute_sync`` separated the interpolation from
        # ``_prepare_one_sync_node``'s guard, and a guard in another function
        # protects nothing the day a second caller appears.
        # ``_safe_graph_identifier`` is idempotent on an already-valid
        # identifier, so this is one regex per BATCH (not per row) and fails
        # closed to ``Code``.
        label = _safe_graph_identifier(label, default="Code")
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
    return nodes_synced


def _resolve_sync_edge_label(node_type: str) -> str:
    """Node-type -> Cypher label resolution shared by an edge's endpoints.

    Extracted verbatim from ``execute_sync``'s edge-grouping loop.
    """
    label = _TYPE_TO_TABLE.get(node_type) or "".join(
        word.capitalize() for word in node_type.replace("_", " ").split()
    )
    return _safe_graph_identifier(label, default="Code")


def _prepare_one_sync_edge(
    graph: Any, u: Any, v: Any, data: dict
) -> "tuple[tuple[str, str, str], dict] | None":
    """Compute ``(edge_key, entry)`` for one edge, or ``None`` to skip.

    Extracted verbatim from ``execute_sync``'s edge-grouping loop body.
    """
    etype = _safe_graph_identifier(str(data.get("relationship", "rel")).upper())
    if not etype:
        return None

    u_type = str(graph.nodes[u].get("node_type", "")).lower()
    v_type = str(graph.nodes[v].get("node_type", "")).lower()
    u_label = _resolve_sync_edge_label(u_type)
    v_label = _resolve_sync_edge_label(v_type)

    # Bare labels, no leading ``:``. The writer re-validates and adds the
    # colon itself, so the value that crosses this boundary is an identifier
    # rather than a Cypher fragment that only LOOKS like one.
    return (etype, u_label or "Code", v_label or "Code"), {"uid": u, "vid": v}


def _group_sync_edges(graph: Any) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    """Extracted verbatim: the edge-grouping loop of ``execute_sync``."""
    edges_by_type: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for u, v, data in graph.edges(data=True):
        prepared = _prepare_one_sync_edge(graph, u, v, data)
        if prepared is None:
            continue
        edge_key, entry = prepared
        if edge_key not in edges_by_type:
            edges_by_type[edge_key] = []
        edges_by_type[edge_key].append(entry)
    return edges_by_type


def _write_synced_edge_batches(
    ctx: PipelineContext,
    db: Any,
    edges_by_type: dict[tuple[str, str, str], list[dict[str, Any]]],
) -> int:
    """Extracted verbatim: the edge MERGE batch-write loop of ``execute_sync``."""
    edges_synced = 0
    for (etype, u_label, v_label), batch in edges_by_type.items():
        # Re-validated HERE, at the point of interpolation, for the same reason
        # as ``_write_synced_node_batches``: extracting this writer out of
        # ``execute_sync`` separated the f-string from
        # ``_prepare_one_sync_edge``'s guard, and a guard in another function
        # protects nothing the day a second caller builds ``edges_by_type``
        # some other way. All three interpolated identifiers are re-checked --
        # the relationship type as well as both end labels.
        # ``_safe_graph_identifier`` is idempotent on an already-valid
        # identifier, so for every value the current preparer emits these
        # return their input unchanged, at one regex per BATCH, not per row.
        # An identifier that does NOT survive validation yields an unusable
        # (never an injected) query, which the batch loop below already logs.
        etype = _safe_graph_identifier(etype)
        u_label = _safe_graph_identifier(u_label, default="Code")
        v_label = _safe_graph_identifier(v_label, default="Code")
        query = f"MATCH (a:{u_label} {{id: $uid}}), (b:{v_label} {{id: $vid}}) MERGE (a)-[r:{etype}]->(b)"
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
    return edges_synced


def _sweep_stale_codebase_nodes(ctx: PipelineContext, db: Any) -> None:
    """Extracted verbatim: the stale-codebase-node sweep of ``execute_sync``."""
    if "ingestion_timestamp" not in ctx.metadata:
        return
    ts = ctx.metadata["ingestion_timestamp"]
    workspace_path = ctx.config.workspace_path
    try:
        db.execute(
            "MATCH (n:Code) WHERE n.file_path STARTS WITH $workspace_path AND (n.last_seen_timestamp < $ts OR n.last_seen_timestamp IS NULL) DETACH DELETE n",
            {"workspace_path": workspace_path, "ts": ts},
        )
        logger.info("Sweep complete: deleted stale codebase nodes.")
    except Exception as exc:
        logger.debug("Failed to sweep stale nodes: error_type=%s", type(exc).__name__)


async def execute_sync(
    ctx: PipelineContext, deps: dict[str, PhaseResult]
) -> dict[str, Any]:
    """Phase 12: Persist to the configured graph backend."""

    if not ctx.config.persist_to_ladybug:
        return {"status": "skipped", "reason": "persistence disabled"}

    db = _resolve_sync_backend(ctx)
    if db is None:
        return {"status": "skipped", "reason": "graph backend not available"}
    graph = ctx.graph

    authority_backend = getattr(db, "_authority", db)
    if authority_backend.__class__.__name__ == "EpistemicGraphBackend":
        return _sync_via_epistemic_backend(ctx, db, graph)

    nodes_by_group = _group_sync_nodes(ctx, db, graph)
    nodes_synced = _write_synced_node_batches(ctx, db, nodes_by_group)

    edges_by_type = _group_sync_edges(graph)
    edges_synced = _write_synced_edge_batches(ctx, db, edges_by_type)

    _sweep_stale_codebase_nodes(ctx, db)

    return {"nodes_synced": nodes_synced, "edges_synced": edges_synced}


sync_phase = PipelinePhase(
    name="sync",
    # shacl_gate runs the SHACL ingestion gate (quarantine) before commit.
    deps=["centrality", "embedding", "registry", "shacl_gate"],
    execute_fn=execute_sync,
)
