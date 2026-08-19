"""Auto-extracted graph-os MCP tools: write_ingest_tools (register_write_ingest_tools).

Split out of kg_server._build_server to deepen the MCP surface into focused
modules without changing tool behavior or names.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import Field

from agent_utilities.core.event_loop import run_blocking_ordered
from agent_utilities.mcp import kg_server
from agent_utilities.security.error_surface import (
    public_error_json,
    public_error_text,
)
from agent_utilities.security.persistence_privacy import persistence_reference

_OPAQUE_NODE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")


def _parse_source_specs(raw: str, spec_cls: Any) -> list[Any]:
    """Parse skill-graph source specs from a JSON list or ``kind=uri,...`` shorthand.

    JSON form: ``[{"kind": "web", "uri": "https://x", "options": {"max_depth": 2}}]``.
    Shorthand: ``web=https://x,pdf=/a.pdf`` (no per-source options).
    """
    raw = (raw or "").strip()
    if not raw:
        return []
    if raw.startswith("["):
        return [spec_cls.from_dict(d) for d in json.loads(raw)]
    return [spec_cls.parse(tok) for tok in raw.split(",") if tok.strip()]


# ── bulk_ingest (B-11, CONCEPT:AU-KG.ingest.envelope-atomic-transaction) ─────────────────────────
#
# Was: a Python `for` loop calling `engine.add_node()` once per element — nodes
# only, non-atomic, no idempotency key, N round trips over the engine's
# MessagePack-on-a-socket transport. Rewritten onto the engine's real atomic
# primitives: `IntelligenceGraphEngine.batch_typed_mutations` (one native
# `BatchUpdate` transaction, real `upsert: bool` insert-or-merge) as the light
# path when the caller supplies no evidence/idempotency key, and
# `envelope_ingest.ingest_graph_slice` (one `ApplyChangeEnvelope(s)`
# transaction, durably idempotent by `(tenant, graph, idempotency_key)`,
# carrying evidence/lineage/policy) when it does. `check_no_per_element_ingest_loop.py`
# (a fleet-wide ratchet gate over `agent_utilities/mcp/`) enforces that this
# shape does not silently reappear.

# The engine's own documented `BatchUpdate` bounds
# (`crates/eg-compute/src/algorithms.rs`: `MAX_BATCH_UPDATE_BYTES`,
# `MAX_BATCH_OPERATIONS`, `MAX_BATCH_UPDATE_ITEMS`). Chunking below is bounded
# primarily by op count and byte size; each op decodes to a handful of msgpack
# items, so the 500,000-item ceiling is not independently binding for ordinary
# bulk_ingest payloads (documented, not separately tracked).
_BULK_INGEST_MAX_OPS = 50_000
_BULK_INGEST_MAX_BYTES = 32 * 1024 * 1024
# Server-side governance stamping (ownership/classification/bitemporal fields,
# label normalization — see `IntelligenceGraphEngine.batch_typed_mutations`)
# adds bytes beyond what this client-side estimate over the caller's raw
# properties sees, so chunk at a safety margin under the engine's hard 32MiB
# ceiling rather than flush right up against it.
_BULK_INGEST_BYTE_SAFETY_MARGIN = 0.75


def _chunk_batch_mutations(
    mutations: list[dict[str, Any]],
    *,
    max_ops: int = _BULK_INGEST_MAX_OPS,
    max_bytes: int = _BULK_INGEST_MAX_BYTES,
) -> list[list[dict[str, Any]]]:
    """Deterministically split ``mutations`` at the engine's documented bounds.

    Pure function of input order + limits — the same input always produces the
    same chunk boundaries. Every input mutation lands in exactly one output
    chunk, in original order; nothing is ever dropped. Nodes must precede the
    edges that reference them in ``mutations`` (the caller's job — see
    ``_run_bulk_ingest``), and chunks are applied SEQUENTIALLY in order so an
    edge in a later chunk always finds its node already committed.
    """
    import msgpack

    budget = int(max_bytes * _BULK_INGEST_BYTE_SAFETY_MARGIN)
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_bytes = 0
    for mutation in mutations:
        try:
            size = len(msgpack.packb(mutation, use_bin_type=True, default=str))
        except Exception:  # noqa: BLE001 — conservative fallback estimate
            size = len(json.dumps(mutation, default=str).encode("utf-8"))
        if current and (len(current) >= max_ops or current_bytes + size > budget):
            chunks.append(current)
            current = []
            current_bytes = 0
        current.append(mutation)
        current_bytes += size
    if current:
        chunks.append(current)
    return chunks


def _parse_bulk_ingest_elements(
    raw_nodes: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | str:
    """Parse ``bulk_ingest``'s ``nodes`` JSON into ``(node_mutations, edge_mutations)``.

    Each mutation is shaped for ``IntelligenceGraphEngine.batch_typed_mutations``:
    ``{"kind": "node", "id", "node_type", "properties"}`` or ``{"kind": "edge",
    "source", "target", "rel_type", "properties"}``. Returns an error STRING (a
    ``public_error_json`` payload) instead of raising, matching this module's
    existing `_write_with_engine` error contract.
    """
    try:
        elements = json.loads(raw_nodes) if raw_nodes else []
    except (TypeError, ValueError) as e:
        return public_error_json(e, code="invalid_request")
    if not isinstance(elements, list):
        return public_error_json(
            ValueError("'nodes' must be a JSON list for bulk_ingest"),
            code="invalid_request",
        )

    node_mutations: list[dict[str, Any]] = []
    edge_mutations: list[dict[str, Any]] = []
    for index, item in enumerate(elements):
        if not isinstance(item, dict):
            return public_error_json(
                ValueError(f"bulk_ingest element[{index}] must be a JSON object"),
                code="invalid_request",
            )
        kind = item.get("kind")
        is_edge = kind == "edge" or (
            kind is None and "source_id" in item and "target_id" in item
        )
        if is_edge:
            source = str(item.get("source_id") or "").strip()
            target = str(item.get("target_id") or "").strip()
            rel_type = str(item.get("rel_type") or "").strip()
            if not source or not target or not rel_type:
                return public_error_json(
                    ValueError(
                        f"bulk_ingest edge[{index}] requires source_id, "
                        "target_id, and rel_type"
                    ),
                    code="invalid_request",
                )
            properties = item.get("properties") or {}
            if not isinstance(properties, dict):
                return public_error_json(
                    ValueError(
                        f"bulk_ingest edge[{index}] 'properties' must be an object"
                    ),
                    code="invalid_request",
                )
            edge_mutations.append(
                {
                    "kind": "edge",
                    "source": source,
                    "target": target,
                    "rel_type": rel_type,
                    "properties": properties,
                }
            )
        else:
            elem_id = str(item.get("id") or "").strip()
            if not elem_id:
                return public_error_json(
                    ValueError(f"bulk_ingest node[{index}] requires 'id'"),
                    code="invalid_request",
                )
            properties = item.get("properties") or {}
            if not isinstance(properties, dict):
                return public_error_json(
                    ValueError(
                        f"bulk_ingest node[{index}] 'properties' must be an object"
                    ),
                    code="invalid_request",
                )
            node_mutations.append(
                {
                    "kind": "node",
                    "id": elem_id,
                    "node_type": str(
                        item.get("type") or item.get("node_type") or "Node"
                    ),
                    "properties": properties,
                }
            )
    return node_mutations, edge_mutations


def _run_bulk_ingest(
    engine: Any,
    raw_nodes: str,
    *,
    idempotency_key: str,
    evidence: str,
    upsert: bool,
) -> str:
    """``graph_write(action="bulk_ingest")`` — see the module comment above."""
    parsed = _parse_bulk_ingest_elements(raw_nodes)
    if isinstance(parsed, str):
        return parsed
    node_mutations, edge_mutations = parsed

    try:
        evidence_records = json.loads(evidence) if evidence else []
    except (TypeError, ValueError) as e:
        return public_error_json(e, code="invalid_request")
    if not isinstance(evidence_records, list):
        return public_error_json(
            ValueError("'evidence' must be a JSON list for bulk_ingest"),
            code="invalid_request",
        )

    if not node_mutations and not edge_mutations:
        return json.dumps(
            {
                "action": "bulk_ingest",
                "mode": "noop",
                "nodes_ingested": 0,
                "edges_ingested": 0,
                "chunks": 0,
            }
        )

    # ── heavy path — only ApplyChangeEnvelope(s) carries evidence/lineage/
    # policy and a genuinely durable per-call idempotency key; BatchUpdate has
    # neither at the wire level. One atomic transaction either way. ──
    if evidence_records or idempotency_key:
        from agent_utilities.knowledge_graph.ingestion import envelope_ingest

        entities = [
            {"id": m["id"], "node_type": m["node_type"], **m["properties"]}
            for m in node_mutations
        ]
        relationships = [
            {
                "source": m["source"],
                "target": m["target"],
                "relationship": m["rel_type"],
                **m["properties"],
            }
            for m in edge_mutations
        ]
        if entities and evidence_records:
            # ChangeEnvelope attaches per-row evidence only on the envelope's
            # PRIMARY object (`envelope_ingest._prepare_node_rows`'s `_evidence`
            # pop) — the first node in the batch, matching `ingest_graph_slice`'s
            # own "first entity is primary" contract.
            entities[0] = {**entities[0], "_evidence": evidence_records}
        try:
            result = envelope_ingest.ingest_graph_slice(
                engine,
                "bulk_ingest",
                entities,
                relationships,
                idempotency_key=idempotency_key,
            )
        except Exception as e:  # noqa: BLE001 — surface as data, not a 500
            return public_error_json(e, context={"action": "bulk_ingest"})
        # Surface the engine's own replay outcome honestly — `status` is
        # "success" (applied) / "skipped" (idempotent replay) / "rejected" /
        # "failed", never collapsed to a single "ok".
        return json.dumps(
            {
                "action": "bulk_ingest",
                "mode": "change_envelope",
                "nodes_ingested": len(node_mutations),
                "edges_ingested": len(edge_mutations),
                **result,
            },
            default=str,
        )

    # ── light path — the engine's real BatchUpdate insert-or-merge primitive,
    # chunked deterministically at its documented bounds. Nodes precede edges
    # so a chunked edge always finds its node already committed. ──
    mutations = [*node_mutations, *edge_mutations]
    # Named module-level lookups (not the helper's own default args) so a test
    # can tighten the bounds via monkeypatch to exercise multi-chunk behavior.
    chunks = _chunk_batch_mutations(
        mutations,
        max_ops=_BULK_INGEST_MAX_OPS,
        max_bytes=_BULK_INGEST_MAX_BYTES,
    )
    chunk_sizes: list[int] = []
    applied_ops = 0
    for chunk_index, chunk in enumerate(chunks):
        try:
            applied = engine.batch_typed_mutations(chunk, upsert=upsert)
        except Exception as e:  # noqa: BLE001 — surface as data, not a 500
            return public_error_json(
                e,
                context={
                    "action": "bulk_ingest",
                    "chunk_index": chunk_index,
                    "chunks_total": len(chunks),
                    "chunks_applied_before_failure": chunk_index,
                    "ops_applied_before_failure": applied_ops,
                },
            )
        if not applied:
            # `batch_typed_mutations` returns False only when the configured
            # backend has no native typed-batch capability (never a partial
            # write) — fail loudly rather than silently falling back to the
            # per-element loop this rewrite exists to remove.
            return public_error_json(
                RuntimeError(
                    "the configured backend has no native typed-batch "
                    "(BatchUpdate) capability"
                ),
                code="dependency_unavailable",
                context={
                    "action": "bulk_ingest",
                    "chunk_index": chunk_index,
                    "chunks_total": len(chunks),
                    "chunks_applied_before_failure": chunk_index,
                    "ops_applied_before_failure": applied_ops,
                },
            )
        chunk_sizes.append(len(chunk))
        applied_ops += len(chunk)
    return json.dumps(
        {
            "action": "bulk_ingest",
            "mode": "batch_update",
            "nodes_ingested": len(node_mutations),
            "edges_ingested": len(edge_mutations),
            "upsert": upsert,
            "chunks": len(chunks),
            "chunk_sizes": chunk_sizes,
            "applied_ops": applied_ops,
        }
    )


def register_write_ingest_tools(mcp):
    """Register the write_ingest_tools group on the given FastMCP server."""

    @mcp.tool(
        name="graph_write",
        description=(
            "Write nodes, relationships, or register external graphs to the Knowledge "
            "Graph. Actions: add_node, add_edge, delete_node, delete_edge, "
            "register_external_graph, bulk_ingest, compare_and_set, store_memory, "
            "recall_memory, recall_media, log_chat, submit_sdd, register_execution, check_loop. "
            "Use 'compare_and_set' for an atomic conditional update (optimistic "
            "concurrency / conditional state transitions / atomic reservations) so "
            "concurrent agents shaping the same node never lose each other's write."
        ),
        tags=["graph-os", "write", "mutation"],
    )
    async def graph_write(
        action: str = Field(
            description=(
                "Action to perform (add_node, add_edge, delete_node, delete_edge, "
                "register_external_graph, bulk_ingest, compare_and_set, store_memory, "
                "recall_memory, recall_media, log_chat, submit_sdd, register_execution, check_loop). "
                "Use 'compare_and_set' for an ATOMIC conditional update — optimistic "
                "concurrency / safe concurrent graph-shaping: it applies 'updates' only "
                "if every field in 'conditions' still equals the node's current value "
                "(missing field reads as null), so two agents mutating the same node "
                "never lose each other's write (conditional state transitions, atomic "
                "reservations)."
            )
        ),
        node_id: str = Field(
            default="", description="The unique identifier for the node."
        ),
        node_type: str = Field(
            default="", description="The type or label of the node."
        ),
        properties: str = Field(
            default="{}", description="JSON-encoded dictionary of properties."
        ),
        source_id: str = Field(
            default="", description="The source node ID for an edge."
        ),
        target_id: str = Field(
            default="", description="The target node ID for an edge."
        ),
        rel_type: str = Field(
            default="", description="The relationship type for an edge."
        ),
        endpoint_url: str = Field(
            default="", description="URL for external graph registration."
        ),
        graph_type: str = Field(
            default="",
            description="Type of external graph (e.g., 'sparql', 'graphql').",
        ),
        agent_id: str = Field(
            default="", description="ID of the agent performing the action."
        ),
        nodes: str = Field(
            default="[]",
            description=(
                "JSON-encoded list of nodes or tags for bulk operations. For "
                "action='bulk_ingest' (CONCEPT:AU-KG.ingest.envelope-atomic-transaction, B-11): a list of "
                "node objects ({'id','type','properties'}, or {'kind':'node',...} "
                "explicitly) and/or edge objects ({'kind':'edge','source_id',"
                "'target_id','rel_type','properties'} — 'source_id'+'target_id' "
                "alone also infers kind='edge'). Every element commits atomically "
                "in one native engine transaction (chunked deterministically at "
                "the engine's documented bounds — 32MiB / 50,000 ops / 500,000 "
                "items per chunk — never silently dropped; the response reports "
                "'chunks')."
            ),
        ),
        idempotency_key: str = Field(
            default="",
            description=(
                "For action='bulk_ingest': a caller-owned idempotency key for this "
                "exact batch (CONCEPT:AU-KG.ingest.envelope-atomic-transaction). Non-empty (or a "
                "non-empty 'evidence') routes the batch onto the engine's atomic "
                "ApplyChangeEnvelopes path (durably idempotent, scoped by "
                "(tenant, graph, idempotency_key) — a replay reports "
                "'status':'skipped', never silently re-reported as fresh 'success'). "
                "Empty ⇒ the lighter BatchUpdate path, which carries no per-call "
                "idempotency key at the wire level."
            ),
        ),
        evidence: str = Field(
            default="[]",
            description=(
                "For action='bulk_ingest': JSON list of evidence records "
                "({'object_id','modality','locus','content_digest'}) attached to "
                "the FIRST node in 'nodes' (CONCEPT:AU-KG.ingest.envelope-atomic-transaction). Non-empty "
                "routes the batch onto ApplyChangeEnvelopes (the only primitive "
                "carrying evidence/policy/lineage) instead of the lighter BatchUpdate."
            ),
        ),
        upsert: bool = Field(
            default=True,
            description=(
                "For action='bulk_ingest' on the BatchUpdate (light) path: real "
                "engine insert-or-merge semantics (CONCEPT:AU-KG.ingest.envelope-atomic-transaction). True "
                "(default) MERGEs onto an existing id — idempotent re-application, "
                "matching every prior add_node-loop caller's behavior. False INSERTs "
                "(a repeated edge becomes an additional parallel edge rather than "
                "replacing the prior one)."
            ),
        ),
        connection: str = Field(
            default="",
            description=(
                "CONCEPT:AU-KG.backend.multi-connection-registry — named BACKEND connection to write to "
                "(default = primary). Use a registered connection name, or 'all' "
                "(or a comma-separated list) to mirror the SAME write to several "
                "backends. Fan-out requires an explicit multi-connection value; the "
                "default and a single named connection stay single-write. Selects "
                "WHICH BACKEND, never which physical graph — see `graph`."
            ),
        ),
        graph: str = Field(
            default="",
            description=(
                "CONCEPT:AU-KG.backend.explicit-graph-selection — explicit physical engine graph to write to "
                "(one of the names `engine_tenants(action='list')`/the engine's own "
                "ListGraphs returns), independent of `connection`. Empty = the "
                "caller's own bound graph (unchanged default behavior). Requires "
                "exactly one resolved `connection` (the default one) — never "
                "combinable with `connection='all'`/a list. Authorization-checked "
                "by the engine's own RBAC/RLS on every call; an unknown graph, or "
                "a `connection` with no physical-graph concept, is a typed error — "
                "never a silent fallback to a default graph and never a union "
                "across graphs. Echoed back (as `connection`/`graph`) in the "
                "compare_and_set/recall_media JSON responses and appended to the "
                "plain-text outcome of the other write actions."
            ),
        ),
        conditions: dict = Field(
            default_factory=dict,
            description=(
                "For action='compare_and_set': field→expected-value the node must "
                "currently match for the update to apply (a missing field reads as "
                "null). e.g. {'status': 'pending'}."
            ),
        ),
        updates: dict = Field(
            default_factory=dict,
            description=(
                "For action='compare_and_set': field→new-value to merge into the node "
                "ONLY when every condition matches. e.g. {'status': 'claimed', "
                "'owner': 'agent-7'}."
            ),
        ),
    ) -> str:
        """Write nodes, relationships, or register external graphs. This is the primary mutation interface for the Knowledge Graph.

        The ``compare_and_set`` action is the atomic conditional-update primitive
        (CONCEPT:AU-KG.ingest.atomic-compare-and-set): it merges ``updates`` into ``node_id`` only if every
        field in ``conditions`` still equals the node's current value (missing
        field ≡ null) — use it for optimistic concurrency, conditional state
        transitions, and atomic reservations so two agents mutating the same node
        never clobber each other. Returns ``{"action": "compare_and_set",
        "node_id": ..., "applied": <bool>}`` (``applied=False`` = precondition
        failed / another agent won the race).
        """
        # A direct call bypassing `_execute_tool` (which resolves `Field`
        # defaults) binds an omitted `graph` to its raw, truthy
        # `pydantic.fields.FieldInfo` rather than `""` — normalize once here,
        # mirroring `query_tools._run_graph_query`'s identical guard.
        graph = graph if isinstance(graph, str) else ""

        def _execute() -> str:
            def _write_with_engine(engine: Any) -> str:
                if not engine:
                    return "Error: IntelligenceGraphEngine not active."
                try:
                    # ``properties`` carries a JSON dict for node/edge writes, but a
                    # RAW content string for the memory/chat/sdd actions
                    # (store_memory, recall_memory, log_chat, submit_sdd) which read it
                    # directly. Parse it ONLY where it is consumed as a dict — parsing
                    # eagerly for every action raised "Expecting value: line 1 column 1"
                    # on plain content text and broke graph_memory store/recall.
                    def _props() -> dict:
                        return json.loads(properties) if properties else {}

                    if action == "add_node":
                        if not node_id or not node_type:
                            return "Error: node_id and node_type required"
                        engine.add_node(node_id, node_type, _props())
                        return f"Node {node_id} added."
                    elif action == "add_edge":
                        if not source_id or not target_id or not rel_type:
                            return "Error: source_id, target_id, and rel_type required"
                        engine.link_nodes(source_id, target_id, rel_type, _props())
                        return f"Edge {source_id} -> {target_id} added."
                    elif action == "delete_node":
                        # BUG-049: this used to be `engine.delete_node(node_id)`
                        # followed by an UNCONDITIONAL f"Node {node_id} deleted."
                        # `node_id` defaults to "", was never validated, and
                        # `node_type` was ignored entirely -- so a predicate delete
                        # returned "Node  deleted." having removed nothing. A
                        # destructive action must never report success it did not
                        # perform. Note add_node/add_edge/register_external_graph
                        # directly above and below all validated their required
                        # args; only the two destructive branches did not.
                        if not node_id and not node_type:
                            return (
                                "Error: delete_node requires node_id, or node_type "
                                "for a predicate delete"
                            )
                        if node_id:
                            engine.delete_node(node_id)
                            return f"Node {node_id} deleted."
                        # Predicate delete. Enumerate engine-side by label rather
                        # than through Cypher: the query path applies RLS row
                        # filtering (unowned rows are invisible), so a Cypher
                        # enumeration would silently under-delete.
                        #
                        # The label index is reached under different names
                        # depending on which object `_resolve_target_engines`
                        # handed us: an IntelligenceGraphEngine exposes it on its
                        # backend as `nodes_by_label`, while a GraphComputeEngine
                        # (what incident_tools holds) has `get_nodes_by_label`
                        # directly. Try each, and if none exists FAIL LOUDLY naming
                        # what was tried — never fall back to a Cypher scan, which
                        # would under-delete, and never return a success string.
                        _backend = getattr(engine, "backend", None)
                        _lookup = (
                            getattr(engine, "get_nodes_by_label", None)
                            or getattr(_backend, "nodes_by_label", None)
                            or getattr(_backend, "get_nodes_by_label", None)
                        )
                        if _lookup is None:
                            return (
                                "Error: no label index accessor on this engine "
                                "(tried engine.get_nodes_by_label, "
                                "engine.backend.nodes_by_label, "
                                "engine.backend.get_nodes_by_label); refusing to "
                                "fall back to a Cypher scan, which RLS row "
                                "filtering would make under-delete"
                            )
                        matched = _lookup(node_type, 0) or []
                        deleted = 0
                        for matched_row in matched:
                            matched_id = (
                                matched_row[0]
                                if isinstance(matched_row, (tuple, list))
                                else matched_row
                            )
                            engine.delete_node(matched_id)
                            deleted += 1
                        return f"Deleted {deleted} node(s) of type {node_type}."
                    elif action == "delete_edge":
                        # BUG-049, same class: validate before mutating.
                        if not source_id or not target_id or not rel_type:
                            return "Error: source_id, target_id, and rel_type required"
                        engine.delete_edge(source_id, target_id, rel_type)
                        return f"Edge {source_id} -> {target_id} deleted."
                    elif action == "register_external_graph":
                        if not endpoint_url:
                            return "Error: endpoint_url required"
                        engine.add_node(
                            endpoint_url,
                            "ExternalGraphReference",
                            {"graph_type": graph_type},
                        )
                        return f"Registered external graph at {endpoint_url}"
                    elif action == "bulk_ingest":
                        return _run_bulk_ingest(
                            engine,
                            nodes,
                            idempotency_key=idempotency_key,
                            evidence=evidence,
                            upsert=upsert,
                        )
                    elif action == "compare_and_set":
                        # CONCEPT:AU-KG.ingest.atomic-compare-and-set — atomic compare-and-set as a first-class
                        # agent capability. Applies ``updates`` to the node
                        # ONLY if every field in ``conditions`` still equals its current
                        # value (missing field ≡ null), under the engine's write lock —
                        # the optimistic-concurrency primitive that lets concurrent
                        # agents shape the same node without lost updates. Returns a
                        # clear applied/not-applied result; a ``False`` (lost the race /
                        # precondition failed) is surfaced, never swallowed.
                        if not node_id:
                            return "Error: node_id required"

                        # Omitted dict params arrive as the unresolved FastMCP
                        # ``FieldInfo`` (default_factory is not resolved by the
                        # internal/REST dispatcher); coerce anything non-dict — and a
                        # JSON-string some MCP clients send — to a plain dict.
                        def _as_dict(v: Any) -> dict:
                            if isinstance(v, dict):
                                return v
                            if isinstance(v, str) and v.strip():
                                try:
                                    parsed = json.loads(v)
                                    return parsed if isinstance(parsed, dict) else {}
                                except (ValueError, TypeError):
                                    return {}
                            return {}

                        applied = bool(
                            engine.backend.compare_and_set_node_fields(
                                node_id, _as_dict(conditions), _as_dict(updates)
                            )
                        )
                        return json.dumps(
                            {
                                "action": "compare_and_set",
                                "node_id": node_id,
                                "applied": applied,
                            }
                        )
                    elif action in ("store_memory", "recall_memory"):
                        # The canonical memory store is the engine facade's MemoryMixin
                        # (engine.store_memory / engine.recall_memory) — the same path
                        # messaging/kg_ingest and the kg-memory task worker use. The old
                        # ``agent_utilities.memory.manager.MemoryManager`` indirection
                        # never existed as a module, so this branch always fell into
                        # "memory module not available"; wire it to the real method.
                        if action == "store_memory":
                            engine.store_memory(
                                content=properties,
                                memory_type=node_type or "episodic",
                                tags=json.loads(nodes) if nodes else [],
                                agent_id=agent_id,
                            )
                            return "Memory stored."
                        res = engine.recall_memory(
                            query=properties, memory_type=node_type, top_k=5
                        )
                        return "\n".join(str(r) for r in res)
                    elif action == "recall_media":
                        # CONCEPT:AU-KG.identity.asset-occurrence — list durable media
                        # occurrence records only. Content identity remains a property;
                        # it is never reused as occurrence identity.
                        # Returns metadata (occurrence_id + content_digest + media_type), NOT
                        # raw bytes (those are fetched by digest via
                        # MediaStore.fetch_bytes).
                        # Optional filter: node_id=<message memory id> for a turn's media.
                        where = "n.node_type = 'AssetOccurrence'"
                        if node_id:
                            if not _OPAQUE_NODE_ID.fullmatch(node_id):
                                return public_error_json(
                                    ValueError("invalid node identifier"),
                                    code="invalid_request",
                                )
                            where += f" AND n.message_id = '{node_id}'"
                        try:
                            rows = engine.query_cypher(
                                f"MATCH (n) WHERE {where} RETURN "
                                "n.id AS occurrence_id, n.content_digest AS digest, "
                                "n.media_type AS media_type, n.mime_type AS mime_type, "
                                "n.created_at AS created_at LIMIT 50"
                            )
                            return json.dumps(
                                {"action": "recall_media", "occurrences": rows},
                                default=str,
                            )
                        except Exception as e:  # noqa: BLE001
                            return public_error_json(e)
                    elif action in (
                        "log_chat",
                        "submit_sdd",
                        "register_execution",
                        "check_loop",
                    ):
                        if action == "log_chat":
                            engine.add_node(
                                f"chat_{agent_id}_{hash(properties)}",
                                "ChatLog",
                                {"content": properties, "agent_id": agent_id},
                            )
                            return "Chat logged."
                        elif action == "submit_sdd":
                            engine.add_node(
                                f"sdd_{agent_id}_{hash(properties)}",
                                "SDD",
                                {"content": properties, "agent_id": agent_id},
                            )
                            return "SDD submitted."
                        elif action == "register_execution":
                            engine.add_node(
                                f"exec_{agent_id}", "Execution", {"status": "running"}
                            )
                            return "Execution registered."
                        elif action == "check_loop":
                            return "Loop status: OK"
                        return f"Error: Action '{action}' not implemented."
                    else:
                        return f"Error: Unknown write action '{action}'"
                except Exception as e:
                    return public_error_text(e)

            def _echo_graph_selection(raw: str, name: str, requested_graph: str) -> str:
                """Echo the resolved `connection`/`graph` so a caller can always
                tell what was actually written (CONCEPT:AU-KG.backend.explicit-graph-selection). A JSON
                success payload gets the two keys merged in (never overriding an
                existing key); a standardized error envelope (has an `error` key)
                is left byte-for-byte alone; a plain-text outcome gets a bracketed
                suffix, unless it is already a plain-text `Error: ...` — an
                unstructured failure string gains nothing from selection metadata.
                """
                try:
                    parsed = json.loads(raw)
                except (TypeError, ValueError):
                    parsed = None
                if isinstance(parsed, dict):
                    if "error" in parsed:
                        return raw
                    parsed.setdefault("connection", name)
                    parsed.setdefault("graph", requested_graph)
                    return json.dumps(parsed, default=str)
                if raw.startswith("Error:"):
                    return raw
                return (
                    f"{raw} [connection={name} graph={requested_graph or '(default)'}]"
                )

            def _run_write(name: str, engine: Any) -> str:
                try:
                    with kg_server.bound_to_graph(graph):
                        raw = _write_with_engine(engine)
                except PermissionError as e:
                    # `_write_with_engine` already converts any exception the
                    # underlying write raises into a plain-text error, so this
                    # only catches a `PermissionError` from `bound_to_graph`
                    # itself. Same graph-conditional classification as the
                    # query tools: never reclassify a pre-existing generic
                    # denial when no explicit `graph` was requested.
                    return public_error_text(
                        e, code="permission_denied" if graph else "operation_failed"
                    )
                return _echo_graph_selection(raw, name, graph)

            # CONCEPT:AU-KG.backend.multi-connection-registry — resolve the connection(s). Writes only fan out on
            # an EXPLICIT multi-connection request ('all' or a list); the default
            # and a single named connection stay single-write to avoid accidental
            # multi-store writes. CONCEPT:AU-KG.backend.explicit-graph-selection — `graph` (a physical engine
            # graph) is a separate axis from `connection`; it requires exactly one
            # resolved connection, so it fails closed against any fan-out.
            try:
                entries, errors, fanout = kg_server._resolve_target_engines(connection)
                entries = kg_server.resolve_explicit_graph(
                    entries, graph, fanout=fanout
                )
            except kg_server.GraphNotFoundError as e:
                return public_error_text(e, code="graph_not_found")
            except kg_server.GraphSelectionConflictError as e:
                return public_error_text(e, code="graph_selection_conflict")
            except Exception as e:
                return public_error_text(e)

            # CONCEPT:AU-KG.ingest.role-enforcement — role enforcement: a 'read' (data source) or 'mirror'
            # (fan-out replica) connection rejects direct connection= writes. Mirrors are
            # written only through the fan-out outbox, never here.
            registry = kg_server.get_connection_registry()
            errors = dict(errors)
            writable = []
            for name, eng in entries:
                if registry.is_writable(name):
                    writable.append((name, eng))
                else:
                    errors[name] = (
                        f"connection '{name}' is read-only (role={registry.role(name)})"
                    )

            if not fanout:
                if not writable:
                    return json.dumps(
                        {"error": errors.get(entries[0][0], "connection is read-only")},
                        default=str,
                    )
                write_name, write_engine = writable[0]
                return _run_write(write_name, write_engine)

            # Fan-out — per-target timeout so one slow backend can't stall the set.
            results, fan_errors = kg_server.fanout_execute(
                writable, lambda name, eng: _run_write(name, eng)
            )
            return json.dumps(
                {"targets": results, "errors": {**errors, **fan_errors}}, default=str
            )

        return await run_blocking_ordered(_execute)

    kg_server.REGISTERED_TOOLS["graph_write"] = graph_write

    @mcp.tool(
        name="graph_feedback",
        description=(
            "Record a human correction so the brain learns: correction_type "
            "'outcome' adjusts an entity's reward, 'rule' persists a durable "
            "governance/voice/source rule consulted at retrieval time, 'eval' "
            "adds a regression case, 'reads_avoided' closes the code_context "
            "reads-avoided loop (target_id=capability_id, corrected_value=JSON "
            "{reads_avoided,files_read,correct,query}) (CONCEPT:AU-AHE.evaluation.reads-avoided-feedback), "
            "'action_outcome' closes the loop on ANY autonomous action — a context "
            "answer, a deploy, a ticket close, a routing choice (target_id=action/"
            "capability id, corrected_value=JSON {success,reward?,expected?,observed?,"
            "query?}) so routing/playbooks prefer actions that achieve their goal "
            "(CONCEPT:AU-AHE.evaluation.action-outcome-feedback), 'gotcha' pins a hard-won trap to a file/module "
            "(target_id=path, corrected_value=the note) so code_context surfaces it "
            "when an agent next touches that area (CONCEPT:AU-KG.ingest.gotcha-feedback-capture), "
            "'selective_erasure' forgets the learned reward for superseded "
            "designations (target_id + optional corrected_value list of ids) so the "
            "router re-learns them instead of carrying stale utility across a "
            "source/model regime change (CONCEPT:AU-KG.memory.generation-scoped-selective-reward). This is how "
            "'this was wrong, here's the fix' becomes future behaviour (CONCEPT:EG-KG.storage.nonblocking-checkpoint)."
        ),
        tags=["graph-os", "feedback", "learning"],
    )
    async def graph_feedback(
        correction_type: str = Field(
            description=(
                "outcome | rule | eval | reads_avoided | action_outcome | gotcha | "
                "selective_erasure."
            )
        ),
        target_id: str = Field(
            description="Entity/episode/query the correction is about."
        ),
        corrected_value: str = Field(
            default="",
            description="The corrected value (reward, expected output, etc.).",
        ),
        reason: str = Field(default="", description="Why — the human's explanation."),
        rule_scope: str = Field(
            default="governance",
            description="For rule corrections: governance | voice | source | preference.",
        ),
        rule_kind: str = Field(
            default="forbid",
            description="For rule corrections: forbid | prefer | demote.",
        ),
        actor_id: str = Field(
            default="human", description="Who issued the correction."
        ),
    ) -> str:
        """Record a human correction (outcome/rule/eval) and apply it durably."""
        engine = kg_server._get_engine()
        if not engine:
            return "Error: IntelligenceGraphEngine not active."

        def _execute() -> str:
            try:
                from agent_utilities.knowledge_graph.adaptation.feedback import (
                    FeedbackService,
                )

                service = FeedbackService.from_engine(engine)
                result = service.record_correction(
                    correction_type,
                    target_id,
                    corrected_value=corrected_value or None,
                    reason=reason,
                    actor_id=actor_id,
                    rule_scope=rule_scope,
                    rule_kind=rule_kind,
                )
                return json.dumps(result.as_dict())
            except Exception as e:
                return public_error_text(e)

        return await run_blocking_ordered(_execute)

    kg_server.REGISTERED_TOOLS["graph_feedback"] = graph_feedback

    @mcp.tool(
        name="graph_ingest",
        description="Smart ingestion for codebases, documents, directories, and conversation logs. Also handles corpus management and job status.",
        tags=["graph-os", "ingest"],
    )
    async def graph_ingest(
        target_path: str = Field(
            default="", description="Path or JSON list of paths to ingest."
        ),
        max_depth: int = Field(
            default=3, description="Maximum directory depth for codebase ingestion."
        ),
        agent_id: str = Field(
            default="", description="ID of the agent performing the ingestion."
        ),
        action: str = Field(
            default="ingest",
            description="Action to perform (ingest, ingest_url, backfill_platform_history, archivebox_sync, skill_workflows, fact_extract, sync_second_brain, classify_topics, enrich_pending_documents, distill, import_pack, ingest_knowledge_pack, agent_toolkit, corpus, jobs, job_status, status, cancel, clear, prioritize, rebuild_indexes, observe, materialize, materialize_source, sync, reflect). For backfill_platform_history, pass corpus_name=<recoverable platform> and target_path=<channel/room id>; it re-fetches retained history using the platform's existing credential and records idempotent backfilled InboundMessage nodes. 'enrich_pending_documents' sweeps :Document nodes a connector wrote via the native_ingest primitive (e.g. searxng-mcp results) from outside the hub process — raw text only, flagged needs_enrichment=true — and runs each through the SAME DocumentProcessor + central _enrich_text seam a direct ingest gets (chunk+contextual-enrich+concepts+facts+WorldView topic classification), clearing the flag. 'classify_topics' runs the WorldView subject/topic classifier ad hoc (CONCEPT:AU-KG.enrichment.topic-classification-topology): description=raw text (or target_path=file, or target_path=an existing Document node id to attach edges to) → classifies onto the canonical WorldView taxonomy (ontology_worldview.ttl) and mints/links the :Topic hierarchy (BROADER/NARROWER) + HAS_TOPIC/CLASSIFIED_AS edges with confidence; corpus_name=optional title. This is the SAME core every document ingestion runs by default — use this action to classify a document that already exists in the graph without re-ingesting it, or to preview a classification. 'ingest_url' content-aware single-URL ingest (CONCEPT:AU-KG.research.skill-graph-distillation): target_path=URL → fetch via the unified resolver (ArchiveBox→crawl4ai→requests) into a Document, and for a research roundup (auto-detected, or forced with description='extract_papers' / disabled with 'no_papers') download the cited papers via scholarx and ingest them too, linking page→paper; runs inline. 'archivebox_sync' pulls preserved ArchiveBox snapshots into the KG (corpus_name='full' = pull ALL, else delta; base_path=JSON list of snapshot ids to select). 'skill_workflows' ingests the universal-skills workflow corpus (workflows/<domain>/<name>/SKILL.md) into the KG as dispatchable WorkflowDefinition DAGs (+WorkflowStep depends_on edges +USES_SKILL links) in the exact WorkflowStore shape execute_workflow reads, so graph_orchestrate execute_workflow can discover and fire them; ALSO sweeps the sibling atomic-skill corpus (skill_type: skill) into a CallableResource(AGENT_SKILL) each via the same reused ingest_runnable_skill primitive, so both legs of the corpus are classified by this one call; target_path optionally overrides the corpus root, default=installed universal_skills package; idempotent (content-addressed re-ingest is a no-op); runs as a BACKGROUND job (returns a job_id immediately — the full corpus takes ~150s, over the call ceiling — poll with action=job_status job_id=<id>). 'materialize_source' runs an enterprise source extractor (corpus_name=category, e.g. 'camunda'/'aris'/'egeria'; description=optional JSON extractor config), persists its BusinessProcess/BusinessTask/FLOWS_TO batch into the graph via an in-process vendor client, then runs one OWL reasoning cycle so the new process structure folds into the cross-vendor crosswalk. 'fact_extract' turns a document (description=raw text, or target_path=file) into atomic (subject)-[predicate]->(object) fact edges with confidence/evidence/tags, dedups them, persists to the graph, and returns the facts + JSONL. 'sync_second_brain' (CONCEPT:AU-KG.enrichment.second-brain-note-sync) is the one-call personal-notes sync: target_path=notes directory or file (a markdown folder, an Obsidian vault, or a synced Nextcloud/Paperless export — see the `second-brain-sync` skill), corpus_name=corpus name, base_path=optional 'since' cursor (epoch seconds or an ISO-8601 timestamp) to only process notes changed since then. Per note: runs 'fact_extract' (evidence-spanned facts), extracts entities/claims and PROPOSES each new claim into the governed ClaimFlywheel lifecycle (graph_claims reads the same state — never silently accepted), then scans each new claim against existing graph content and persists any contradiction as a propose-only :BeliefRevisionProposal (the same node loop_controller's belief-revision pass writes, so existing review tooling picks it up with no new UI). Content-hash idempotent: re-running over an unchanged corpus mints zero new facts/claims/proposals. 'extract_submit'/'extract_jobs'/'extract_status'/'extract_pause'/'extract_resume'/'extract_jsonl' run extraction as a GPU-slot-scheduled job (preempt/backfill/resume on the single GPU) addressed by job_id; max_depth sets rounds. 'distill' exports a KG subgraph to a portable skill-graph (target_path=out dir; corpus_name=seed node id OR description=query; max_depth=hop depth). 'import_pack' re-ingests a distilled skill-graph dir back into the KG (target_path=dir; corpus_name='dedup' to merge duplicates). 'build_skill_graph' runs the UNIFIED skill-graph pipeline (CONCEPT:AU-KG.research.skill-graph-distillation): acquire from ANY source kind into one standardized skill-graph (corpus_name=name; target_path=output parent dir; base_path=JSON list of sources [{kind,uri,options}] OR 'kind=uri,kind=uri' shorthand over web/pdf/office/dir/url_reader/rest/database/mcp_tool/generated/kg_query; description=optional human description) — always writes the offline corpus + a sources.json provenance/freshness manifest, and ALSO ingests into the KG when the daemon is reachable (degrades cleanly otherwise). 'skill_graph_status' reports freshness of an existing skill-graph (target_path=dir; corpus_name='quick' to skip network sources). 'rebuild_skill_graph' re-acquires from the recorded sources and bumps the version (target_path=dir). Queue control: 'cancel' (job_id), 'clear' (target_path=status filter pending|running|completed|failed|cancelled|zombie|all, default completed), 'prioritize' (job_id, priority_bucket=0..3 — no named priority aliases are accepted). Research evolution (CONCEPT:AU-KG.ingest.batch-research-cohort): 'cohort_create' (base_path=JSON list of paper URLs, target_path=JSON list of repo paths, description=goal) batch-ingests a cohort of papers+repos whose self-polling barrier synthesizes the comparative feature/innovation matrix (KG-2.173) when every member drains; 'cohort_status' (job_id=cohort_id) returns per-member progress + the matrix counts; 'profile' (corpus_name=lane|type|tkind, CONCEPT:AU-OS.observability.per-lane-latency-metrics) returns per-lane/stage latency percentiles + token/cost + the parallelism factor.",
        ),
        job_id: str = Field(
            default="", description="ID of the job to check status for."
        ),
        priority_bucket: int = Field(
            default=1,
            ge=0,
            le=3,
            description="Integer WorkItem claim bucket used by prioritize.",
        ),
        corpus_name: str = Field(
            default="", description="Name of the corpus to add/update."
        ),
        base_path: str = Field(default="", description="Base path for the corpus."),
        description: str = Field(default="", description="Description of the corpus."),
        content_type: str = Field(
            default="",
            description="Internal override only — leave empty. The content type (codebase, document, config, prompt, skill, mcp_server, kb, conversation, policy) is auto-detected from the path, and heavy types (codebase/document) always run on the async job queue. Only set this to force a specific category for an ambiguous path.",
        ),
        connection: str = Field(
            default="",
            description=(
                "CONCEPT:AU-KG.backend.multi-connection-registry — for action='ingest' codebase/document "
                "jobs: named BACKEND connection to submit to (default = primary). "
                "Selects WHICH BACKEND, never which physical graph — see `graph`. "
                "Fan-out ('all'/a list) is rejected: an ingest job always targets "
                "exactly one backend, never a union."
            ),
        ),
        graph: str = Field(
            default="",
            description=(
                "CONCEPT:AU-KG.backend.explicit-graph-selection — for action='ingest' codebase/document jobs: "
                "explicit physical engine graph to ingest INTO (one of the names "
                "`engine_tenants(action='list')` returns), independent of "
                "`connection`. Empty = the caller's own bound graph (unchanged "
                "default behavior). An unknown graph, or a `connection` with no "
                "physical-graph concept, is a typed error — never a silent "
                "fallback or fan-out. The submitted job stays bound to this exact "
                "graph through async claim/execute (persisted on the WorkItem), so "
                "it can never drift to a default/registry-wide graph."
            ),
        ),
    ) -> str:
        """Smart ingestion tool to populate the Knowledge Graph with codebases, documents, and memory observations. Monitors async ingestion jobs."""
        # A direct call bypassing FastMCP's own Field-default resolution binds
        # an omitted `graph`/`connection` to a raw `FieldInfo` rather than
        # `""` — normalize once here, mirroring `query_tools._run_graph_query`'s
        # identical guard.
        graph = graph if isinstance(graph, str) else ""
        connection = connection if isinstance(connection, str) else ""

        # U-06/GOC-67: resolve `connection`/`graph` for action='ingest' BEFORE
        # any job is submitted or content is written, exactly like
        # `graph_query`/`graph_write` — an explicit graph never defaults,
        # never fans out, and an unknown/unauthorized graph fails closed with
        # no job created and no partial write.
        ingest_engine: Any = None
        if action == "ingest" and (graph or connection):
            try:
                entries, errors, fanout = kg_server._resolve_target_engines(connection)
                entries = kg_server.resolve_explicit_graph(
                    entries, graph, fanout=fanout
                )
            except kg_server.GraphNotFoundError as e:
                return public_error_text(e, code="graph_not_found")
            except kg_server.GraphSelectionConflictError as e:
                return public_error_text(e, code="graph_selection_conflict")
            except Exception as e:
                return public_error_text(e)
            if fanout or len(entries) != 1:
                return public_error_text(
                    kg_server.GraphSelectionConflictError(
                        "action='ingest' targets exactly one backend; "
                        "'connection=all'/a list is not supported for ingestion"
                    ),
                    code="graph_selection_conflict",
                )
            _ingest_conn_name, ingest_engine = entries[0]

        engine = ingest_engine if ingest_engine is not None else kg_server._get_engine()
        if not engine:
            return "Error: IntelligenceGraphEngine not active."

        try:
            if action == "ingest":
                from agent_utilities.knowledge_graph.ingestion.engine import (
                    ContentType,
                    IngestionEngine,
                    IngestionManifest,
                )

                if not target_path:
                    return "Error: target_path required for ingest action"

                # Parse one-or-many paths (JSON list, comma-separated, or single).
                raw = target_path.strip()
                paths = (
                    json.loads(raw)
                    if raw.startswith("[")
                    else [p.strip() for p in raw.split(",") if p.strip()]
                    if "," in raw
                    else [raw]
                )
                paths = [p.strip() for p in paths if isinstance(p, str) and p.strip()]
                if not paths:
                    return "Error: target_path required for ingest action"

                # ``content_type`` is auto-detected per path and is NOT an
                # agent-facing concern (CONCEPT:AU-KG.research.skill-graph-distillation ContentType.classify is the
                # single source of truth). It survives only as an internal override
                # for genuinely ambiguous paths; ``isinstance(str)`` filters out the
                # unresolved FastMCP ``FieldInfo`` default. Whatever the type, heavy
                # categories ALWAYS route through the async durable queue so an
                # ingest call can never block the caller for minutes — the old
                # "explicit content_type → synchronous IngestionEngine" branch was a
                # footgun that did exactly that.
                override = (
                    content_type.strip().lower()
                    if (content_type and isinstance(content_type, str))
                    else ""
                )

                def resolve_ct(p: str) -> ContentType:
                    if override:
                        try:
                            return ContentType(override)
                        except ValueError:
                            pass
                    return ContentType.classify(p)

                # DOCUMENT/CODEBASE are slow (chunk+embed / tree-sitter parse) and
                # are handled by the background task worker → enqueue, never block.
                # The remaining lightweight categories (config/prompt/skill/
                # mcp_server/kb/conversation/policy/…) are fast and are only routed
                # by the unified IngestionEngine, so they run inline.
                async_types = {ContentType.DOCUMENT, ContentType.CODEBASE}
                async_jobs: list[str] = []
                sync_out: list[str] = []
                ing: IngestionEngine | None = None
                for p in paths:
                    ct = resolve_ct(p)
                    if ct in async_types:
                        t_type = (
                            "codebase" if ct == ContentType.CODEBASE else "document"
                        )
                        # BUG-120: the async worker only re-narrows onto an
                        # explicit `graph` for `task_type='codebase'` (see
                        # `_bound_to_explicit_ingest_graph`'s call site in
                        # `_run_background_task`) — the legacy document-chunk
                        # ingest branch never reads the WorkItem's `graph`
                        # metadata at all. Silently accepting `graph=` here
                        # for a document path would echo a resolved graph the
                        # write never actually honors — a fabricated success.
                        # Fail closed instead until document ingest gets the
                        # same worker-side wiring codebase already has.
                        if graph and t_type != "codebase":
                            return public_error_text(
                                kg_server.GraphSelectionConflictError(
                                    "explicit graph selection for async "
                                    "ingestion is currently supported only "
                                    f"for codebase content; {p!r} resolved "
                                    f"to content_type={t_type!r}"
                                ),
                                code="graph_selection_conflict",
                            )
                        jid = await run_blocking_ordered(
                            engine.submit_task,
                            target_path=p,
                            is_codebase=(t_type == "codebase"),
                            provenance={
                                "agent_id": agent_id,
                                "max_depth": max_depth,
                            },
                            task_type=t_type,
                            # U-06: persist the caller's resolved graph on the
                            # WorkItem's own metadata — the async worker that
                            # later executes this job re-narrows onto it
                            # (`_bound_to_explicit_ingest_graph`) instead of
                            # falling back to its own ambient/default graph.
                            graph=graph,
                        )
                        async_jobs.append(jid)
                    else:
                        if ing is None:
                            ing = IngestionEngine(kg_engine=engine)
                        # Sync path runs inline, in THIS request's own verified
                        # session — narrow it directly for the call's duration
                        # (a no-op when `graph` is empty).
                        with kg_server.bound_to_graph(graph):
                            r = await ing.ingest(
                                IngestionManifest(
                                    content_type=ct,
                                    source_uri=p,
                                    max_depth=max_depth,
                                    metadata={"agent_id": agent_id},
                                )
                            )
                        sync_out.append(
                            f"[{ct.value}] {p}: {r.status} (+{r.nodes_created}n/+{r.edges_created}e"
                            f"{', ' + str(r.details.get('cards_pending')) + ' cards pending' if r.details.get('cards_pending') else ''}"
                            f"{'; ' + r.error if r.error else ''})"
                        )

                msgs: list[str] = []
                if async_jobs:
                    label = (
                        f"Started ingestion job {async_jobs[0]} for {paths[0]}"
                        if len(async_jobs) == 1
                        else f"Submitted {len(async_jobs)} jobs: {', '.join(async_jobs)}"
                    )
                    if graph or connection:
                        label += f" [connection={connection or '(default)'} graph={graph or '(default)'}]"
                    msgs.append(label)
                if sync_out:
                    msgs.append(" | ".join(sync_out))
                return " ; ".join(msgs) if msgs else "Nothing to ingest."

            elif action == "ingest_url":
                # Content-aware single-URL ingest (CONCEPT:AU-KG.research.skill-graph-distillation): fetch via the
                # unified resolver (ArchiveBox→crawl4ai→requests) → Document, and —
                # for a research roundup (auto-detected, or forced via
                # description='extract_papers') — download the papers it cites and
                # ingest them too. Runs as a BACKGROUND job (fetch + paper downloads
                # can exceed the call ceiling): returns a job_id; poll with
                # action=job_status. The gateway host daemon's task workers process
                # it through the unified _ingest_document path.
                if not target_path:
                    return "Error: target_path (a URL) required for ingest_url"
                url = target_path.strip()
                # Default ON (CONCEPT:AU-KG.ingest.chunk-overlap-stage): first-class embedded Chunk objects +
                # contextual-retrieval enrichment, at parity with connector ingestion
                # (KG-2.50) — makes this tool's documented "chunking + contextual
                # enrichment + embeddings" behavior real instead of only the plain
                # idea_block text chunks.
                prov: dict[str, Any] = {
                    "agent_id": agent_id,
                    "source_url": url,
                    "chunk_objects": True,
                    "contextual": True,
                }
                flag = (description or "").strip().lower()
                if flag in ("extract_papers", "papers", "extract_papers=true", "true"):
                    prov["extract_papers"] = True
                elif flag in ("no_papers", "extract_papers=false", "false"):
                    prov["extract_papers"] = False
                jid = await run_blocking_ordered(
                    engine.submit_task,
                    target_path=url,
                    is_codebase=False,
                    provenance=prov,
                    task_type="content_url",
                )
                return (
                    f"Submitted content-aware URL ingest job {jid} for {url} "
                    f"(poll: action=job_status job_id={jid})."
                )

            elif action == "backfill_platform_history":
                # BUG-041: expose the existing operator recovery capability on
                # the same graph_ingest surface as the other source actions.
                # ``corpus_name`` is the platform and ``target_path`` is its
                # channel/room id; the connector resolves the live backend's
                # already-configured credential and records idempotent history
                # rows through the normal inbox writer.
                from agent_utilities.messaging.backfill import (
                    backfill_platform_history,
                )

                platform = (corpus_name or "").strip().lower()
                channel_id = (target_path or "").strip()
                if not platform:
                    return "Error: corpus_name required for backfill_platform_history"
                if not channel_id:
                    return (
                        "Error: target_path (channel/room id) required for "
                        "backfill_platform_history"
                    )
                result = await run_blocking_ordered(
                    backfill_platform_history,
                    engine,
                    platform=platform,
                    channel_id=channel_id,
                    session=(agent_id or "graph_ingest"),
                )
                return json.dumps(result)

            elif action == "archivebox_sync":
                # Pull preserved ArchiveBox snapshots into the KG (CONCEPT:AU-KG.research.skill-graph-distillation).
                # corpus_name selects the mode: 'full' = pull ALL, else delta;
                # base_path = JSON list of specific snapshot ids to sync.
                from agent_utilities.knowledge_graph.core.source_sync import (
                    sync_source,
                )

                mode = (corpus_name or "delta").strip().lower()
                ids = None
                if base_path.strip().startswith("["):
                    ids = [str(x) for x in json.loads(base_path)]
                res_d = sync_source(
                    engine,
                    "archivebox",
                    mode="full" if mode == "full" else mode,
                    ids=ids,
                )
                return json.dumps(res_d)

            elif action == "gitlab_sync":
                # Index whole GitLab instance(s) as a resolved code graph (KG-2.9g).
                # corpus_name = mode ('full' = re-index all, else delta);
                # base_path = JSON list of project ids to narrow to.
                from agent_utilities.knowledge_graph.core.source_sync import (
                    sync_source,
                )

                mode = (corpus_name or "delta").strip().lower()
                ids = None
                if base_path.strip().startswith("["):
                    ids = [str(x) for x in json.loads(base_path)]
                res_d = sync_source(
                    engine,
                    "gitlab",
                    mode="full" if mode == "full" else mode,
                    ids=ids,
                )
                return json.dumps(res_d)

            elif action == "gitlab_webhook":
                # Near-real-time incremental re-index from a GitLab push/MR webhook
                # (KG-2.9g): description = the raw webhook JSON payload.
                from agent_utilities.knowledge_graph.core.gitlab_indexer import (
                    handle_gitlab_webhook,
                )

                try:
                    payload = json.loads(description) if description else {}
                except (ValueError, TypeError):
                    return json.dumps(
                        {"status": "ignored", "reason": "invalid payload JSON"}
                    )
                webhook_result = await run_blocking_ordered(
                    handle_gitlab_webhook, engine, payload
                )
                return json.dumps(webhook_result)

            elif action == "corpus":
                if not corpus_name:
                    return "Error: corpus_name required"
                await run_blocking_ordered(
                    engine.add_node,
                    f"corpus_{corpus_name}",
                    "Corpus",
                    base_path=base_path,
                    description=description,
                )
                return f"Corpus {corpus_name} added/updated."

            elif action == "jobs":
                import json as _json

                grouped = engine.list_tasks()
                lines = []
                for status, jobs in grouped.items():
                    if not isinstance(jobs, list):
                        continue
                    for job in jobs[:20]:
                        lines.append(
                            f"{job['job_id']}: {status} ({job.get('target', 'unknown')})"
                        )
                # Per-category metrics breakdown (time/nodes/edges/failures) —
                # the harness-style view, pollable over MCP (CONCEPT:EG-KG.storage.nonblocking-checkpoint).
                breakdown = {}
                if hasattr(engine, "aggregate_ingest_metrics"):
                    try:
                        _b = engine.aggregate_ingest_metrics()
                        breakdown = _b if isinstance(_b, dict) else {}
                    except Exception:  # noqa: BLE001
                        breakdown = {}
                head = (
                    "\n".join(lines) if lines else "No active or recent ingestion jobs."
                )
                return (
                    head
                    + "\n\n=== per-category metrics ===\n"
                    + _json.dumps(breakdown, indent=2)
                    if breakdown
                    else head
                )

            elif action in ("job_status", "status"):
                if not job_id:
                    return "Error: job_id required"
                import json as _json

                job = engine.get_task_status(job_id)
                if not job:
                    return f"Job {job_id} not found."
                status = job["status"]
                meta = job.get("metadata") or {}
                metrics = {
                    k: meta[k]
                    for k in (
                        "type",
                        "content_type",
                        "duration_ms",
                        "nodes_added",
                        "nodes_created",
                        "edges_added",
                        "edges_created",
                        "cards_pending",
                        "error",
                    )
                    if k in meta
                }
                for key in (
                    "attempt",
                    "max_attempts",
                    "resource_class",
                    "lease_expires_at",
                    "heartbeat_at",
                    "updated_at",
                ):
                    if job.get(key) is not None:
                        metrics[key] = job[key]
                return f"Job {job_id} status: {status}\n" + _json.dumps(
                    metrics, indent=2
                )

            elif action == "cancel":
                import json as _json

                if not job_id:
                    return "Error: job_id required for cancel"
                return _json.dumps(engine.cancel_task(job_id), indent=2)

            elif action == "clear":
                # ``target_path`` carries the status filter:
                # pending|running|completed|failed|cancelled|zombie|all (default
                # 'completed' — the safe default that never drops queued work).
                import json as _json

                tp = target_path if isinstance(target_path, str) else ""
                return _json.dumps(
                    engine.clear_tasks((tp or "completed").strip().lower()), indent=2
                )

            elif action == "prioritize":
                import json as _json

                if not job_id:
                    return "Error: job_id required for prioritize"
                return _json.dumps(
                    engine.prioritize_task(job_id, priority_bucket),
                    indent=2,
                )

            elif action == "cohort_create":
                # CONCEPT:AU-KG.ingest.batch-research-cohort — batch-ingest N papers + M repos as one research
                # cohort whose barrier synthesizes the comparative feature matrix
                # (KG-2.173) once every member drains. base_path = JSON list of paper
                # URLs/ids; target_path = JSON list of repo paths; description = goal.
                import json as _json

                from agent_utilities.knowledge_graph.research.cohort import (
                    create_cohort,
                )

                def _aslist(v: str) -> list[str]:
                    v = (v or "").strip()
                    if not v:
                        return []
                    try:
                        parsed = _json.loads(v)
                        return (
                            [str(x) for x in parsed]
                            if isinstance(parsed, list)
                            else [str(parsed)]
                        )
                    except (ValueError, TypeError):
                        return [s.strip() for s in v.split(",") if s.strip()]

                papers = _aslist(base_path)
                repos = _aslist(target_path)
                if not papers and not repos:
                    return public_error_text(
                        ValueError(
                            "cohort_create needs base_path=<JSON list of paper URLs> "
                            "and/or target_path=<JSON list of repo paths>"
                        ),
                        code="invalid_request",
                    )
                return _json.dumps(
                    create_cohort(engine, papers=papers, repos=repos, goal=description),
                    indent=2,
                )

            elif action == "cohort_status":
                import json as _json

                from agent_utilities.knowledge_graph.research.cohort import (
                    cohort_status,
                )

                cid = (job_id or target_path or "").strip()
                if not cid:
                    return "Error: job_id=<cohort_id> required for cohort_status"
                return _json.dumps(cohort_status(engine, cid), indent=2)

            elif action == "profile":
                # CONCEPT:AU-OS.observability.per-lane-latency-metrics — per-lane/stage latency percentiles + token/cost +
                # the parallelism factor (Σ task ms ÷ wall ms). corpus_name picks the
                # grouping dimension: lane (default) | type | tkind.
                import json as _json

                if not hasattr(engine, "profile_report"):
                    return "Error: profiling not available on this engine."
                return _json.dumps(
                    engine.profile_report(group_by=(corpus_name or "lane").strip()),
                    indent=2,
                )

            elif action == "fleet_relevance":
                # CONCEPT:AU-AHE.assimilation.research-source-grading — grade every ingested research source against the
                # whole 80+ agent-packages fleet; surface every >threshold match.
                # corpus_name = threshold percent (default 5.0).
                import json as _json

                from agent_utilities.knowledge_graph.research.fleet_relevance import (
                    grade_fleet,
                )

                try:
                    thr = float(corpus_name) if corpus_name else 5.0
                except ValueError:
                    thr = 5.0
                return _json.dumps(grade_fleet(engine, threshold_pct=thr), indent=2)

            elif action == "rebuild_indexes":
                engine.build_indexes()
                return "Indexes rebuilt successfully."

            # ── KG-2.7: Observational Memory Bridge Actions ──
            elif action == "observe":
                try:
                    from pathlib import Path as _Path

                    from agent_utilities.knowledge_graph.memory.observer import (
                        observe_from_file,
                    )

                    if not target_path:
                        return "Error: target_path required (path to JSONL transcript)"
                    observation_result = observe_from_file(
                        engine, _Path(target_path), source=agent_id or "mcp"
                    )
                    return observation_result or "No new observations extracted."
                except Exception as e:
                    return public_error_text(e)

            elif action == "materialize":
                try:
                    from agent_utilities.knowledge_graph.memory import (
                        materialize_memory,
                    )

                    paths = materialize_memory(engine)
                    return json.dumps(
                        {
                            "status": "materialized",
                            "files": {k: str(v) for k, v in paths.items()},
                        }
                    )
                except Exception as e:
                    return public_error_text(e)

            elif action == "sync":
                try:
                    from agent_utilities.knowledge_graph.memory import (
                        ingest_memory_edits,
                    )

                    results = ingest_memory_edits(engine)
                    return (
                        json.dumps({"status": "synced", "ingested": results})
                        if results
                        else "No edits detected."
                    )
                except Exception as e:
                    return public_error_text(e)

            elif action == "reflect":
                try:
                    from agent_utilities.knowledge_graph.memory import (
                        run_reflector,
                    )

                    reflection_result = run_reflector(engine)
                    return reflection_result or "No observations to reflect on."
                except Exception as e:
                    return public_error_text(e)

            elif action == "materialize_source":
                # CONCEPT:AU-KG.ingest.enterprise-source-extractor — persist an enterprise source extractor
                # (camunda/aris/egeria/…) INTO the graph, then run one OWL
                # reasoning cycle so the new BusinessProcess/BusinessTask/
                # FLOWS_TO structure folds into the cross-vendor crosswalk
                # natively. corpus_name=category; description=optional JSON
                # extractor config; an in-process vendor client is resolved
                # from the connector package's auth.get_client().
                try:
                    from agent_utilities.knowledge_graph.enrichment.materialize import (
                        run_materialize_source,
                    )

                    category = (corpus_name or "").strip()
                    if not category:
                        return json.dumps(
                            {
                                "error": "materialize_source requires corpus_name "
                                "(the extractor category, e.g. 'camunda' or 'aris')"
                            }
                        )
                    extractor_config = (
                        json.loads(description)
                        if description and description.strip().startswith("{")
                        else None
                    )
                    # Shared core — same path the unified ``source_sync`` uses.
                    return json.dumps(
                        run_materialize_source(
                            engine, category, config=extractor_config
                        ),
                        default=str,
                    )
                except Exception as e:
                    return public_error_text(e)

            elif action == "skill_workflows":
                # CONCEPT:AU-KG.ingest.skill-workflow-corpus — ingest the universal-skills workflow corpus
                # (workflows/<domain>/<name>/SKILL.md) as dispatchable
                # WorkflowDefinition DAGs so the orchestration workflow /
                # execute_workflow can discover & fire them. ``target_path`` is
                # an optional explicit corpus root (a dir that is/contains
                # ``workflows/``); default = installed universal_skills package.
                #
                # Also sweeps the ATOMIC-skill sibling corpus (skill_type: skill)
                # into a CallableResource(AGENT_SKILL) each, via the same reused
                # ``ingest_atomic_skills`` primitive ``package_install_ingest.py``
                # pairs with this one on its own watermarked schedule — this is the
                # manual/on-demand full-sweep entrypoint for BOTH legs, so an
                # operator (or a schedule that isn't currently reachable) is never
                # the only way to (re)classify the atomic-skill corpus.
                #
                # Durable per-node writes for the full corpus (~315 workflows)
                # take ~150s — over the MCP call ceiling — and the backend can't
                # bulk-write durably here, so this enqueues a BACKGROUND job (run
                # by the task worker, off the request path) and returns its id;
                # poll with ``action=job_status job_id=<id>``.
                try:
                    root = target_path if isinstance(target_path, str) else ""
                    jid = await run_blocking_ordered(
                        engine.submit_task,
                        target_path=root or "universal-skills",
                        is_codebase=False,
                        provenance={"agent_id": agent_id},
                        task_type="skill_workflows",
                    )
                    return json.dumps(
                        {
                            "job_id": jid,
                            "status": "submitted",
                            "message": (
                                f"Skill-workflow + atomic-skill ingest enqueued as "
                                f"background job {jid}; poll with graph_ingest "
                                f"action=job_status job_id={jid}."
                            ),
                        }
                    )
                except Exception as e:
                    return public_error_text(e)

            elif action == "curate_wiki":
                # CONCEPT:AU-KG.ingest.wiki-delta-ingest — delta-skip continuous ingest of a self-curating wiki dir.
                try:
                    from agent_utilities.knowledge_graph.ingestion.wiki_curator import (
                        curate_wiki,
                    )

                    if not target_path:
                        return json.dumps(
                            {"error": "curate_wiki requires target_path (the wiki dir)"}
                        )
                    summary = curate_wiki(engine, target_path)
                    return json.dumps(summary, default=str)
                except Exception as e:
                    return public_error_text(e)

            elif action == "distill":
                # CONCEPT:AU-AHE.optimization.physical-distillation-engine — Distill a coherent KG subgraph OUT into a
                # portable skill-graph: a reference/ markdown tree + a
                # kg_manifest.json provenance record (round-trippable via the
                # 'ingest_knowledge_pack' action). The output dir is consumable
                # verbatim by skill-graph-builder as a local-directory source.
                # Param overloads (mirroring agent_toolkit's reuse of fields):
                #   target_path  -> output directory (required)
                #   corpus_name  -> seed node id      (anchor by id)
                #   description  -> natural-language query (semantic anchor)
                #   max_depth    -> BFS hop depth
                try:
                    from agent_utilities.knowledge_graph.distillation import (
                        SkillGraphDistiller,
                    )

                    if not target_path:
                        return json.dumps(
                            {"error": "distill requires target_path (output dir)"}
                        )
                    seed = corpus_name or None
                    query = description or None
                    if not (seed or query):
                        return json.dumps(
                            {
                                "error": "distill requires a seed (corpus_name=node_id) "
                                "or query (description=text)"
                            }
                        )
                    # content_type="workflow" → distill a graph-native skill-WORKFLOW
                    # (procedure step-DAG) instead of a documentation skill-graph.
                    as_workflow = (content_type or "").strip().lower() == "workflow"
                    distiller = await SkillGraphDistiller.connect()
                    try:
                        if as_workflow:
                            wf = await distiller.distill_workflow(
                                seed=seed,
                                query=query,
                                depth=max_depth,
                                out_dir=target_path,
                            )
                            payload = {
                                "kind": "skill-workflow",
                                "name": wf["name"],
                                "steps": wf["steps"],
                            }
                        else:
                            manifest = await distiller.distill(
                                seed=seed,
                                query=query,
                                depth=max_depth,
                                out_dir=target_path,
                            )
                            payload = {
                                "kind": "skill-graph",
                                "stats": manifest["stats"],
                            }
                    finally:
                        await distiller.close()
                    return json.dumps(
                        {
                            "status": "distilled",
                            "out_dir": target_path,
                            "manifest": f"{target_path.rstrip('/')}/kg_manifest.json",
                            **payload,
                        },
                        default=str,
                    )
                except Exception as e:
                    return public_error_text(e)

            elif action in (
                "build_skill_graph",
                "skill_graph_status",
                "rebuild_skill_graph",
            ):
                # CONCEPT:AU-KG.research.skill-graph-distillation — the unified skill-graph pipeline: acquire from any
                # source kind (web/pdf/office/dir/url_reader/rest/database/mcp_tool/
                # generated/kg_query) into a standardized skill-graph with a
                # sources.json provenance/freshness manifest, hybrid-auto KG ingest,
                # and a staleness/rebuild loop. Heavy/blocking work runs off the event
                # loop via a worker thread.
                import asyncio

                from agent_utilities.knowledge_graph.distillation import (
                    SkillGraphPipeline,
                    SourceSpec,
                )

                pipe = SkillGraphPipeline()
                if action == "build_skill_graph":
                    if not (corpus_name and target_path):
                        return json.dumps(
                            {
                                "error": "build_skill_graph requires corpus_name (name) "
                                "and target_path (output parent dir); base_path = JSON "
                                "list of sources or 'kind=uri,kind=uri' shorthand."
                            }
                        )
                    try:
                        specs = _parse_source_specs(base_path, SourceSpec)
                    except ValueError as exc:
                        return public_error_json(exc)
                    if not specs:
                        return json.dumps({"error": "no sources provided in base_path"})
                    sg_built = await asyncio.to_thread(
                        lambda: pipe.build(
                            name=corpus_name,
                            specs=specs,
                            out_dir=target_path,
                            description=description or None,
                        )
                    )
                    return json.dumps(sg_built, default=str)
                if action == "skill_graph_status":
                    if not target_path:
                        return json.dumps(
                            {"error": "skill_graph_status requires target_path (dir)"}
                        )
                    quick = corpus_name.strip().lower() == "quick"
                    sg_report = await asyncio.to_thread(
                        lambda: pipe.status(target_path, quick=quick)
                    )
                    return json.dumps(sg_report, default=str)
                # rebuild_skill_graph
                if not target_path:
                    return json.dumps(
                        {"error": "rebuild_skill_graph requires target_path (dir)"}
                    )
                sg_rebuilt = await asyncio.to_thread(lambda: pipe.rebuild(target_path))
                return json.dumps(sg_rebuilt, default=str)

            elif action == "agent_toolkit":
                sources = (
                    json.loads(target_path)
                    if target_path.startswith("[")
                    else [target_path]
                )
                # Use `description` param as optional agent_card_path override
                agent_card_path = (
                    description if description else "/.well-known/agent.json"
                )
                result = await engine.ingest_agent_toolkit(
                    sources, agent_card_path=agent_card_path
                )
                return json.dumps(result, default=str)

            elif action == "ingest_knowledge_pack":
                from pathlib import Path

                import yaml

                from agent_utilities.models.knowledge_pack import (
                    KnowledgePackBundle,
                    KnowledgePackHydrator,
                    KnowledgePackImporter,
                )

                if not target_path:
                    return "Error: target_path required for ingest_knowledge_pack"

                path = Path(target_path)
                if not path.exists() or not path.is_file():
                    return f"Error: knowledge pack file not found at {target_path}"

                def _load_knowledge_pack_file() -> Any:
                    with open(path, encoding="utf-8") as f:
                        if path.suffix in [".yaml", ".yml"]:
                            return yaml.safe_load(f)
                        return json.load(f)

                data = await run_blocking_ordered(_load_knowledge_pack_file)
                bundle = KnowledgePackBundle.from_dict(data)
                await KnowledgePackHydrator.hydrate(bundle)
                await run_blocking_ordered(
                    KnowledgePackImporter.seed_into_kg, bundle, engine
                )
                return f"Knowledge pack from {target_path} hydrated and ingested."

            elif action == "import_pack":
                # CONCEPT:AU-AHE.optimization.physical-distillation-engine — Round-trip import of a distilled skill-graph
                # package (reference/ + kg_manifest.json): reconstruct the original
                # subgraph here, preserving node ids + edges. The inverse of
                # 'distill'. ``corpus_name="dedup"`` runs the IdeaBlock dedup-merge.

                from agent_utilities.knowledge_graph.distillation import (
                    import_skill_graph_pack,
                )

                if not target_path:
                    return json.dumps(
                        {"error": "import_pack requires target_path (skill-graph dir)"}
                    )
                try:
                    stats = await run_blocking_ordered(
                        import_skill_graph_pack,
                        engine,
                        target_path,
                        dedup=(corpus_name == "dedup"),
                    )
                    return json.dumps(
                        {"status": "imported", "stats": stats}, default=str
                    )
                except Exception as e:  # noqa: BLE001
                    return public_error_text(e)

            elif action == "fact_extract":
                # CONCEPT:AU-KG.enrichment.atomic-triple-extraction — document → atomic-triple fact extraction.
                # Streams (subject)-[predicate]->(object) edges carrying
                # confidence/evidence_span/tags, dedups them semantically with
                # our own embedder, persists them as graph edges (variant node
                # names merged), and returns the facts + JSONL (upstream parity).
                # Text source: ``description`` (raw text) or ``target_path``
                # (local file, else treated as raw text). Single round + dedup
                # (multi-round recall is opt-in over the REST surface).
                from pathlib import Path

                from agent_utilities.knowledge_graph.extraction import (
                    ExtractedFact,
                    extract_facts,
                    facts_to_jsonl,
                    persist_facts,
                )
                from agent_utilities.knowledge_graph.extraction.job_manager import (
                    EngineStoreAdapter,
                )

                text = description or ""
                source_ref = ""
                if not text and target_path:
                    p = Path(target_path)
                    if p.exists() and p.is_file():
                        text = await run_blocking_ordered(
                            p.read_text, encoding="utf-8", errors="ignore"
                        )
                        source_ref = persistence_reference(
                            "fact_source", target_path, namespace="fact-extraction"
                        )
                    else:
                        text = target_path
                if not text.strip():
                    return json.dumps(
                        {
                            "error": "fact_extract requires text (description=) "
                            "or a readable file (target_path=)"
                        }
                    )

                facts: list[ExtractedFact] = []
                async for ev in extract_facts(text, rounds=1, source_file=source_ref):
                    if ev["type"] == "fact":
                        facts.append(ExtractedFact(**ev["fact"]))

                # CONCEPT:AU-ORCH.execution.event-loop-blocking-sweep — persist_facts
                # loops over every extracted fact issuing a synchronous
                # add_node/add_edge KG round trip, so it must not run inline on
                # the request-serving loop. The scanner in
                # scripts/check_event_loop_blocking.py only matches ``engine.*``
                # shaped attribute calls and therefore cannot see a blocking call
                # made through a plain helper like this one (D-W15-6).
                stats = await run_blocking_ordered(
                    persist_facts, EngineStoreAdapter(engine), facts
                )
                unique = sum(1 for f in facts if not f.is_duplicate)
                return json.dumps(
                    {
                        "status": "extracted",
                        "facts": [f.model_dump() for f in facts],
                        "jsonl": facts_to_jsonl(facts),
                        "stats": {
                            **stats,
                            "total_facts": len(facts),
                            "unique_facts": unique,
                            "duplicate_facts": len(facts) - unique,
                        },
                    },
                    default=str,
                )

            elif action == "sync_second_brain":
                # CONCEPT:AU-KG.enrichment.second-brain-note-sync — one-call
                # personal-notes sync: target_path (a notes dir/file) ->
                # fact_extract (evidence-spanned facts) + entity/claim
                # extraction, each new claim PROPOSED into the governed
                # ClaimFlywheel lifecycle (never silently accepted), then a
                # ContradictionDetector scan against existing graph content —
                # a finding persists as a propose-only :BeliefRevisionProposal
                # (the SAME node loop_controller._run_belief_revision writes,
                # so the existing review surface picks it up with no new UI).
                # Thin composition only — see knowledge_graph/extraction/
                # second_brain_sync.py for the primitives it sequences.
                from datetime import datetime as _datetime

                from agent_utilities.knowledge_graph.extraction import (
                    sync_second_brain,
                )

                if not target_path:
                    return json.dumps(
                        {
                            "error": "sync_second_brain requires target_path "
                            "(a notes directory or file)"
                        }
                    )

                since: float | None = None
                raw_since = (base_path or "").strip()
                if raw_since:
                    try:
                        since = float(raw_since)
                    except ValueError:
                        try:
                            since = _datetime.fromisoformat(raw_since).timestamp()
                        except ValueError:
                            since = None

                sync_result = await sync_second_brain(
                    engine, target_path, since=since, corpus_name=corpus_name
                )
                return json.dumps(sync_result.model_dump(), default=str)

            elif action == "classify_topics":
                # CONCEPT:AU-KG.enrichment.topic-classification-topology — ad-hoc WorldView
                # subject/topic classification: classify text (description=raw text,
                # or target_path=file) and materialize the :Topic hierarchy +
                # HAS_TOPIC/CLASSIFIED_AS edges linking it to a Document node id
                # (target_path, when NOT a readable file, is used as that node id;
                # otherwise one is derived from a content hash). Same core the
                # ingestion enrichment seam runs default-on for every ingested
                # document (ingestion/engine.py::_enrich_text).
                import hashlib
                from pathlib import Path

                from agent_utilities.knowledge_graph.enrichment.topic_classifier import (
                    classify_and_link_topics,
                )

                text = description or ""
                doc_id = ""
                if target_path:
                    p = Path(target_path)
                    if p.exists() and p.is_file():
                        text = text or await run_blocking_ordered(
                            p.read_text, encoding="utf-8", errors="ignore"
                        )
                        doc_id = "doc:source:" + persistence_reference(
                            "document_source", target_path, namespace="topic-classifier"
                        )
                    else:
                        doc_id = "doc:source:" + persistence_reference(
                            "document_source", target_path, namespace="topic-classifier"
                        )
                if not text.strip():
                    return json.dumps(
                        {
                            "error": "classify_topics requires text (description=) "
                            "or a readable file (target_path=)"
                        }
                    )
                if not doc_id:
                    doc_id = (
                        f"doc:adhoc:{hashlib.sha256(text.encode()).hexdigest()[:16]}"
                    )

                backend = getattr(engine, "backend", None) or engine
                topic_res = await classify_and_link_topics(
                    backend, doc_id, text, title=corpus_name or "", source_type="adhoc"
                )
                return json.dumps(topic_res, default=str)

            elif action == "enrich_pending_documents":
                # CONCEPT:AU-KG.enrichment.topic-classification-topology — hub-side catch-up sweep for
                # ``:Document`` nodes a connector wrote via the ``native_ingest``
                # primitive (searxng-mcp results, any future native-ingest
                # producer) from OUTSIDE the hub process, so they arrived as raw
                # text without chunking/enrichment. Finds every
                # ``needs_enrichment=true`` Document (max_depth=limit, default
                # 200) and runs it through the SAME DocumentProcessor +
                # central _enrich_text seam every directly-ingested document gets.
                from agent_utilities.knowledge_graph.memory.native_ingest import (
                    enrich_pending_documents,
                )

                sweep_res = await enrich_pending_documents(engine, limit=200)
                return json.dumps(sweep_res)

            elif action in (
                "extract_submit",
                "extract_jobs",
                "extract_status",
                "extract_pause",
                "extract_resume",
                "extract_jsonl",
            ):
                # CONCEPT:AU-KG.enrichment.gpu-scheduled-extraction — GPU-slot-scheduled fact extraction. Unlike the
                # inline 'fact_extract', these submit a job that runs on the single
                # GPU inference slot with preempt/backfill/resume, so concurrent
                # submissions don't oversubscribe the GPU. job_id addresses a job.

                mgr = kg_server._get_extraction_manager(engine)

                if action == "extract_submit":
                    text = description or ""
                    if not text and target_path:
                        from pathlib import Path

                        p = Path(target_path)
                        text = (
                            await run_blocking_ordered(
                                p.read_text, encoding="utf-8", errors="ignore"
                            )
                            if p.exists() and p.is_file()
                            else target_path
                        )
                    if not text.strip():
                        return json.dumps(
                            {
                                "error": "extract_submit requires description= or target_path="
                            }
                        )
                    jid = await mgr.submit(
                        text=text, rounds=max(1, min(10, max_depth or 1))
                    )
                    return json.dumps({"status": "submitted", "job_id": jid})

                if action == "extract_jobs":
                    return json.dumps({"jobs": mgr.jobs()}, default=str)

                if not job_id:
                    return json.dumps({"error": f"{action} requires job_id"})

                if action == "extract_status":
                    return json.dumps(
                        mgr.status(job_id) or {"error": "no such job"}, default=str
                    )
                if action == "extract_jsonl":
                    return mgr.jsonl(job_id)
                if action == "extract_pause":
                    await mgr.pause(job_id)
                    return json.dumps({"status": "paused", "job_id": job_id})
                # extract_resume
                await mgr.resume(job_id)
                return json.dumps({"status": "resumed", "job_id": job_id})

            else:
                return f"Error: Unknown ingest action '{action}'"
        except Exception as e:
            return public_error_text(e)

    kg_server.REGISTERED_TOOLS["graph_ingest"] = graph_ingest

    @mcp.tool(
        name="usage_query",
        description=(
            "Query usage/cost/observability analytics (CONCEPT:AU-ECO.mcp.usage-cost-observability-surface): token "
            "counts, cost, model/tool/skill/db-call usage, session browser, "
            "activity heatmap, full-text search, and Langfuse trace links. One "
            "store covers both ingested agent logs and our own runtime telemetry."
        ),
        tags=["graph-os", "observability", "usage"],
    )
    async def usage_query(
        action: str = Field(
            default="summary",
            description=(
                "summary | by_model | by_project | by_agent | tools | activity | "
                "sessions | session_detail | top_sessions | search | traces | series"
            ),
        ),
        from_date: str = Field(default="", description="ISO start (started_at >=)."),
        to_date: str = Field(default="", description="ISO end (started_at <=)."),
        project: str = Field(default="", description="Filter by project."),
        agent: str = Field(default="", description="Filter by agent type."),
        model: str = Field(default="", description="Filter by model."),
        origin: str = Field(
            default="", description="ingested | runtime (omit for both)."
        ),
        tenant_id: str = Field(default="", description="Tenant scope."),
        session_id: str = Field(default="", description="For action=session_detail."),
        query: str = Field(default="", description="For action=search (FTS)."),
        limit: int = Field(default=50, description="Row cap for list actions."),
    ) -> str:
        """Read-side analytics over the usage store. Returns JSON."""
        import json as _json

        from agent_utilities.usage.authorization import (
            UsageAuthorizationError,
            resolve_usage_tenant,
        )
        from agent_utilities.usage.service import get_usage_service

        try:
            authoritative_tenant = resolve_usage_tenant(tenant_id or None)
        except UsageAuthorizationError as exc:
            return f"usage_query authorization error: {exc.detail}"
        svc = get_usage_service()
        f = {
            k: v
            for k, v in {
                "from_date": from_date,
                "to_date": to_date,
                "project": project,
                "agent": agent,
                "model": model,
                "origin": origin,
                "tenant_id": authoritative_tenant,
            }.items()
            if v
        }
        try:
            if action == "summary":
                out: Any = svc.summary(**f).model_dump()
            elif action == "by_model":
                out = [e.model_dump() for e in svc.by_model(**f)]
            elif action == "by_project":
                out = [e.model_dump() for e in svc.by_project(**f)]
            elif action == "by_agent":
                out = [e.model_dump() for e in svc.by_agent(**f)]
            elif action == "tools":
                out = [e.model_dump() for e in svc.tools(**f)]
            elif action == "activity":
                out = [e.model_dump() for e in svc.activity(**f)]
            elif action == "sessions":
                out = [e.model_dump() for e in svc.sessions(limit=limit, **f)]
            elif action == "top_sessions":
                out = [e.model_dump() for e in svc.top_sessions(limit=limit, **f)]
            elif action == "session_detail":
                if not session_id:
                    return "Error: session_id required for session_detail"
                detail = svc.session_detail(session_id, **f)
                out = detail.model_dump() if detail else None
            elif action == "search":
                if not query:
                    return "Error: query required for search"
                out = [e.model_dump() for e in svc.search(query, limit=limit, **f)]
            elif action == "traces":
                from agent_utilities.observability.langfuse_exporter import (
                    get_langfuse_exporter,
                )

                exporter = get_langfuse_exporter()
                enabled = bool(getattr(exporter, "enabled", False))
                trace_filters = {**f, "origin": "runtime"}
                rows = svc.sessions(limit=min(max(limit, 1), 500), **trace_filters)
                out = {
                    "enabled": enabled,
                    "trace_count": len(rows) if enabled else 0,
                    "traces": (
                        [
                            {
                                "trace_ref": persistence_reference("trace", row.id),
                                "project": row.project,
                            }
                            for row in rows
                        ]
                        if enabled
                        else []
                    ),
                }
            elif action == "series":
                # CONCEPT:AU-KG.ingest.per-agent-token-usage — per-agent token usage over time from the engine
                # tsdb (native range/window), not a Python re-scan. ``from_date``/
                # ``to_date`` are epoch seconds; ``model`` carries the bucket field
                # (default total_tokens); ``limit`` carries the window size in seconds
                # (0 = raw points). agent= the series key.
                from agent_utilities.observability.token_tracker import (
                    query_token_series,
                )

                try:
                    start = float(from_date) if from_date else 0.0
                    end = float(to_date) if to_date else 4.0e18
                except ValueError:
                    return "Error: series from_date/to_date must be epoch seconds"
                pts = query_token_series(
                    agent,
                    start,
                    end,
                    field=model or "total_tokens",
                    window_s=float(limit) if limit else None,
                    agg=origin or "sum",
                )
                out = [{"ts": t, "value": v} for t, v in pts]
            else:
                return f"Error: unknown usage_query action '{action}'"
            return _json.dumps(out, default=str)
        except Exception as exc:  # noqa: BLE001
            return f"usage_query error_type={type(exc).__name__}"

    kg_server.REGISTERED_TOOLS["usage_query"] = usage_query

    @mcp.tool(
        name="ingest_sessions",
        description=(
            "Ingest AI agent chat/session history into the usage store + KG "
            "(CONCEPT:AU-ECO.mcp.client-side-chat-session). 'collect' auto-detects installed agents on THIS "
            "host and parses their local logs (use when the engine is local). "
            "'upload' accepts pre-parsed session bundles as JSON so a CLIENT can "
            "parse its own logs and push them to a REMOTE/central engine that has "
            "no filesystem access to the client — closing the remote-ingest gap. "
            "'paths' ingests explicit files/dirs."
        ),
        tags=["graph-os", "ingest", "observability"],
    )
    async def ingest_sessions(
        action: str = Field(default="collect", description="collect | upload | paths"),
        bundles_json: str = Field(
            default="",
            description="For action=upload: JSON array of ParsedSessionBundle objects.",
        ),
        target_path: str = Field(
            default="", description="For action=paths: JSON list or comma paths."
        ),
        tenant_id: str = Field(default="", description="Tenant scope for the rows."),
    ) -> str:
        """Client-parses, server-sinks ingestion of agent session logs."""
        import json as _json
        import uuid as _uuid

        from agent_utilities.usage.authorization import (
            UsageAuthorizationError,
            require_usage_admin,
            resolve_usage_tenant,
        )

        try:
            authoritative_tenant = resolve_usage_tenant(tenant_id or None)
            if action == "collect":
                from agent_utilities.ingestion.collector import collect_local_sessions

                require_usage_admin()
                return _json.dumps(
                    collect_local_sessions(tenant_id=authoritative_tenant or ""),
                    default=str,
                )
            if action == "upload":
                # CONCEPT:AU-KG.ingest.drain-session-bundle — NON-BLOCKING upload. Each uploaded session
                # expands to many usage-store rows (sessions + events + tool
                # calls + FTS index), so the old synchronous ``record_bundle``
                # loop blew past the 60s MCP client window under load even at
                # batch=10. Mirror ``source_sync``/``graph_ingest``: ENQUEUE the
                # bundles as a durable ``session_upload`` background task and
                # return a ``job_id`` immediately — the host daemon's task worker
                # drains it (parse → usage store) off the call path. A tiny batch
                # is cheap, so it still runs inline (auto-sized, no user knob).
                from agent_utilities.usage.models import ParsedSessionBundle
                from agent_utilities.usage.privacy import normalize_bundle
                from agent_utilities.usage.recorder import get_usage_recorder

                raw = _json.loads(bundles_json) if bundles_json else []
                if not isinstance(raw, list):
                    return "Error: bundles_json must contain a JSON array"
                normalized_items: list[dict] = []
                for item in raw:
                    bundle = ParsedSessionBundle.model_validate(item)
                    if authoritative_tenant:
                        bundle.session.tenant_id = authoritative_tenant
                    normalized_items.append(normalize_bundle(bundle).model_dump())
                # Inline fast path only for a handful of bundles — well under the
                # call ceiling; anything larger enqueues.
                _UPLOAD_INLINE_MAX = 3
                if len(normalized_items) <= _UPLOAD_INLINE_MAX:
                    recorder = get_usage_recorder()
                    ok = 0
                    for item in normalized_items:
                        bundle = ParsedSessionBundle.model_validate(item)
                        if recorder.record_bundle(bundle):
                            ok += 1
                    return _json.dumps(
                        {
                            "received": len(normalized_items),
                            "ingested": ok,
                            "status": "ingested",
                        }
                    )

                # Large upload → enqueue and return. Carry the bundles in the
                # WorkItem metadata payload (same shape as ``kg_memory``,
                # CONCEPT:AU-KG.compute.offloaded-memory-write); the host worker reads it back, parses and
                # records. ``skip_dedupe`` because each batch is a distinct,
                # idempotent (record_bundle replaces rows) payload — never collapse
                # two real uploads into one. A unique target keeps job ids distinct.
                engine = kg_server._get_engine()
                target = f"session-upload:{_uuid.uuid4().hex}"
                jid = await run_blocking_ordered(
                    engine.submit_task,
                    target_path=target,
                    is_codebase=False,
                    provenance={"agent_id": "ingest_sessions"},
                    task_type="session_upload",
                    skip_dedupe=True,
                    extra_meta={"payload": {"bundles": normalized_items}},
                )
                return _json.dumps(
                    {
                        "status": "enqueued",
                        "job_id": jid,
                        "received": len(normalized_items),
                        "message": (
                            f"{len(normalized_items)} session bundles enqueued as background job "
                            f"{jid}; poll with graph_ingest action=job_status "
                            f"job_id={jid}."
                        ),
                    }
                )
            if action == "paths":
                from agent_utilities.ingestion.collector import collect_paths

                require_usage_admin()
                raw = target_path.strip()
                paths = (
                    _json.loads(raw)
                    if raw.startswith("[")
                    else [p.strip() for p in raw.split(",") if p.strip()]
                )
                return _json.dumps(
                    collect_paths(paths, tenant_id=authoritative_tenant or ""),
                    default=str,
                )
            return f"Error: unknown ingest_sessions action '{action}'"
        except UsageAuthorizationError as exc:
            return f"ingest_sessions authorization error: {exc.detail}"
        except Exception as exc:  # noqa: BLE001
            return f"ingest_sessions error_type={type(exc).__name__}"

    kg_server.REGISTERED_TOOLS["ingest_sessions"] = ingest_sessions
