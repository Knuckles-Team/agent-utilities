"""Pydantic request/response models for the ``/api/graph/ingest/*`` and
``/api/graph/write/*`` REST route families.

CONTEXT (why this module exists): every route in this family is mounted by
``agent_utilities.mcp.kg_server._mount_rest_routes`` as a raw Starlette
``app.add_route(...)`` handler, so today the live OpenAPI 3.1.0 spec documents
**zero** of these paths — FastAPI only introspects routes registered via
``add_api_route(..., response_model=...)``. These models are the typed
contract a later lane wires onto those routes with
``add_api_route(response_model=...)`` so the spec finally documents them.
Nothing here changes runtime behavior; it only describes the wire contract
each handler already enforces.

DERIVATION METHOD: for every route, the handler function in ``kg_server.py``
was read line-by-line to see exactly which JSON keys it pulls out of
``request.json()`` (or the raw request for GET/DELETE) and which keyword
arguments it forwards to ``_execute_tool(<tool_name>, **kwargs)``. Two shapes
occur:

* **Full passthrough** (``/graph/write``, ``/graph/ingest`` — the base,
  action-routed endpoints): the handler forwards the ENTIRE parsed body as
  ``_execute_tool(tool_name, **body)``. The real contract is therefore the
  underlying tool's own signature (``graph_write``/``graph_ingest`` in
  ``agent_utilities/mcp/tools/write_ingest_tools.py``), which is what these
  models mirror field-for-field. ``_execute_tool`` itself validates the
  forwarded kwargs against that signature and raises
  ``UnsupportedToolFieldError`` (mapped to HTTP 400) for anything the tool
  does not declare — so a field the model doesn't know about is NOT silently
  dropped by the server, only by an overly-strict client-side schema. To
  avoid a schema-vs-server strictness mismatch, these two models use
  ``extra="allow"``: unknown fields survive validation and are forwarded
  exactly as the handler already forwards them, and the *server* remains the
  single source of truth on which fields a given ``action`` actually accepts.

* **Cherry-picked / single-action** (every granular ``/graph/write/{node,
  edge,bulk,chat,execution}`` and ``/graph/ingest/{submit,corpus,observe,
  agent-toolkit,materialize,reflect,sync}`` route): the handler hard-codes
  the ``action`` and reads only a fixed, small set of keys off the body via
  ``body.get(...)``, ignoring everything else — proven by inspection, not
  assumed. These models therefore use the pydantic default (``extra``
  unset, i.e. ignored) rather than ``allow``: preserving an unread field
  would misrepresent the request as having been forwarded when the handler
  demonstrably never looks at it.

KNOWN LIVE DEFECT ENCODED CORRECTLY (not replicated): ``graph_write_node_endpoint``/
``graph_write_delete_node_endpoint`` previously called ``_execute_tool("graph_write",
id=...)`` while the tool declares the parameter ``node_id`` — every call failed
closed with ``UnsupportedToolFieldError``. The handler was already fixed to pass
``node_id=`` before this module was written; ``GraphWriteNodeRequest`` below
names the field ``node_id`` (never ``id``), and
``tests/unit/gateway/test_schemas_graph_ingest.py`` pins the correct name so a
regression trips a test, not just a runtime 400.

OTHER WIRE-NAME MISMATCHES FOUND (same class of bug-shape, but currently
INTENTIONAL renamings rather than defects — see the field docstrings below for
each):

* ``POST /graph/write/chat`` accepts wire field ``content``, which
  ``graph_write_chat_endpoint`` maps onto the tool's ``properties`` parameter
  (``properties=body.get("content", "")``). Modeled as ``content`` (the wire
  name a caller actually uses), with a note on the underlying mapping.
* ``POST /graph/ingest/agent-toolkit`` accepts wire fields ``sources`` and
  ``agent_card_path``, which ``graph_ingest_agent_toolkit_endpoint`` maps onto
  the tool's ``target_path`` and ``description`` parameters respectively.
  Modeled as ``sources``/``agent_card_path`` (the wire names), with a note on
  the underlying mapping.

KNOWN DATA-LOSS GAP (reported, not fixed here — out of this lane's scope,
which may not edit ``kg_server.py``): ``graph_write_bulk_endpoint`` forwards
only ``nodes`` to ``graph_write(action="bulk_ingest", ...)``. The underlying
tool also accepts ``idempotency_key``, ``evidence``, ``upsert``, ``connection``,
and ``graph`` for this same action (see the tool's own docstring), but the
granular ``/graph/write/bulk`` route never reads or forwards them — a caller
sending an idempotency key or explicit evidence on this route gets it
silently discarded server-side, always taking the non-idempotent
``BatchUpdate`` (light) path with ``upsert=True``. Callers who need those
options must use the base ``POST /graph/write`` (``action="bulk_ingest"``)
endpoint instead, whose model (``GraphWriteRequest``) does carry them. This
is documented on ``GraphWriteBulkRequest`` below.

TWO ROUTES IN SCOPE HAVE NO REQUEST MODEL BY DESIGN: ``GET /graph/ingest/jobs``
(``graph_ingest_jobs_endpoint``) and ``GET /graph/ingest/job/{job_id}``
(``graph_ingest_job_status_endpoint``) are both GET with no JSON body — the
latter's only input is the ``job_id`` path parameter, already typed by the
route pattern itself. Both return ``GraphToolResponse`` on success; no
dedicated request model is defined for either.

RESPONSE SHAPE: every route in both families returns the SAME envelope on
success — ``{"status": "success", "result": safe_json_load(res)}`` — built
in ``kg_server.py`` right after each ``_execute_tool(...)`` call.
``safe_json_load`` JSON-decodes ``res`` when it is a JSON-parseable string and
returns it unchanged otherwise (most of these tools return a human-readable
plain string, e.g. ``"Node n1 added."``; several return a JSON-encoded
object, e.g. bulk_ingest's chunk-count summary). Because ``result``'s actual
shape varies per route AND per the tool's internal branch (e.g. bulk_ingest's
``noop``/``change_envelope``/``batch_update`` modes each add different keys),
it is modeled permissively as ``Any`` on one SHARED ``GraphToolResponse`` —
the stable part is the envelope, not the payload inside it. A stricter
per-route response type is deliberately NOT invented here; see
``GraphToolResponse``'s own docstring.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "GraphToolResponse",
    "GraphWriteRequest",
    "GraphWriteNodeRequest",
    "GraphWriteEdgeRequest",
    "GraphWriteEdgeDeleteRequest",
    "GraphWriteBulkRequest",
    "GraphWriteChatRequest",
    "GraphWriteExecutionRequest",
    "GraphIngestRequest",
    "GraphIngestSubmitRequest",
    "GraphIngestCorpusRequest",
    "GraphIngestObserveRequest",
    "GraphIngestAgentToolkitRequest",
    "GraphIngestNoOpRequest",
]


# ══════════════════════════════════════════════════════════════════════════
# Shared response envelope
# ══════════════════════════════════════════════════════════════════════════


class GraphToolResponse(BaseModel):
    """The uniform success envelope every route in this module returns.

    Built by ``kg_server.py`` as
    ``JSONResponse({"status": "success", "result": safe_json_load(res)})``
    immediately after each route's ``_execute_tool(...)`` call succeeds. A
    failed call never reaches this shape — it returns a differently-shaped
    error JSON body (built by ``_external_error_response``/
    ``public_error_payload``) with a non-2xx status code instead, so this
    model documents the success case only.

    ``result`` is genuinely free-form and deliberately typed ``Any`` rather
    than given an invented structure: most of the underlying ``graph_write``/
    ``graph_ingest`` tool actions return a plain human-readable string (e.g.
    ``"Node n1 added."``, ``"Edge a -> b deleted."``), while others return a
    JSON-encoded object whose keys vary by internal branch (e.g.
    ``bulk_ingest``'s ``noop``/``change_envelope``/``batch_update`` modes each
    carry a different key set; ``job_status`` returns a status line followed
    by a pretty-printed JSON metrics block, which ``safe_json_load`` leaves as
    a plain string since the whole thing is not valid JSON). Treat ``result``
    as route- and action-specific free text/JSON to inspect at read time, not
    as a typed contract.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"status": "success", "result": "Node n1 added."},
                {
                    "status": "success",
                    "result": {
                        "action": "bulk_ingest",
                        "mode": "batch_update",
                        "nodes_ingested": 3,
                        "edges_ingested": 1,
                        "upsert": True,
                        "chunks": 1,
                        "chunk_sizes": [4],
                        "applied_ops": 4,
                    },
                },
            ]
        },
    )

    status: Literal["success"] = Field(
        description=(
            "Always 'success' on this envelope. A failed tool call or "
            "malformed request produces a separate error-shaped JSON body "
            "with a non-2xx HTTP status instead of this model."
        )
    )
    result: Any = Field(
        description=(
            "The underlying graph-os tool's return value, JSON-decoded when "
            "it was a JSON-parseable string and passed through unchanged "
            "otherwise (``safe_json_load``). Shape varies by route and, for "
            "the action-routed base endpoints, by the caller's own "
            "``action`` — see this route's tool implementation in "
            "``agent_utilities/mcp/tools/write_ingest_tools.py`` for the "
            "exact shape a given action produces."
        )
    )


# ══════════════════════════════════════════════════════════════════════════
# Base, full action-routed endpoints — POST /graph/write, POST /graph/ingest
# ══════════════════════════════════════════════════════════════════════════


class GraphWriteRequest(BaseModel):
    """Request body for ``POST /graph/write`` — the base, action-routed
    mutation endpoint (``graph_write_endpoint``).

    The handler forwards the ENTIRE parsed JSON body to
    ``_execute_tool("graph_write", **body)`` with no field filtering, so this
    model mirrors the ``graph_write`` MCP tool's own signature
    (``agent_utilities/mcp/tools/write_ingest_tools.py``) field-for-field.
    Unlike the granular single-action routes below, ``properties``/``nodes``/
    ``evidence`` here are the RAW JSON-STRING types the tool itself declares
    (this route does no ``dict``/``list`` → JSON-string coercion the way the
    granular routes do via their handler's ``_to_json_str`` helper) — send an
    already-JSON-encoded string for those fields, not a nested object.

    ``extra="allow"``: this is a full ``**body`` passthrough, so an unknown
    field reaches ``_execute_tool`` exactly as sent; the server's own
    ``UnsupportedToolFieldError`` → HTTP 400 is the real validation for a
    field a given ``action`` doesn't accept, not this schema.
    """

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "examples": [
                {
                    "action": "add_node",
                    "node_id": "n1",
                    "node_type": "Concept",
                    "properties": '{"label": "example"}',
                },
                {
                    "action": "compare_and_set",
                    "node_id": "n1",
                    "conditions": {"status": "pending"},
                    "updates": {"status": "claimed", "owner": "agent-7"},
                },
            ]
        },
    )

    action: str = Field(
        description=(
            "Action to perform: add_node, add_edge, delete_node, "
            "delete_edge, register_external_graph, bulk_ingest, "
            "compare_and_set, store_memory, recall_memory, recall_media, "
            "log_chat, submit_sdd, register_execution, or check_loop. Each "
            "action reads a different subset of the fields below; an "
            "unrecognized action returns a plain-text 'Unknown write "
            "action' result rather than a validation error. Use "
            "'compare_and_set' for an atomic conditional update (optimistic "
            "concurrency) so concurrent agents never lose each other's "
            "write to the same node."
        )
    )
    node_id: str = Field(
        default="",
        description=(
            "The node's unique identifier. Required (non-empty) for "
            "add_node, delete_node, and compare_and_set."
        ),
    )
    node_type: str = Field(
        default="",
        description=(
            "The node's type/label. Required (non-empty) for add_node; for "
            "delete_node, either node_id or node_type must be set (a "
            "node_type with no node_id performs a predicate delete of every "
            "node of that type)."
        ),
    )
    properties: str = Field(
        default="{}",
        description=(
            "JSON-ENCODED STRING (not a nested object) of node/edge "
            "properties for add_node/add_edge. For store_memory/"
            "recall_memory/log_chat/submit_sdd this instead carries RAW "
            "content/query text, not JSON."
        ),
    )
    source_id: str = Field(
        default="",
        description="Edge source node id. Required for add_edge/delete_edge.",
    )
    target_id: str = Field(
        default="",
        description="Edge target node id. Required for add_edge/delete_edge.",
    )
    rel_type: str = Field(
        default="",
        description="Edge relationship type. Required for add_edge/delete_edge.",
    )
    endpoint_url: str = Field(
        default="",
        description="External graph endpoint URL. Required for register_external_graph.",
    )
    graph_type: str = Field(
        default="",
        description=(
            "External graph kind for register_external_graph, e.g. "
            "'sparql' or 'graphql'."
        ),
    )
    agent_id: str = Field(
        default="", description="ID of the agent performing the action."
    )
    nodes: str = Field(
        default="[]",
        description=(
            "JSON-ENCODED STRING (not a nested list) list of node/edge "
            "objects for action='bulk_ingest'. Each element is either "
            "{'id','type','properties'} (or {'kind':'node',...} "
            "explicitly), or {'kind':'edge','source_id','target_id',"
            "'rel_type','properties'} — 'source_id'+'target_id' alone also "
            "infers kind='edge'. Committed atomically in one engine "
            "transaction (chunked at documented engine bounds; never "
            "silently dropped — the response reports 'chunks')."
        ),
    )
    idempotency_key: str = Field(
        default="",
        description=(
            "For action='bulk_ingest': a caller-owned idempotency key for "
            "this exact batch. Non-empty (or a non-empty 'evidence') routes "
            "the batch onto the engine's durably-idempotent "
            "ApplyChangeEnvelopes path, scoped by (tenant, graph, "
            "idempotency_key) — a replay reports 'status':'skipped', never "
            "silently re-reported as fresh success. Empty uses the lighter "
            "BatchUpdate path, which has no per-call idempotency key."
        ),
    )
    evidence: str = Field(
        default="[]",
        description=(
            "For action='bulk_ingest': JSON-encoded STRING list of evidence "
            "records ({'object_id','modality','locus','content_digest'}) "
            "attached to the first node in 'nodes'. Non-empty routes the "
            "batch onto ApplyChangeEnvelopes instead of the lighter "
            "BatchUpdate path."
        ),
    )
    upsert: bool = Field(
        default=True,
        description=(
            "For action='bulk_ingest' on the BatchUpdate (light) path: True "
            "(default) MERGEs onto an existing id (idempotent); False "
            "INSERTs (a repeated edge becomes an additional parallel edge "
            "rather than replacing the prior one)."
        ),
    )
    connection: str = Field(
        default="",
        description=(
            "Named backend connection to write to (default = primary). Use "
            "a registered connection name, or 'all'/a comma-separated list "
            "to mirror the same write to several backends. Selects WHICH "
            "BACKEND, never which physical graph — see 'graph'."
        ),
    )
    graph: str = Field(
        default="",
        description=(
            "Explicit physical engine graph to write to, independent of "
            "'connection'. Empty = the caller's own bound graph. Requires "
            "exactly one resolved 'connection' — never combinable with "
            "connection='all'/a list. An unknown graph is a typed error, "
            "never a silent fallback."
        ),
    )
    conditions: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "For action='compare_and_set': field to expected-value the "
            "node must currently match for 'updates' to apply (a missing "
            "field reads as null), e.g. {'status': 'pending'}."
        ),
    )
    updates: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "For action='compare_and_set': field to new-value to merge "
            "into the node ONLY when every condition in 'conditions' "
            "matches, e.g. {'status': 'claimed', 'owner': 'agent-7'}."
        ),
    )


class GraphIngestRequest(BaseModel):
    """Request body for ``POST /graph/ingest`` — the base, action-routed
    ingestion endpoint (``graph_ingest_endpoint``).

    Like ``GraphWriteRequest``, the handler forwards the entire parsed body
    to ``_execute_tool("graph_ingest", **body)`` with no field filtering, so
    this mirrors the ``graph_ingest`` MCP tool's signature. That tool's
    ``action`` accepts a much larger vocabulary than this module's granular
    routes cover (ingest, ingest_url, backfill_platform_history, corpus,
    jobs, job_status, cancel, clear, prioritize, rebuild_indexes, observe,
    materialize, materialize_source, sync, reflect, agent_toolkit,
    ingest_knowledge_pack, fact_extract, distill, and more — see the tool's
    own docstring for the full, actively-growing list). ``action`` is
    deliberately typed ``str`` rather than a ``Literal`` enum here: the tool
    grows new actions independently of this schema, and an unrecognized
    action fails at the tool layer with a descriptive error rather than a
    validation error, so pinning an enum would only risk silently going
    stale.

    ``extra="allow"`` for the same reason as ``GraphWriteRequest``: this is
    a full ``**body`` passthrough.
    """

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "examples": [
                {
                    "action": "ingest",
                    "target_path": "/repos/agent-utilities",
                    "max_depth": 3,
                    "agent_id": "agent-7",
                },
                {"action": "job_status", "job_id": "job-abc123"},
            ]
        },
    )

    target_path: str = Field(
        default="",
        description=(
            "Path, or JSON-encoded STRING list of paths, to ingest. "
            "Required (non-empty) for action='ingest'. Heavy content types "
            "(codebase/document) are always routed to the async job queue; "
            "lighter categories (config/prompt/skill/mcp_server/kb/"
            "conversation/policy) run inline."
        ),
    )
    max_depth: int = Field(
        default=3, description="Maximum directory depth for codebase ingestion."
    )
    agent_id: str = Field(
        default="", description="ID of the agent performing the ingestion."
    )
    action: str = Field(
        default="ingest",
        description=(
            "Action to perform. 'ingest' (default) is the common case "
            "modeled by the granular /graph/ingest/submit route. See this "
            "model's own docstring for the much larger action vocabulary "
            "this base endpoint alone exposes."
        ),
    )
    job_id: str = Field(
        default="",
        description="ID of the job to check status for/cancel/prioritize.",
    )
    priority_bucket: int = Field(
        default=1,
        ge=0,
        le=3,
        description="Integer WorkItem claim bucket (0-3) used by action='prioritize'.",
    )
    corpus_name: str = Field(
        default="", description="Name of the corpus to add/update for action='corpus'."
    )
    base_path: str = Field(
        default="", description="Base path for the corpus, for action='corpus'."
    )
    description: str = Field(
        default="",
        description="Free-text description of the corpus, for action='corpus'.",
    )
    content_type: str = Field(
        default="",
        description=(
            "Internal override only — leave empty. The content type is "
            "auto-detected from target_path; set this only to force a "
            "specific category for an ambiguous path."
        ),
    )
    connection: str = Field(
        default="",
        description=(
            "For action='ingest' codebase/document jobs: named backend "
            "connection to submit to (default = primary). Fan-out "
            "('all'/a list) is rejected — an ingest job always targets "
            "exactly one backend."
        ),
    )
    graph: str = Field(
        default="",
        description=(
            "For action='ingest' codebase/document jobs: explicit physical "
            "engine graph to ingest into, independent of 'connection'. "
            "Empty = the caller's own bound graph. The submitted job stays "
            "bound to this exact graph through async claim/execute."
        ),
    )


# ══════════════════════════════════════════════════════════════════════════
# Granular /graph/write/* endpoints
# ══════════════════════════════════════════════════════════════════════════


class GraphWriteNodeRequest(BaseModel):
    """Request body for ``POST /graph/write/node`` (``graph_write_node_endpoint``).

    Hard-codes ``action="add_node"`` server-side and calls
    ``_execute_tool("graph_write", action="add_node", node_id=..., "
    "node_type=..., properties=...)``.

    REGRESSION GUARD: the tool declares this identifier parameter
    ``node_id`` (never ``id``) — a prior mismatch (the handler used to pass
    ``id=``) made every call to this route fail closed with
    ``UnsupportedToolFieldError``, fixed before this schema was written. The
    field below is named ``node_id`` to match the tool exactly and is
    covered by a regression test.

    ``node_id``/``node_type`` are marked required here (unlike the
    underlying tool, which defaults both to ``""``) because the tool's own
    add_node branch unconditionally returns ``"Error: node_id and node_type
    required"`` when either is empty — this route can never meaningfully
    succeed without them, so failing fast with a 422 is more honest than a
    200 response whose ``result`` is an error string.
    """

    model_config = ConfigDict(extra="ignore")

    node_id: str = Field(
        min_length=1,
        description="The unique identifier for the node. Forwarded to the tool as 'node_id', not 'id'.",
        json_schema_extra={"examples": ["n1"]},
    )
    node_type: str = Field(
        min_length=1,
        description="The type/label of the node.",
        json_schema_extra={"examples": ["Concept"]},
    )
    properties: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "JSON object of node properties. The handler JSON-encodes this "
            "into a string before forwarding it to the graph_write tool "
            "(which declares 'properties' as a JSON-encoded string, not a "
            "nested object) — send a plain object here, not a pre-encoded "
            "string."
        ),
        json_schema_extra={"examples": [{"label": "example"}]},
    )


class GraphWriteEdgeRequest(BaseModel):
    """Request body for ``POST /graph/write/edge`` (``graph_write_edge_endpoint``,
    action='add_edge').

    ``source_id``/``target_id``/``rel_type`` are marked required here for the
    same reason as ``GraphWriteNodeRequest.node_id``/``node_type``: the
    tool's add_edge branch unconditionally errors when any of the three is
    empty.

    Note ``POST /graph/write/edge`` and ``DELETE /graph/write/edge`` share a
    path but accept DIFFERENT bodies — see ``GraphWriteEdgeDeleteRequest``
    for the DELETE (action='delete_edge') body, which has no 'properties'.
    """

    model_config = ConfigDict(extra="ignore")

    source_id: str = Field(min_length=1, description="The edge's source node id.")
    target_id: str = Field(min_length=1, description="The edge's target node id.")
    rel_type: str = Field(min_length=1, description="The edge's relationship type.")
    properties: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "JSON object of edge properties. JSON-encoded server-side "
            "before being forwarded to the graph_write tool."
        ),
    )


class GraphWriteEdgeDeleteRequest(BaseModel):
    """Request body for ``DELETE /graph/write/edge``
    (``graph_write_delete_edge_endpoint``, action='delete_edge').

    Same path as ``POST /graph/write/edge`` (``GraphWriteEdgeRequest``) but a
    different body: no 'properties' field, since the handler never reads one
    for a delete.
    """

    model_config = ConfigDict(extra="ignore")

    source_id: str = Field(min_length=1, description="The edge's source node id.")
    target_id: str = Field(min_length=1, description="The edge's target node id.")
    rel_type: str = Field(min_length=1, description="The edge's relationship type.")


class GraphWriteBulkRequest(BaseModel):
    """Request body for ``POST /graph/write/bulk`` (``graph_write_bulk_endpoint``,
    action='bulk_ingest').

    KNOWN GAP: this route forwards ONLY ``nodes`` to the underlying
    ``graph_write`` tool call — ``idempotency_key``, ``evidence``, ``upsert``,
    ``connection``, and ``graph`` (all of which the tool accepts for
    action='bulk_ingest') are not read by this handler at all, so sending
    them here has no effect; the write always takes the non-idempotent
    BatchUpdate (light) path with ``upsert=True``. Callers who need those
    options must use ``POST /graph/write`` with ``action="bulk_ingest"``
    instead (``GraphWriteRequest``), whose model carries them. Reported as a
    finding rather than fixed here — this lane may not edit
    ``kg_server.py``.
    """

    model_config = ConfigDict(extra="ignore")

    nodes: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "List of node and/or edge objects to commit atomically in one "
            "engine transaction. A node element is {'id','type',"
            "'properties'} (or {'kind':'node',...} explicitly); an edge "
            "element is {'kind':'edge','source_id','target_id','rel_type',"
            "'properties'} — 'source_id'+'target_id' alone also infers "
            "kind='edge'. An empty list is a valid no-op (the response "
            "reports mode='noop')."
        ),
        json_schema_extra={
            "examples": [
                [
                    {"id": "n1", "type": "Concept", "properties": {"label": "a"}},
                    {
                        "kind": "edge",
                        "source_id": "n1",
                        "target_id": "n2",
                        "rel_type": "RELATES_TO",
                    },
                ]
            ]
        },
    )


class GraphWriteChatRequest(BaseModel):
    """Request body for ``POST /graph/write/chat`` (``graph_write_chat_endpoint``,
    action='log_chat').

    WIRE-NAME MAPPING: the wire field is ``content`` — the handler maps it
    onto the tool's ``properties`` parameter
    (``properties=body.get("content", "")``). Modeled with the wire name a
    caller actually sends, not the tool's internal parameter name.
    """

    model_config = ConfigDict(extra="ignore")

    agent_id: str = Field(default="", description="ID of the agent logging the chat.")
    content: str = Field(
        default="",
        description=(
            "The chat content to log. Forwarded to the graph_write tool as "
            "its 'properties' parameter — this route renames it on the "
            "wire for readability."
        ),
    )


class GraphWriteExecutionRequest(BaseModel):
    """Request body for ``POST /graph/write/execution``
    (``graph_write_execution_endpoint``, action='register_execution')."""

    model_config = ConfigDict(extra="ignore")

    agent_id: str = Field(
        default="", description="ID of the agent whose execution is being registered."
    )


# ══════════════════════════════════════════════════════════════════════════
# Granular /graph/ingest/* endpoints
# ══════════════════════════════════════════════════════════════════════════


class GraphIngestSubmitRequest(BaseModel):
    """Request body for ``POST /graph/ingest/submit``
    (``graph_ingest_submit_endpoint``, action='ingest').

    ``target_path`` is marked required here (unlike the underlying tool,
    which defaults it to ``""``) because the tool's ingest branch
    unconditionally returns ``"Error: target_path required for ingest
    action"`` when it is empty.
    """

    model_config = ConfigDict(extra="ignore")

    target_path: str | list[str] = Field(
        description=(
            "Path to ingest, or a list of paths. A list is JSON-encoded "
            "server-side before being forwarded to the graph_ingest tool "
            "(which accepts a single path string or a JSON-encoded string "
            "list); send either form here directly."
        ),
        json_schema_extra={"examples": ["/repos/agent-utilities"]},
    )
    max_depth: int = Field(
        default=3, description="Maximum directory depth for codebase ingestion."
    )
    agent_id: str = Field(
        default="", description="ID of the agent performing the ingestion."
    )


class GraphIngestCorpusRequest(BaseModel):
    """Request body for ``POST /graph/ingest/corpus``
    (``graph_ingest_corpus_endpoint``, action='corpus').

    ``corpus_name`` is marked required here (unlike the underlying tool,
    which defaults it to ``""``) because the tool's corpus branch
    unconditionally returns ``"Error: corpus_name required"`` when it is
    empty.
    """

    model_config = ConfigDict(extra="ignore")

    corpus_name: str = Field(
        min_length=1, description="Name of the corpus to add/update."
    )
    base_path: str = Field(
        default="", description="Base filesystem path for the corpus."
    )
    description: str = Field(
        default="", description="Free-text description of the corpus."
    )


class GraphIngestObserveRequest(BaseModel):
    """Request body for ``POST /graph/ingest/observe``
    (``graph_ingest_observe_endpoint``, action='observe').

    ``target_path`` is marked required here (unlike the underlying tool,
    which defaults it to ``""``) because the tool's observe branch
    unconditionally returns ``"Error: target_path required (path to JSONL
    transcript)"`` when it is empty.
    """

    model_config = ConfigDict(extra="ignore")

    target_path: str = Field(
        min_length=1,
        description="Path to a JSONL transcript to extract observations from.",
    )
    agent_id: str = Field(
        default="",
        description="Source label recorded on the extracted observations (defaults to 'mcp' if omitted entirely).",
    )


class GraphIngestAgentToolkitRequest(BaseModel):
    """Request body for ``POST /graph/ingest/agent-toolkit``
    (``graph_ingest_agent_toolkit_endpoint``, action='agent_toolkit').

    WIRE-NAME MAPPING: the wire fields are ``sources`` and
    ``agent_card_path`` — the handler maps them onto the tool's
    ``target_path`` and ``description`` parameters respectively
    (``target_path=_to_json_str(body.get("sources", [])),
    description=body.get("agent_card_path", "")``). Modeled with the wire
    names a caller actually sends, not the tool's internal parameter names.
    """

    model_config = ConfigDict(extra="ignore")

    sources: list[str] = Field(
        default_factory=list,
        description=(
            "List of agent-toolkit source URIs/paths to ingest. JSON-"
            "encoded server-side before being forwarded to the "
            "graph_ingest tool as its 'target_path' parameter."
        ),
    )
    agent_card_path: str = Field(
        default="",
        description=(
            "Optional override path for the agent card manifest. Forwarded "
            "to the graph_ingest tool as its 'description' parameter; when "
            "omitted/empty, the tool itself defaults to "
            "'/.well-known/agent.json'."
        ),
    )


class GraphIngestNoOpRequest(BaseModel):
    """Shared request body for the three ``/graph/ingest/*`` routes whose
    handlers read NO fields at all from the request body:
    ``POST /graph/ingest/materialize`` (action='materialize'),
    ``POST /graph/ingest/reflect`` (action='reflect'), and
    ``POST /graph/ingest/sync`` (action='sync').

    Each of these handlers calls ``_execute_tool("graph_ingest",
    action=<fixed>)`` directly, without ever parsing ``request.json()`` —
    unlike every other route in this module, a body is not merely optional
    here, it is never even read. Any JSON object (or no body at all) is
    accepted; this model documents that contract as an intentionally-empty,
    fully-permissive shape rather than omitting a model for these three
    routes, so all nine ``/graph/ingest/*`` routes in this lane's scope have
    an explicit, documented request contract.
    """

    model_config = ConfigDict(extra="allow")
