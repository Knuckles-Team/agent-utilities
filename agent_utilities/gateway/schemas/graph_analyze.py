"""Typed request/response models for the ``/api/graph/analyze/*`` and
``/api/graph/search/*`` REST route families.

CONCEPT:AU-KG.query.typed-rest-schemas — these routes are mounted in
``agent_utilities/mcp/kg_server.py::_mount_rest_routes`` (~lines 5043-5300) with raw
Starlette ``app.add_route(...)`` handlers that carry no schema, so none of them appear
in the live OpenAPI document. Every model below was derived by reading the actual
handler for the route (what it pulls out of the request body / query string) and, for
the handlers that dispatch through ``_execute_tool("<tool>", **kwargs)``, the signature
of the underlying registered tool in ``agent_utilities/mcp/tools/`` — not invented.

Response shape, once you follow the dispatch chain, turns out to be uniform for both
families:

* Every ``/api/graph/analyze/*`` route here ultimately calls one of the
  ``graph_analyze`` / ``graph_code`` / ``graph_research`` / ``graph_evaluate`` /
  ``graph_explain`` tools (``agent_utilities/mcp/tools/analysis_tools.py`` and
  ``analyze_suite.py``), every one of which returns an
  :class:`~agent_utilities.models.evidence_bundle.EvidenceBundle`. The handler then does
  ``safe_json_load(res)`` (``kg_server.py::safe_json_load``), which calls
  ``res.model_dump()`` for anything with a ``model_dump`` method — so the REST
  ``result`` field is always that EvidenceBundle's dict form. Hence one shared
  :class:`AnalyzeResponse` for the whole family.
* Every ``/api/graph/search/*`` route here calls the ``graph_search`` tool
  (``agent_utilities/mcp/tools/query_tools.py``), which returns a plain, already
  human-formatted **string** (score-sorted result blocks, a "No results found for
  query: ..." message, an "Error: Unknown search mode '...'" message, or a
  newline-joined capability list for ``mode='discover'``) — never JSON. Passed through
  ``safe_json_load``, a non-JSON string round-trips unchanged. Hence one shared
  :class:`SearchTextResponse`.

Both success envelopes are ``{"status": "success", "result": ...}``
(``JSONResponse({"status": "success", "result": safe_json_load(res)})`` — identical
across every handler in this module). The error envelope (raised exceptions) is a
separate, already-typed ``OperationResult``-shaped payload produced by
``agent_utilities.security.error_surface.public_error_payload`` — out of scope here
since it is common to the whole gateway, not specific to analyze/search, and already
has its own schema (``agent_utilities/protocols/epistemic_operations/_generated.py``).

**Permissiveness.** Two request bodies are dispatched to a tool with a fixed signature
and no ``**kwargs`` (``graph_analyze``, ``graph_search``) via a raw ``**body`` splat in
``_execute_tool``; ``kg_server._validate_tool_kwargs_against_signature`` (U-74) already
rejects any field outside that signature with a 400 today, so :class:`GraphAnalyzeRequest`
and :class:`GraphSearchRequest` use ``extra="forbid"`` — this mirrors, not tightens,
existing behavior. Every other request model here backs a hand-written handler that
pulls named keys out of the body with ``.get(...)`` and silently ignores anything else
— those use ``extra="allow"`` so a caller sending extra/legacy fields today keeps
working unchanged.

Two route bodies are read but never actually used by their handler
(``/graph/analyze/evolve-model``, ``/graph/analyze/forecast``, ``/graph/analyze/causal``,
``/graph/analyze/invariant`` don't call ``request.json()`` at all) — modeled as
:class:`NoParamsRequest`, an intentionally empty, ``extra="allow"`` model.

**Unresolved contract note.** ``graph_analyze``'s own tool description advertises a
``distill_memory`` action with ``submit``/``poll_job_id`` parameters, but the tool's
actual signature (``analysis_tools.py`` ``graph_analyze``) only declares
``action``/``query``/``top_k``/``node_id``/``depth``/``target`` — there is no wire field
for ``submit``/``poll_job_id``. Whether ``distill_memory`` is reachable through this
tool at all (vs. only through some other, unmounted surface) could not be determined
from this route's code; :class:`GraphAnalyzeRequest` models the six fields the tool
signature actually declares and does not attempt to guess a ``submit``/``poll_job_id``
wire shape that no handler here reads.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_utilities.models.evidence_bundle import EvidenceBundle

__all__ = [
    "AnalyzeResponse",
    "SearchTextResponse",
    "GraphAnalyzeRequest",
    "ResearchQueryTopKRequest",
    "BlastRadiusQuery",
    "InspectQuery",
    "CallGraphQuery",
    "SimilarCodeQuery",
    "RoutesQuery",
    "AdrRequest",
    "HarnessGateRequest",
    "CodeContextRequest",
    "ExplainRequest",
    "ScopeTopKQuery",
    "ContextRequest",
    "EvaluateRequest",
    "NoParamsRequest",
    "GraphSearchRequest",
    "SearchQueryTopKRequest",
    "SearchDiscoverRequest",
]


# ══════════════════════════════════════════════════════════════════
# Shared response envelopes
# ══════════════════════════════════════════════════════════════════


class AnalyzeResponse(BaseModel):
    """Success envelope shared by every ``/api/graph/analyze/*`` route in this module.

    Every handler here ends with
    ``JSONResponse({"status": "success", "result": safe_json_load(res)})`` where
    ``res`` is the :class:`~agent_utilities.models.evidence_bundle.EvidenceBundle`
    returned by the ``graph_analyze``/``graph_code``/``graph_research``/
    ``graph_evaluate``/``graph_explain`` tool the route dispatches to — every one of
    those tools is typed ``-> EvidenceBundle`` (see ``analysis_tools.py``,
    ``analyze_suite.py``). ``safe_json_load`` calls ``.model_dump()`` on it, so
    ``result`` is that bundle's dict form, not a free-form blob.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": "success",
                    "result": {
                        "answer_candidate": "graph-os has 47 registered tools.",
                        "claims": [],
                        "evidence_spans": [],
                        "source_authority": {},
                        "contradictions": [],
                        "confidence": None,
                        "freshness": {},
                        "policy_exclusions": [],
                        "reasoning_trace": [],
                        "next_actions": [],
                        "error": None,
                    },
                }
            ]
        }
    )

    status: str = Field(
        description="Always 'success' on this envelope; a failed call raises before "
        "reaching this response and is instead surfaced as a non-2xx "
        "OperationResult-shaped error payload (see module docstring)."
    )
    result: EvidenceBundle = Field(
        description="The EvidenceBundle produced by the underlying analysis tool: "
        "answer_candidate, supporting claims/evidence_spans, confidence, "
        "freshness, and reasoning_trace. See EvidenceBundle's own docstring "
        "(agent_utilities/models/evidence_bundle.py) for the full contract."
    )


class SearchTextResponse(BaseModel):
    """Success envelope shared by every ``/api/graph/search/*`` route in this module.

    The ``graph_search`` tool (``query_tools.py``) returns a plain, already
    human-formatted string — never structured JSON — so ``result`` is genuinely
    free-form text here, not an object. Typical shapes: newline/``---``-separated
    ``"[Type] Name (ID: ...) - Score: 0.87\\n<description>"`` blocks, a bare
    ``"No results found for query: '<query>'"`` / ``"Error: Unknown search mode
    '<mode>'"`` message, or (``mode='discover'``) a ``"- name: description"`` list.
    A caller wanting structured results should prefer the graph_search
    ``mode='compiled'`` path (which returns a citation-bearing text bundle) or a
    typed ``/api/graph/analyze/*`` route instead of parsing this text.
    """

    status: str = Field(description="Always 'success' on this envelope.")
    result: str = Field(
        description="The raw formatted search result text (or a 'No results found' / "
        "'Error: Unknown search mode' message on a benign miss — those are NOT "
        "HTTP errors, they are 200s with an explanatory result string)."
    )


# ══════════════════════════════════════════════════════════════════
# POST /api/graph/analyze  (graph_analyze_endpoint → _execute_tool("graph_analyze", **body))
# ══════════════════════════════════════════════════════════════════


class GraphAnalyzeRequest(BaseModel):
    """Body of ``POST /api/graph/analyze`` — the ``graph_analyze`` tool's structural/
    operational action surface (``analysis_tools.py::graph_analyze``). The endpoint
    splats the JSON body directly onto the tool's exact signature (``action``,
    ``query``, ``top_k``, ``node_id``, ``depth``, ``target``); any other field is
    already rejected today with HTTP 400 (U-74's ``UnsupportedToolFieldError``), so
    ``extra='forbid'`` here mirrors existing behavior rather than tightening it.
    """

    model_config = ConfigDict(extra="forbid")

    action: str = Field(
        default="inspect",
        description="One of: inspect | enrichment_coverage | process_writeback | "
        "placement_plan | infra_sweep | security_scan | distill_memory | readiness. "
        "An action outside this set is NOT rejected — the tool returns a 200 with a "
        "graceful EvidenceBundle carrying an 'error' field instead — so this is "
        "documented as plain str rather than a Literal to avoid rejecting a value "
        "the live tool currently tolerates.",
    )
    query: str = Field(
        default="", description="Query or path the analysis action operates on."
    )
    top_k: int = Field(default=10, description="Result or complexity bound.")
    node_id: str = Field(default="", description="Optional anchor node id.")
    depth: int = Field(default=2, description="Traversal depth.")
    target: str = Field(default="", description="Analysis target (action-specific).")


# ══════════════════════════════════════════════════════════════════
# POST /api/graph/analyze/synthesize, /api/graph/analyze/deep-extract
#   → _execute_tool("graph_research", action=<fixed>, query=..., top_k=...)
# ══════════════════════════════════════════════════════════════════


class ResearchQueryTopKRequest(BaseModel):
    """Body shared by ``POST /api/graph/analyze/synthesize`` (``graph_research``
    action ``synthesize``) and ``POST /api/graph/analyze/deep-extract`` (action
    ``deep_extract``). Both handlers read only ``query``/``top_k`` from the body via
    ``.get(...)`` and silently ignore any other key, so ``extra='allow'`` preserves
    that today's callers sending extra fields keep working unchanged.
    """

    model_config = ConfigDict(extra="allow")

    query: str = Field(
        default="", description="Source / topic for the research action."
    )
    top_k: int = Field(default=10, description="Result count / complexity budget.")


# ══════════════════════════════════════════════════════════════════
# GET /api/graph/analyze/blast-radius, /inspect, /call-graph, /similar-code, /routes,
#     /code-metrics, /arch-report — query-string parameter groupings.
# ══════════════════════════════════════════════════════════════════


class BlastRadiusQuery(BaseModel):
    """Query parameters of ``GET /api/graph/analyze/blast-radius`` — dispatches to
    ``graph_code`` action ``blast_radius`` (transitive impact of a node)."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(
        default="", description="The :Code node id to compute blast radius for."
    )
    depth: int = Field(default=2, description="Traversal depth bound.")


class InspectQuery(BaseModel):
    """Query parameters of ``GET /api/graph/analyze/inspect`` — dispatches to
    ``graph_analyze`` action ``inspect``."""

    model_config = ConfigDict(extra="allow")

    target: str = Field(default="", description="What to inspect (action-specific).")


class CallGraphQuery(BaseModel):
    """Query parameters of ``GET /api/graph/analyze/call-graph`` — dispatches to
    ``graph_code`` action ``call_graph`` (the type/scope-resolved call/inheritance
    graph for a symbol). The handler accepts the traversal direction under either
    ``direction`` or the legacy ``target`` query-param name (``direction`` wins when
    both are present); this model keeps both fields to preserve that today.
    """

    model_config = ConfigDict(extra="allow")

    id: str = Field(default="", description="The symbol (:Code node) id.")
    direction: str | None = Field(
        default=None,
        description="callees | callers | inherits. Preferred name; wins over "
        "'target' when both are supplied.",
    )
    target: str | None = Field(
        default=None,
        description="Legacy alias for 'direction', used only when 'direction' is "
        "absent. Defaults to 'callees' when neither is supplied.",
    )


class SimilarCodeQuery(BaseModel):
    """Query parameters of ``GET /api/graph/analyze/similar-code`` — dispatches to
    ``graph_code`` action ``similar_code`` (model-free MinHash/LSH near-clone
    neighbours of a symbol)."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(default="", description="The symbol (:Code node) id.")
    top_k: int = Field(default=10, description="Max neighbours to return.")


class RoutesQuery(BaseModel):
    """Query parameters of ``GET /api/graph/analyze/routes`` — dispatches to
    ``graph_code`` action ``routes`` with NO parameters (the HTTP route→handler→
    service graph is returned in full every time). Intentionally empty; kept as a
    model for documentation symmetry with the other GET routes in this family.
    """

    model_config = ConfigDict(extra="allow")


class ScopeTopKQuery(BaseModel):
    """Query parameters shared by ``GET /api/graph/analyze/code-metrics``
    (``graph_code`` action ``code_metrics`` — god nodes / communities / bridges over
    the :Code subgraph) and ``GET /api/graph/analyze/arch-report`` (action
    ``arch_report`` — the regenerable architecture report). Both handlers resolve the
    scope from either ``scope`` or the legacy ``target`` query-param name (``scope``
    wins when both are present, empty string when neither is supplied).
    """

    model_config = ConfigDict(extra="allow")

    scope: str | None = Field(
        default=None,
        description="Optional file_path/source_system substring to scope the "
        "report to. Preferred name; wins over 'target' when both are supplied.",
    )
    target: str | None = Field(
        default=None,
        description="Legacy alias for 'scope', used only when 'scope' is absent.",
    )
    top_k: int = Field(default=10, description="Section/result size bound.")


# ══════════════════════════════════════════════════════════════════
# POST /api/graph/analyze/adr, /harness-gate, /code-context, /explain, /context,
#      /evaluate
# ══════════════════════════════════════════════════════════════════


class AdrRequest(BaseModel):
    """Body of ``POST /api/graph/analyze/adr`` — Architecture Decision Record CRUD
    via ``graph_code`` action ``adr``. Supplying ``title`` creates/updates a record;
    an empty body lists existing ADRs. The wire field names (``title``/``status``/
    ``decision``) are remapped by the handler onto the tool's generic ``query``/
    ``target``/``node_id`` parameters respectively.
    """

    model_config = ConfigDict(extra="allow")

    title: str = Field(
        default="", description="ADR title. Non-empty creates a new ADR record."
    )
    status: str = Field(
        default="", description="ADR status filter/update (e.g. proposed, accepted)."
    )
    decision: str = Field(default="", description="The decision text/id for this ADR.")


class HarnessGateRequest(BaseModel):
    """Body of ``POST /api/graph/analyze/harness-gate`` — validates a candidate
    harness-evolution state against the concentration/no-regression/pathology SHACL
    gate (``graph_evaluate`` action ``harness_gate``). The handler forwards the WHOLE
    JSON body verbatim (``json.dumps(body)``) as the tool's ``query`` string, so this
    model is intentionally permissive: any body shape the gate logic understands is
    passed through unmodified. ``edits``/``variants``/``pathologies`` are the
    documented top-level keys but are not required by the handler.
    """

    model_config = ConfigDict(extra="allow")

    edits: list[Any] = Field(
        default_factory=list,
        description="Candidate harness edits to gate (shape defined by the SHACL "
        "gate, not this endpoint).",
    )
    variants: list[Any] | None = Field(
        default=None, description="Optional harness variants under evaluation."
    )
    pathologies: list[Any] | None = Field(
        default=None, description="Optional known pathologies to check against."
    )


class CodeContextRequest(BaseModel):
    """Body of ``POST /api/graph/analyze/code-context`` — the synthesized, cited
    codebase Q&A (``graph_code`` action ``code_context``, KG-2.134). ``intent`` and
    ``cross_repo`` are combined by the handler into the tool's single ``target``
    string (``"<intent>+xrepo"`` when ``cross_repo`` is truthy).
    """

    model_config = ConfigDict(extra="allow")

    query: str = Field(default="", description="The question about the codebase.")
    intent: str = Field(
        default="how",
        description="how | usage | impact — the kind of answer wanted.",
    )
    node_id: str | None = Field(
        default=None, description="Optional :Code anchor node id."
    )
    top_k: int = Field(default=10, description="Result count bound.")
    depth: int = Field(default=2, description="Traversal depth bound.")
    cross_repo: bool = Field(
        default=False,
        description="When true, widen the answer across the whole fleet instead of "
        "just this repo (appends '+xrepo' to the resolved intent).",
    )


class ExplainRequest(BaseModel):
    """Body of ``POST /api/graph/analyze/explain`` — the universal context plane:
    routes a question to its domain provider and returns one grounded, cited answer
    (``graph_explain`` action ``explain``). ``domain``/``intent`` are combined by the
    handler into the tool's single ``target`` string (``"<domain>:<intent>"`` when
    ``domain`` is non-empty, else the bare ``intent``).
    """

    model_config = ConfigDict(extra="allow")

    query: str = Field(default="", description="The question to answer.")
    domain: str = Field(
        default="",
        description="Optional explicit domain provider, e.g. 'code', 'ops', "
        "'deploy', 'entity', 'capability'. Inferred from the question when omitted.",
    )
    intent: str = Field(
        default="",
        description="Optional domain-specific intent, e.g. 'usage', 'why', 'status'.",
    )
    node_id: str | None = Field(default=None, description="Optional anchor node id.")
    top_k: int = Field(default=10, description="Result count bound.")
    depth: int = Field(default=2, description="Traversal depth bound.")


class ContextRequest(BaseModel):
    """Body of ``POST /api/graph/analyze/context`` — a synthesized context bundle
    (``graph_explain`` action ``context``)."""

    model_config = ConfigDict(extra="allow")

    target: str = Field(default="", description="What to build context around.")
    query: str = Field(default="", description="The question driving the context.")
    top_k: int = Field(default=10, description="Result count bound.")


class EvaluateRequest(BaseModel):
    """Body of ``POST /api/graph/analyze/evaluate`` — score outputs against the
    reliability/eval corpus (``graph_evaluate`` action ``evaluate``). The handler
    reads only ``target``; the underlying tool's ``query``/``top_k``/``node_id``/
    ``depth`` parameters are not reachable from this route.
    """

    model_config = ConfigDict(extra="allow")

    target: str = Field(
        default="", description="The subject being evaluated (id / JSON / state)."
    )


# ══════════════════════════════════════════════════════════════════
# POST /api/graph/analyze/evolve-model, /forecast, /causal, /invariant
#   → _execute_tool("graph_evaluate", action=<fixed>)  — body is NEVER read.
# ══════════════════════════════════════════════════════════════════


class NoParamsRequest(BaseModel):
    """Body of ``POST /api/graph/analyze/evolve-model``, ``/forecast``, ``/causal``,
    and ``/invariant``. Each handler calls ``_execute_tool("graph_evaluate",
    action=<fixed action>)`` directly and never parses the request body at all — so
    today ANY body (including none, or malformed JSON) is accepted and silently
    ignored. Modeled as an intentionally empty, ``extra='allow'`` shape rather than
    omitting a request model, so the OpenAPI doc for these routes is explicit that
    no body fields are consumed instead of just missing a schema.
    """

    model_config = ConfigDict(extra="allow")


# ══════════════════════════════════════════════════════════════════
# POST /api/graph/search  (graph_search_endpoint → _execute_tool("graph_search", **body))
# ══════════════════════════════════════════════════════════════════


class GraphSearchRequest(BaseModel):
    """Body of ``POST /api/graph/search`` — the full ``graph_search`` tool surface
    (``query_tools.py::graph_search``). Like ``GraphAnalyzeRequest``, the endpoint
    splats the JSON body directly onto the tool's exact signature and U-74 already
    rejects any field outside it with HTTP 400, so ``extra='forbid'`` mirrors, rather
    than tightens, existing behavior.
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        description="Natural language search query or concept ID. Required — the "
        "tool declares this parameter with no default."
    )
    mode: str = Field(
        default="hybrid",
        description="Search strategy. One of: hybrid (default, semantic+keyword) | "
        "hyde | deep | concept | analogy | memory | discover | dci | latent | sira | "
        "hard_negatives | rerank | adore | chrono_ids | compiled. An unrecognized "
        "mode is NOT rejected — the tool returns a 200 with an 'Error: Unknown "
        "search mode ...' result string — so this is plain str rather than a "
        "Literal to avoid rejecting a value the live tool currently tolerates.",
    )
    top_k: int = Field(default=10, description="Maximum results to return.")
    self_correct: bool = Field(
        default=False,
        description="Run a self-correcting second retrieval pass at the deep "
        "threshold when the quality gate fails.",
    )
    as_of: str = Field(
        default="",
        description="Optional ISO-8601 instant; recency decay is measured relative "
        "to this time instead of now.",
    )
    connection: str = Field(
        default="",
        description="Named backend connection to search (default = primary), or "
        "'all'/a comma-separated list to fan out across connections.",
    )
    graph: str = Field(
        default="",
        description="Explicit physical engine graph, independent of 'connection'. "
        "Empty = the caller's own bound graph.",
    )
    token_budget: int = Field(
        default=0,
        description="mode='compiled' only: token budget the assembled bundle must "
        "fit inside. 0 uses the compiler's default budget.",
    )


# ══════════════════════════════════════════════════════════════════
# POST /api/graph/search/concept, /analogy, /memory, /dci, /discover
#   → _execute_tool("graph_search", query=..., mode=<fixed>, top_k=...)
# ══════════════════════════════════════════════════════════════════


class SearchQueryTopKRequest(BaseModel):
    """Body shared by ``POST /api/graph/search/concept``, ``/analogy``, ``/memory``,
    and ``/dci`` — each fixes ``graph_search``'s ``mode`` to its own route name and
    reads only ``query``/``top_k`` from the body via ``.get(...)``, silently ignoring
    anything else (hence ``extra='allow'``).
    """

    model_config = ConfigDict(extra="allow")

    query: str = Field(default="", description="Natural language search query.")
    top_k: int = Field(default=10, description="Maximum results to return.")


class SearchDiscoverRequest(BaseModel):
    """Body of ``POST /api/graph/search/discover`` — cross-references ``query``
    against all ingested content via ``graph_search`` mode ``discover``. Unlike its
    sibling granular search routes, the handler does not read ``top_k`` at all (a
    ``top_k`` field in the body is accepted but has no effect), so it is not part of
    this model — only ``query`` is read.
    """

    model_config = ConfigDict(extra="allow")

    query: str = Field(default="", description="Natural language search query.")
