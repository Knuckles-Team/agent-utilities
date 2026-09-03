"""Pydantic request/response models for the core ``/api/graph/*`` REST surface
plus its non-graph siblings (SCHEMA LANE 3/3).

``kg_server._mount_rest_routes`` mounts these routes with raw Starlette
``app.add_route(...)`` (see ``agent_utilities/mcp/kg_server.py``), which carries
no request/response schema at all — the live OpenAPI 3.1.0 spec documents
**zero** ``/api/graph/*`` paths as a result. This module is the typed
prerequisite for wiring ``response_model=``/body-typed parameters onto those
routes (a later, single-owner lane); it does not touch ``kg_server.py`` itself.

Every model here was derived by reading the actual handler in
``agent_utilities/mcp/kg_server.py`` and, where the handler dispatches through
``_execute_tool("<tool>", **body)``, the real target tool signature in
``REGISTERED_TOOLS`` (``agent_utilities/mcp/tools/*.py``) — never invented.
See each model's docstring for its source handler/tool and the exact
permissiveness contract (``extra="allow"`` vs ``extra="forbid"``) proven
against that handler's code, not assumed.

The top-level ``/tools`` route name is used by two unrelated handlers in this
codebase that happen to share the same leaf path — see
:class:`ToolsCatalogResponse` vs :class:`ToolCatalogItem`.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agent_utilities.models.evidence_bundle import EvidenceBundle

__all__ = [
    # graph_query / graph_query/federated
    "GraphQueryFederatedRequest",
    "EvidenceBundleEnvelope",
    # graph_code / graph_research / graph_evaluate / graph_explain
    "GraphCodeRequest",
    "GraphResearchRequest",
    "GraphEvaluateRequest",
    "GraphExplainRequest",
    # graph_observe
    "GraphObserveRequest",
    "GraphObserveResponse",
    # graph_orchestrate
    "GraphOrchestrateRequest",
    "GraphOrchestrateResponse",
    # graph_configure (generic + the three bespoke granular twins)
    "GraphConfigureRequest",
    "GraphConfigureResponse",
    "GraphConfigureSecretRequest",
    "SetSecretResult",
    "GraphConfigureSecretResponse",
    "GraphConfigureVaultSyncRequest",
    "VaultSyncResult",
    "GraphConfigureVaultSyncResponse",
    "HookDoctorEntry",
    "GraphConfigureDoctorResponse",
    # /tools (two distinct handlers sharing one leaf path — see docstrings)
    "McpServerToolInfo",
    "BuiltinToolInfo",
    "SkillCatalogEntry",
    "ToolsCatalogResponse",
    "ToolCatalogItem",
    # /goals + /goals/{goal_id}/cancel + /goals/{goal_id}/iterations
    "GoalRecord",
    "CreateGoalRequest",
    "CreateGoalResponse",
    "ListGoalsResponse",
    "CancelGoalResponse",
    # /connector/run + /connector/sources
    "ConnectorRunRequest",
    "ConnectorRunResult",
    "ConnectorRunResponse",
    "ConnectorSourcesResult",
    "ConnectorSourcesResponse",
    # /sessions
    "SessionRecord",
    "SessionsListQuery",
]


# ══════════════════════════════════════════════════════════════════
# Shared envelope
# ══════════════════════════════════════════════════════════════════


class EvidenceBundleEnvelope(BaseModel):
    """The ``{"status": "success", "result": <EvidenceBundle>}`` envelope shared
    by ``/graph/query``, ``/graph/query/federated``, ``/graph/code``,
    ``/graph/research``, ``/graph/evaluate``, and ``/graph/explain``.

    Every one of those routes dispatches a tool whose declared return type is
    :class:`~agent_utilities.models.evidence_bundle.EvidenceBundle`; the
    handler serializes it with ``safe_json_load``, which calls
    ``.model_dump()`` on a Pydantic model rather than round-tripping through
    JSON text — so ``result`` is exactly an ``EvidenceBundle`` dump, not a
    freeform dict. Reused verbatim here (not redefined) per the "share one
    definition across identical shapes" rule — see
    ``agent_utilities/models/evidence_bundle.py`` for the fully-described
    field set.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["success"] = Field(
        description="Always 'success' on this path; failures return a "
        "different, non-2xx error body (see `_external_error_response`) "
        "and never reach this envelope shape."
    )
    result: EvidenceBundle = Field(
        description="The dispatched tool's EvidenceBundle answer, dumped to "
        "a plain dict by the handler's `safe_json_load(res)` call."
    )


# ══════════════════════════════════════════════════════════════════
# 1. graph_query / graph_query/federated
# ══════════════════════════════════════════════════════════════════


class GraphQueryFederatedRequest(BaseModel):
    """Request body for ``POST /graph/query/federated`` — a narrower,
    fixed-scope twin of ``graph_query`` (``graph_query_federated_endpoint`` in
    ``kg_server.py``).

    ``extra="forbid"`` mirrors the handler's explicit allowlist for ``query``,
    ``params``, and ``reference_id``. Unknown fields fail before dispatch.

    Also unlike ``/graph/query``, ``params`` here is converted through
    ``_to_json_str`` server-side, so it genuinely accepts EITHER a JSON
    object or a pre-encoded string.
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        default="",
        description="Read-only query string; always run with `scope='federated'` "
        "server-side (not caller-settable on this route).",
    )
    params: dict[str, Any] | str = Field(
        default_factory=dict,
        description="Query parameters as a JSON object OR a pre-encoded "
        "JSON string — the handler normalizes either via `_to_json_str` "
        "before dispatch (unlike the stricter `/graph/query`, which only "
        "accepts a pre-encoded string).",
    )
    reference_id: str = Field(
        default="",
        description="ExternalGraphReference id identifying the federated target.",
    )


# ══════════════════════════════════════════════════════════════════
# 2. graph_code / graph_research / graph_evaluate / graph_explain
#    (the focused analyze-suite tools — all share one action-core shape)
# ══════════════════════════════════════════════════════════════════

GraphCodeAction = Literal[
    "code_context",
    "cross_repo_usages",
    "call_graph",
    "similar_code",
    "routes",
    "change_coupling",
    "code_evolution",
    "blast_radius",
    "code_metrics",
    "arch_report",
    "adr",
]
GraphResearchAction = Literal[
    "synthesize",
    "deep_extract",
    "background_research",
    "relevance_sweep",
    "research_ingest",
    "evolve_variants",
    "track_citations",
    "spawn_background",
    "night_shift",
    "contradictions",
]
GraphEvaluateAction = Literal[
    "evaluate",
    "evaluate_alpha",
    "evaluate_harness",
    "guard_corpus",
    "harness_gate",
    "check_constraints",
    "specialize",
    "world_model_rollout",
    "latent_efficiency_benchmark",
    "assimilation_benchmark",
    "evolve_model",
    "evolve_code",
    "forecast",
    "causal",
    "invariant",
    "quant_crypto",
    "quant_exchange",
    "quant_microstructure",
    "quant_strategy",
    "quant_regime",
    "quant_insider",
]
GraphExplainAction = Literal["explain", "context", "executable_rag", "recommend"]


class GraphCodeRequest(BaseModel):
    """Request body for ``POST /graph/code`` — REST twin of the ``graph_code``
    MCP tool (``agent_utilities/mcp/tools/analyze_suite.py``), mounted via the
    shared ``_make_action_endpoint("graph_code")`` factory in ``kg_server.py``.

    ``extra="forbid"``: PROVEN — the factory blind-splats ``**body`` into
    ``_execute_tool("graph_code", **body)``, but the tool function itself
    declares exactly ``action``/``query``/``top_k``/``node_id``/``depth``/
    ``target`` with no ``**kwargs`` catch-all, so
    ``_validate_tool_kwargs_against_signature`` raises ``UnsupportedToolFieldError``
    (mapped to a deterministic 400) for anything else — the "blind splat" is
    strict in practice, not permissive.
    """

    model_config = ConfigDict(extra="forbid")

    action: GraphCodeAction = Field(
        default="code_context",
        description="Code-intelligence operation to run.",
    )
    query: str = Field(
        default="", description="Question / area / symbol name / repo path."
    )
    top_k: int = Field(default=10, description="Result count.")
    node_id: str = Field(
        default="",
        description="Symbol/:Code node id (used by call_graph, similar_code, "
        "blast_radius).",
    )
    depth: int = Field(default=2, description="Traversal depth (blast_radius).")
    target: str = Field(
        default="",
        description="'how'|'usage'|'impact' (code_context) or "
        "'callees'|'callers'|'inherits' (call_graph).",
    )


class GraphResearchRequest(BaseModel):
    """Request body for ``POST /graph/research`` — REST twin of the
    ``graph_research`` MCP tool. Same factory/strictness proof as
    :class:`GraphCodeRequest` (``extra="forbid"``, verified against the tool's
    signature, no ``**kwargs``).
    """

    model_config = ConfigDict(extra="forbid")

    action: GraphResearchAction = Field(
        default="synthesize",
        description="Research/assimilation pipeline operation to run.",
    )
    query: str = Field(default="", description="Source / topic / artifact.")
    top_k: int = Field(default=10, description="Complexity budget / result count.")
    node_id: str = Field(default="", description="Optional node id.")
    depth: int = Field(default=2, description="Optional traversal depth.")
    target: str = Field(
        default="",
        description="Optional target (e.g. a local markdown vault root for "
        "'night_shift').",
    )


class GraphEvaluateRequest(BaseModel):
    """Request body for ``POST /graph/evaluate`` — REST twin of the
    ``graph_evaluate`` MCP tool. Same factory/strictness proof as
    :class:`GraphCodeRequest` (``extra="forbid"``).
    """

    model_config = ConfigDict(extra="forbid")

    action: GraphEvaluateAction = Field(
        default="evaluate",
        description="Evaluation / world-model / forecasting operation to run.",
    )
    query: str = Field(
        default="",
        description="Subject of the evaluation (JSON / id / start state).",
    )
    top_k: int = Field(default=10, description="Steps / result count.")
    node_id: str = Field(default="", description="Optional node id.")
    depth: int = Field(default=2, description="Optional traversal depth.")
    target: str = Field(default="", description="Optional target.")


class GraphExplainRequest(BaseModel):
    """Request body for ``POST /graph/explain`` — REST twin of the
    ``graph_explain`` MCP tool (the universal cited-context plane). Same
    factory/strictness proof as :class:`GraphCodeRequest` (``extra="forbid"``).
    """

    model_config = ConfigDict(extra="forbid")

    action: GraphExplainAction = Field(
        default="explain",
        description="'explain' (one cited answer), 'context' (synthesized "
        "context bundle), 'executable_rag' (grounded multi-hop retrieval), "
        "or 'recommend' (ranked items, not a single answer).",
    )
    query: str = Field(
        default="", description="The question, or intent/history for 'recommend'."
    )
    top_k: int = Field(default=10, description="Result count.")
    node_id: str = Field(default="", description="Optional anchor node id.")
    depth: int = Field(default=2, description="Optional traversal depth.")
    target: str = Field(
        default="",
        description="'domain:intent' (e.g. 'ops:why', 'code:usage'), a bare "
        "intent with the domain inferred, or 'domains' to list providers.",
    )


# ══════════════════════════════════════════════════════════════════
# 3. graph_observe
# ══════════════════════════════════════════════════════════════════


class GraphObserveRequest(BaseModel):
    """Request body for ``POST /graph/observe`` — REST twin of the
    ``graph_observe`` MCP tool (``agent_utilities/mcp/tools/analyze_suite.py``),
    mounted via the same ``_make_action_endpoint`` factory.

    ``extra="forbid"``: PROVEN the same way as :class:`GraphCodeRequest` — the
    tool signature is exactly ``action``/``query``/``top_k`` with no
    ``**kwargs``.
    """

    model_config = ConfigDict(extra="forbid")

    action: Literal[
        "trace_rootcause", "prompt_regression", "failure_cluster", "error_detail"
    ] = Field(
        default="trace_rootcause",
        description="Observability query to run over the trace/score subgraph.",
    )
    query: str = Field(
        default="",
        description="Agent/capability filter for 'trace_rootcause', or the "
        "opaque `error.detail_ref` to resolve for 'error_detail'.",
    )
    top_k: int = Field(default=20, description="Max rows/clusters to return.")


class GraphObserveResponse(BaseModel):
    """Response envelope for ``POST /graph/observe``.

    Unlike the EvidenceBundle-returning tools above, ``graph_observe``'s own
    declared return type is a plain ``str``: for 'trace_rootcause'/
    'prompt_regression'/'failure_cluster' it is ``json.dumps(...)`` of a
    structured analytics payload (JSON-parsed by `safe_json_load` into a
    dict); for 'error_detail' it can ALSO return a plain non-JSON prose
    string (a permission-denied message, or "Error: Unknown observe
    action ..."), which `safe_json_load` passes through unparsed. `result`
    is genuinely free-form on the dict side (trace analytics rows vary by
    query) — modeled permissively rather than inventing structure.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["success"] = Field(description="Always 'success' on this path.")
    result: dict[str, Any] | str = Field(
        description="A structured analytics payload (dict, shape depends on "
        "`action` — free-form, not modeled further) for the three trace "
        "actions, OR a plain prose string for 'error_detail' failures / an "
        "unknown-action message."
    )


# ══════════════════════════════════════════════════════════════════
# 4. graph_orchestrate
# ══════════════════════════════════════════════════════════════════


class GraphOrchestrateRequest(BaseModel):
    """Request body for ``POST /graph/orchestrate`` — REST twin of the
    ``graph_orchestrate`` MCP tool (``agent_utilities/mcp/tools/analysis_tools.py``),
    which resolves and executes one governed local-vLLM delegation.

    ``extra="forbid"``: PROVEN — ``graph_orchestrate_endpoint`` blind-splats
    ``**body`` into ``_execute_tool("graph_orchestrate", **body)``, but the
    tool's signature (17 named parameters, no ``**kwargs``) means an unknown
    field raises ``UnsupportedToolFieldError`` deep inside `_execute_tool`
    (the endpoint's generic ``except Exception`` still maps it to a REST
    error, just not the deterministic-400 shape the factory-based endpoints
    above get — worth a follow-up, out of scope for this schema-only lane).
    """

    model_config = ConfigDict(extra="forbid")

    task: str = Field(default="", description="Task for the delegated agent.")
    agent_name: str = Field(
        default="",
        description="Optional registered agent name. Empty resolves the "
        "best ingested skill/workflow/fleet-tool from the KG's unified "
        "capability ranking.",
    )
    skill_name: str = Field(
        default="",
        description="Exact ingested AGENT_SKILL name. Mutually exclusive "
        "with `agent_name`.",
    )
    tool_server: str = Field(
        default="",
        description="Exact configured MCP server catalog entry to bind to "
        "`skill_name`, or the server hosting a resolved/ranked fleet tool.",
    )
    execution_mode: Literal["auto", "pydantic_graph"] = Field(
        default="auto",
        description="'auto' selects the leanest governed route; "
        "'pydantic_graph' requires `skill_name`, `tool_server`, and a "
        "non-empty `allowed_tools`.",
    )
    max_steps: int = Field(default=30, description="Maximum delegated tool-loop steps.")
    context: str = Field(
        default="",
        description="Curated inline context injected into the delegated agent.",
    )
    budget_tokens: int = Field(
        default=0, description="Hard total-token budget; zero uses the runtime default."
    )
    context_ref: str = Field(
        default="", description="Persisted ContextBlob id to resolve and inject."
    )
    allowed_tools: str | list[str] = Field(
        default="",
        description="Least-privilege tool allow-list as a JSON list or "
        "comma-separated string.",
    )
    required_tools: str | list[str] = Field(
        default="",
        description="Tools that must each have recorded ToolCall provenance "
        "before the run can succeed. Must be a subset of `allowed_tools`.",
    )
    cred_ref: str = Field(
        default="",
        description="Reference to an ephemeral credential in the secrets backend.",
    )
    open_channel: bool = Field(
        default=False,
        description="Open a native bidirectional message channel for this run.",
    )
    reasoning_effort: str = Field(
        default="",
        description="'low'/'medium'/'high' to turn reasoning ON for a "
        "delegated execution; empty = off / inherit the model's setting.",
    )
    model_class: str = Field(
        default="standard",
        description="Required configured model class: 'economy' | 'standard'.",
    )
    response_format: Literal["text", "json"] = Field(
        default="text",
        description="'text' for ordinary prose, 'json' for one "
        "Pydantic-validated JSON object.",
    )
    grounding: Literal["required", "best_effort", "none"] = Field(
        default="required",
        description="'required' fails the run closed if mandatory evidence "
        "compilation fails; 'best_effort'/'none' explicitly opt into "
        "degraded, marked-as-such operation.",
    )


class GraphOrchestrateResponse(BaseModel):
    """Response envelope for ``POST /graph/orchestrate``.

    The tool's own return type is ``str`` (``json.dumps(payload)``); the
    handler JSON-parses it via ``safe_json_load``. `payload` is explicitly
    documented by the tool as including "resolution, run/session handles,
    tool-call provenance, and any approval request" plus `output`/`mermaid` —
    genuinely free-form across execution modes, so `result` is modeled
    permissively rather than inventing a fixed key set.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["success"] = Field(description="Always 'success' on this path.")
    result: dict[str, Any] = Field(
        description="Free-form delegation result — resolution info, run/session "
        "handles, tool-call provenance, any approval request, plus `output` "
        "and `mermaid` keys (always present, possibly null/empty)."
    )


# ══════════════════════════════════════════════════════════════════
# 5. graph_configure (generic + the three bespoke granular twins this lane
#    owns: doctor, secret, vault-sync)
# ══════════════════════════════════════════════════════════════════


class GraphConfigureRequest(BaseModel):
    """Request body for ``POST /graph/configure`` — REST twin of the
    ``graph_configure`` MCP tool (``agent_utilities/mcp/tools/analysis_tools.py``).

    ``extra="forbid"``: PROVEN — ``graph_configure_endpoint`` blind-splats
    ``**body`` into ``_execute_tool("graph_configure", **body)``, and the
    tool's signature is exactly ``action``/``config_key``/``config_value``
    with no ``**kwargs``, so an unrecognized field is rejected deep inside
    `_execute_tool` the same way as `graph_orchestrate` above.

    ⚠ SECRET HANDLING: `config_value` is the generic payload/secret carrier
    for actions like `set_secret`/`vault_sync` — see the bespoke
    :class:`GraphConfigureSecretRequest`/:class:`GraphConfigureVaultSyncRequest`
    models below for the routes that exist specifically to carry secret
    material, and for confirmation that the RESPONSE never echoes it back.
    This generic model is documented as a general configuration surface, not
    a secrets endpoint, and should not be used to build examples containing
    real credentials.
    """

    model_config = ConfigDict(extra="forbid")

    action: str = Field(
        default="register_mcp",
        description="Configuration operation. Not a closed enum in the tool "
        "itself (plain `str`, not `Literal`) — known values include "
        "set_secret, vault_sync, register_mcp, install_hooks, "
        "uninstall_hooks, doctor (note: handled but NOT listed in the "
        "tool's own description string — a minor doc gap found while "
        "deriving this model), config_doctor, system_doctor, health, "
        "preflight, generate_config, get_config, set_config, list_config, "
        "add_connection, remove_connection, list_connections, "
        "discover_connection_schema, and the Stardog "
        "push_to_stardog/pull_from_stardog/stardog_sparql/"
        "stardog_export_graph/stardog_import_graph actions.",
        json_schema_extra={"examples": ["doctor", "list_connections"]},
    )
    config_key: str = Field(
        default="",
        description="The key/ID of the configuration or secret being acted "
        "on (connection name, schema-pack name, etc. — action-dependent).",
    )
    config_value: str = Field(
        default="",
        description="JSON-encoded string payload for the action — MAY carry "
        "secret material for `set_secret`/`vault_sync` (see the bespoke "
        "models below); never logged or echoed by this tool's own success "
        "responses.",
    )


class GraphConfigureResponse(BaseModel):
    """Response envelope for ``POST /graph/configure``.

    The tool returns ``str`` (JSON-encoded); `result` shape varies per
    `action` (the tool has 20+ branches with different return shapes) —
    modeled permissively rather than inventing one structure.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["success"] = Field(description="Always 'success' on this path.")
    result: dict[str, Any] | str = Field(
        description="Action-dependent payload; free-form dict for nearly "
        "every action, occasionally a bare string."
    )


class GraphConfigureSecretRequest(BaseModel):
    """Request body for ``POST /graph/configure/secret`` — REST twin of
    ``graph_configure(action='set_secret')`` (``graph_configure_secret_endpoint``
    in ``kg_server.py``).

    ``extra="allow"``: PROVEN — the handler reads exactly `config_key` and
    `config_value` via `body.get(...)` and forwards only those two as fixed
    kwargs; any other body field is silently ignored, never rejected.

    ⚠ SECRETS: `config_value` IS the raw secret value being written. This
    field is intentionally undocumented with any example value, and no
    field on this model or :class:`GraphConfigureSecretResponse` echoes it —
    confirmed by reading the handler chain down to
    ``create_secrets_client().set(config_key, config_value)``, whose success
    path returns only ``{"status": "success", "action": "set_secret",
    "stored": true}``.
    """

    model_config = ConfigDict(extra="allow")

    config_key: str = Field(
        default="",
        description="The secret's key/name (e.g. an env var name, or "
        "'xai/...' to route to the xAI-scoped secrets client). NOT a secret "
        "value itself.",
    )
    config_value: str = Field(
        default="",
        description="THE SECRET VALUE to store. Write-only: never returned "
        "in any response. Do not populate this field with example data in "
        "documentation or client generators.",
    )


class SetSecretResult(BaseModel):
    """The exact, verified `result` payload of a successful `set_secret` —
    deliberately minimal so nothing beyond this shape is ever assumed safe
    to surface for a secrets-write endpoint.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["success"] = Field(description="Always 'success' on this path.")
    action: Literal["set_secret"] = Field(description="Echoes the action performed.")
    stored: bool = Field(description="Whether the secret was written. Never the value.")


class GraphConfigureSecretResponse(BaseModel):
    """Response envelope for ``POST /graph/configure/secret``. See
    :class:`SetSecretResult` — this NEVER carries the secret value.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["success"] = Field(description="Always 'success' on this path.")
    result: SetSecretResult = Field(
        description="Confirmation only — no secret material."
    )


class GraphConfigureVaultSyncRequest(BaseModel):
    """Request body for ``POST /graph/configure/vault-sync`` — REST twin of
    ``graph_configure(action='vault_sync')`` (``graph_configure_vault_sync_endpoint``
    in ``kg_server.py``, CONCEPT:AU-OS.deployment.vault-first-routine-genesis).

    ``extra="allow"``: PROVEN the same way as :class:`GraphConfigureSecretRequest`
    — only `config_key`/`config_value` are read.

    ⚠ SECRETS: `config_value` is a JSON-encoded string of
    ``{"env_keys": [...], "values": {KEY: VALUE}, "overwrite": bool}`` —
    `values` MAY carry raw secret material being seeded. Confirmed safe on
    the way out: :class:`VaultSyncResult` below only ever contains
    ``vault://<service>/<KEY>`` REFERENCES and key NAMES (`present`/
    `written`/`missing`), never the values themselves (verified against
    ``SecretsClient.vault_sync``'s docstring and return statement).
    """

    model_config = ConfigDict(extra="allow")

    config_key: str = Field(
        default="",
        description="Logical service name (the `apps/<service>` secrets-store path segment).",
    )
    config_value: str = Field(
        default="",
        description='JSON-encoded `{"env_keys": [...], "values": {KEY: VALUE}, '
        '"overwrite": bool}`. `values` MAY carry raw secret material being '
        "seeded — write-only, never echoed back (see `VaultSyncResult`).",
        json_schema_extra={
            "examples": ['{"env_keys": ["API_TOKEN"], "overwrite": false}']
        },
    )


class VaultSyncResult(BaseModel):
    """The exact, verified `result` payload of a successful `vault_sync` —
    only vault:// reference URIs and key NAMES, never secret values (see
    :class:`GraphConfigureVaultSyncRequest` docstring for how this was
    confirmed).
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["success"] = Field(description="Always 'success' on this path.")
    action: Literal["vault_sync"] = Field(description="Echoes the action performed.")
    service: str = Field(description="The service name this sync ran against.")
    refs: dict[str, str] = Field(
        description="`{KEY: 'vault://<service>/<KEY>'}` for every env key — "
        "resolvable references, never the underlying values."
    )
    present: list[str] = Field(
        description="Env key names that already existed in the store and were kept as-is."
    )
    written: list[str] = Field(
        description="Env key names that were just written by this call."
    )
    missing: list[str] = Field(
        description="Env key names with neither a stored value nor a supplied one."
    )


class GraphConfigureVaultSyncResponse(BaseModel):
    """Response envelope for ``POST /graph/configure/vault-sync``. See
    :class:`VaultSyncResult` — references and key names only, never values.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["success"] = Field(description="Always 'success' on this path.")
    result: VaultSyncResult = Field(
        description="References + key names only — no secret material."
    )


class HookDoctorEntry(BaseModel):
    """One agent surface's hook-installation health, as returned by
    ``HookInstaller.doctor()`` (``agent_utilities/ecosystem/hook_installer.py``).
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Human-readable agent-surface name.")
    path: str | None = Field(
        description="Filesystem path to that surface's hook config, or null "
        "when the surface has no on-disk config (e.g. an integrated surface)."
    )
    status: Literal["healthy", "stale", "not_installed", "integrated", "n/a"] = Field(
        description="'healthy' (contains 'agent-utilities'), 'stale' (config "
        "exists but doesn't reference agent-utilities), 'not_installed' (no "
        "file yet), 'integrated' (agent-terminal-ui, no separate config "
        "file), or 'n/a'."
    )
    size_bytes: int | None = Field(
        default=None,
        description="Config file size in bytes; present only when `path` "
        "exists on disk.",
    )


class GraphConfigureDoctorResponse(BaseModel):
    """Response envelope for ``POST /graph/configure/doctor`` — REST twin of
    ``graph_configure(action='doctor')`` (``graph_configure_doctor_endpoint``
    in ``kg_server.py``, no request body). `result` is keyed by internal
    agent-surface identifiers (e.g. 'claude-code', 'agent-terminal-ui') —
    modeled as an open mapping rather than a closed key set since the agent
    surface roster can grow.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["success"] = Field(description="Always 'success' on this path.")
    result: dict[str, HookDoctorEntry] = Field(
        description="Agent-surface id -> hook health entry."
    )


# ══════════════════════════════════════════════════════════════════
# 6. /tools — TWO DISTINCT HANDLERS SHARE THIS LEAF PATH
#
# `_mount_rest_routes` (kg_server.py) mounts `GET /tools` -> `get_tools_endpoint`,
# which returns a CATEGORIZED dict (`_ToolsPayload`: mcp_tools/builtin_tools/
# skills/skill_graphs/skill_workflows/section_status). Under the gateway this
# is reachable at `/api/tools` (`register_graph_routes(app, prefix="/api")`).
#
# SEPARATELY, `agent_utilities/server/routers/core.py` registers its OWN
# `GET /tools` (`list_tools`) directly on the app with NO prefix, returning a
# FLAT `list[dict]` of Tool+Skill KG nodes. THIS is the one
# `agent-terminal-ui/agent_terminal_ui/client.py:534`'s `AgentClient.list_tools()`
# actually calls (confirmed via
# `agent-terminal-ui/tests/test_graph_route_wiring.py::
# test_the_servers_own_unprefixed_routers_keep_their_paths`, which asserts
# `/tools`, not `/api/tools`, and treats it as one of "the server's own
# unprefixed routers").
#
# Both are modeled below since this lane's brief explicitly named `/tools`
# among "YOUR ROUTES" (the `_mount_rest_routes` table) AND explicitly warned
# about the agent-terminal-ui flat-list contract — which turned out to
# belong to the OTHER handler. See the report for this lane for the full
# writeup; `agent_utilities/server/routers/core.py` is out of this lane's
# edit scope (not on the touch list), so wiring either response_model in is
# left to the lane that owns that file / kg_server.py.
# ══════════════════════════════════════════════════════════════════


class McpServerToolInfo(BaseModel):
    """One entry in `ToolsCatalogResponse.mcp_tools`. Despite the field name,
    this has always been one row per CONFIGURED MCP SERVER (an `mcpServers`
    entry), never an individual MCP tool — see `_build_tools_payload_sync`'s
    docstring in `kg_server.py`.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="The configured MCP server's name.")
    type: Literal["MCP Server"] = Field(description="Constant discriminator.")
    launch_mode: Literal["subprocess", "remote"] = Field(
        description="'subprocess' for a stdio-transport server, 'remote' otherwise."
    )
    command: str = Field(
        description="'[configured]' for a stdio server (the raw command is "
        "never exposed — privacy), else empty."
    )
    args: list[str] = Field(
        description="['[configured]'] for a stdio server, else empty — same "
        "opaque-presence-marker rule as `command`."
    )
    status: Literal["active", "disabled"] = Field(
        description="Derived from the per-user toggle preference AND the "
        "catalog row's own configured `enabled` flag."
    )
    enabled: bool = Field(description="Boolean form of `status`.")


class BuiltinToolInfo(BaseModel):
    """One entry in `ToolsCatalogResponse.builtin_tools` — a native,
    in-process Python callable under `agent_utilities/tools/*.py` (never
    MCP-discovered).
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="The tool module's file stem.")
    type: Literal["Built-in Tool"] = Field(description="Constant discriminator.")
    file_path: str = Field(description="Synthetic 'tool://<stem>' locator.")
    status: Literal["enabled", "disabled"] = Field(
        description="Derived from the per-user toggle preference."
    )
    enabled: bool = Field(description="Boolean form of `status`.")


class SkillCatalogEntry(BaseModel):
    """One entry in `ToolsCatalogResponse.skills` / `skill_graphs` /
    `skill_workflows`. Sourced from a `SKILL.md` file's YAML frontmatter
    (`_parse_skill_md`), so the exact key set varies by what a given skill's
    frontmatter declares — modeled with the fields every entry is known to
    carry, `extra="allow"` for the rest rather than dropping real frontmatter
    data.
    """

    model_config = ConfigDict(extra="allow")

    id: str = Field(
        description="Stable skill identifier, used for toggle-state lookups."
    )
    name: str = Field(default="", description="Human-readable skill name.")
    type: Literal["Agent Skill", "Skill Workflow", "Skill Graph"] = Field(
        description="Which of the three corpora this entry came from."
    )
    enabled: bool = Field(description="Per-user toggle preference for this skill.")


class ToolsCatalogResponse(BaseModel):
    """Response body for the `GET /tools` mounted by `_mount_rest_routes`
    (`get_tools_endpoint` in `kg_server.py`) — reachable at `/api/tools`
    under the gateway's default `/api` prefix. See the module-level note
    above `McpServerToolInfo` for why this is NOT the flat-list contract
    agent-terminal-ui depends on (that's :class:`ToolCatalogItem` instead).

    NOT wrapped in a `{"status", "result"}` envelope — this handler returns
    the catalog dict directly as the JSON body.
    """

    model_config = ConfigDict(extra="forbid")

    mcp_tools: list[McpServerToolInfo] = Field(
        description="One entry per configured MCP server (see `McpServerToolInfo`)."
    )
    builtin_tools: list[BuiltinToolInfo] = Field(
        description="One entry per native in-process tool module."
    )
    skills: list[SkillCatalogEntry] = Field(
        description="Atomic AGENT_SKILL entries from the universal-skills corpus."
    )
    skill_graphs: list[SkillCatalogEntry] = Field(
        description="Skill-graph entries from the skill-graphs corpus."
    )
    skill_workflows: list[SkillCatalogEntry] = Field(
        description="Skill-workflow entries from the universal-skills corpus."
    )
    section_status: dict[str, str] = Field(
        description="Per-section 'ok'/'unavailable' read status, so a caller "
        "can distinguish a genuinely empty section from one whose source "
        "could not be read this time."
    )


class ToolCatalogItem(BaseModel):
    """One item in the FLAT list returned by `GET /tools` as registered in
    `agent_utilities/server/routers/core.py::list_tools` — the route
    `agent-terminal-ui`'s `AgentClient.list_tools()` actually calls (see the
    module-level note above `McpServerToolInfo`). Response shape for that
    route is `list[ToolCatalogItem]`, NOT wrapped in any envelope — modeling
    it as one would "improve" it into a breaking change for a shipped
    frontend (`ToolsSidebar._populate_tree` in agent-terminal-ui reads
    `item.get("description", "")` directly off each flat item).
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="The underlying :Tool or :Skill node's id.")
    name: str = Field(description="Display name.")
    description: str = Field(
        default="",
        description="Free-text description — read directly by "
        "agent-terminal-ui's sidebar for both search filtering and the "
        "rendered tree label.",
    )
    source_name: str = Field(
        default="",
        description="The owning MCP server name (tools) or skill category (skills).",
    )
    type: Literal["tool", "skill"] = Field(
        description="Which KG node label this came from."
    )


# ══════════════════════════════════════════════════════════════════
# 7. /goals (GET+POST), /goals/{goal_id}/cancel, /goals/{goal_id}/iterations
# ══════════════════════════════════════════════════════════════════


class GoalRecord(BaseModel):
    """One autonomous goal-loop record, as returned by both `GET /goals`
    (a flat list of these) and `GET /goals/{goal_id}/iterations` (one of
    these, or a 404 error body — not modeled here). Sourced from
    ``agent_utilities/core/sessions.py``'s in-memory `active_goals` entries
    AND the KG-Loop-node fallback (`_goal_row_to_entry`) — the two shapes
    are IDENTICAL except `created_at`, which the KG-derived fallback never
    populates (hence optional here, not required).

    ``extra="allow"``: the in-memory and KG-derived producers are
    independently maintained dicts; nothing in the read path enforces a
    closed key set, so a future field addition to either producer should
    not be silently dropped by this model.
    """

    model_config = ConfigDict(extra="allow")

    goal_id: str = Field(description="The goal's unique id.")
    session_id: str = Field(description="The durable session this goal is attached to.")
    status: str = Field(
        description="Lifecycle status, e.g. 'submitted', 'queued', 'running', "
        "'succeeded', 'failed', 'cancelled', 'ready'."
    )
    objective: str = Field(description="The natural-language goal objective.")
    owner_host: str = Field(
        default="",
        description="The dispatch worker host currently (or last) owning this goal.",
    )
    created_at: float | None = Field(
        default=None,
        description="Unix timestamp the goal was created. Populated for "
        "goals still tracked in-memory on the host that created them; "
        "absent (null) for a goal rehydrated from its KG Loop node only.",
    )
    iterations: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Recorded iteration steps for this goal run (free-form "
        "per-iteration shape).",
    )
    total_iterations: int = Field(
        default=0, description="Count of iterations run so far."
    )
    total_duration_ms: int = Field(
        default=0,
        description="Cumulative wall-clock duration across iterations, in ms.",
    )
    total_tool_calls: int = Field(
        default=0, description="Cumulative tool-call count across iterations."
    )
    summary: str = Field(
        default="", description="Latest human-readable status summary."
    )
    error: str = Field(
        default="", description="Error detail when `status` is 'failed'."
    )


class CreateGoalRequest(BaseModel):
    """Request body for ``POST /goals`` — launches a new backgrounded
    autonomous goal-loop (``create_goal`` in ``agent_utilities/core/sessions.py``).

    ``extra="allow"``: PROVEN — the handler reads exactly `objective`,
    `max_iterations`, `validation_cmd`, and `constraints` via `body.get(...)`;
    any other field is silently ignored.
    """

    model_config = ConfigDict(extra="allow")

    objective: str = Field(
        description="The goal to pursue, in natural language. REQUIRED — an "
        "empty/missing value returns 400.",
        json_schema_extra={"examples": ["Get the failing suite green again"]},
    )
    max_iterations: int | None = Field(
        default=None,
        description="Cap on loop iterations. Omitted/falsy keeps the "
        "GoalSpec's own default.",
    )
    validation_cmd: str | None = Field(
        default=None,
        description="Shell command the loop runs to check whether the "
        "objective is satisfied.",
    )
    constraints: list[Any] | None = Field(
        default=None,
        description="Optional list of constraint entries carried through to "
        "the GoalSpec, sanitized before storage.",
    )


class CreateGoalResponse(BaseModel):
    """Response body for a successful ``POST /goals``. NOT wrapped in a
    `{"status", "result"}` envelope — the handler returns this shape directly.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["success"] = Field(description="Always 'success' on this path.")
    goal_id: str = Field(description="The newly created goal's id.")
    session_id: str = Field(description="The newly created durable session's id.")
    objective: str = Field(description="The parsed objective (echoed back).")
    validation_cmd: str = Field(
        description="The parsed validation command, empty string if none was supplied."
    )
    dispatch: dict[str, Any] = Field(
        description="The queue-dispatch handle from `enqueue_agent_turn` — "
        "free-form, dispatch-backend-dependent."
    )


class ListGoalsResponse(BaseModel):
    """Wrapper purely for documentation/testing purposes — ``GET /goals``
    itself returns a bare JSON array, not an object. Use `.root` conceptually
    as `list[GoalRecord]`; prefer constructing/validating
    `list[GoalRecord]` directly against the route's real body.
    """

    model_config = ConfigDict(extra="forbid")

    goals: list[GoalRecord] = Field(
        description="Every active + durable goal, merged (KG-derived entries "
        "first, then in-memory entries override by id) — NOT the route's "
        "literal JSON shape (which is the bare array); this field exists so "
        "the array's item type has one documented, importable name."
    )


class CancelGoalResponse(BaseModel):
    """Response body for ``POST /goals/{goal_id}/cancel``
    (``cancel_goal`` in ``agent_utilities/core/sessions.py``). A goal-not-found
    404 returns `{"error": "Active goal run not found"}` instead — not
    modeled here (error shape, not the success contract).
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["success"] = Field(description="Always 'success' on this path.")
    message: str = Field(description="Human-readable confirmation.")


# ══════════════════════════════════════════════════════════════════
# 8. /connector/run, /connector/sources
# ══════════════════════════════════════════════════════════════════


class ConnectorRunRequest(BaseModel):
    """Request body for ``POST /connector/run`` — REST twin of
    ``source_connector(action='run')`` (``connector_run_endpoint`` in
    ``kg_server.py``, CONCEPT:AU-ECO.connector.document-source-framework–4.29).

    ``extra="allow"``: PROVEN — the handler reads exactly `source_type`,
    `config`, `connector_id`, `contextual`, and `incremental` via
    `body.get(...)`; anything else is silently ignored.

    ⚠ `config` MAY CARRY CREDENTIALS: unlike GraphQL sources (which resolve
    endpoint/auth/TLS from runtime secret profiles per the tool's own
    description), REST/database/web connector configs can legitimately need
    literal connection strings, API keys, or basic-auth material inline —
    there is no separate secret-reference convention enforced at this
    boundary the way `graph_configure/secret` has. Flagged here rather than
    invented a redaction scheme the code doesn't actually implement; a
    caller populating this field with real credentials should treat the
    whole request (and any provenance/log capture of it) as sensitive.
    """

    model_config = ConfigDict(extra="allow")

    source_type: str = Field(
        default="",
        description="Connector type: filesystem | web | rest | database | "
        "graphql_document | 'mcp:<package>' | mcp_tool. REQUIRED for a real "
        "run — empty returns an in-body error (not a 4xx; see "
        "`ConnectorRunResult`).",
    )
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Connector-specific configuration (e.g. {'root': '/docs'} "
        "or {'base_url': 'https://...'}). MAY contain credentials for "
        "non-GraphQL source types — see this model's docstring.",
    )
    connector_id: str = Field(
        default="",
        description="Stable id for incremental checkpoint storage (optional).",
    )
    contextual: bool = Field(
        default=True,
        description="Enable contextual-retrieval enrichment during ingestion.",
    )
    incremental: bool = Field(
        default=True,
        description="Use the connector's resumable poll/watermark instead of a full load.",
    )


class ConnectorRunResult(BaseModel):
    """The `result` payload of `POST /connector/run` — the ingestion
    `details` mapping from `OntologySystem.run_connector`. `**details` is
    connector-dependent (extra allowed) on top of the four fields every
    connector run reports.
    """

    model_config = ConfigDict(extra="allow")

    status: str = Field(description="Ingestion run status (e.g. 'success', 'failed').")
    error: str | None = Field(
        description="Error detail when the run failed, else null."
    )
    nodes_created: int = Field(description="KG nodes created by this run.")
    edges_created: int = Field(description="KG edges created by this run.")


class ConnectorRunResponse(BaseModel):
    """Response envelope for ``POST /connector/run``."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success"] = Field(description="Always 'success' on this path.")
    result: ConnectorRunResult = Field(
        description="The connector run's ingestion result."
    )


class ConnectorSourcesResult(BaseModel):
    """The `result` payload of `GET /connector/sources` — the registered
    connector-type inventory (`source_connector(action='list')`).
    """

    model_config = ConfigDict(extra="forbid")

    connectors: list[str] = Field(description="Registered connector type names.")


class ConnectorSourcesResponse(BaseModel):
    """Response envelope for ``GET /connector/sources``
    (``connector_sources_endpoint`` in ``kg_server.py``, no request body/params).
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["success"] = Field(description="Always 'success' on this path.")
    result: ConnectorSourcesResult = Field(description="The connector type inventory.")


# ══════════════════════════════════════════════════════════════════
# 9. /sessions
# ══════════════════════════════════════════════════════════════════


class SessionRecord(BaseModel):
    """One durable agent-session row, as returned in the flat list from
    ``GET /sessions`` (``get_all_sessions`` in ``agent_utilities/core/sessions.py``).
    Field set matches the `sessions` table schema exactly (see
    `_connect_db`'s `CREATE TABLE` statement in that module) plus the two
    boolean coercions the handler applies (`background`/`needs_input` are
    stored as 0/1 and converted to real booleans before the response is built).
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="The session's unique id.")
    title: str = Field(default="", description="Display title.")
    created_at: float = Field(description="Unix timestamp the session was created.")
    updated_at: float = Field(
        description="Unix timestamp of the session's last update."
    )
    model: str = Field(default="", description="Model id used for this session.")
    mode: str = Field(default="ask", description="Session mode, e.g. 'ask'.")
    workspace: str = Field(
        default="", description="Workspace path the session is scoped to."
    )
    turn_count: int = Field(default=0, description="Number of turns recorded so far.")
    status: str = Field(
        default="active",
        description="Lifecycle status, e.g. 'active', 'queued', 'cancelled'.",
    )
    background: bool = Field(
        description="Whether this session runs a backgrounded goal loop "
        "(coerced from the stored 0/1 integer)."
    )
    needs_input: bool = Field(
        description="Whether the session is blocked waiting on user input "
        "(coerced from the stored 0/1 integer)."
    )
    last_response_preview: str = Field(
        default="", description="Truncated preview of the most recent response."
    )
    goal_id: str = Field(
        default="",
        description="The attached goal's id, empty when this session has none.",
    )
    metadata_json: str = Field(
        default="{}", description="JSON-encoded string of additional session metadata."
    )
    tenant_id: str = Field(
        default="", description="Owning tenant id (AU-P0-5 row-level scoping column)."
    )


class SessionsListQuery(BaseModel):
    """Query parameters for ``GET /sessions``. The handler clamps both
    values server-side rather than rejecting an out-of-range value, so this
    model mirrors that behavior in its field descriptions instead of adding
    a stricter validator that would reject something the real endpoint
    happily accepts and clamps.
    """

    model_config = ConfigDict(extra="allow")

    limit: int = Field(
        default=500,
        description="Max rows to return. Server-clamped to [1, 2000] — an "
        "out-of-range value is not rejected, just clamped.",
    )
    offset: int = Field(
        default=0,
        description="Row offset for pagination. Server-clamped to >= 0.",
    )
