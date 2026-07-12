"""Full low-level epistemic-graph engine surface as MCP tools + REST routes.

agent-utilities IS the native API/MCP layer for the epistemic-graph engine. The
curated high-level ``graph_*`` / ``ontology_*`` / ``object_*`` tools cover the
synthesised, agent-facing operations; this module ADDS complete 1:1 coverage of
the engine's *low-level* capability surface so no engine method is reachable only
from a Python import.

The engine speaks length-prefixed MessagePack over UDS/TCP; the pure-Python
``epistemic_graph`` client wraps every wire ``Method`` (``crates/eg-types/src/
protocol.rs``) as a method on one of its sub-clients (``.nodes``, ``.edges``,
``.graph``, ``.analytics``, ``.lifecycle``, ``.reasoning``, ``.ledger``,
``.channels``, ``.tenants``, ``.resharding``, ``.consensus``, ``.finance``,
``.datascience``, ``.query``, ``.txn``, ``.timeseries``, ``.rdf``, ``.streaming``,
``.blob``, ``.broker``, ``.rbac``, ``.admin``, ``.graphlearn``). That client is
the source of truth for "what the engine can do".

Design (anti-sprawl, anti-drift):

- One action-routed MCP tool per engine domain (``engine_<domain>``), each a thin
  generic dispatcher that resolves the engine client and calls
  ``getattr(client.<domain>, action)(**params)``. The action set per domain is
  DISCOVERED by introspecting the client sub-client class — no hand-maintained
  method list to rot (CONCEPT:AU-KG.compute.engine-surface-manifest).
- Every tool gets its REST twin registered into ``ACTION_TOOL_ROUTES`` in the SAME
  call, so the surface-parity gate stays green and ``_mount_rest_routes`` mounts
  ``POST /engine/<domain>`` automatically.
- Both surfaces dispatch through the one ``_execute_tool`` core — no parallel
  implementation.
- The verbose 1:1 surface (one ``engine_<domain>_<method>`` MCP tool per engine
  method, opt-in via ``MCP_TOOL_MODE=verbose``/``both``) is generated from
  :data:`ENGINE_DOMAINS` by the graph-os verbose builder + the action manifest
  generator (``scripts/gen_graphos_manifest.py``).
- Per-action scope/policy (AU-P0-6): every domain is classified ADMIN or
  normal (see :data:`ADMIN_DOMAINS`); ADMIN actions (tenant lifecycle, cluster
  resharding, zero-trust consensus/identity, RBAC policy administration,
  ops backup/restore) are denied to an acting identity that lacks the
  ``kg:admin`` scope/role, fail-closed. Reads and normal writes stay open to
  any actor exactly as before — see :func:`_enforce_admin_scope`.
- Bounded connection pool (AU-P0-6): the per-graph engine client is kept warm
  in an LRU-bounded pool (``KG_ENGINE_TOOL_POOL_SIZE``, default 16 — see
  :func:`_client_for`) instead of an unbounded forever-cache, so
  connection/thread count does not grow without limit as graph cardinality
  grows.

CONCEPT:AU-ECO.mcp.full-api-mcp-surface — Full engine API + MCP surface (REST + MCP in lockstep)
CONCEPT:AU-KG.compute.engine-surface-manifest — Engine surface manifest (client-introspection source of truth)
"""

from __future__ import annotations

import base64
import inspect
import json
import re
import threading
from typing import Any

from pydantic import Field

from agent_utilities.mcp import kg_server

# ── Domain → client sub-client class map ─────────────────────────────────────
# domain name == the attribute on ``SyncEpistemicGraphClient`` == the REST sub-path.
# The class is introspected (below) for its public async methods = the actions.
_DOMAIN_CLASSES: dict[str, str] = {
    "nodes": "NodeClient",
    "edges": "EdgeClient",
    "graph": "GraphOperationsClient",
    "analytics": "AnalyticsClient",
    "lifecycle": "LifecycleClient",
    "reasoning": "ReasoningClient",
    "ledger": "LedgerClient",
    "channels": "ChannelsClient",
    "tenants": "MultiTenantClient",
    "resharding": "ReshardingClient",
    "consensus": "ConsensusClient",
    "finance": "FinanceClient",
    "datascience": "DataScienceClient",
    "mining": "MiningClient",
    "query": "QueryClient",
    "txn": "TxnClient",
    "timeseries": "TimeSeriesClient",
    "rdf": "RdfClient",
    "streaming": "StreamingClient",
    "blob": "BlobClient",
    # AU-P0-6: previously-missing namespaces (audited gap #3). Domain name ==
    # the attribute on ``SyncEpistemicGraphClient`` (checked against the
    # installed ``epistemic_graph.client`` module — NOT a guess): the graph-
    # learning sub-client is exposed as ``.graphlearn`` (no underscore), so
    # the domain/tool/REST-path name below is ``graphlearn`` to match it.
    # rbac/admin are ADMIN_DOMAINS (see below) — gated BEFORE being newly
    # exposed, per the audit's explicit ordering.
    "broker": "BrokerClient",
    "rbac": "RbacClient",
    "admin": "AdminClient",
    "graphlearn": "GraphLearnClient",
}

_DOMAIN_BLURB: dict[str, str] = {
    "nodes": "node CRUD, batch/union reads, degree/neighbour queries",
    "edges": "edge CRUD, temporal invalidate/supersede, batch reads",
    "graph": "graph algorithms, AST parse/index, semantic+embedding compute",
    "analytics": "centrality + (personalized) PageRank",
    "lifecycle": "prune/decay/evict, batch_update, context view, (de)serialize",
    "reasoning": "forward-chaining OWL/RDFS Datalog closure",
    "ledger": "audit ledger get/clear/apply",
    "channels": "dynamic agent communication channels",
    "tenants": "multi-tenant graph create/delete/list",
    "resharding": "M3 catalog/reshard/rebalance admin (redb)",
    "consensus": "zero-trust identity + multisig mutation",
    "finance": "quantitative finance (optimize/risk/regime/signals/HFT/derivatives)",
    "datascience": "estimators + primitives + training kernels",
    "mining": "association-rule mining (Apriori/FP-Growth/Eclat: support/confidence/lift)",
    "query": "SQL / Cypher / GraphQL / UQL / unified cross-modal query / federation",
    "txn": "server-side OCC ACID transactions",
    "timeseries": "native TSDB append/range/window/asof/gapfill",
    "rdf": "RDF triples + SPARQL + OWL reasoning",
    "streaming": "CDC / continuous queries / watch / triggers",
    "blob": "streamed content-addressed media blobs",
    "broker": "native message broker: exchange/queue/stream admin + routed publish/consume",
    "rbac": "RBAC policy administration: roles + resource/action grants (ADMIN)",
    "admin": "ops/maintenance: online backup + restore (ADMIN)",
    "graphlearn": "KAN graph-learning: fit/predict a learned per-feature edge-function link predictor",
}


def _discover_domains() -> dict[str, list[str]]:
    """Introspect the engine client sub-client classes → ``{domain: [methods]}``.

    The ``epistemic_graph`` client mirrors the wire protocol 1:1, so its public
    async methods ARE the engine's invocable capability surface. Discovering them
    (instead of hand-listing 200+ methods) keeps this surface drift-free: a new
    engine method shows up automatically once the client wraps it.
    """
    try:
        from epistemic_graph import client as _client_mod
    except Exception:  # noqa: BLE001 — engine client absent ⇒ register nothing
        return {}

    out: dict[str, list[str]] = {}
    for domain, class_name in _DOMAIN_CLASSES.items():
        cls = getattr(_client_mod, class_name, None)
        if cls is None:
            continue
        methods = sorted(
            name
            for name, member in inspect.getmembers(cls, inspect.iscoroutinefunction)
            if not name.startswith("_")
        )
        if methods:
            out[domain] = methods
    return out


# The declarative engine-surface manifest (source of truth for the tools AND the
# verbose action manifest generator). CONCEPT:AU-KG.compute.engine-surface-manifest.
ENGINE_DOMAINS: dict[str, list[str]] = _discover_domains()


# ── Engine client resolution (bounded LRU pool, AU-P0-6) ─────────────────────
# Was: one synchronous client cached PER GRAPH forever in a plain ``dict`` —
# connection/thread/socket count grew without bound as graph cardinality grew
# (audited gap #2). Now: an LRU-bounded warm pool (reusing the same primitive
# ``TenantEnginePool`` already used for the L1 compute-engine pool — one
# tested bounded-pool implementation, not a second one) sized from
# ``KG_ENGINE_TOOL_POOL_SIZE`` (default 16); the least-recently-used
# connection is evicted (and closed) once the pool is at capacity, so the
# resident connection count is capped regardless of how many distinct graphs
# are ever touched.
_DEFAULT_GRAPH_POOL_KEY = "__default__"


def _client_factory(pool_key: str) -> Any:
    """Connect a fresh ``SyncEpistemicGraphClient`` for one pool key.

    ``pool_key`` is the opaque cache key used by :func:`_client_for` (never
    empty — see :data:`_DEFAULT_GRAPH_POOL_KEY`); it is translated back to the
    real ``graph`` argument (``None`` for the deployment default) before
    calling the centralized resolver (CONCEPT:AU-OS.deployment.engine-resolver-auto-provision
    ``client_connect_kwargs``) so a remote/sharded/insecure deployment is
    honoured. Connect-only — never autostarts an engine; if the engine is
    down, ``connect`` raises and the caller surfaces a clean error.
    """
    from epistemic_graph.client import SyncEpistemicGraphClient

    from agent_utilities.knowledge_graph.core.engine_resolver import (
        client_connect_kwargs,
    )

    graph = None if pool_key == _DEFAULT_GRAPH_POOL_KEY else pool_key
    kwargs = client_connect_kwargs(None, graph)
    return SyncEpistemicGraphClient.connect(**kwargs)


def _client_evict(pool_key: str, client: Any) -> None:
    """Best-effort close of an evicted/discarded wire client."""
    close = getattr(client, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001 — eviction must never raise
            pass


def _client_pool_capacity() -> int:
    try:
        from agent_utilities.core.config import config

        return int(getattr(config, "kg_engine_tool_pool_size", 16) or 0)
    except Exception:  # noqa: BLE001 — default to a small bounded pool
        return 16


_CLIENT_POOL_LOCK = threading.Lock()
_CLIENT_POOL: Any = None  # lazily-built ``TenantEnginePool``, module-private


def _get_client_pool() -> Any:
    global _CLIENT_POOL
    if _CLIENT_POOL is None:
        with _CLIENT_POOL_LOCK:
            if _CLIENT_POOL is None:
                from agent_utilities.knowledge_graph.core.tenant_engine_pool import (
                    TenantEnginePool,
                )

                _CLIENT_POOL = TenantEnginePool(
                    capacity=_client_pool_capacity(),
                    factory=_client_factory,
                    on_evict=_client_evict,
                )
    return _CLIENT_POOL


def reset_client_pool() -> None:
    """Drop the pool singleton, closing every warm client first (test helper)."""
    global _CLIENT_POOL
    with _CLIENT_POOL_LOCK:
        if _CLIENT_POOL is not None:
            _CLIENT_POOL.clear()
        _CLIENT_POOL = None


def _client_for(graph: str) -> Any:
    """Return a warm ``SyncEpistemicGraphClient`` bound to ``graph`` from the
    bounded LRU pool (AU-P0-6 — see :func:`_get_client_pool`). Thread-safe."""
    pool = _get_client_pool()
    return pool.acquire(graph_name=graph or _DEFAULT_GRAPH_POOL_KEY)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, bytes | bytearray):
        return {"__bytes_b64__": base64.b64encode(bytes(obj)).decode("ascii")}
    return str(obj)


# ── Per-action scope/policy (AU-P0-6) ─────────────────────────────────────────
# Domains whose ENTIRE action surface is privileged/administrative — mirrors
# the engine's own separation of tenant/cluster/identity/governance ops from
# ordinary graph reads+writes, rather than inventing a parallel classification
# (CONCEPT:AU-KG.compute.engine-surface-manifest). Every method under one of
# these domains requires :data:`ENGINE_ADMIN_SCOPE`, with NO per-method
# carve-out: even a read here (``tenants.list``, ``resharding.catalog_list``,
# ``rbac.list``) discloses cluster/tenant/identity topology a normal actor
# should not see.
#
# No machine-readable per-method capability/role metadata ships from the
# engine client (checked: ``BrokerClient``/``RbacClient``/``AdminClient``/
# ``MultiTenantClient``/``ReshardingClient``/``ConsensusClient`` carry doc
# comments only, no ``requires_role``/``is_admin``/capability descriptor), so
# this is an explicit, hand-maintained map — called out here as such per
# AU-P0-6's guidance.
ADMIN_DOMAINS: frozenset[str] = frozenset(
    {"tenants", "resharding", "consensus", "rbac", "admin"}
)

# The remaining known domains are ordinary graph reads+writes, open to any
# actor exactly as before this workstream — AU-P0-6 closes the ADMIN gap, it
# does not newly restrict reads/writes.
_NORMAL_DOMAINS: frozenset[str] = frozenset(
    {
        "nodes",
        "edges",
        "graph",
        "analytics",
        "lifecycle",
        "reasoning",
        "ledger",
        "channels",
        "finance",
        "datascience",
        "mining",
        "query",
        "txn",
        "timeseries",
        "rdf",
        "streaming",
        "blob",
        "broker",
        "graphlearn",
    }
)

#: Scope an acting identity must carry (as an ``admin`` role on its
#: :class:`~agent_utilities.security.brain_context.ActorContext`, or this
#: literal scope string in an active
#: :class:`~agent_utilities.knowledge_graph.core.session.GraphSession`) to
#: invoke an ADMIN-domain action.
ENGINE_ADMIN_SCOPE = "kg:admin"

# Best-effort read-verb prefixes for the (unenforced, introspection-only)
# read/mutate label — see :func:`action_policy`. Reads and normal writes are
# BOTH open to non-admin actors, so getting one of these wrong never opens or
# closes an enforcement gate; it only affects the descriptive ``mutate``/
# ``scope`` fields surfaced when an action is listed.
_READ_VERB_PREFIXES: tuple[str, ...] = (
    "get",
    "has",
    "list",
    "read",
    "query",
    "search",
    "sql",
    "sparql",
    "range",
    "sort",
    "pagerank",
    "centrality",
    "topological",
    "compute_stats",
    "var",
    "metrics",
    "reason",
    "estimate",
    "describe",
    "export",
    "fetch",
    "find",
    "lookup",
    "resolve",
    "peek",
    "watch",
    "subscribe",
    "sample",
    "validate",
    "check",
    "diff",
    "preview",
    "count",
    "stats",
    "predict",
)


def _is_admin_domain(domain: str) -> bool:
    """Whether every action under ``domain`` requires :data:`ENGINE_ADMIN_SCOPE`.

    Fail-closed (AU-P0-6 guardrail): a domain this map has not explicitly
    classified as normal — e.g. a future engine namespace added to
    ``_DOMAIN_CLASSES`` before this map is updated — is treated as ADMIN by
    default rather than silently open.
    """
    if domain in ADMIN_DOMAINS:
        return True
    return domain not in _NORMAL_DOMAINS


def _is_read_action(action: str) -> bool:
    """Best-effort read/mutate label for policy introspection — NOT itself an
    enforcement gate (see :data:`_READ_VERB_PREFIXES`)."""
    name = action.lower()
    return name.startswith(_READ_VERB_PREFIXES)


def action_policy(domain: str, action: str) -> dict[str, Any]:
    """The scope/policy classification for one ``engine_<domain>`` action.

    Returns a dict with ``admin`` (the only ENFORCED bit — see
    :func:`_enforce_admin_scope`), the best-effort ``mutate`` read/write
    label, and the ``scope`` string this action is documented under
    (``kg:admin`` / ``kg:write`` / ``kg:read``).
    """
    admin = _is_admin_domain(domain)
    mutate = not _is_read_action(action)
    scope = ENGINE_ADMIN_SCOPE if admin else ("kg:write" if mutate else "kg:read")
    return {
        "domain": domain,
        "action": action,
        "admin": admin,
        "mutate": mutate,
        "scope": scope,
    }


def _enforce_admin_scope(domain: str, action: str) -> None:
    """Deny an ADMIN-domain action unless the acting identity carries the
    admin role/scope (AU-P0-6). No-op for normal domains. Fail-closed — see
    :func:`_is_admin_domain`.

    Checked against BOTH the ambient
    :class:`~agent_utilities.security.brain_context.ActorContext`
    (``current_actor()`` — set by ``kg_server._execute_tool``'s ``use_actor``
    for the REST/token-authenticated path, or the privileged ``SYSTEM_ACTOR``
    default when nothing has scoped the request) and, when one is active, the
    explicit :class:`~agent_utilities.knowledge_graph.core.session.GraphSession`
    scopes — either one satisfying the gate is enough, mirroring
    ``GraphSession.require_scope``'s own admin-role bypass.
    """
    if not _is_admin_domain(domain):
        return

    from agent_utilities.security.brain_context import current_actor

    actor = current_actor()
    if "admin" in (actor.roles or ()):
        return

    session = None
    try:
        from agent_utilities.knowledge_graph.core.session import current_session

        session = current_session()
    except Exception:  # noqa: BLE001 — session module optional in minimal installs
        session = None
    if session is not None and (
        "admin" in (session.actor.roles or ()) or ENGINE_ADMIN_SCOPE in session.scopes
    ):
        return

    raise PermissionError(
        f"engine_{domain}.{action} is an ADMIN-only action (tenants/resharding/"
        f"consensus/rbac/admin family, CONCEPT:AU-P0-6) — requires the "
        f"{ENGINE_ADMIN_SCOPE!r} scope or an 'admin' role on the acting "
        f"identity (actor={actor.actor_id!r}, roles={sorted(actor.roles)!r})."
    )


# ── Unbounded global-analytics OOM guard ──────────────────────────────────
# ``engine_analytics(action="pagerank", params_json="{}")`` — an unbounded global
# PageRank over the live ~139k-node graph — OOM-killed the epistemic-graph engine
# (exitCode 137). ``AnalyticsClient.pagerank``/``betweenness_centrality``/
# ``degree_centrality_all`` (``epistemic_graph.client``) take NO kwarg that scopes
# them to anything smaller than the whole graph, so every call to one of them is
# unbounded by construction; ``personalized_pagerank`` is bounded ONLY when it
# carries a non-empty ``seed_nodes`` frontier. Rather than let the engine OOM
# (silent, ungraceful), fail loud client-side before the RPC — mirrors the
# engine's own ``RESULT_TOO_LARGE`` node-dump guard, just enforced here because
# these are compute calls the engine has no size-check for.
_UNBOUNDED_ANALYTICS_ACTIONS = frozenset(
    {"pagerank", "betweenness_centrality", "degree_centrality_all"}
)


def _reject_unbounded_analytics(
    domain: str, action: str, params: dict[str, Any]
) -> str | None:
    """Non-``None`` ⇒ a clear, typed rejection message; ``None`` ⇒ let it through.

    Only the ``analytics`` domain's whole-graph algorithms are gated. Bounded/
    legitimate analytics calls (``degree_centrality`` for one node, a
    ``personalized_pagerank`` with a seeded frontier, anything outside
    ``analytics``) are never touched.
    """
    if domain != "analytics":
        return None
    if action in _UNBOUNDED_ANALYTICS_ACTIONS:
        return (
            f"engine_analytics.{action} has no way to scope itself below the "
            "ENTIRE graph (no top_k/limit/nodes/subset parameter exists on it) — "
            "an unbounded call previously OOM-killed the engine (exitCode 137). "
            "Rejected fail-loud instead of risking another crash. Use a bounded "
            "alternative: 'personalized_pagerank' with a non-empty 'seed_nodes' "
            "frontier ([[node_id, weight], ...]), or scope your read to a node "
            "subset first (e.g. graph_analyze(action='blast_radius')/'inspect')."
        )
    if action == "personalized_pagerank" and not params.get("seed_nodes"):
        return (
            "engine_analytics.personalized_pagerank requires a non-empty "
            "'seed_nodes' ([[node_id, weight], ...]) — without one the call is "
            "unbounded over the entire graph, which previously OOM-killed the "
            "engine (exitCode 137). Pass a seeded frontier to scope it."
        )
    return None


# ── UQL server-side text-embedding pre-pass (F2, CONCEPT:AU-KG.query.uql-rank-text-preembed) ──
#
# ``RANK BY ~"some text"`` lowers to `Op::RankEmbed` and is resolved by an embedder
# bound on the engine's `PlanCtx` — but nothing binds one there, so it always fails
# with "no server-side text embedder is bound on this query" even though an
# embedder IS configured and used everywhere else (``graph_search`` embeds the
# query CLIENT-side via ``create_embedding_model`` then ranks). Rather than wait on
# an engine-side `PlanCtx::with_embedder` wiring, mirror the `graph_search` pattern
# here: pre-embed every quoted-text RANK leg in Python, with the SAME embedder (so
# the vector dimension matches the stored node embeddings), and rewrite it to an
# inline literal vector before the query reaches the engine. Native by default — no
# opt-in flag — and fails loud if the embedder is unavailable (never silently drops
# the RANK leg).
_UQL_RANK_TEXT_OPEN_RE = re.compile(r'~\s*"')


def _uql_rank_text_spans(text: str) -> list[tuple[int, int, str]]:
    """Find every quoted-text RANK leg ``~"…"`` in a UQL string.

    A leg is ``~`` immediately (whitespace allowed) followed by a double-quoted
    string — mirrors ``eg-plan``'s parser (`RANK BY ~<vector_ref>` where a `"`
    lookahead is the `Text` arm, `crates/eg-plan/src/uql/parser.rs::parse_vector_ref`)
    and its lexer's double-quoted-string escaping (`lexer.rs::lex_string`): a
    doubled quote (``""``) or a backslash-escaped ``\\"``/``\\\\`` is an escaped
    character, anything else closes the string. Returns ``(start, end, literal)``
    triples — ``start``/``end`` span the WHOLE ``~"…"`` run (for replacement),
    ``literal`` is the unescaped text to embed. An inline vector (``~[…]``) or a
    bare handle (``~handle``) never matches — neither has a `"` right after the
    optional whitespace — so both pass through untouched.
    """
    spans: list[tuple[int, int, str]] = []
    n = len(text)
    for m in _UQL_RANK_TEXT_OPEN_RE.finditer(text):
        start = m.start()
        i = m.end()  # just past the opening quote
        out_chars: list[str] = []
        closed = False
        while i < n:
            c = text[i]
            if c == '"':
                if i + 1 < n and text[i + 1] == '"':
                    out_chars.append('"')
                    i += 2
                    continue
                i += 1
                closed = True
                break
            if c == "\\" and i + 1 < n and text[i + 1] in ('"', "\\"):
                out_chars.append(text[i + 1])
                i += 2
                continue
            out_chars.append(c)
            i += 1
        if closed:
            spans.append((start, i, "".join(out_chars)))
    return spans


def _collect_unified_rank_text(ops: Any, pending: list[dict[str, Any]]) -> None:
    """Recurse a structured ``unified`` plan (list of op dicts) collecting every
    ``{"Rank": {"query": "<text>"}}`` leg's field dict — including inside
    ``FuseRrf`` branches — so its ``query`` can be overwritten with an embedded
    vector in place. A ``Rank.query`` that's already a list (the normal inline-
    vector form) is left untouched.
    """
    if not isinstance(ops, list):
        return
    for op in ops:
        if not isinstance(op, dict):
            continue
        rank = op.get("Rank")
        if isinstance(rank, dict) and isinstance(rank.get("query"), str):
            pending.append(rank)
        fuse = op.get("FuseRrf")
        if isinstance(fuse, dict):
            for branch in fuse.get("branches") or []:
                _collect_unified_rank_text(branch, pending)


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch-embed ``texts`` with the SAME embedder :func:`create_embedding_model`
    resolves everywhere else (``graph_search``, ingestion, enrichment, …) so the
    vector dimension matches the stored node embeddings. Raises loud — never
    silently drops a RANK leg — if no embedder can be constructed/reached.
    """
    from agent_utilities.core.embedding_utilities import create_embedding_model

    try:
        model = create_embedding_model()
        batch = getattr(model, "get_text_embedding_batch", None)
        vectors = (
            batch(texts)
            if callable(batch)
            else [model.get_text_embedding(t) for t in texts]
        )
    except Exception as exc:  # noqa: BLE001 — degrade loud, never silent-drop the RANK leg
        raise RuntimeError(
            "UQL 'RANK BY ~\"text\"' needs a server-side text embedder — the engine "
            "has none bound (CONCEPT:AU-KG.query.uql-rank-text-preembed) and "
            "pre-embedding it client-side "
            "(agent_utilities.core.embedding_utilities.create_embedding_model) failed "
            f"too: {exc}"
        ) from exc
    return [list(v) for v in vectors]


def _embed_uql_rank_text(
    domain: str, action: str, params: dict[str, Any]
) -> dict[str, Any]:
    """Pre-embed every quoted-text ``RANK BY ~"…"`` leg before a ``uql``/``unified``
    query dispatches (F2 fix — see the module comment above). The ONE chokepoint
    for both UQL surfaces: :func:`_dispatch` calls this unconditionally for
    ``engine_query`` before invoking the engine client, so neither surface
    duplicates the embed-and-rewrite logic.
    """
    if domain != "query":
        return params
    if action == "uql" and isinstance(params.get("text"), str):
        text = params["text"]
        spans = _uql_rank_text_spans(text)
        if spans:
            vectors = _embed_texts([literal for _, _, literal in spans])
            rewritten = text
            for (start, end, _literal), vector in reversed(
                list(zip(spans, vectors, strict=True))
            ):
                rendered = "[" + ",".join(f"{float(c):.8f}" for c in vector) + "]"
                rewritten = rewritten[:start] + "~" + rendered + rewritten[end:]
            params["text"] = rewritten
    elif action == "unified" and isinstance(params.get("plan"), list):
        pending: list[dict[str, Any]] = []
        _collect_unified_rank_text(params["plan"], pending)
        if pending:
            vectors = _embed_texts([rank_field["query"] for rank_field in pending])
            for rank_field, vector in zip(pending, vectors, strict=True):
                rank_field["query"] = [float(c) for c in vector]
    return params


def _dispatch(
    domain: str, methods: set[str], action: str, params_json: str, graph: str
) -> str:
    """Generic dispatch into one engine sub-client method. The ONE engine core."""
    if not action:
        return json.dumps(
            {
                "domain": domain,
                "actions": sorted(methods),
                "admin_domain": _is_admin_domain(domain),
            }
        )
    if action not in methods:
        return json.dumps(
            {
                "error": f"unknown action {action!r} for engine_{domain}",
                "actions": sorted(methods),
            }
        )
    # AU-P0-6: fail-closed ADMIN gate — raises PermissionError (not returned as
    # JSON error data) so a denial is unambiguous and cannot be masked by a
    # caller pattern-matching on ``{"error": ...}`` engine-degrade payloads.
    _enforce_admin_scope(domain, action)
    try:
        params = json.loads(params_json) if params_json else {}
    except (TypeError, ValueError) as exc:
        return json.dumps({"error": f"invalid params_json: {exc}"})
    if not isinstance(params, dict):
        return json.dumps({"error": "params_json must decode to a JSON object"})
    guard_msg = _reject_unbounded_analytics(domain, action, params)
    if guard_msg is not None:
        return json.dumps(
            {
                "error": guard_msg,
                "domain": domain,
                "action": action,
                "guard": "RESULT_TOO_LARGE",
            }
        )
    # F2: pre-embed any quoted-text RANK leg (`RANK BY ~"text"`) in a `uql`/
    # `unified` query — see the helper's module comment above. Fails loud (a
    # JSON error payload), never silently drops the RANK leg.
    try:
        params = _embed_uql_rank_text(domain, action, params)
    except Exception as exc:  # noqa: BLE001 — surface as engine-style error data, not a 500
        return json.dumps({"error": str(exc), "domain": domain, "action": action})
    try:
        client = _client_for(graph)
    except Exception as exc:  # noqa: BLE001 — engine unreachable is a normal degrade
        return json.dumps({"error": f"engine unavailable: {exc}"})
    fn = getattr(getattr(client, domain), action, None)
    if not callable(fn):
        return json.dumps(
            {"error": f"engine_{domain} has no callable action {action!r}"}
        )
    try:
        result = fn(**params)
    except TypeError as exc:
        return json.dumps({"error": f"bad arguments for {domain}.{action}: {exc}"})
    except Exception as exc:  # noqa: BLE001 — surface engine errors as data, not 500
        return json.dumps({"error": str(exc), "domain": domain, "action": action})
    return json.dumps(result, default=_json_default)


def _make_domain_tool(domain: str, methods: list[str]):
    method_set = set(methods)

    async def _engine_domain_tool(
        action: str = Field(
            default="",
            description=f"engine_{domain} method to call (empty ⇒ list actions).",
        ),
        params_json: str = Field(
            default="{}",
            description="JSON object of keyword arguments for the method.",
        ),
        graph: str = Field(
            default="",
            description="Target graph name (empty ⇒ the deployment default graph).",
        ),
    ) -> str:
        return _dispatch(domain, method_set, action, params_json, graph)

    return _engine_domain_tool


def register_engine_tools(mcp) -> None:
    """Register the full low-level engine surface — one ``engine_<domain>`` tool
    per engine client sub-client, each with its ``/engine/<domain>`` REST twin.

    REGISTERED_TOOLS + ACTION_TOOL_ROUTES are populated in lockstep so the
    surface-parity gate stays green. CONCEPT:AU-ECO.mcp.full-api-mcp-surface.
    """
    for domain, methods in ENGINE_DOMAINS.items():
        tool_name = f"engine_{domain}"
        fn = _make_domain_tool(domain, methods)
        fn.__name__ = tool_name
        blurb = _DOMAIN_BLURB.get(domain, "")
        admin_note = (
            f" ADMIN domain — every action requires the {ENGINE_ADMIN_SCOPE!r} "
            "scope/role (AU-P0-6); denied otherwise."
            if _is_admin_domain(domain)
            else ""
        )
        description = (
            f"Low-level epistemic-graph engine surface for the '{domain}' domain"
            + (f" ({blurb})" if blurb else "")
            + ". Action-routed 1:1 over the epistemic_graph client: set 'action' to "
            f"the method name and 'params_json' to its JSON kwargs. Actions: "
            f"{', '.join(sorted(methods))}."
            + admin_note
            + " (CONCEPT:AU-ECO.mcp.full-api-mcp-surface)"
        )
        tags = ["graph-os", "engine", domain]
        if _is_admin_domain(domain):
            tags.append("admin")
        mcp.tool(name=tool_name, description=description, tags=tags)(fn)
        kg_server.REGISTERED_TOOLS[tool_name] = fn
        kg_server.ACTION_TOOL_ROUTES[tool_name] = f"/engine/{domain}"
