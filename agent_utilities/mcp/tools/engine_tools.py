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
``.blob``). That client is the source of truth for "what the engine can do".

Design (anti-sprawl, anti-drift):

- One action-routed MCP tool per engine domain (``engine_<domain>``), each a thin
  generic dispatcher that resolves the engine client and calls
  ``getattr(client.<domain>, action)(**params)``. The action set per domain is
  DISCOVERED by introspecting the client sub-client class — no hand-maintained
  method list to rot (CONCEPT:KG-2.278).
- Every tool gets its REST twin registered into ``ACTION_TOOL_ROUTES`` in the SAME
  call, so the surface-parity gate stays green and ``_mount_rest_routes`` mounts
  ``POST /engine/<domain>`` automatically.
- Both surfaces dispatch through the one ``_execute_tool`` core — no parallel
  implementation.
- The verbose 1:1 surface (one ``engine_<domain>_<method>`` MCP tool per engine
  method, opt-in via ``MCP_TOOL_MODE=verbose``/``both``) is generated from
  :data:`ENGINE_DOMAINS` by the graph-os verbose builder + the action manifest
  generator (``scripts/gen_graphos_manifest.py``).

CONCEPT:ECO-4.99 — Full engine API + MCP surface (REST + MCP in lockstep)
CONCEPT:KG-2.278 — Engine surface manifest (client-introspection source of truth)
"""

from __future__ import annotations

import base64
import inspect
import json
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
    "query": "QueryClient",
    "txn": "TxnClient",
    "timeseries": "TimeSeriesClient",
    "rdf": "RdfClient",
    "streaming": "StreamingClient",
    "blob": "BlobClient",
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
    "query": "SQL / Cypher / GraphQL / UQL / unified cross-modal query / federation",
    "txn": "server-side OCC ACID transactions",
    "timeseries": "native TSDB append/range/window/asof/gapfill",
    "rdf": "RDF triples + SPARQL + OWL reasoning",
    "streaming": "CDC / continuous queries / watch / triggers",
    "blob": "streamed content-addressed media blobs",
}


def _discover_domains() -> dict[str, list[str]]:
    """Introspect the engine client sub-client classes → ``{domain: [methods]}``.

    The ``epistemic_graph`` client mirrors the wire protocol 1:1, so its public
    async methods ARE the engine's invokable capability surface. Discovering them
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
# verbose action manifest generator). CONCEPT:KG-2.278.
ENGINE_DOMAINS: dict[str, list[str]] = _discover_domains()


# ── Engine client resolution (cached per target graph) ───────────────────────
_CLIENT_LOCK = threading.Lock()
_CLIENTS: dict[str, Any] = {}


def _client_for(graph: str) -> Any:
    """Return a cached ``SyncEpistemicGraphClient`` bound to ``graph``.

    Connects via the centralized resolver (CONCEPT:OS-5.63 ``client_connect_kwargs``)
    so a remote/sharded/insecure deployment is honoured. Connect-only — never
    autostarts an engine; if the engine is down, ``connect`` raises and the caller
    surfaces a clean error.
    """
    key = graph or ""
    with _CLIENT_LOCK:
        existing = _CLIENTS.get(key)
        if existing is not None:
            return existing
        from epistemic_graph.client import SyncEpistemicGraphClient

        from agent_utilities.knowledge_graph.core.engine_resolver import (
            client_connect_kwargs,
        )

        kwargs = client_connect_kwargs(None, graph or None)
        client = SyncEpistemicGraphClient.connect(**kwargs)
        _CLIENTS[key] = client
        return client


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (bytes, bytearray)):
        return {"__bytes_b64__": base64.b64encode(bytes(obj)).decode("ascii")}
    return str(obj)


def _dispatch(
    domain: str, methods: set[str], action: str, params_json: str, graph: str
) -> str:
    """Generic dispatch into one engine sub-client method. The ONE engine core."""
    if not action:
        return json.dumps({"domain": domain, "actions": sorted(methods)})
    if action not in methods:
        return json.dumps(
            {
                "error": f"unknown action {action!r} for engine_{domain}",
                "actions": sorted(methods),
            }
        )
    try:
        params = json.loads(params_json) if params_json else {}
    except (TypeError, ValueError) as exc:
        return json.dumps({"error": f"invalid params_json: {exc}"})
    if not isinstance(params, dict):
        return json.dumps({"error": "params_json must decode to a JSON object"})
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
    surface-parity gate stays green. CONCEPT:ECO-4.99.
    """
    for domain, methods in ENGINE_DOMAINS.items():
        tool_name = f"engine_{domain}"
        fn = _make_domain_tool(domain, methods)
        fn.__name__ = tool_name
        blurb = _DOMAIN_BLURB.get(domain, "")
        description = (
            f"Low-level epistemic-graph engine surface for the '{domain}' domain"
            + (f" ({blurb})" if blurb else "")
            + ". Action-routed 1:1 over the epistemic_graph client: set 'action' to "
            f"the method name and 'params_json' to its JSON kwargs. Actions: "
            f"{', '.join(sorted(methods))}. (CONCEPT:ECO-4.99)"
        )
        mcp.tool(
            name=tool_name, description=description, tags=["graph-os", "engine", domain]
        )(fn)
        kg_server.REGISTERED_TOOLS[tool_name] = fn
        kg_server.ACTION_TOOL_ROUTES[tool_name] = f"/engine/{domain}"
