"""New engine-surface MCP tools — CONCEPT:KG-2.310.

epistemic-graph v2.2.0 grew several engine capabilities that agent-utilities
should expose to the fleet as first-class, purpose-shaped MCP tools rather than
leaving them reachable only through the generic ``engine_<domain>`` 1:1 surface
(``engine_tools.py``). This module ADDS those ergonomic wrappers:

* ``graph_broker``          — the message broker (declare exchange/queue, bind,
  publish, consume, stats — AMQP-style; distinct from the agent-to-agent
  ``graph_bus`` in ``bus_tools.py``).
* ``graph_kvcache``         — the shared, content-addressed KV-cache over the
  EG-187 HTTP surface, driven through the KG-2.306
  :class:`~agent_utilities.kvcache.EpistemicGraphKVBackend` connector.
* ``graph_federated_search``— federated search fanned across registered external
  graph references.
* ``graph_promql``          — PromQL instant/range metric queries (observability).
* ``graph_traces``          — distributed-trace search / fetch (observability).
* ``graph_gis``             — geospatial route / tile / geo-task ops.
* ``graph_memory``          — the EG-318 memory surface: episodic→semantic memory
  (create-summary / consolidate / maintain), the spatial scene graph
  (add-scene-object / world-transform), and RL trajectories (start-trajectory /
  append-step / discounted-return), plus their reads.

Design (matches the rest of ``agent_utilities/mcp/tools``):

* **Reuse the existing engine transport** — every client-backed tool resolves the
  same :class:`~epistemic_graph.client.SyncEpistemicGraphClient` that
  ``engine_tools`` uses (via :func:`engine_tools._client_for`); the KV-cache tool
  reuses the KG-2.306 HTTP connector. No new transport is invented.
* **Additive + gated / graceful degradation** — the v2.2.0 surfaces may not be
  present in the connected engine build (or wired into the ``epistemic_graph``
  client yet). Each tool probes a small set of candidate ``client.<sub>.<method>``
  paths and, when none resolve, returns a clean ``degraded`` payload instead of
  raising — so the fleet can call these today and they light up automatically once
  the engine ships the capability.
* **Two surfaces** — each tool registers a ``/graph/<name>`` REST twin in
  ``ACTION_TOOL_ROUTES`` (auto-mounted by the generic factory), keeping MCP⇄REST
  parity like every other tool.

CONCEPT:KG-2.310 — MCP surface for the new engine ops (broker / kvcache /
federated-search / promql / traces / gis).
"""

from __future__ import annotations

import base64
import json
from typing import Any

from pydantic import Field

from agent_utilities.mcp import kg_server

# Candidate ``(sub_client_attr, method_attr)`` probe lists per logical action. The
# engine build / client may expose the surface under any of several plausible
# namespaces; the first callable found wins and everything else degrades cleanly.
_FEDERATED_SEARCH_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("search", "federated_search"),
    ("query", "federated_search"),
    ("federation", "search"),
    ("query", "federated"),
)
_PROMQL_INSTANT_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("observability", "promql"),
    ("metrics", "promql"),
    ("observability", "query"),
    ("promql", "query"),
)
_PROMQL_RANGE_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("observability", "promql_range"),
    ("metrics", "query_range"),
    ("observability", "query_range"),
    ("promql", "query_range"),
)
_TRACES_SEARCH_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("observability", "search_traces"),
    ("traces", "search"),
    ("observability", "query_traces"),
)
_TRACES_GET_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("observability", "get_trace"),
    ("traces", "get"),
    ("observability", "trace"),
)

# graph_memory — EG-318 memory / scene / trajectory ops. Each logical action maps
# to a small probe list of ``(sub_client_attr, method_attr)`` paths the engine
# build / client may plausibly expose the EG-318 wire Method under (the client
# wraps each wire ``Method`` as a snake_case method on a sub-client). The first
# callable found wins; when none resolve the tool degrades cleanly. Any action not
# listed here falls back to probing the three sub-clients with the action name
# directly, so read ops (get_summary / get_scene / get_trajectory / …) light up
# by name once the engine ships them.
_MEMORY_ACTION_CANDIDATES: dict[str, tuple[tuple[str, str], ...]] = {
    # episodic → semantic memory (CreateSummaryNode / Consolidate / Maintain)
    "create_summary": (
        ("memory", "create_summary"),
        ("memory", "create_summary_node"),
    ),
    "consolidate": (("memory", "consolidate"),),
    "maintain": (("memory", "maintain"),),
    # spatial scene graph (AddSceneObject / world transform)
    "add_scene_object": (
        ("scene", "add_object"),
        ("scene", "add_scene_object"),
        ("memory", "add_scene_object"),
    ),
    "world_transform": (
        ("scene", "world_transform"),
        ("scene", "set_world_transform"),
    ),
    # trajectories / RL episodes (StartTrajectory / AppendStep / discounted return)
    "start_trajectory": (
        ("trajectory", "start"),
        ("trajectory", "start_trajectory"),
    ),
    "append_step": (
        ("trajectory", "append_step"),
        ("trajectory", "append"),
    ),
    "discounted_return": (("trajectory", "discounted_return"),),
}


# ── Transport resolution (reuse the one engine client; injectable for tests) ──
def _client(graph: str) -> Any:
    """Resolve the shared ``SyncEpistemicGraphClient`` for ``graph``.

    Delegates to :func:`engine_tools._client_for` so these tools ride the exact
    same connect path (CONCEPT:OS-5.63 resolver, connection caching) as the
    low-level ``engine_<domain>`` tools — one transport, no reinvention. Isolated
    behind this thin indirection so unit tests can inject a mock client
    (CONCEPT:KG-2.310).
    """
    from agent_utilities.mcp.tools import engine_tools

    return engine_tools._client_for(graph)


def _kv_backend() -> Any:
    """Build the KG-2.306 KV-cache connector from the engine's EG-187 environment.

    Isolated behind this indirection so unit tests can inject a fake backend
    (CONCEPT:KG-2.310).
    """
    from agent_utilities.kvcache import EpistemicGraphKVBackend

    return EpistemicGraphKVBackend.from_env()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, bytes | bytearray):
        return {"__bytes_b64__": base64.b64encode(bytes(obj)).decode("ascii")}
    return str(obj)


def _degraded(surface: str, action: str, tried: list[str]) -> str:
    """Clean 'this engine build lacks the surface' payload (never raise)."""
    return json.dumps(
        {
            "surface": surface,
            "action": action,
            "degraded": True,
            "error": (
                f"engine surface {surface!r} is not available in this engine build "
                "(no matching client method); this tool degrades cleanly and will "
                "activate once the engine ships the capability"
            ),
            "tried": tried,
        }
    )


def _resolve(client: Any, candidates: tuple[tuple[str, str], ...]) -> Any:
    """Return the first callable ``client.<sub>.<method>`` among ``candidates``.

    Returns ``None`` when none of the candidate surfaces are present so the caller
    can degrade gracefully (CONCEPT:KG-2.310).
    """
    for sub_attr, meth_attr in candidates:
        sub = getattr(client, sub_attr, None)
        if sub is None:
            continue
        fn = getattr(sub, meth_attr, None)
        if callable(fn):
            return fn
    return None


def _invoke(
    *,
    surface: str,
    action: str,
    graph: str,
    candidates: tuple[tuple[str, str], ...],
    params: dict[str, Any],
) -> str:
    """Resolve the client, dispatch to the first present surface, JSON the result.

    Every failure mode is returned as data, never raised: engine unreachable →
    ``error``; surface absent → ``degraded``; bad kwargs / engine error →
    ``error`` (CONCEPT:KG-2.310).
    """
    try:
        client = _client(graph)
    except Exception as exc:  # noqa: BLE001 — engine down is a normal degrade
        return json.dumps(
            {"surface": surface, "action": action, "error": f"engine unavailable: {exc}"}
        )
    fn = _resolve(client, candidates)
    if fn is None:
        return _degraded(surface, action, [f"{a}.{m}" for a, m in candidates])
    try:
        result = fn(**params)
    except TypeError as exc:
        return json.dumps(
            {"surface": surface, "action": action, "error": f"bad arguments: {exc}"}
        )
    except Exception as exc:  # noqa: BLE001 — surface engine errors as data
        return json.dumps({"surface": surface, "action": action, "error": str(exc)})
    return json.dumps(
        {"surface": surface, "action": action, "result": result}, default=_json_default
    )


def _drop_empty(**kwargs: Any) -> dict[str, Any]:
    """Keep only kwargs the caller actually supplied (non-empty string / non-None)."""
    return {k: v for k, v in kwargs.items() if v not in ("", None)}


def register_engine_surface_tools(mcp) -> None:
    """Register the KG-2.310 engine-surface tools + their REST twins.

    Each tool is added to ``REGISTERED_TOOLS`` and mapped to a ``/graph/<name>``
    route in ``ACTION_TOOL_ROUTES`` (auto-mounted by the generic REST factory), so
    MCP and REST stay in lockstep. CONCEPT:KG-2.310.
    """

    # ══════════════════════════════════════════════════════════════════
    # graph_broker — engine message broker (exchanges / queues / streams)
    # ══════════════════════════════════════════════════════════════════
    @mcp.tool(
        name="graph_broker",
        description=(
            "CONCEPT:KG-2.310 — the epistemic-graph engine message broker "
            "(AMQP-style exchanges + queues + streams), distinct from the "
            "agent-to-agent 'graph_bus'. Action-routed 1:1 over the engine broker "
            "surface: set 'action' to the broker method — e.g. 'declare_exchange' "
            "(exchange [+exchange_type]), 'declare_queue' (queue), 'bind' (queue + "
            "exchange [+routing_key]), 'publish' (exchange + routing_key + payload), "
            "'consume' (queue [+max_messages,+ack]), 'stats' / 'list_queues' / "
            "'list_exchanges'. Extra kwargs go via params_json. Degrades cleanly when "
            "the engine build has no broker surface."
        ),
        tags=["graph-os", "engine", "broker", "messaging"],
    )
    def graph_broker(
        action: str = Field(
            default="stats",
            description="Broker method: declare_exchange | declare_queue | bind | "
            "publish | consume | stats | list_queues | list_exchanges | ...",
        ),
        exchange: str = Field(default="", description="Exchange name."),
        queue: str = Field(default="", description="Queue name."),
        routing_key: str = Field(default="", description="Routing/binding key."),
        payload: str = Field(default="", description="Message body (publish)."),
        exchange_type: str = Field(
            default="", description="Exchange type: direct | fanout | topic (declare)."
        ),
        params_json: str = Field(
            default="{}",
            description="JSON object of extra kwargs, e.g. {\"max_messages\":10,"
            "\"ack\":true,\"durable\":true}. Merged over the typed fields.",
        ),
        graph: str = Field(
            default="", description="Target graph (empty ⇒ deployment default)."
        ),
    ) -> str:
        """Thin wrapper over the engine broker surface (CONCEPT:KG-2.310)."""
        if not action:
            return json.dumps({"surface": "broker", "error": "action is required"})
        try:
            extra = json.loads(params_json) if params_json else {}
        except (TypeError, ValueError) as exc:
            return json.dumps({"surface": "broker", "error": f"invalid params_json: {exc}"})
        if not isinstance(extra, dict):
            return json.dumps(
                {"surface": "broker", "error": "params_json must decode to an object"}
            )
        params = _drop_empty(
            exchange=exchange,
            queue=queue,
            routing_key=routing_key,
            payload=payload,
            exchange_type=exchange_type,
        )
        params.update(extra)
        return _invoke(
            surface="broker",
            action=action,
            graph=graph,
            candidates=(("broker", action),),
            params=params,
        )

    kg_server.REGISTERED_TOOLS["graph_broker"] = graph_broker
    kg_server.ACTION_TOOL_ROUTES["graph_broker"] = "/graph/broker"

    # ══════════════════════════════════════════════════════════════════
    # graph_kvcache — shared content-addressed KV-cache (EG-187 / KG-2.306)
    # ══════════════════════════════════════════════════════════════════
    @mcp.tool(
        name="graph_kvcache",
        description=(
            "CONCEPT:KG-2.310 — the engine's shared, content-addressed KV-cache over "
            "the EG-187 HTTP surface, driven through the KG-2.306 "
            "EpistemicGraphKVBackend connector. Actions: 'get' (key → base64 block "
            "bytes or miss), 'put' (key + value_b64 → stored bool), 'contains'/'exists' "
            "(key → bool), 'stats' (occupancy + dedup counters). The connector already "
            "degrades every transport error to a cache miss, so this tool never raises."
        ),
        tags=["graph-os", "engine", "kvcache"],
    )
    def graph_kvcache(
        action: str = Field(
            default="stats", description="get | put | contains | exists | stats"
        ),
        key: str = Field(default="", description="Opaque block key (get/put/contains/exists)."),
        value_b64: str = Field(
            default="", description="Base64-encoded block bytes to store (put)."
        ),
    ) -> str:
        """Thin wrapper over the KG-2.306 KV-cache connector (CONCEPT:KG-2.310)."""
        try:
            backend = _kv_backend()
        except Exception as exc:  # noqa: BLE001 — mis-config degrades, never raises
            return json.dumps(
                {"surface": "kvcache", "action": action, "error": f"kvcache unavailable: {exc}"}
            )
        try:
            if action == "get":
                if not key:
                    return json.dumps({"surface": "kvcache", "error": "key required"})
                blob = backend.get(key)
                return json.dumps(
                    {
                        "surface": "kvcache",
                        "action": action,
                        "hit": blob is not None,
                        "value_b64": (
                            base64.b64encode(blob).decode("ascii")
                            if blob is not None
                            else None
                        ),
                    }
                )
            if action == "put":
                if not key:
                    return json.dumps({"surface": "kvcache", "error": "key required"})
                try:
                    raw = base64.b64decode(value_b64) if value_b64 else b""
                except (ValueError, TypeError) as exc:
                    return json.dumps(
                        {"surface": "kvcache", "error": f"invalid value_b64: {exc}"}
                    )
                return json.dumps(
                    {"surface": "kvcache", "action": action, "stored": bool(backend.put(key, raw))}
                )
            if action in ("contains", "exists"):
                if not key:
                    return json.dumps({"surface": "kvcache", "error": "key required"})
                probe = backend.exists if action == "exists" else backend.contains
                return json.dumps(
                    {"surface": "kvcache", "action": action, "present": bool(probe(key))}
                )
            if action == "stats":
                stats = backend.stats()
                data = (
                    stats.model_dump()
                    if hasattr(stats, "model_dump")
                    else dict(stats)
                )
                return json.dumps(
                    {"surface": "kvcache", "action": action, "result": data},
                    default=_json_default,
                )
            return json.dumps({"surface": "kvcache", "error": f"unknown action {action!r}"})
        finally:
            close = getattr(backend, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001 — best-effort cleanup
                    pass

    kg_server.REGISTERED_TOOLS["graph_kvcache"] = graph_kvcache
    kg_server.ACTION_TOOL_ROUTES["graph_kvcache"] = "/graph/kvcache"

    # ══════════════════════════════════════════════════════════════════
    # graph_federated_search — search across registered external graphs
    # ══════════════════════════════════════════════════════════════════
    @mcp.tool(
        name="graph_federated_search",
        description=(
            "CONCEPT:KG-2.310 — federated search fanned across registered external "
            "graph references. Provide a natural-language / keyword 'query'; optionally "
            "scope to specific 'references' (comma-separated ExternalGraphReference ids) "
            "and cap with 'top_k'. Extra engine kwargs via params_json. Degrades cleanly "
            "when the engine build has no federated-search surface."
        ),
        tags=["graph-os", "engine", "search", "federated"],
    )
    def graph_federated_search(
        query: str = Field(description="Search query (natural language or keywords)."),
        references: str = Field(
            default="",
            description="Comma-separated external graph reference ids (empty ⇒ all).",
        ),
        top_k: int = Field(default=10, description="Max results to return."),
        params_json: str = Field(
            default="{}", description="JSON object of extra engine kwargs."
        ),
        graph: str = Field(
            default="", description="Target graph (empty ⇒ deployment default)."
        ),
    ) -> str:
        """Thin wrapper over the engine federated-search surface (CONCEPT:KG-2.310)."""
        try:
            extra = json.loads(params_json) if params_json else {}
        except (TypeError, ValueError) as exc:
            return json.dumps(
                {"surface": "federated_search", "error": f"invalid params_json: {exc}"}
            )
        if not isinstance(extra, dict):
            return json.dumps(
                {"surface": "federated_search", "error": "params_json must decode to an object"}
            )
        refs = [r.strip() for r in references.split(",") if r.strip()]
        params: dict[str, Any] = {"query": query, "top_k": int(top_k)}
        if refs:
            params["references"] = refs
        params.update(extra)
        return _invoke(
            surface="federated_search",
            action="search",
            graph=graph,
            candidates=_FEDERATED_SEARCH_CANDIDATES,
            params=params,
        )

    kg_server.REGISTERED_TOOLS["graph_federated_search"] = graph_federated_search
    kg_server.ACTION_TOOL_ROUTES["graph_federated_search"] = "/graph/federated-search"

    # ══════════════════════════════════════════════════════════════════
    # graph_promql — observability: PromQL metric queries
    # ══════════════════════════════════════════════════════════════════
    @mcp.tool(
        name="graph_promql",
        description=(
            "CONCEPT:KG-2.310 — query the engine's observability metrics with PromQL. "
            "action='instant' (a single evaluation at 'time', default now) or 'range' "
            "(over start..end at 'step'). Extra engine kwargs via params_json. Degrades "
            "cleanly when the engine build has no metrics/PromQL surface."
        ),
        tags=["graph-os", "engine", "observability", "metrics"],
    )
    def graph_promql(
        query: str = Field(description="A PromQL expression."),
        action: str = Field(default="instant", description="instant | range"),
        time: str = Field(default="", description="Evaluation time (instant), RFC3339/unix."),
        start: str = Field(default="", description="Range start (range)."),
        end: str = Field(default="", description="Range end (range)."),
        step: str = Field(default="", description="Range step, e.g. '30s' (range)."),
        params_json: str = Field(
            default="{}", description="JSON object of extra engine kwargs."
        ),
        graph: str = Field(
            default="", description="Target graph (empty ⇒ deployment default)."
        ),
    ) -> str:
        """Thin wrapper over the engine PromQL surface (CONCEPT:KG-2.310)."""
        try:
            extra = json.loads(params_json) if params_json else {}
        except (TypeError, ValueError) as exc:
            return json.dumps({"surface": "promql", "error": f"invalid params_json: {exc}"})
        if not isinstance(extra, dict):
            return json.dumps({"surface": "promql", "error": "params_json must decode to an object"})
        if action == "range":
            params = _drop_empty(query=query, start=start, end=end, step=step)
            candidates = _PROMQL_RANGE_CANDIDATES
        elif action == "instant":
            params = _drop_empty(query=query, time=time)
            candidates = _PROMQL_INSTANT_CANDIDATES
        else:
            return json.dumps({"surface": "promql", "error": f"unknown action {action!r}"})
        params.update(extra)
        return _invoke(
            surface="promql",
            action=action,
            graph=graph,
            candidates=candidates,
            params=params,
        )

    kg_server.REGISTERED_TOOLS["graph_promql"] = graph_promql
    kg_server.ACTION_TOOL_ROUTES["graph_promql"] = "/graph/promql"

    # ══════════════════════════════════════════════════════════════════
    # graph_traces — observability: distributed-trace search / fetch
    # ══════════════════════════════════════════════════════════════════
    @mcp.tool(
        name="graph_traces",
        description=(
            "CONCEPT:KG-2.310 — search or fetch distributed traces from the engine's "
            "observability surface. action='search' (filter by 'service'/'operation'/"
            "free-form 'query', capped by 'limit') or 'get' (a single 'trace_id'). Extra "
            "engine kwargs via params_json. Degrades cleanly when the engine build has "
            "no trace surface."
        ),
        tags=["graph-os", "engine", "observability", "traces"],
    )
    def graph_traces(
        action: str = Field(default="search", description="search | get"),
        trace_id: str = Field(default="", description="Trace id (action='get')."),
        service: str = Field(default="", description="Service name filter (search)."),
        operation: str = Field(default="", description="Operation/span name filter (search)."),
        query: str = Field(default="", description="Free-form filter expression (search)."),
        limit: int = Field(default=20, description="Max traces to return (search)."),
        params_json: str = Field(
            default="{}", description="JSON object of extra engine kwargs."
        ),
        graph: str = Field(
            default="", description="Target graph (empty ⇒ deployment default)."
        ),
    ) -> str:
        """Thin wrapper over the engine trace surface (CONCEPT:KG-2.310)."""
        try:
            extra = json.loads(params_json) if params_json else {}
        except (TypeError, ValueError) as exc:
            return json.dumps({"surface": "traces", "error": f"invalid params_json: {exc}"})
        if not isinstance(extra, dict):
            return json.dumps({"surface": "traces", "error": "params_json must decode to an object"})
        if action == "get":
            if not trace_id:
                return json.dumps({"surface": "traces", "error": "trace_id required"})
            params: dict[str, Any] = {"trace_id": trace_id}
            candidates = _TRACES_GET_CANDIDATES
        elif action == "search":
            params = _drop_empty(service=service, operation=operation, query=query)
            params["limit"] = int(limit)
            candidates = _TRACES_SEARCH_CANDIDATES
        else:
            return json.dumps({"surface": "traces", "error": f"unknown action {action!r}"})
        params.update(extra)
        return _invoke(
            surface="traces",
            action=action,
            graph=graph,
            candidates=candidates,
            params=params,
        )

    kg_server.REGISTERED_TOOLS["graph_traces"] = graph_traces
    kg_server.ACTION_TOOL_ROUTES["graph_traces"] = "/graph/traces"

    # ══════════════════════════════════════════════════════════════════
    # graph_gis — geospatial route / tile / geo-task ops
    # ══════════════════════════════════════════════════════════════════
    @mcp.tool(
        name="graph_gis",
        description=(
            "CONCEPT:KG-2.310 — the engine's GIS surface. Action-routed 1:1 over the "
            "engine geo methods: e.g. 'route' (from + to [+profile]), 'tile' (z/x/y), "
            "'nearest' (lat + lon [+limit]), 'geo_task' (a named geospatial job). All "
            "structured args go via params_json (e.g. {\"from\":[lat,lon],\"to\":[lat,"
            "lon]}). Degrades cleanly when the engine build has no GIS surface."
        ),
        tags=["graph-os", "engine", "gis", "geospatial"],
    )
    def graph_gis(
        action: str = Field(
            default="route",
            description="GIS method: route | tile | nearest | geo_task | ...",
        ),
        params_json: str = Field(
            default="{}",
            description="JSON object of kwargs for the GIS method (coordinates, "
            "profile, tile z/x/y, task name, ...).",
        ),
        graph: str = Field(
            default="", description="Target graph (empty ⇒ deployment default)."
        ),
    ) -> str:
        """Thin wrapper over the engine GIS surface (CONCEPT:KG-2.310)."""
        if not action:
            return json.dumps({"surface": "gis", "error": "action is required"})
        try:
            params = json.loads(params_json) if params_json else {}
        except (TypeError, ValueError) as exc:
            return json.dumps({"surface": "gis", "error": f"invalid params_json: {exc}"})
        if not isinstance(params, dict):
            return json.dumps({"surface": "gis", "error": "params_json must decode to an object"})
        return _invoke(
            surface="gis",
            action=action,
            graph=graph,
            candidates=(("gis", action), ("geo", action)),
            params=params,
        )

    kg_server.REGISTERED_TOOLS["graph_gis"] = graph_gis
    kg_server.ACTION_TOOL_ROUTES["graph_gis"] = "/graph/gis"

    # ══════════════════════════════════════════════════════════════════
    # graph_memory — EG-318 memory / scene / trajectory engine ops
    # ══════════════════════════════════════════════════════════════════
    @mcp.tool(
        name="graph_memory",
        description=(
            "CONCEPT:KG-2.310 — the engine's EG-318 memory surface: episodic→semantic "
            "memory, the spatial scene graph, and RL trajectories. Action-routed 1:1 "
            "over the engine memory methods (dashes normalize to underscores): "
            "'create_summary' (episodic nodes → a summary node — node_ids [+window]), "
            "'consolidate' (roll episodic into semantic memory), 'maintain' (decay / "
            "prune / re-index the memory store), 'add_scene_object' (object_id [+pose/"
            "transform/parent]), 'world_transform' (object_id + transform → world pose), "
            "'start_trajectory' (agent/episode → trajectory_id), 'append_step' "
            "(trajectory_id + step {state,action,reward,...}), 'discounted_return' "
            "(trajectory_id [+gamma]). Read ops (e.g. 'get_summary', 'get_scene', "
            "'get_trajectory') route by action name too. Structured args go via "
            "params_json. Degrades cleanly when the engine build has no memory surface."
        ),
        tags=["graph-os", "engine", "memory", "scene", "trajectory"],
    )
    def graph_memory(
        action: str = Field(
            default="consolidate",
            description="Memory method: create_summary | consolidate | maintain | "
            "add_scene_object | world_transform | start_trajectory | append_step | "
            "discounted_return | get_* | ...",
        ),
        params_json: str = Field(
            default="{}",
            description="JSON object of kwargs for the memory method, e.g. "
            '{"node_ids":["n1","n2"]}, {"object_id":"o1","transform":[...]}, '
            '{"trajectory_id":"t1","step":{"state":...,"action":...,"reward":1.0}}, '
            'or {"trajectory_id":"t1","gamma":0.99}.',
        ),
        graph: str = Field(
            default="", description="Target graph (empty ⇒ deployment default)."
        ),
    ) -> str:
        """Thin wrapper over the engine EG-318 memory surface (CONCEPT:KG-2.310)."""
        action = (action or "").strip().replace("-", "_")
        if not action:
            return json.dumps({"surface": "memory", "error": "action is required"})
        try:
            params = json.loads(params_json) if params_json else {}
        except (TypeError, ValueError) as exc:
            return json.dumps({"surface": "memory", "error": f"invalid params_json: {exc}"})
        if not isinstance(params, dict):
            return json.dumps(
                {"surface": "memory", "error": "params_json must decode to an object"}
            )
        candidates = _MEMORY_ACTION_CANDIDATES.get(
            action,
            (("memory", action), ("scene", action), ("trajectory", action)),
        )
        return _invoke(
            surface="memory",
            action=action,
            graph=graph,
            candidates=candidates,
            params=params,
        )

    kg_server.REGISTERED_TOOLS["graph_memory"] = graph_memory
    kg_server.ACTION_TOOL_ROUTES["graph_memory"] = "/graph/memory"
