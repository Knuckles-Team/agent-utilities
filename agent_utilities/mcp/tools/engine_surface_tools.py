"""New engine-surface MCP tools — CONCEPT:AU-KG.coordination.engine-message-broker.

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

CONCEPT:AU-KG.coordination.engine-message-broker — MCP surface for the new engine ops (broker / kvcache /
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
    same connect path (CONCEPT:AU-OS.deployment.engine-resolver-auto-provision resolver, connection caching) as the
    low-level ``engine_<domain>`` tools — one transport, no reinvention. Isolated
    behind this thin indirection so unit tests can inject a mock client
    (CONCEPT:AU-KG.coordination.engine-message-broker).
    """
    from agent_utilities.mcp.tools import engine_tools

    return engine_tools._client_for(graph)


def _kv_backend() -> Any:
    """Build the KG-2.306 KV-cache connector from the engine's EG-187 environment.

    Isolated behind this indirection so unit tests can inject a fake backend
    (CONCEPT:AU-KG.coordination.engine-message-broker).
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
    can degrade gracefully (CONCEPT:AU-KG.coordination.engine-message-broker).
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
    ``error`` (CONCEPT:AU-KG.coordination.engine-message-broker).
    """
    try:
        client = _client(graph)
    except Exception as exc:  # noqa: BLE001 — engine down is a normal degrade
        return json.dumps(
            {
                "surface": surface,
                "action": action,
                "error": f"engine unavailable: {exc}",
            }
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


def _run_coro(coro: Any) -> Any:
    """Run an async coroutine from a sync MCP handler (loop-running or not)."""
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(lambda: asyncio.run(coro)).result()


# CONCEPT:AU-KG.memory.unified-memory-crud-core — the unified memory-CRUD core. graph_memory's recall/store/link
# actions route into the SAME ``graph_write`` tool the REST ``/graph/write/memory``
# [``/recall``] twins and the harness ``kg_memory_recall``/``kg_memory_store`` tools
# already use — one core, no fourth memory surface. ``link`` reuses graph_write's
# ``add_edge`` so relating two memories rides the same mutation path.
_MEMORY_CRUD_ACTIONS = ("recall", "store", "link")


def _memory_crud(action: str, params: dict[str, Any]) -> str:
    """Dispatch recall/store/link into the shared ``graph_write`` memory core."""
    if action == "store":
        call = dict(
            action="store_memory",
            agent_id=params.get("agent_id", params.get("agent", "")),
            node_type=params.get("memory_type", params.get("type", "")),
            properties=params.get("content", params.get("properties", "")),
            nodes=json.dumps(params.get("tags", [])),
        )
    elif action == "recall":
        call = dict(
            action="recall_memory",
            properties=params.get("query", params.get("content", "")),
            node_type=params.get("memory_type", params.get("type", "")),
        )
    else:  # link
        src = params.get("source", params.get("source_id", ""))
        tgt = params.get("target", params.get("target_id", ""))
        if not src or not tgt:
            return json.dumps(
                {
                    "surface": "memory",
                    "action": "link",
                    "error": "link requires 'source' and 'target' (memory node ids)",
                }
            )
        call = dict(
            action="add_edge",
            source_id=src,
            target_id=tgt,
            rel_type=params.get("rel_type", "RELATES_TO"),
            properties=json.dumps(params.get("properties", {})),
        )
    try:
        result = _run_coro(kg_server._execute_tool("graph_write", **call))
    except Exception as exc:  # noqa: BLE001 — surface engine/core errors as data
        return json.dumps({"surface": "memory", "action": action, "error": str(exc)})
    return json.dumps(
        {"surface": "memory", "action": action, "result": result}, default=_json_default
    )


def _pick_warm_fork_sandbox(preferred: str = "") -> Any:
    """Return the cheapest available warm-fork rung, or ``None`` (CONCEPT:AU-KG.coordination.warm-fork-fanout).

    Reuses the ORCH-1.86 sandbox registry (``default_sandboxes()``, cheapest-first)
    and selects the first backend whose capabilities advertise ``warm_fork`` and
    which is available on this host. ``preferred`` pins a rung by name when set.
    """
    try:
        from agent_utilities.rlm.sandboxes.registry import default_sandboxes
    except Exception:  # noqa: BLE001 — subsystem unimportable ⇒ degrade cleanly
        return None

    forkable = [
        sb
        for sb in default_sandboxes()
        if getattr(getattr(sb, "capabilities", None), "warm_fork", False)
    ]
    if preferred:
        forkable = [sb for sb in forkable if sb.name == preferred] or forkable
    for sb in forkable:
        try:
            if sb.is_available():
                return sb
        except Exception:  # noqa: BLE001 — an unprobeable rung is simply skipped
            continue
    return None


def _fork_fanout(branches: list[Any], seed_vars: dict[str, Any], preferred: str) -> str:
    """Warm a fork parent once and fan out ``branches``, returning per-branch results.

    Backed by the ORCH-1.86..93 warm-fork primitive: the base
    :class:`ForkableSandbox.execute` warms-or-reuses one parent (copy-on-write) and
    forks a child per branch. Fan-out is concurrent ``execute`` calls sharing that
    one warm parent. Degrades cleanly to a structured ``unavailable`` payload when no
    warm-fork rung is available on this host (CONCEPT:AU-KG.coordination.warm-fork-fanout).
    """
    sb = _pick_warm_fork_sandbox(preferred)
    if sb is None:
        return json.dumps(
            {
                "surface": "fork",
                "degraded": True,
                "error": (
                    "no warm-fork rung available on this host (forkserver needs a "
                    "POSIX-fork-capable interpreter; container_fork needs a docker "
                    "runtime; firecracker needs a reachable forkd + KVM), and the "
                    "epistemic-graph engine client exposes no warm-fork primitive"
                ),
                "followup": (
                    "spike: surface a first-class engine warm-fork/KV-cache-fork op "
                    "on the epistemic_graph client (LMCacheMPConnector snapshot → "
                    "branch), then route graph_fork to it when present, keeping the "
                    "local ForkableSandbox rungs as the fallback"
                ),
                "branch_count": len(branches),
            }
        )

    from agent_utilities.rlm.sandboxes.base import SandboxEnv

    async def _run_all() -> list[dict[str, Any]]:
        import asyncio

        async def _one(idx: int, snippet: Any) -> dict[str, Any]:
            try:
                result = await sb.execute(
                    str(snippet), SandboxEnv(vars=dict(seed_vars))
                )
                return {
                    "index": idx,
                    "ok": result.error is None,
                    "stdout": result.stdout,
                    "error": result.error,
                    "vars": result.updated_vars,
                }
            except Exception as exc:  # noqa: BLE001 — one branch never fails the set
                return {"index": idx, "ok": False, "error": str(exc)}

        return await asyncio.gather(*(_one(i, s) for i, s in enumerate(branches)))

    try:
        results = _run_coro(_run_all())
    except Exception as exc:  # noqa: BLE001 — infra death → structured error, no crash
        return json.dumps({"surface": "fork", "sandbox": sb.name, "error": str(exc)})
    return json.dumps(
        {
            "surface": "fork",
            "sandbox": sb.name,
            "branch_count": len(results),
            "branches": results,
        },
        default=_json_default,
    )


def _crossmodal_fork_fanout(
    branches: list[Any],
    seed_vars: dict[str, Any],
    preferred: str,
    context_query: str,
    candidate_var: str,
) -> str:
    """Retrieve an engine cross-modal candidate set ONCE, then warm-fork ``branches`` over it.

    The agent-utilities side of the epistemic-graph cross-modal seam
    (CONCEPT:AU-ORCH.sandbox.crossmodal-fork-fanout): a vector+graph+text candidate set is
    retrieved once for ``context_query`` and forked into every branch as ``candidate_var`` —
    branches reuse that one context with no recompute. Falls back to the structured degraded
    payload when no warm-fork rung is available or the engine retriever is unreachable.
    """
    try:
        from agent_utilities.runtime.crossmodal_fork import CrossModalForkFanout
    except Exception as exc:  # noqa: BLE001 — capability unimportable ⇒ degrade cleanly
        return json.dumps(
            {"surface": "fork", "context_query": context_query, "error": str(exc)}
        )

    fanout = CrossModalForkFanout()

    async def _run() -> Any:
        return await fanout.fan_out(
            context_query,
            [str(b) for b in branches],
            preferred=preferred,
            candidate_var=candidate_var,
            extra_vars=seed_vars or None,
        )

    try:
        res = _run_coro(_run())
    except Exception as exc:  # noqa: BLE001 — engine/infra death → structured error, no crash
        return json.dumps(
            {"surface": "fork", "context_query": context_query, "error": str(exc)}
        )
    return json.dumps(
        {
            "surface": "fork",
            "context_query": context_query,
            "candidate_var": candidate_var,
            "candidate_count": res.candidate_count,
            "retrieval_calls": res.retrieval_calls,
            "reused_without_recompute": res.reused_without_recompute,
            "degraded": res.degraded,
            "error": res.error,
            "sandbox": res.sandbox,
            "branch_count": len(res.branches),
            "branches": [
                {
                    "index": b.index,
                    "ok": b.ok,
                    "stdout": b.stdout,
                    "error": b.error,
                    "output": b.output,
                }
                for b in res.branches
            ],
        },
        default=_json_default,
    )


def register_engine_surface_tools(mcp) -> None:
    """Register the KG-2.310 engine-surface tools + their REST twins.

    Each tool is added to ``REGISTERED_TOOLS`` and mapped to a ``/graph/<name>``
    route in ``ACTION_TOOL_ROUTES`` (auto-mounted by the generic REST factory), so
    MCP and REST stay in lockstep. CONCEPT:AU-KG.coordination.engine-message-broker.
    """

    # ══════════════════════════════════════════════════════════════════
    # graph_broker — engine message broker (exchanges / queues / streams)
    # ══════════════════════════════════════════════════════════════════
    @mcp.tool(
        name="graph_broker",
        description=(
            "CONCEPT:AU-KG.coordination.engine-message-broker — the epistemic-graph engine message broker "
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
            description='JSON object of extra kwargs, e.g. {"max_messages":10,'
            '"ack":true,"durable":true}. Merged over the typed fields.',
        ),
        graph: str = Field(
            default="", description="Target graph (empty ⇒ deployment default)."
        ),
    ) -> str:
        """Thin wrapper over the engine broker surface (CONCEPT:AU-KG.coordination.engine-message-broker)."""
        if not action:
            return json.dumps({"surface": "broker", "error": "action is required"})
        try:
            extra = json.loads(params_json) if params_json else {}
        except (TypeError, ValueError) as exc:
            return json.dumps(
                {"surface": "broker", "error": f"invalid params_json: {exc}"}
            )
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
            "CONCEPT:AU-KG.coordination.engine-message-broker — the engine's shared, content-addressed KV-cache over "
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
        key: str = Field(
            default="", description="Opaque block key (get/put/contains/exists)."
        ),
        value_b64: str = Field(
            default="", description="Base64-encoded block bytes to store (put)."
        ),
    ) -> str:
        """Thin wrapper over the KG-2.306 KV-cache connector (CONCEPT:AU-KG.coordination.engine-message-broker)."""
        try:
            backend = _kv_backend()
        except Exception as exc:  # noqa: BLE001 — mis-config degrades, never raises
            return json.dumps(
                {
                    "surface": "kvcache",
                    "action": action,
                    "error": f"kvcache unavailable: {exc}",
                }
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
                    {
                        "surface": "kvcache",
                        "action": action,
                        "stored": bool(backend.put(key, raw)),
                    }
                )
            if action in ("contains", "exists"):
                if not key:
                    return json.dumps({"surface": "kvcache", "error": "key required"})
                probe = backend.exists if action == "exists" else backend.contains
                return json.dumps(
                    {
                        "surface": "kvcache",
                        "action": action,
                        "present": bool(probe(key)),
                    }
                )
            if action == "stats":
                stats = backend.stats()
                data = (
                    stats.model_dump() if hasattr(stats, "model_dump") else dict(stats)
                )
                return json.dumps(
                    {"surface": "kvcache", "action": action, "result": data},
                    default=_json_default,
                )
            return json.dumps(
                {"surface": "kvcache", "error": f"unknown action {action!r}"}
            )
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
            "CONCEPT:AU-KG.coordination.engine-message-broker — federated search fanned across registered external "
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
        """Thin wrapper over the engine federated-search surface (CONCEPT:AU-KG.coordination.engine-message-broker)."""
        try:
            extra = json.loads(params_json) if params_json else {}
        except (TypeError, ValueError) as exc:
            return json.dumps(
                {"surface": "federated_search", "error": f"invalid params_json: {exc}"}
            )
        if not isinstance(extra, dict):
            return json.dumps(
                {
                    "surface": "federated_search",
                    "error": "params_json must decode to an object",
                }
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
            "CONCEPT:AU-KG.coordination.engine-message-broker — query the engine's observability metrics with PromQL. "
            "action='instant' (a single evaluation at 'time', default now) or 'range' "
            "(over start..end at 'step'). Extra engine kwargs via params_json. Degrades "
            "cleanly when the engine build has no metrics/PromQL surface."
        ),
        tags=["graph-os", "engine", "observability", "metrics"],
    )
    def graph_promql(
        query: str = Field(description="A PromQL expression."),
        action: str = Field(default="instant", description="instant | range"),
        time: str = Field(
            default="", description="Evaluation time (instant), RFC3339/unix."
        ),
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
        """Thin wrapper over the engine PromQL surface (CONCEPT:AU-KG.coordination.engine-message-broker)."""
        try:
            extra = json.loads(params_json) if params_json else {}
        except (TypeError, ValueError) as exc:
            return json.dumps(
                {"surface": "promql", "error": f"invalid params_json: {exc}"}
            )
        if not isinstance(extra, dict):
            return json.dumps(
                {"surface": "promql", "error": "params_json must decode to an object"}
            )
        if action == "range":
            params = _drop_empty(query=query, start=start, end=end, step=step)
            candidates = _PROMQL_RANGE_CANDIDATES
        elif action == "instant":
            params = _drop_empty(query=query, time=time)
            candidates = _PROMQL_INSTANT_CANDIDATES
        else:
            return json.dumps(
                {"surface": "promql", "error": f"unknown action {action!r}"}
            )
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
            "CONCEPT:AU-KG.coordination.engine-message-broker — search or fetch distributed traces from the engine's "
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
        operation: str = Field(
            default="", description="Operation/span name filter (search)."
        ),
        query: str = Field(
            default="", description="Free-form filter expression (search)."
        ),
        limit: int = Field(default=20, description="Max traces to return (search)."),
        params_json: str = Field(
            default="{}", description="JSON object of extra engine kwargs."
        ),
        graph: str = Field(
            default="", description="Target graph (empty ⇒ deployment default)."
        ),
    ) -> str:
        """Thin wrapper over the engine trace surface (CONCEPT:AU-KG.coordination.engine-message-broker)."""
        try:
            extra = json.loads(params_json) if params_json else {}
        except (TypeError, ValueError) as exc:
            return json.dumps(
                {"surface": "traces", "error": f"invalid params_json: {exc}"}
            )
        if not isinstance(extra, dict):
            return json.dumps(
                {"surface": "traces", "error": "params_json must decode to an object"}
            )
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
            return json.dumps(
                {"surface": "traces", "error": f"unknown action {action!r}"}
            )
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
            "CONCEPT:AU-KG.coordination.engine-message-broker — the engine's GIS surface. Action-routed 1:1 over the "
            "engine geo methods: e.g. 'route' (from + to [+profile]), 'tile' (z/x/y), "
            "'nearest' (lat + lon [+limit]), 'geo_task' (a named geospatial job). All "
            'structured args go via params_json (e.g. {"from":[lat,lon],"to":[lat,'
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
        """Thin wrapper over the engine GIS surface (CONCEPT:AU-KG.coordination.engine-message-broker)."""
        if not action:
            return json.dumps({"surface": "gis", "error": "action is required"})
        try:
            params = json.loads(params_json) if params_json else {}
        except (TypeError, ValueError) as exc:
            return json.dumps(
                {"surface": "gis", "error": f"invalid params_json: {exc}"}
            )
        if not isinstance(params, dict):
            return json.dumps(
                {"surface": "gis", "error": "params_json must decode to an object"}
            )
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
            "CONCEPT:AU-KG.coordination.engine-message-broker — the engine's EG-318 memory surface: episodic→semantic "
            "memory, the spatial scene graph, and RL trajectories. Action-routed 1:1 "
            "over the engine memory methods (dashes normalize to underscores): "
            "'create_summary' (episodic nodes → a summary node — node_ids [+window]), "
            "'consolidate' (roll episodic into semantic memory), 'maintain' (decay / "
            "prune / re-index the memory store), 'add_scene_object' (object_id [+pose/"
            "transform/parent]), 'world_transform' (object_id + transform → world pose), "
            "'start_trajectory' (agent/episode → trajectory_id), 'append_step' "
            "(trajectory_id + step {state,action,reward,...}), 'discounted_return' "
            "(trajectory_id [+gamma]). Read ops (e.g. 'get_summary', 'get_scene', "
            "'get_trajectory') route by action name too. UNIFIED memory-CRUD "
            "(CONCEPT:AU-KG.memory.unified-memory-crud-core) — 'store' (agent_id + content [+memory_type,+tags]), "
            "'recall' (query [+memory_type]), 'link' (source + target [+rel_type]) — "
            "route into the SAME graph_write memory core as the REST "
            "/graph/write/memory[/recall] twins and the harness kg_memory_recall/store "
            "tools (one core, no separate surface). Structured args go via params_json. "
            "Degrades cleanly when the engine build has no memory surface."
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
        """Thin wrapper over the engine EG-318 memory surface (CONCEPT:AU-KG.coordination.engine-message-broker)."""
        action = (action or "").strip().replace("-", "_")
        if not action:
            return json.dumps({"surface": "memory", "error": "action is required"})
        try:
            params = json.loads(params_json) if params_json else {}
        except (TypeError, ValueError) as exc:
            return json.dumps(
                {"surface": "memory", "error": f"invalid params_json: {exc}"}
            )
        if not isinstance(params, dict):
            return json.dumps(
                {"surface": "memory", "error": "params_json must decode to an object"}
            )
        # CONCEPT:AU-KG.memory.unified-memory-crud-core — unified memory-CRUD short-circuit: recall/store/link go
        # to the shared graph_write memory core (same as REST + harness), not the
        # engine EG-318 surface.
        if action in _MEMORY_CRUD_ACTIONS:
            return _memory_crud(action, params)
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

    # ══════════════════════════════════════════════════════════════════
    # graph_mine — data-mining surface (CONCEPT:EG-KG.mining.frequent-itemset-mining)
    # ══════════════════════════════════════════════════════════════════
    @mcp.tool(
        name="graph_mine",
        description=(
            "CONCEPT:EG-KG.mining.frequent-itemset-mining — the unified data-mining surface "
            "over the engine, compute-near-data (mining runs where the graph lives). "
            "Actions: 'associate' (association rules), 'cluster' (clustering), "
            "'anomaly' (outlier detection), 'classify_fit'/'classify_predict' "
            "(classification), 'reduce' (dimensionality reduction), 'sequence' "
            "(sequential-pattern mining). "
            "• associate — frequent-itemset + rules (Apriori/FP-Growth/Eclat; support, "
            "confidence, lift). Provide 'transactions' (baskets of item labels) OR a "
            "graph-derived 'source' {node_label, direction(out|in|any), "
            "item_field(label|prop:<key>), relation, limit}. writeback ⇒ :AssociationRule nodes. "
            "• cluster — DBSCAN(default)/hierarchical/gmm/kmedoids over 'features' (a row "
            "matrix) OR a vector 'source' {node_label, limit} (the stored embeddings of "
            "those nodes — cross-modal 'cluster the vectors of these nodes'). Params: "
            "eps/min_pts (dbscan), k, linkage(single|complete|average), max_iter, seed. "
            "writeback ⇒ :Cluster nodes linked to members. Returns "
            "{clusters:[{cluster_id,members,centroid,score}], labels, ...} (gmm adds "
            "responsibilities). "
            "• anomaly — zscore(default)/isoforest/lof/ocsvm over 'features', a 1-D 'values' "
            "series (tsdb RCA), OR a vector 'source'. Params: k(lof), n_trees/sample_size/seed "
            "(isoforest), nu/kernel(rbf|linear)/gamma (ocsvm), threshold. writeback ⇒ "
            ":Anomaly nodes linked to their source. Returns {rows:[{id,anomaly_score,"
            "is_anomaly}], n_anomalies, threshold, ...}. "
            "• classify_fit — PREDICTIVE fit → model blob: gaussiannb(default)/multinomialnb/"
            "knn/logistic/svc over 'x' (rows) OR a vector 'source', plus integer 'y' labels. "
            "Params: k(knn), alpha(mnb), lr/epochs/l2(logistic), c(svc). Returns {model, "
            "classes, ...} — pass 'model' to classify_predict. "
            "• classify_predict — apply a fitted 'model' to 'x' OR a vector 'source' "
            "(cross-modal 'classify these nodes by their embeddings'). writeback ⇒ "
            ":Classification nodes linked to source. Returns {rows:[{id,label,proba}], classes}. "
            "• reduce — DESCRIPTIVE row transform: svd(default)/lda(supervised — needs "
            "'labels')/umap/tsne over 'x' OR a vector 'source' (reduce node vectors for the "
            "graphviz). Params: n_components, n_neighbors/min_dist(umap), perplexity/lr(tsne), "
            "epochs, seed. writeback ⇒ :Embedding2D nodes. Returns {rows:[{id,coords}], "
            "n_components, ...} (svd adds singular_values). UMAP/t-SNE are approximate, small-N. "
            "• sequence — frequent ORDERED subsequences: prefixspan(default)/gsp (both agree) "
            "over 'sequences' (time-ordered lists of item labels, repeats allowed) OR a "
            "graph-derived 'source' {node_label, direction(out|in|any), item_field, relation, "
            "limit} (each node's chronological neighbor history becomes one sequence — "
            "'what reliably follows what'). Params: min_support. writeback ⇒ :SequentialPattern "
            "nodes linked to their item nodes. Returns {patterns:[{items,support,count}], "
            "n_sequences, n_patterns, ...}. "
            "REST twins: POST /api/mining/{associate,cluster,anomaly,classify_fit,"
            "classify_predict,reduce,sequence} (same _execute_tool core). Degrades cleanly on a "
            "no-mining engine build."
        ),
        tags=["graph-os", "engine", "mining", "clustering", "anomaly", "data-mining"],
    )
    def graph_mine(
        action: str = Field(
            default="associate",
            description="Mining action: 'associate' | 'cluster' | 'anomaly' | "
            "'classify_fit' | 'classify_predict' | 'reduce' | 'sequence'.",
        ),
        params_json: str = Field(
            default="{}",
            description="JSON object of mining kwargs, e.g. "
            '{"transactions":[["bread","milk"],["bread","butter"]],'
            '"min_support":0.5,"algorithm":"fpgrowth"} (associate); '
            '{"features":[[0,0],[10,10]],"algorithm":"dbscan","eps":1.0,"min_pts":2} '
            'or {"source":{"node_label":"Doc"},"algorithm":"kmedoids","k":3,'
            '"writeback":true} (cluster); '
            '{"values":[1,1,1,100],"algorithm":"zscore"} or '
            '{"source":{"node_label":"Metric"},"algorithm":"isoforest"} (anomaly); '
            '{"x":[[0,0],[10,10]],"y":[0,1],"algorithm":"logistic"} (classify_fit); '
            '{"model":{...},"x":[[0.1,0.1]]} (classify_predict); '
            '{"x":[[..]],"algorithm":"svd","n_components":2} or '
            '{"source":{"node_label":"Doc"},"algorithm":"umap","writeback":true} (reduce); '
            '{"sequences":[["login","browse","purchase"]],"min_support":0.5} or '
            '{"source":{"node_label":"Session"},"algorithm":"gsp","writeback":true} (sequence).',
        ),
        graph: str = Field(
            default="", description="Target graph (empty ⇒ deployment default)."
        ),
    ) -> str:
        """Thin action-router over the engine mining surface (CONCEPT:EG-KG.mining.frequent-itemset-mining)."""
        action = (action or "").strip().replace("-", "_") or "associate"
        try:
            params = json.loads(params_json) if params_json else {}
        except (TypeError, ValueError) as exc:
            return json.dumps(
                {"surface": "mining", "error": f"invalid params_json: {exc}"}
            )
        if not isinstance(params, dict):
            return json.dumps(
                {"surface": "mining", "error": "params_json must decode to an object"}
            )
        return _invoke(
            surface="mining",
            action=action,
            graph=graph,
            candidates=((("mining", action),)),
            params=params,
        )

    kg_server.REGISTERED_TOOLS["graph_mine"] = graph_mine
    # REST twin path: POST {prefix}/mining/associate (mounted bespoke in kg_server so
    # a natural mining body works while dispatching the SAME _execute_tool core).
    kg_server.ACTION_TOOL_ROUTES["graph_mine"] = "/mining/associate"

    # ══════════════════════════════════════════════════════════════════
    # graph_fork — warm-fork / KV-cache fan-out (CONCEPT:AU-KG.coordination.warm-fork-fanout)
    # ══════════════════════════════════════════════════════════════════
    @mcp.tool(
        name="graph_fork",
        description=(
            "CONCEPT:AU-KG.coordination.warm-fork-fanout — warm-fork fan-out over the ORCH-1.86..93 warm-fork "
            "primitive (LMCache KV / copy-on-write sandboxes): pay warm-up ONCE for a "
            "parent context, then fork N copy-on-write branches to run per-branch "
            "computations concurrently and return each branch's result. Provide either "
            "'branches_json' (a JSON list of per-branch code snippets) or 'code' + 'n' "
            "(run the same snippet across n branches); 'vars_json' seeds the shared "
            "namespace forked into every branch; 'sandbox' optionally pins a rung "
            "(forkserver | container_fork | firecracker), else the cheapest available "
            "warm-fork rung is used. Set 'context_query' to retrieve an engine cross-modal "
            "candidate set (vector+graph+text fusion) ONCE and fork it into every branch as "
            "'candidate_var' (default 'candidates') — the branches reuse that one context with "
            "no recompute (CONCEPT:AU-ORCH.sandbox.crossmodal-fork-fanout). Degrades cleanly "
            "(structured 'unavailable') when no warm-fork rung is available on this host."
        ),
        tags=["graph-os", "engine", "fork", "warm-fork", "fanout"],
    )
    def graph_fork(
        code: str = Field(
            default="",
            description="A single code snippet run on each of 'n' branches (ignored "
            "when 'branches_json' is provided).",
        ),
        n: int = Field(
            default=0, description="Fan-out count when using 'code' (branches to fork)."
        ),
        branches_json: str = Field(
            default="[]",
            description="JSON list of per-branch code snippets; overrides code/n.",
        ),
        vars_json: str = Field(
            default="{}",
            description="JSON object seeding the namespace forked into every branch.",
        ),
        sandbox: str = Field(
            default="",
            description="Preferred warm-fork rung name (empty ⇒ cheapest available).",
        ),
        context_query: str = Field(
            default="",
            description="Optional: retrieve an engine cross-modal candidate set (vector+graph"
            "+text) for this query ONCE and fork it into every branch (reused, no recompute).",
        ),
        candidate_var: str = Field(
            default="candidates",
            description="Namespace name the cross-modal candidate set is bound to in each branch "
            "(only used when context_query is set).",
        ),
    ) -> str:
        """Thin verb over the warm-fork primitive (CONCEPT:AU-KG.coordination.warm-fork-fanout)."""
        try:
            branches = json.loads(branches_json) if branches_json else []
        except (TypeError, ValueError) as exc:
            return json.dumps(
                {"surface": "fork", "error": f"invalid branches_json: {exc}"}
            )
        if not isinstance(branches, list):
            return json.dumps(
                {"surface": "fork", "error": "branches_json must decode to a list"}
            )
        if not branches:
            if code and int(n) > 0:
                branches = [code] * int(n)
            else:
                return json.dumps(
                    {
                        "surface": "fork",
                        "error": "provide branches_json (list) or code + n (>0)",
                    }
                )
        try:
            seed_vars = json.loads(vars_json) if vars_json else {}
        except (TypeError, ValueError) as exc:
            return json.dumps({"surface": "fork", "error": f"invalid vars_json: {exc}"})
        if not isinstance(seed_vars, dict):
            return json.dumps(
                {"surface": "fork", "error": "vars_json must decode to an object"}
            )
        if context_query.strip():
            return _crossmodal_fork_fanout(
                branches,
                seed_vars,
                sandbox.strip(),
                context_query.strip(),
                candidate_var.strip() or "candidates",
            )
        return _fork_fanout(branches, seed_vars, sandbox.strip())

    kg_server.REGISTERED_TOOLS["graph_fork"] = graph_fork
    kg_server.ACTION_TOOL_ROUTES["graph_fork"] = "/graph/fork"
