# CONCEPT:KG-2.58 - Tenant-partitioned engine sharding with HRW graph-to-shard routing and tenant to named-graph placement over GRAPH_SERVICE_ENDPOINTS
# CONCEPT:OS-5.28 - Shard topology visibility with per-shard reachability status surfaces and per-endpoint engine gauges and counters
"""Tenant-partitioned engine shard topology.

CONCEPT:KG-2.58 — Tenant-Partitioned Engine Sharding. Stage-2 scaling for the
epistemic-graph compute tier: N independent engine processes ("shards") behind
the client-side HRW router that already ships in ``epistemic_graph.pool``.

Partition model (the one sentence that matters)::

    tenant  →  named graph  →  HRW (rendezvous hash)  →  shard endpoint

* The **named graph** is the partition unit. Every engine process keeps its
  own string-keyed named-graph registry, so a graph lives wholly on exactly
  one shard and no cross-shard coordination is ever needed for a single
  graph's operations.
* **Tenancy** enters only by *choosing the graph name*: when a caller does not
  target an explicit graph and an ambient
  :class:`~agent_utilities.security.brain_context.ActorContext` carries a
  tenant, the default graph is mapped to that tenant's graph via
  :func:`tenant_graph_name`. The shard choice is therefore always a pure
  function of the graph name — sync clients here and async
  ``epistemic_graph.pool.ShardRouter`` users agree by construction (this
  module delegates to the very same HRW implementation).
* **Zero-infra default preserved**: with a single endpoint (the default —
  ``GRAPH_SERVICE_ENDPOINTS`` unset) nothing changes: no tenant graph
  mapping, no routing, same socket, same autostart behaviour.

Operational semantics in sharded mode (``GRAPH_SERVICE_ENDPOINTS`` lists 2+
endpoints):

* **Autostart is local-only.** ``EPISTEMIC_GRAPH_AUTOSTART=1`` may only spawn
  an engine for a *local* (``unix://``) endpoint; a configured remote
  (``tcp://``) shard that is unreachable is a hard, fail-loud
  ``ConnectionError`` naming the shard — silently auto-starting a local
  stand-in would split that shard's graphs into invisible islands (same
  convention as the CONCEPT:KG-2.55 task-queue contract).
* **The flock host role is per-host.** ``host_lock.py`` elects the ONE process
  per host that runs daemons and may own the LOCAL engine; it says nothing
  about remote shards.
* **Rebalancing is intentionally out of scope.** HRW keeps key movement
  minimal when endpoints are added/removed, but the engine does NOT migrate
  data: after a topology change, a graph whose HRW winner changed must be
  moved manually — export from the old shard / import on the new one via the
  existing snapshot tooling (``--persist-dir`` checkpoints or
  ``lifecycle.to_msgpack``/``from_msgpack``). Until then the graph re-creates
  empty on its new shard.

Topology visibility (CONCEPT:OS-5.28 — Shard Topology Visibility):
:func:`shard_topology_status` powers the unified daemon status and the
gateway's ``/daemon/shards`` route, and exports the per-shard
``agent_utilities_engine_shard_up{endpoint}`` gauge.
"""

from __future__ import annotations

import logging
import re
import socket
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_GRAPH",
    "DEFAULT_LOCAL_ENDPOINT",
    "is_local_endpoint",
    "probe_endpoint",
    "resolve_endpoints",
    "resolve_routing_graph",
    "shard_endpoint_for",
    "shard_topology_status",
    "sharding_active",
    "tenant_graph_name",
]

#: Fallback default graph when no config object is reachable. The runtime
#: default lives on ``AgentConfig.kg_default_graph`` (KG_DEFAULT_GRAPH).
DEFAULT_GRAPH = "__commons__"

#: The historical single-endpoint default (matches the engine's own fallback).
DEFAULT_LOCAL_ENDPOINT = "unix:///tmp/epistemic-graph.sock"  # nosec B108

# Tenant ids come from JWT claims (org_id/tid/...) and may contain arbitrary
# characters; graph names should stay shell/file/metric friendly.
_TENANT_SLUG_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _config(config: Any = None) -> Any:
    if config is not None:
        return config
    from agent_utilities.core.config import AgentConfig

    return AgentConfig()


def default_graph_name(config: Any = None) -> str:
    """The configured default graph (``KG_DEFAULT_GRAPH``, default ``__commons__``)."""
    cfg = _config(config)
    return getattr(cfg, "kg_default_graph", DEFAULT_GRAPH) or DEFAULT_GRAPH


# ---------------------------------------------------------------------------
# Endpoint resolution
# ---------------------------------------------------------------------------


def resolve_endpoints(config: Any = None) -> list[str]:
    """Resolve the configured engine endpoint list (1 = today's single mode).

    Precedence: ``graph_service_endpoints`` (GRAPH_SERVICE_ENDPOINTS, comma or
    JSON list) → ``tcp://{graph_service_tcp_addr}`` → ``unix://{socket}`` →
    :data:`DEFAULT_LOCAL_ENDPOINT`. Endpoint strings are used VERBATIM as both
    the HRW hash input and the connect target so every client that hashes the
    same configured list (including async ``ShardRouter`` users) agrees on
    placement — configure them with explicit ``unix://`` / ``tcp://`` schemes.
    """
    cfg = _config(config)
    eps = [
        str(e).strip() for e in (cfg.graph_service_endpoints or []) if str(e).strip()
    ]
    if eps:
        return eps
    if getattr(cfg, "graph_service_tcp_addr", None):
        return [f"tcp://{cfg.graph_service_tcp_addr}"]
    if getattr(cfg, "graph_service_socket", None):
        return [f"unix://{cfg.graph_service_socket}"]
    return [DEFAULT_LOCAL_ENDPOINT]


def sharding_active(config: Any = None) -> bool:
    """True when 2+ endpoints are configured (multi-shard mode)."""
    return len(resolve_endpoints(config)) > 1


def is_local_endpoint(endpoint: str) -> bool:
    """True for endpoints that are local-by-construction (unix sockets).

    Only local endpoints are eligible for ``EPISTEMIC_GRAPH_AUTOSTART`` in
    sharded mode: a ``tcp://`` shard (even on a loopback address) is treated
    as remote and must be managed by its own host's daemon.
    """
    ep = endpoint.strip()
    return not ep.startswith("tcp://")


# ---------------------------------------------------------------------------
# Tenant → graph naming discipline
# ---------------------------------------------------------------------------


def tenant_graph_name(tenant: str | None, base: str = DEFAULT_GRAPH) -> str:
    """Map a tenant id onto its per-tenant named graph: ``tenant__<t>__<base>``.

    The single naming rule for tenant-scoped graph placement: facade, backends
    and the engine client all use this helper so a tenant's data consistently
    lands on ONE named graph (and therefore, via HRW, on one shard). An empty
    or unset tenant returns ``base`` unchanged — single-tenant deployments are
    byte-for-byte unaffected.
    """
    if not tenant:
        return base
    slug = _TENANT_SLUG_RE.sub("_", tenant.strip()).strip("_").lower()
    if not slug:
        return base
    return f"tenant__{slug}__{base}"


def resolve_routing_graph(graph_name: str | None, config: Any = None) -> str:
    """Resolve the effective graph (= HRW routing key) for an engine client.

    Resolution order (CONCEPT:KG-2.58):

    1. An explicit, non-default ``graph_name`` — the operation targets a named
       graph; use it verbatim.
    2. The ambient :class:`ActorContext` tenant — the default graph is mapped
       to the tenant's graph via :func:`tenant_graph_name`.
    3. The configured default graph (``KG_DEFAULT_GRAPH``).
    """
    default = default_graph_name(config)
    if graph_name and graph_name != default:
        return graph_name
    from agent_utilities.security.brain_context import current_actor

    tenant = current_actor().tenant_id
    if tenant:
        return tenant_graph_name(tenant, base=default)
    return graph_name or default


# ---------------------------------------------------------------------------
# HRW shard selection — delegates to epistemic_graph.pool.ShardRouter
# ---------------------------------------------------------------------------

_router_cache: dict[tuple[str, ...], Any] = {}


def _hrw_router(endpoints: tuple[str, ...]) -> Any:
    """One cached ``ShardRouter`` per endpoint set, used ONLY for HRW hashing.

    Delegating to the ShardRouter implementation (rather than re-implementing
    the hash here) guarantees sync clients and async pool users can never
    drift on shard placement. The router's connection pools are never
    initialized — no sockets are opened by hashing.
    """
    router = _router_cache.get(endpoints)
    if router is None:
        from epistemic_graph.pool import ShardRouter

        router = _router_cache[endpoints] = ShardRouter(list(endpoints))
    return router


def shard_endpoint_for(graph_name: str, endpoints: Sequence[str]) -> str:
    """Pick the owning shard endpoint for ``graph_name`` (HRW; deterministic).

    With a single endpoint this is the identity function (zero-infra default).
    """
    eps = [e for e in endpoints if e]
    if not eps:
        raise ValueError("shard_endpoint_for requires at least one endpoint")
    if len(eps) == 1:
        return eps[0]
    return _hrw_router(tuple(eps))._get_shard_endpoint(graph_name)


# ---------------------------------------------------------------------------
# Topology visibility (CONCEPT:OS-5.28)
# ---------------------------------------------------------------------------


def probe_endpoint(endpoint: str, timeout: float = 0.5) -> bool:
    """Transport-level reachability probe (raw connect; no auth handshake).

    Cheap by design: a TCP/UDS connect proves the engine process is listening
    without spending an authenticated RPC per scrape.
    """
    ep = endpoint.strip()
    try:
        if ep.startswith("tcp://"):
            host, _, port = ep[6:].rpartition(":")
            with socket.create_connection(
                (host or "127.0.0.1", int(port)), timeout=timeout
            ):
                return True
        path = ep[7:] if ep.startswith("unix://") else ep
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            sock.connect(path)
            return True
        finally:
            sock.close()
    except (OSError, ValueError):
        return False


def record_shard_connect(endpoint: str, up: bool) -> None:
    """Export shard reachability observed on a real client connect attempt."""
    try:
        from agent_utilities.observability.gateway_metrics import ENGINE_SHARD_UP

        ENGINE_SHARD_UP.labels(endpoint=endpoint).set(1.0 if up else 0.0)
    except Exception:  # pragma: no cover - metrics must never break clients
        logger.debug("Could not export shard-up metric for %s", endpoint)


def shard_topology_status(
    config: Any = None, probe: bool = True, timeout: float = 0.5
) -> dict[str, Any]:
    """Shard topology + per-shard reachability for status/health surfaces.

    Returns::

        {
          "mode": "single" | "sharded",
          "default_graph": "...",
          "endpoints": [
            {"endpoint": "...", "local": bool, "reachable": bool, "breaker": "..."},
            ...
          ],
        }

    Also refreshes ``agent_utilities_engine_shard_up{endpoint}`` when probing.
    The host flock governs only the LOCAL engine; remote shards are reported,
    never managed, from here.
    """
    cfg = _config(config)
    endpoints = resolve_endpoints(cfg)
    entries: list[dict[str, Any]] = []
    for ep in endpoints:
        entry: dict[str, Any] = {"endpoint": ep, "local": is_local_endpoint(ep)}
        if probe:
            up = probe_endpoint(ep, timeout=timeout)
            entry["reachable"] = up
            record_shard_connect(ep, up)
        try:
            from agent_utilities.knowledge_graph.core.engine_breaker import get_breaker

            entry["breaker"] = get_breaker(ep).state
        except Exception:  # pragma: no cover - status must stay best-effort
            pass
        entries.append(entry)
    return {
        "mode": "sharded" if len(endpoints) > 1 else "single",
        "default_graph": default_graph_name(cfg),
        "endpoints": entries,
    }
