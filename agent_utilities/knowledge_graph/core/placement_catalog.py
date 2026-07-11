# CONCEPT:AU-KG.sharding.tenant-partitioned-sharding-hrw - Tenant-partitioned engine sharding with HRW graph-to-shard routing and tenant to named-graph placement over GRAPH_SERVICE_ENDPOINTS
"""Engine placement-catalog consumer (DIST-P2-2b).

The engine (epistemic-graph ``src/raft/placement.rs``, DIST-P2-1) now owns an
authoritative **PlacementCatalog**: a durable, versioned "this tenant's
keyspace lives here" record with routing epochs, online move (snapshot →
catch-up → fenced cutover), and virtual partitions (one tenant can span
groups). AU must be a CONSUMER of that authority, never a second one — this
module is the client-side seam that makes that true.

:func:`resolve_placement` is the ONE entrypoint (mirrors the "one resolver"
discipline of :mod:`.engine_resolver`):

1. **Cache** — a short-TTL ``(endpoint, epoch)`` answer for this partition key
   (``(tenant, sub_key)`` — the same split the engine uses,
   :func:`split_tenant_key`), so a hot path does not round-trip the catalog on
   every call.
2. **Engine catalog** — on a cache miss/expiry, ask the engine's placement
   route op. ANY reachable configured endpoint can answer (the catalog is
   cluster-wide, not per-shard), so every endpoint is tried in HRW-preference
   order until one responds.
3. **Static HRW ring** (:func:`.shard_topology.shard_endpoint_for`) — the
   BOOTSTRAP/fallback, used only when the catalog is disabled, every contact
   endpoint is unreachable, or the engine doesn't advertise one at all (an
   older engine — see the wire-contract note below). Never a second
   authority: this module makes no independent placement *decision*, it just
   picks somewhere to ask when there is nobody left to ask.

A caller that discovers its cached placement is stale (a request rejected for
an epoch mismatch, i.e. the engine's ``redirect_if_stale``) re-resolves with
``resolve_placement(..., force_refresh=True)`` — this bypasses the cache,
re-queries the catalog (presenting the previously-cached epoch so the engine
can answer with a redirect), and returns the fresh ``(endpoint, epoch)`` to
reconnect and retry against.

Wire contract (forward-looking). AU calls the engine's generic RPC dispatch
with method name ``"PlacementRoute"`` and params ``{"tenant", "sub_key",
"client_epoch"}`` — the same thin-namespace idiom the engine's own admin
clients already use (``AdminClient.backup`` -> ``_send("Backup", ...)``,
``ReshardingClient.catalog_list`` -> ``_send("CatalogList")``). **Today's
shipped engine has no such wire ``Method`` yet** — ``PlacementCatalog::route``
is presently consumed only INSIDE the engine, by ``MultiRaft``'s own
cross-group dispatch (``src/raft/placement.rs`` documents AU-side consumption
as the separate follow-up this module implements). So every real call here
fails, and the fallback contract kicks in exactly as designed: ANY failure
(missing attribute, RPC error, connection error, timeout) means "this engine
doesn't advertise a catalog", and resolution falls back to the static HRW
ring — today's deployments are byte-for-byte unaffected. The moment the
engine adds the wire method, AU starts consuming it with no further AU-side
change (a catalog-aware ``client.placement.route(...)`` namespace is tried
first, ahead of the raw ``_send`` fallback, for a nicer typed client later).

Hermetic under the unit suite: with ``AGENT_UTILITIES_TESTING`` set (the whole
suite's default, ``tests/conftest.py``) and no explicit ``client_factory``
override, :func:`resolve_placement` skips the real network round-trip and
goes straight to HRW — the same hermetic-guard convention
``engine_resolver.setting_autostart`` uses to keep the unit suite from ever
touching a real engine. A test that wants to exercise the catalog path passes
its own ``client_factory`` (a mock), which always bypasses the guard.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "PlacementResult",
    "invalidate",
    "resolve_placement",
    "split_tenant_key",
]

#: Default cache TTL when ``AgentConfig.placement_catalog_ttl_s`` is absent
#: (e.g. a bare ``_config`` stub in a caller's tests) — short by design, per
#: the task's guardrail: a moved partition must be discovered again quickly.
_DEFAULT_TTL_S = 5.0


def split_tenant_key(graph_name: str) -> tuple[str, str]:
    """Split ``graph_name`` into ``(tenant, sub_key)``.

    MUST agree with the engine's own ``raft::placement::split_tenant_key``: the
    substring before the FIRST ``:`` is the tenant, the rest is the
    workspace/session/entity sub-key that hashes into a tenant's partition
    range. A name with no ``:`` (or an empty tenant before it) is its own
    tenant AND sub-key.
    """
    if ":" in graph_name:
        tenant, _, rest = graph_name.partition(":")
        if tenant:
            return tenant, rest
    return graph_name, graph_name


@dataclass(frozen=True)
class PlacementResult:
    """A resolved endpoint for one partition key.

    ``source`` is ``"catalog"`` when the engine's PlacementCatalog answered
    explicitly, or ``"hrw"`` when this is the static bootstrap/fallback ring
    (catalog disabled, unreachable/unsupported, or it explicitly has no
    placement for this tenant). ``epoch`` is ``0`` for an HRW answer — there is
    no routing epoch outside the catalog.
    """

    endpoint: str
    epoch: int
    source: str
    group: Any | None = None


@dataclass
class _CacheEntry:
    result: PlacementResult
    expires_at: float


# Keyed by (endpoints tuple, tenant, sub_key) -> the resolved placement. A
# process-wide cache (mirrors the module-level HRW router cache in
# shard_topology._router_cache) — short TTL keeps it from ever going stale for
# long, and every entry is independently invalidated/refreshed.
_cache: dict[tuple[Any, ...], _CacheEntry] = {}
_cache_lock = threading.Lock()


def _cache_key(
    endpoints: tuple[str, ...], tenant: str, sub_key: str
) -> tuple[Any, ...]:
    return (endpoints, tenant, sub_key)


def invalidate(graph_name: str | None = None) -> None:
    """Drop cached placement(s). ``None`` clears the whole cache (tests /
    a full topology reconfigure); otherwise drops every endpoint-set entry
    for ``graph_name``'s ``(tenant, sub_key)``."""
    with _cache_lock:
        if graph_name is None:
            _cache.clear()
            return
        tenant, sub_key = split_tenant_key(graph_name)
        for key in [k for k in _cache if k[1] == tenant and k[2] == sub_key]:
            del _cache[key]


def _catalog_enabled(config: Any) -> bool:
    return bool(getattr(config, "placement_catalog_enabled", True))


def _catalog_ttl_s(config: Any) -> float:
    ttl = getattr(config, "placement_catalog_ttl_s", _DEFAULT_TTL_S)
    try:
        ttl = float(ttl)
    except (TypeError, ValueError):
        return _DEFAULT_TTL_S
    return ttl if ttl > 0 else _DEFAULT_TTL_S


def _hermetic_testing_guard(client_factory: Callable[[str], Any] | None) -> bool:
    """True when the real network round-trip must be skipped.

    Mirrors ``engine_resolver.setting_autostart``'s own testing guard: the
    unit suite sets ``AGENT_UTILITIES_TESTING=true`` and must never dial a
    real socket. A caller that explicitly injects ``client_factory`` (this
    module's own tests, or a caller that wants to exercise the catalog path
    against an in-process fake) opts back in — only the DEFAULT real-connect
    path is guarded.
    """
    if client_factory is not None:
        return False
    from agent_utilities.core.config import setting

    return setting("AGENT_UTILITIES_TESTING", "false").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _default_connect(endpoint: str, auth_secret: str | None) -> Any:
    """Open a short-lived client to ``endpoint`` for a catalog query only."""
    from epistemic_graph.client import SyncEpistemicGraphClient

    kwargs: dict[str, Any] = {"auth_secret": auth_secret}
    if endpoint.startswith("tcp://"):
        kwargs["tcp_addr"] = endpoint[6:]
    elif endpoint.startswith("unix://"):
        kwargs["socket_path"] = endpoint[7:]
    else:
        kwargs["socket_path"] = endpoint
    return SyncEpistemicGraphClient.connect(**kwargs)


def _catalog_call(client: Any, tenant: str, sub_key: str, client_epoch: int) -> Any:
    """Issue the placement-route RPC on an already-connected ``client``.

    Tries a friendly ``client.placement.route(...)`` namespace first (the
    shape a catalog-aware client will expose once the engine wires this up),
    then the raw ``_send`` RPC every thin admin namespace in
    ``epistemic_graph.client`` already uses this way. Raises on any failure —
    the caller treats every exception identically (see module docstring).
    """
    placement_ns = getattr(client, "placement", None)
    if placement_ns is not None and hasattr(placement_ns, "route"):
        return placement_ns.route(tenant, sub_key, client_epoch=client_epoch)
    return client._send(
        "PlacementRoute",
        {"tenant": tenant, "sub_key": sub_key, "client_epoch": client_epoch},
    )


def _query_catalog(
    tenant: str,
    sub_key: str,
    contact_endpoints: list[str],
    config: Any,
    *,
    client_factory: Callable[[str], Any] | None,
    client_epoch: int,
) -> PlacementResult | None:
    """Ask the engine's placement catalog for ``(tenant, sub_key)``.

    Returns ``None`` when the catalog cannot be consulted at all (every
    contact endpoint unreachable/unsupported) or explicitly answers "no
    placement for this tenant" — either way the caller falls back to HRW.
    Tries each of ``contact_endpoints`` in order and stops at the first that
    answers (any reachable member of a raft-replicated cluster can answer;
    the catalog is cluster-wide, not per-shard).
    """
    from .graph_compute import resolve_engine_auth  # local: avoid import cycle

    auth_secret, insecure = resolve_engine_auth(config)
    auth_secret = None if insecure else auth_secret

    for endpoint in contact_endpoints:
        client = None
        owns_client = client_factory is None
        try:
            client = (
                client_factory(endpoint)
                if client_factory is not None
                else _default_connect(endpoint, auth_secret)
            )
            answer = _catalog_call(client, tenant, sub_key, client_epoch)
        except Exception as exc:  # noqa: BLE001 — best-effort: try next / HRW
            logger.debug(
                "placement-catalog query to %s failed (%s) — trying next "
                "endpoint or falling back to HRW",
                endpoint,
                exc,
            )
            continue
        finally:
            if client is not None and owns_client:
                try:
                    client.close()
                except Exception:  # noqa: BLE001 — best-effort teardown
                    pass

        if not isinstance(answer, dict) or not answer.get("explicit"):
            # A well-formed answer from a catalog-aware engine saying "no
            # explicit placement for this tenant" — HRW is authoritative for
            # it, definitively (not because the catalog is unreachable).
            return None
        target = answer.get("endpoint")
        if not target:
            return None
        return PlacementResult(
            endpoint=str(target),
            epoch=int(answer.get("epoch") or 0),
            source="catalog",
            group=answer.get("group"),
        )
    return None


def resolve_placement(
    graph_name: str,
    endpoints: list[str] | tuple[str, ...],
    config: Any = None,
    *,
    force_refresh: bool = False,
    client_factory: Callable[[str], Any] | None = None,
) -> PlacementResult:
    """Resolve ``graph_name``'s owning endpoint — the engine catalog first,
    the static HRW ring as bootstrap/fallback only.

    ``force_refresh=True`` bypasses the cache and re-queries the catalog —
    call this after a data request comes back rejected for a stale epoch
    (the engine's fenced-cutover redirect) to get the fresh
    ``(endpoint, epoch)`` to reconnect and retry against.

    ``client_factory``, when given, is called with an endpoint string and
    must return a connected client exposing the placement-route RPC (see
    :func:`_catalog_call`) — the injection seam tests use to mock the engine
    without a live connection; it also opts out of the hermetic testing guard
    (see module docstring).
    """
    eps = tuple(e for e in endpoints if e)
    if not eps:
        raise ValueError("resolve_placement requires at least one endpoint")

    from .shard_topology import shard_endpoint_for

    if len(eps) == 1:
        # Zero-infra / single-endpoint: identity, no catalog round trip —
        # there is nowhere else the graph could live.
        return PlacementResult(endpoint=eps[0], epoch=0, source="hrw")

    if config is None:
        from agent_utilities.core.config import AgentConfig

        config = AgentConfig()

    tenant, sub_key = split_tenant_key(graph_name)
    key = _cache_key(eps, tenant, sub_key)

    if not force_refresh:
        with _cache_lock:
            entry = _cache.get(key)
            if entry is not None and entry.expires_at > time.monotonic():
                return entry.result

    hrw_pick = shard_endpoint_for(graph_name, list(eps))

    if not _catalog_enabled(config) or _hermetic_testing_guard(client_factory):
        result = PlacementResult(endpoint=hrw_pick, epoch=0, source="hrw")
    else:
        client_epoch = 0
        if force_refresh:
            with _cache_lock:
                prior = _cache.get(key)
            if prior is not None:
                client_epoch = prior.result.epoch
        # Bootstrap contact order: the static HRW pick first (most likely
        # already reachable/local), then the rest of the configured ring —
        # any reachable node can answer, so order is just a preference.
        contact_order = [hrw_pick] + [e for e in eps if e != hrw_pick]
        answer = _query_catalog(
            tenant,
            sub_key,
            contact_order,
            config,
            client_factory=client_factory,
            client_epoch=client_epoch,
        )
        result = (
            answer
            if answer is not None
            else PlacementResult(endpoint=hrw_pick, epoch=0, source="hrw")
        )

    with _cache_lock:
        _cache[key] = _CacheEntry(
            result=result, expires_at=time.monotonic() + _catalog_ttl_s(config)
        )
    return result
