# CONCEPT:AU-KG.sharding.tenant-partitioned-sharding-hrw - Tenant-partitioned engine sharding with HRW graph-to-shard routing and tenant to named-graph placement over GRAPH_SERVICE_ENDPOINTS
"""Strict consumer of the epistemic-graph placement authority (DIST-P2-2b).

The engine (epistemic-graph ``src/raft/placement.rs``, DIST-P2-1) owns an
authoritative **PlacementCatalog**: a durable, versioned "this tenant's
keyspace lives here" record with routing epochs, online move (snapshot →
catch-up → fenced cutover), and virtual partitions (one tenant can span
groups). AU is a CONSUMER of that authority, never a second one — this module
is the client-side seam that makes that true. The engine returns a complete
route for every graph, including its current unplaced and single-node policy.
This module caches that answer and maps the returned Raft group to deployment
topology. It never hashes, guesses, disables the catalog, or treats an
unreachable authority as permission to choose a shard.

:func:`resolve_placement` is the ONE entrypoint (mirrors the "one resolver"
discipline of :mod:`.engine_resolver`): a short-TTL ``(endpoint, epoch)``
answer is cached per partition key (``(tenant, sub_key)`` — the same split the
engine uses, :func:`split_tenant_key`), so a hot path does not round-trip the
catalog on every call; on a cache miss/expiry every configured contact is
tried, in order, until one returns a validated, authoritative route.

A caller that discovers its cached placement is stale (a request rejected for
an epoch mismatch, i.e. the engine's ``redirect_if_stale``) re-resolves with
``resolve_placement(..., force_refresh=True)`` — this bypasses the cache,
re-queries the catalog (presenting the previously-cached epoch so the engine
can answer with a redirect), and returns the fresh route to reconnect and
retry against.

AU calls the engine's typed ``client.placement.route(tenant, sub_key,
client_epoch=...)`` — no raw-method alias, no fallback dialect. Every answer
is validated (:func:`_validate_answer`) against the requested partition and
against the engine's own fencing invariants before it is trusted or cached;
an invalid, non-authoritative, or mismatched answer is a hard error
(:class:`PlacementAuthorityError`), never a silently-accepted guess.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

from agent_utilities.protocols.epistemic_operations import PlacementRoute

__all__ = [
    "PlacementAuthorityError",
    "PlacementResult",
    "PlacementTopologyError",
    "invalidate",
    "resolve_placement",
    "split_tenant_key",
]

#: Default cache TTL when ``AgentConfig.placement_catalog_ttl_s`` is absent
#: (e.g. a bare ``_config`` fake in a caller's tests) — short by design, per
#: the task's guardrail: a moved partition must be discovered again quickly.
_DEFAULT_TTL_S = 5.0


class PlacementAuthorityError(RuntimeError):
    """No configured engine returned a valid authoritative route."""


class PlacementTopologyError(RuntimeError):
    """An authoritative group cannot be mapped to a client endpoint."""


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
    """One engine-authoritative deployment route."""

    endpoint: str
    epoch: int
    group: int
    fencing_token: int
    placed: bool


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


def _catalog_ttl_s(config: Any) -> float:
    try:
        ttl = float(getattr(config, "placement_catalog_ttl_s", _DEFAULT_TTL_S))
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
    path is guarded. Tripping this guard fails closed
    (:class:`PlacementAuthorityError`, see :func:`_query_catalog`) rather than
    fabricating a placement answer — this module never guesses.
    """
    if client_factory is not None:
        return False
    from agent_utilities.core.config import setting

    return setting("AGENT_UTILITIES_TESTING", "false").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _default_connect(
    endpoint: str,
    auth_secret: str,
    config: Any,
    *,
    verified_context: dict[str, Any],
) -> Any:
    from epistemic_graph.client import SyncEpistemicGraphClient

    kwargs: dict[str, Any] = {
        "auth_secret": auth_secret,
        "verified_context": verified_context,
    }
    if endpoint.startswith(("tcp://", "tls://")):
        from .engine_transport import (
            engine_client_transport_kwargs,
            native_endpoint_address,
        )

        kwargs["tcp_addr"] = native_endpoint_address(endpoint)[0]
        kwargs.update(engine_client_transport_kwargs(endpoint, config=config))
    elif endpoint.startswith("unix://"):
        kwargs["socket_path"] = endpoint[7:]
    else:
        kwargs["socket_path"] = endpoint
    return SyncEpistemicGraphClient.connect(**kwargs)


def _catalog_call(client: Any, tenant: str, sub_key: str, client_epoch: int) -> Any:
    """Use the one current typed client contract; no raw-method alias."""
    placement = getattr(client, "placement", None)
    if placement is None or not hasattr(placement, "route"):
        raise PlacementAuthorityError("engine client has no placement authority")
    return placement.route(tenant, sub_key, client_epoch=client_epoch)


def _validate_answer(answer: Any, tenant: str, sub_key: str) -> PlacementRoute:
    try:
        route = PlacementRoute.model_validate(answer)
    except (TypeError, ValueError) as exc:
        raise PlacementAuthorityError("engine returned an invalid placement route") from exc
    if route.authoritative is not True:
        raise PlacementAuthorityError("engine returned a non-authoritative route")
    if route.tenant_ref != tenant or route.partition_ref != sub_key:
        raise PlacementAuthorityError("engine returned a route for another partition")
    if route.fencing_token != route.group or (route.placed and route.epoch == 0):
        raise PlacementAuthorityError("engine returned an invalid placement fence")
    return route


def _map_endpoint(
    group: int,
    contacts: tuple[str, ...],
    config: Any,
) -> str:
    topology = getattr(config, "graph_raft_group_endpoints", None) or {}
    if isinstance(topology, dict):
        target = topology.get(str(group), topology.get(group))
        if target:
            return str(target)
    if len(contacts) == 1:
        return contacts[0]
    raise PlacementTopologyError(
        "authoritative group has no configured client endpoint"
    )


def _request_authority(config: Any) -> tuple[str, dict[str, Any]]:
    from .session import current_session

    session = current_session()
    if session is None or not getattr(session.actor, "authenticated", False):
        raise PlacementAuthorityError("placement lookup requires an authenticated session")
    from .graph_compute import resolve_engine_auth

    return resolve_engine_auth(config), session.engine_verified_context()


def _query_catalog(
    tenant: str,
    sub_key: str,
    contacts: tuple[str, ...],
    config: Any,
    *,
    client_factory: Callable[[str], Any] | None,
    client_epoch: int,
) -> PlacementResult:
    """Ask every configured contact for an authoritative route; never guess.

    Tries each of ``contacts`` in order and returns the first validated,
    authoritative answer (any reachable member of a raft-replicated cluster
    can answer; the catalog is cluster-wide, not per-shard). Under the
    hermetic unit-testing guard (:func:`_hermetic_testing_guard`) with no
    injected ``client_factory``, this fails closed immediately instead of
    dialing a real socket.
    """
    if _hermetic_testing_guard(client_factory):
        raise PlacementAuthorityError(
            "placement catalog lookup skipped under the hermetic testing guard "
            "(AGENT_UTILITIES_TESTING); inject client_factory to exercise it"
        )

    auth_secret: str | None = None
    verified_context: dict[str, Any] | None = None
    if client_factory is None:
        auth_secret, verified_context = _request_authority(config)

    failures = 0
    for contact in contacts:
        client = None
        owns_client = client_factory is None
        try:
            if client_factory is not None:
                client = client_factory(contact)
            else:
                assert auth_secret is not None and verified_context is not None
                client = _default_connect(
                    contact,
                    auth_secret,
                    config,
                    verified_context=verified_context,
                )
            answer = _catalog_call(client, tenant, sub_key, client_epoch)
            route = _validate_answer(answer, tenant, sub_key)
            return PlacementResult(
                endpoint=_map_endpoint(route.group, contacts, config),
                epoch=route.epoch,
                group=route.group,
                fencing_token=route.fencing_token,
                placed=route.placed,
            )
        except PlacementTopologyError:
            raise
        except Exception as exc:  # noqa: BLE001 - try the next configured coordinator
            # Trying the next coordinator is right; discarding WHY this one failed
            # is not. The caller only ever sees "no configured engine returned an
            # authoritative route (N failed)", which is identical for a TLS
            # handshake error, a bad auth secret, an unprovisioned RBAC identity,
            # and a genuinely unreachable engine. Log the real cause per contact.
            logger.warning(
                "placement coordinator did not answer (%s: %s)",
                type(exc).__name__,
                str(exc),
            )
            failures += 1
        finally:
            if client is not None and owns_client:
                try:
                    client.close()
                except Exception:  # noqa: BLE001 - best-effort teardown
                    pass
    raise PlacementAuthorityError(
        f"no configured engine returned an authoritative route ({failures} failed)"
    )


def resolve_placement(
    graph_name: str,
    endpoints: list[str] | tuple[str, ...],
    config: Any = None,
    *,
    force_refresh: bool = False,
    client_factory: Callable[[str], Any] | None = None,
) -> PlacementResult:
    """Resolve ``graph_name``'s owning endpoint through the engine placement
    authority only — never a client-side guess.

    ``force_refresh=True`` bypasses the cache and re-queries the catalog —
    call this after a data request comes back rejected for a stale epoch
    (the engine's fenced-cutover redirect) to get the fresh route to
    reconnect and retry against.

    ``client_factory``, when given, is called with an endpoint string and
    must return a connected client exposing the placement-route RPC (see
    :func:`_catalog_call`) — the injection seam tests use to mock the engine
    without a live connection; it also opts out of the hermetic testing guard
    (see :func:`_hermetic_testing_guard`).
    """
    contacts = tuple(endpoint for endpoint in endpoints if endpoint)
    if not contacts:
        raise ValueError("resolve_placement requires at least one endpoint")
    if config is None:
        from agent_utilities.core.config import AgentConfig

        config = AgentConfig()

    tenant, sub_key = split_tenant_key(graph_name)
    key = _cache_key(contacts, tenant, sub_key)
    if not force_refresh:
        with _cache_lock:
            cached = _cache.get(key)
            if cached is not None and cached.expires_at > time.monotonic():
                return cached.result

    client_epoch = 0
    if force_refresh:
        with _cache_lock:
            prior = _cache.get(key)
        if prior is not None:
            client_epoch = prior.result.epoch

    result = _query_catalog(
        tenant,
        sub_key,
        contacts,
        config,
        client_factory=client_factory,
        client_epoch=client_epoch,
    )
    with _cache_lock:
        _cache[key] = _CacheEntry(
            result=result,
            expires_at=time.monotonic() + _catalog_ttl_s(config),
        )
    return result
