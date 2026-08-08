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
    "discovery_reachable",
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


def _validate_answer(
    answer: Any, tenant: str, sub_key: str
) -> tuple[PlacementRoute, tuple[str, ...]]:
    """Validate the wire answer and split it into the schema-locked
    ``PlacementRoute`` plus its ADR-1 ``endpoints`` extension.

    ``endpoints`` (CONCEPT:AU-KG.sharding.tenant-partitioned-sharding-hrw, ADR-1 / W1.1) is deliberately
    NOT part of ``agent_utilities.protocols.epistemic_operations.PlacementRoute``:
    that schema-generated model is ``extra="forbid"`` (it is digest-pinned
    against the authoritative catalog, shared verbatim with the engine's
    cross-repo-locked DTO, which the engine itself documents as carrying "no
    deployment endpoint material"). Feeding the raw wire dict straight into
    ``model_validate`` would raise on the additive key, so it is stripped out
    and returned separately instead.
    """
    if not isinstance(answer, dict):
        raise PlacementAuthorityError("engine returned an invalid placement route")
    endpoints_raw = answer.get("endpoints", [])
    if not isinstance(endpoints_raw, list) or not all(
        isinstance(e, str) and e for e in endpoints_raw
    ):
        raise PlacementAuthorityError("engine returned invalid placement endpoints")
    core = {key: value for key, value in answer.items() if key != "endpoints"}
    try:
        route = PlacementRoute.model_validate(core)
    except (TypeError, ValueError) as exc:
        raise PlacementAuthorityError(
            "engine returned an invalid placement route"
        ) from exc
    if route.authoritative is not True:
        raise PlacementAuthorityError("engine returned a non-authoritative route")
    if route.tenant_ref != tenant or route.partition_ref != sub_key:
        raise PlacementAuthorityError("engine returned a route for another partition")
    if route.fencing_token != route.group or (route.placed and route.epoch == 0):
        raise PlacementAuthorityError("engine returned an invalid placement fence")
    return route, tuple(endpoints_raw)


def _map_endpoint(
    group: int,
    contacts: tuple[str, ...],
    config: Any,
    route_endpoints: tuple[str, ...] = (),
) -> str:
    """Resolve `group` to a client endpoint (CONCEPT:AU-KG.sharding.tenant-partitioned-sharding-hrw, ADR-1 / W1.1
    resolution order):

    (a) ``GRAPH_RAFT_GROUP_ENDPOINTS`` when it has an explicit entry for
        `group` — an OPERATOR-CONFIGURED OVERRIDE always wins when present
        (the same "explicit config beats an auto-detected default" contract
        every other override in this codebase follows, e.g.
        ``TenantCatalog``'s explicit shard assignment beating the FNV-1a
        hash) — the deployment case this exists for is exactly one where
        engine-discovered addresses are not reachable from this client (NAT,
        an ingress-only network boundary, ...).
    (b) `route_endpoints` (NEW) — the engine's own live, leader-first member
        list for the resolved group (``PlacementRoute.endpoints``),
        authoritative and requiring no static configuration at all.
    (c) The single configured contact, unchanged.

    Raises :class:`PlacementTopologyError` when none apply — multiple
    contacts, no override, and no engine-discovered endpoints yet (e.g. no
    cluster member has self-reported).
    """
    topology = getattr(config, "graph_raft_group_endpoints", None) or {}
    if isinstance(topology, dict):
        target = topology.get(str(group), topology.get(group))
        if target:
            logger.debug(
                "using the static GRAPH_RAFT_GROUP_ENDPOINTS override for group %s "
                "(engine-discovered endpoints, if any, were not used)",
                group,
            )
            return str(target)
    if route_endpoints:
        return route_endpoints[0]
    if len(contacts) == 1:
        return contacts[0]
    raise PlacementTopologyError(
        "authoritative group has no configured client endpoint"
    )


def _request_authority(config: Any) -> tuple[str, dict[str, Any]]:
    from .session import current_session

    session = current_session()
    if session is None or not getattr(session.actor, "authenticated", False):
        raise PlacementAuthorityError(
            "placement lookup requires an authenticated session"
        )
    from .graph_compute import resolve_engine_auth

    return resolve_engine_auth(config), session.engine_verified_context()


#: Substring of the engine's own ``require_admin_capability`` denial
#: (``epistemic-graph/src/server/access.rs::require_admin_capability_with_policy``:
#: ``"ACCESS_DENIED: verified principal lacks admin capability required for
#: '{action}'"``). Matched, never guessed at, against the exact raised text so
#: the admin-capability broker fallback below only ever fires for THIS specific
#: denial — a bad auth secret, an unreachable engine, or a genuine scope denial
#: (``"lacks required scope"``, a DIFFERENT message from a DIFFERENT check —
#: see ``request_identity.py``'s module docstring) must never trigger it.
_ADMIN_CAPABILITY_DENIAL = "lacks admin capability"


def _admin_capability_denied(exc: BaseException | None) -> bool:
    """True when ``exc`` (or its chained cause) is the engine's admin-capability
    denial specifically, not a scope failure, network error, or anything else."""
    seen: set[int] = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if _ADMIN_CAPABILITY_DENIAL in str(current):
            return True
        current = current.__cause__
    return False


def _broker_authority(config: Any) -> tuple[str, dict[str, Any]] | None:
    """Resolve the admin-capability broker's own verified engine authority.

    CONCEPT:AU-OS.identity.idp-role-to-engine-capability-bridge (register
    D-W6-ISO-1) — see ``AgentConfig.kg_admin_broker_oauth2``'s docstring for
    the full "why". Returns ``None`` (never raises) when the broker is not
    configured, so every caller's fallback is a plain "nothing to try", not an
    exception to catch. When configured, mints a REAL, independently
    OIDC-verified :class:`~agent_utilities.security.brain_context.ActorContext`
    for the broker's own OAuth2 client-credentials identity — the exact same
    ``acquire_process_identity_token`` → ``actor_from_bearer_token`` path every
    other external process identity in this codebase already goes through, no
    parallel trust mechanism — and mints its ``GraphSession`` the ordinary way
    (:func:`~agent_utilities.security.request_identity.mint_graph_session`), so
    the returned ``verified_context`` is byte-for-byte the same shape a real
    request's would be, just for the broker's own principal rather than the
    original caller's.
    """
    oauth2 = getattr(config, "kg_admin_broker_oauth2", None)
    if not oauth2:
        return None
    try:
        from agent_utilities.security.request_identity import (
            acquire_process_identity_token,
            mint_actor_from_token_sync,
            mint_graph_session,
        )

        class _BrokerConfigView:
            """Minimal ``.kg_auth_token_ref``/``.kg_identity_oauth2`` shim so the
            broker's distinct OAuth2 block can ride the SAME
            ``acquire_process_identity_token`` resolver every other external
            process identity uses, without that resolver needing to know a
            second config field name exists."""

            kg_auth_token_ref = None
            kg_identity_oauth2 = oauth2

        token = acquire_process_identity_token(_BrokerConfigView())
        broker_actor = mint_actor_from_token_sync(token)
        broker_session = mint_graph_session(broker_actor)
    except Exception as exc:  # noqa: BLE001 - broker unavailable is a fallback miss, not a hard failure
        logger.warning(
            "admin-capability broker identity unavailable (%s: %s)",
            type(exc).__name__,
            exc,
        )
        return None
    from .graph_compute import resolve_engine_auth

    return resolve_engine_auth(config), broker_session.engine_verified_context()


def _attempt_route(
    tenant: str,
    sub_key: str,
    contacts: tuple[str, ...],
    config: Any,
    *,
    client_factory: Callable[[str], Any] | None,
    client_epoch: int,
    auth_secret: str | None,
    verified_context: dict[str, Any] | None,
) -> PlacementResult:
    """Try every configured contact once, under ONE resolved identity.

    Split out of :func:`_query_catalog` so that function can retry the exact
    same contact list under a DIFFERENT (broker) identity on a specific
    denial, without duplicating the try-every-contact loop.
    """
    failures = 0
    last_error: Exception | None = None
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
            route, route_endpoints = _validate_answer(answer, tenant, sub_key)
            return PlacementResult(
                endpoint=_map_endpoint(route.group, contacts, config, route_endpoints),
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
            last_error = exc
        finally:
            if client is not None and owns_client:
                try:
                    client.close()
                except Exception:  # noqa: BLE001 - best-effort teardown
                    pass
    raise PlacementAuthorityError(
        f"no configured engine returned an authoritative route ({failures} failed)"
    ) from last_error


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

    Admin-capability broker fallback (register D-W6-ISO-1): when the caller's
    own identity is denied specifically because the engine has no admin
    capability registered for it (:func:`_admin_capability_denied` — NOT a
    scope failure, NOT a network/transport failure), and
    ``AgentConfig.kg_admin_broker_oauth2`` is configured, retry the identical
    contact list ONE more time under the broker's own verified identity
    (:func:`_broker_authority`). This resolves ROUTING METADATA ONLY — the
    caller's own session (unchanged) still performs the actual data read that
    follows, so per-graph ACL/RLS is enforced exactly as it always was. A
    caller whose own JWT was never verified with ``kg:admin`` never reaches
    this: the scope check (``verified_context.allows_method`` engine-side)
    denies them long before an admin-capability denial could occur, and the
    broker is only ever consulted in response to that specific engine denial.
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

    try:
        return _attempt_route(
            tenant,
            sub_key,
            contacts,
            config,
            client_factory=client_factory,
            client_epoch=client_epoch,
            auth_secret=auth_secret,
            verified_context=verified_context,
        )
    except PlacementAuthorityError as exc:
        if client_factory is not None or not _admin_capability_denied(exc):
            raise
        broker = _broker_authority(config)
        if broker is None:
            raise
        broker_secret, broker_context = broker
        try:
            result = _attempt_route(
                tenant,
                sub_key,
                contacts,
                config,
                client_factory=None,
                client_epoch=client_epoch,
                auth_secret=broker_secret,
                verified_context=broker_context,
            )
        except Exception:
            # The broker fallback failed too (e.g. the broker identity ALSO
            # lacks admin capability, or the engine is genuinely unreachable) —
            # surface the ORIGINAL caller-identity denial, not the broker's,
            # so the error a real user sees still describes their own request.
            raise exc from None
        logger.info(
            "placement route for tenant=%s resolved via the admin-capability "
            "broker (caller identity lacked engine-registered admin capability)",
            tenant,
        )
        return result


def discovery_reachable(
    endpoints: list[str] | tuple[str, ...],
    config: Any = None,
    *,
    client_factory: Callable[[str], Any] | None = None,
) -> bool:
    """True when at least one of `endpoints` answers the engine's
    ``ClusterMembers`` cluster-topology discovery RPC (CONCEPT:AU-KG.sharding.tenant-partitioned-sharding-hrw, ADR-1 /
    W1.1 decision 5).

    Backs the inverted `agent_utilities.deployment.doctor` engine check: a
    multi-contact configuration with no static `GRAPH_RAFT_GROUP_ENDPOINTS`
    map is now OK **iff** discovery answers from a seed — the failure mode
    becomes "discovery unreachable", not "map missing". Tries each endpoint in
    order (mirrors :func:`_query_catalog`'s try-every-contact discipline) and
    returns on the first success; never raises — a probe result, not an
    authoritative route. Respects the SAME hermetic testing guard as
    :func:`resolve_placement`.
    """
    contacts = tuple(endpoint for endpoint in endpoints if endpoint)
    if not contacts or _hermetic_testing_guard(client_factory):
        return False
    if config is None:
        from agent_utilities.core.config import AgentConfig

        config = AgentConfig()

    auth_secret: str | None = None
    verified_context: dict[str, Any] | None = None
    if client_factory is None:
        try:
            auth_secret, verified_context = _request_authority(config)
        except PlacementAuthorityError:
            return False

    for contact in contacts:
        client = None
        owns_client = client_factory is None
        try:
            if client_factory is not None:
                client = client_factory(contact)
            else:
                assert auth_secret is not None and verified_context is not None
                client = _default_connect(
                    contact, auth_secret, config, verified_context=verified_context
                )
            topology = getattr(client, "cluster_topology", None)
            if topology is None or not hasattr(topology, "members"):
                continue
            topology.members()
            return True
        except Exception as exc:  # noqa: BLE001 - try the next seed; a probe never raises
            logger.debug(
                "cluster-topology discovery probe failed for a configured contact "
                "(%s: %s)",
                type(exc).__name__,
                str(exc),
            )
        finally:
            if client is not None and owns_client:
                try:
                    client.close()
                except Exception:  # noqa: BLE001 - best-effort teardown
                    pass
    return False


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
