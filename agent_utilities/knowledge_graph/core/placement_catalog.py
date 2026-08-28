# CONCEPT:AU-KG.sharding.tenant-partitioned-sharding-hrw - Tenant-partitioned engine sharding with HRW graph-to-shard routing and tenant to named-graph placement over GRAPH_SERVICE_ENDPOINTS
"""Strict consumer of the epistemic-graph placement authority (DIST-P2-2b).

The engine (epistemic-graph ``src/raft/placement.rs``, DIST-P2-1) owns an
authoritative **PlacementCatalog**: a durable, versioned "this tenant's
keyspace lives here" record with routing epochs, online move (snapshot →
catch-up → fenced cutover), and virtual partitions (one tenant can span
groups). AU is a CONSUMER of that authority, never a second one — this module
is the client-side seam that makes that true. The engine returns a complete
route for every graph, including its current unplaced and single-node policy.
This module caches that answer and maps a placed Raft group through the same
client's verified ``ClusterMembers`` snapshot. Configured contacts are bootstrap
seeds only. It never hashes, guesses, disables the catalog, or treats an
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
placed answers additionally require the client's verified ``ClusterMembers``
snapshot before an endpoint is exposed;
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

from .cluster_discovery import (
    ClusterDiscoveryError,
    ClusterDiscoverySnapshot,
    ClusterTopologyAuthority,
)

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
    cluster_id: str | None = None
    membership_epoch: int | None = None
    certificate_rotation_epoch: int | None = None
    discovery_expires_at: float | None = None
    reconnect_required: bool = False


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
_discovery_authorities: dict[tuple[float, float], ClusterTopologyAuthority] = {}
_discovery_authority_lock = threading.Lock()


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
        else:
            tenant, sub_key = split_tenant_key(graph_name)
            for key in [k for k in _cache if k[1] == tenant and k[2] == sub_key]:
                del _cache[key]
    # A failed member connect or an explicit stale-route refresh must not
    # continue using the prior leader/certificate snapshot.  Clearing the
    # bounded process cache is safe; the next request re-reads the authenticated
    # ClusterMembers authority and repopulates it per request context.
    with _discovery_authority_lock:
        for authority in _discovery_authorities.values():
            authority.invalidate()


def _catalog_ttl_s(config: Any) -> float:
    try:
        ttl = float(getattr(config, "placement_catalog_ttl_s", _DEFAULT_TTL_S))
    except (TypeError, ValueError):
        return _DEFAULT_TTL_S
    return ttl if ttl > 0 else _DEFAULT_TTL_S


def _discovery_authority(config: Any) -> ClusterTopologyAuthority:
    """Return the bounded discovery consumer for one config policy.

    The authority cache is keyed by policy, and its snapshots are separately
    keyed by the verified tenant/principal/agent binding.  A topology answer
    can therefore never bleed across tenants or survive beyond the explicit
    freshness/certificate bound merely because this module is process-global.
    """
    try:
        max_age_s = float(getattr(config, "graph_discovery_max_age_s", 30.0))
    except (TypeError, ValueError):
        max_age_s = 30.0
    try:
        clock_skew_s = float(getattr(config, "graph_discovery_clock_skew_s", 5.0))
    except (TypeError, ValueError):
        clock_skew_s = 5.0
    key = (max_age_s, clock_skew_s)
    with _discovery_authority_lock:
        authority = _discovery_authorities.get(key)
        if authority is None:
            authority = ClusterTopologyAuthority(
                max_age_s=max_age_s,
                clock_skew_s=clock_skew_s,
            )
            _discovery_authorities[key] = authority
        return authority


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


def _extract_route_core(answer: Any) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Validate the wire shape and split it into ``(core, endpoints)`` — the
    schema-locked fields ``PlacementRoute.model_validate`` will consume, plus
    the ADR-1 ``endpoints`` compatibility extension it deliberately rejects
    (see :func:`_validate_answer`'s docstring for why)."""
    if not isinstance(answer, dict):
        raise PlacementAuthorityError("engine returned an invalid placement route")
    endpoints_raw = answer.get("endpoints", [])
    if not isinstance(endpoints_raw, list) or not all(
        isinstance(e, str) and e for e in endpoints_raw
    ):
        raise PlacementAuthorityError("engine returned invalid placement endpoints")
    core = {key: value for key, value in answer.items() if key != "endpoints"}
    return core, tuple(endpoints_raw)


def _parse_placement_route(core: dict[str, Any]) -> PlacementRoute:
    try:
        return PlacementRoute.model_validate(core)
    except (TypeError, ValueError) as exc:
        raise PlacementAuthorityError(
            "engine returned an invalid placement route"
        ) from exc


def _assert_route_matches_partition(
    route: PlacementRoute, tenant: str, sub_key: str
) -> None:
    if route.authoritative is not True:
        raise PlacementAuthorityError("engine returned a non-authoritative route")
    if route.tenant_ref != tenant or route.partition_ref != sub_key:
        raise PlacementAuthorityError("engine returned a route for another partition")
    if route.fencing_token != route.group or (route.placed and route.epoch == 0):
        raise PlacementAuthorityError("engine returned an invalid placement fence")


def _validate_answer(
    answer: Any, tenant: str, sub_key: str
) -> tuple[PlacementRoute, tuple[str, ...]]:
    """Validate the wire answer and split it into the schema-locked
    ``PlacementRoute`` plus its ADR-1 ``endpoints`` extension.

    ``endpoints`` (a retired ADR-1 compatibility extension) is deliberately
    NOT part of ``agent_utilities.protocols.epistemic_operations.PlacementRoute``:
    that schema-generated model is ``extra="forbid"`` (it is digest-pinned
    against the authoritative catalog, shared verbatim with the engine's
    cross-repo-locked DTO, which the engine itself documents as carrying "no
    deployment endpoint material"). Feeding the raw wire dict straight into
    ``model_validate`` would raise on the additive key, so it is stripped out
    and returned separately instead. Live endpoint selection ignores it and
    consumes only ``ClusterMembers`` through :mod:`.cluster_discovery`.
    """
    core, endpoints = _extract_route_core(answer)
    route = _parse_placement_route(core)
    _assert_route_matches_partition(route, tenant, sub_key)
    return route, endpoints


def _map_endpoint(
    group: int,
    contacts: tuple[str, ...],
    config: Any,
    discovery: ClusterDiscoverySnapshot | None = None,
) -> str:
    """Resolve a placed group only from verified ``ClusterMembers``.

    Configured contacts remain bootstrap seeds.  They are not a per-group
    endpoint authority, and neither the legacy ``GRAPH_RAFT_GROUP_ENDPOINTS``
    map nor the additive ``PlacementRoute.endpoints`` hint is accepted for a
    placed group.  Group zero is the engine's explicit unplaced/control route;
    any configured seed can serve that route, so the first stable contact is
    sufficient and does not claim placement ownership.
    """
    if group == 0:
        if contacts:
            return contacts[0]
        raise PlacementTopologyError("unplaced route has no configured discovery seed")
    if discovery is None:
        raise PlacementTopologyError(
            "placed route has no verified ClusterMembers snapshot"
        )
    try:
        return discovery.endpoint_for_group(group).client_endpoint
    except ClusterDiscoveryError as exc:
        raise PlacementTopologyError(
            f"verified ClusterMembers has no usable endpoint for group {group}"
        ) from exc


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


def _connect_route_client(
    contact: str,
    client_factory: Callable[[str], Any] | None,
    auth_secret: str | None,
    config: Any,
    verified_context: dict[str, Any] | None,
) -> Any:
    if client_factory is not None:
        return client_factory(contact)
    assert auth_secret is not None and verified_context is not None
    return _default_connect(
        contact,
        auth_secret,
        config,
        verified_context=verified_context,
    )


def _resolve_route_discovery(
    client: Any,
    config: Any,
    route: PlacementRoute,
    verified_context: dict[str, Any] | None,
    force_discovery_refresh: bool,
) -> tuple[ClusterDiscoverySnapshot | None, ClusterDiscoverySnapshot | None]:
    """``(discovery, prior_discovery)`` for a placed group; ``(None, None)``
    for an unplaced/group-0 route (untouched — never even queries the
    discovery authority)."""
    if not (route.placed and route.group > 0):
        return None, None
    try:
        authority = _discovery_authority(config)
        discovery_context = verified_context or authority.context_for(client)
        prior_discovery = None
        if discovery_context is not None:
            prior_discovery = authority.last_good_for(
                verified_context=discovery_context,
                expected_cluster_id=getattr(config, "graph_cluster_id", None),
            )
        discovery = authority.read(
            client,
            verified_context=verified_context,
            expected_cluster_id=getattr(config, "graph_cluster_id", None),
            min_placement_epoch=route.epoch,
            force_refresh=force_discovery_refresh,
        )
        return discovery, prior_discovery
    except ClusterDiscoveryError as exc:
        raise PlacementAuthorityError(
            "engine placement route lacks a current verified ClusterMembers snapshot"
        ) from exc


def _reconnect_required(
    prior_discovery: ClusterDiscoverySnapshot | None,
    discovery: ClusterDiscoverySnapshot | None,
) -> bool:
    return bool(
        discovery is not None
        and prior_discovery is not None
        and prior_discovery.cluster_id == discovery.cluster_id
        and (
            prior_discovery.membership_epoch != discovery.membership_epoch
            or prior_discovery.placement_epoch != discovery.placement_epoch
            or prior_discovery.certificate_epoch != discovery.certificate_epoch
        )
    )


def _build_placement_result(
    route: PlacementRoute,
    contacts: tuple[str, ...],
    config: Any,
    discovery: ClusterDiscoverySnapshot | None,
    prior_discovery: ClusterDiscoverySnapshot | None,
) -> PlacementResult:
    return PlacementResult(
        endpoint=_map_endpoint(route.group, contacts, config, discovery),
        epoch=route.epoch,
        group=route.group,
        fencing_token=route.fencing_token,
        placed=route.placed,
        cluster_id=discovery.cluster_id if discovery is not None else None,
        membership_epoch=(
            discovery.membership_epoch if discovery is not None else None
        ),
        certificate_rotation_epoch=(
            discovery.certificate_epoch if discovery is not None else None
        ),
        discovery_expires_at=(
            discovery.expires_at_monotonic if discovery is not None else None
        ),
        reconnect_required=_reconnect_required(prior_discovery, discovery),
    )


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
    force_discovery_refresh: bool = False,
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
            client = _connect_route_client(
                contact, client_factory, auth_secret, config, verified_context
            )
            answer = _catalog_call(client, tenant, sub_key, client_epoch)
            route, _route_endpoints = _validate_answer(answer, tenant, sub_key)
            discovery, prior_discovery = _resolve_route_discovery(
                client, config, route, verified_context, force_discovery_refresh
            )
            return _build_placement_result(
                route, contacts, config, discovery, prior_discovery
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


def _resolve_query_identity(
    config: Any, client_factory: Callable[[str], Any] | None
) -> tuple[str | None, dict[str, Any] | None]:
    """``(auth_secret, verified_context)`` for a fresh catalog query.

    When no ``client_factory`` is injected this is the caller's own resolved
    engine authority. When one IS injected (the single-endpoint production
    reuse seam, which supplies a client only to avoid opening a second
    socket), only the verified context is inherited for discovery binding —
    ``auth_secret`` stays ``None`` (the factory owns the connection's auth),
    and a hermetic fake without a session fails closed to ``None`` context
    rather than raising here (``ClusterTopologyAuthority`` enforces it).
    """
    if client_factory is None:
        return _request_authority(config)
    try:
        _unused_secret, verified_context = _request_authority(config)
    except PlacementAuthorityError:
        verified_context = None
    return None, verified_context


def _should_skip_broker_fallback(client_factory: Callable[[str], Any] | None) -> bool:
    """True when the admin-capability broker retry must NOT fire.

    A caller-supplied ``client_factory`` is used for two UNRELATED reasons:
    (a) hermetic-test injection (``AGENT_UTILITIES_TESTING``), where the
    broker's own real network round-trip must never fire; (b) production
    connection reuse (``graph_compute.py``'s ``reuse_single_endpoint``),
    which is not a test at all and must not silently disable the broker
    fallback. Gate on the actual test signal, not on ``client_factory``'s
    mere presence.
    """
    if client_factory is None:
        return False
    from agent_utilities.core.config import setting

    return setting("AGENT_UTILITIES_TESTING", "false").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _retry_via_admin_broker(
    tenant: str,
    sub_key: str,
    contacts: tuple[str, ...],
    config: Any,
    *,
    client_epoch: int,
    force_discovery_refresh: bool,
    original_exc: PlacementAuthorityError,
) -> PlacementResult:
    """Retry the SAME contact list once under the admin-capability broker's
    own verified identity (``client_factory=None`` always — a fresh,
    independently-identified connection, never the caller-supplied factory).

    Re-raises ``original_exc`` (never the broker's own failure) when the
    broker is unconfigured or its own attempt also fails, so a real user's
    error still describes their own request, not the broker's.
    """
    broker = _broker_authority(config)
    if broker is None:
        raise original_exc
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
            force_discovery_refresh=force_discovery_refresh,
        )
    except Exception:
        raise original_exc from None
    logger.info(
        "placement route for tenant=%s resolved via the admin-capability "
        "broker (caller identity lacked engine-registered admin capability)",
        tenant,
    )
    return result


def _query_catalog(
    tenant: str,
    sub_key: str,
    contacts: tuple[str, ...],
    config: Any,
    *,
    client_factory: Callable[[str], Any] | None,
    client_epoch: int,
    force_discovery_refresh: bool = False,
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

    auth_secret, verified_context = _resolve_query_identity(config, client_factory)

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
            force_discovery_refresh=force_discovery_refresh,
        )
    except PlacementAuthorityError as exc:
        if not _admin_capability_denied(exc):
            raise
        if _should_skip_broker_fallback(client_factory):
            raise
        return _retry_via_admin_broker(
            tenant,
            sub_key,
            contacts,
            config,
            client_epoch=client_epoch,
            force_discovery_refresh=force_discovery_refresh,
            original_exc=exc,
        )


def _probe_discovery_contact(
    contact: str,
    client_factory: Callable[[str], Any] | None,
    auth_secret: str | None,
    config: Any,
    verified_context: dict[str, Any] | None,
) -> bool:
    """True iff ``contact`` answers the engine's ``ClusterMembers`` discovery
    RPC; never raises — a probe result, not an authoritative route."""
    client = None
    owns_client = client_factory is None
    try:
        client = _connect_route_client(
            contact, client_factory, auth_secret, config, verified_context
        )
        _discovery_authority(config).read(
            client,
            verified_context=verified_context,
            expected_cluster_id=getattr(config, "graph_cluster_id", None),
            force_refresh=True,
        )
        return True
    except Exception as exc:  # noqa: BLE001 - try the next seed; a probe never raises
        logger.debug(
            "cluster-topology discovery probe failed for a configured contact (%s: %s)",
            type(exc).__name__,
            str(exc),
        )
        return False
    finally:
        if client is not None and owns_client:
            try:
                client.close()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass


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
    multi-contact configuration is OK **iff** verified discovery answers from
    a seed — the failure mode is "discovery unreachable", regardless of any
    legacy map. A static map is never used as a live authority. Tries each endpoint in
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

    return any(
        _probe_discovery_contact(
            contact, client_factory, auth_secret, config, verified_context
        )
        for contact in contacts
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
        # A placement-cache miss is the reconnect cadence. Refreshing the
        # verified member snapshot on that boundary observes leader/member/
        # certificate changes even when the old endpoint still accepts TCP.
        force_discovery_refresh=True,
    )
    with _cache_lock:
        _cache[key] = _CacheEntry(
            result=result,
            expires_at=time.monotonic() + _catalog_ttl_s(config),
        )
    return result
