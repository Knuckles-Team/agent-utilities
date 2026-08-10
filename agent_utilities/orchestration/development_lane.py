"""Narrow transport for the engine-native RMDD-28 development-lane protocol.

Repository Manager owns the ``LaneAuthority`` lifecycle mapping (allocate,
transition, heartbeat, list, get -- see ``repository_manager.lane_registry``).
AU only exposes a typed transport over the generated
``SyncEpistemicGraphClient``; it never invents owner/fence/quota defaults,
performs a local read-then-CAS, or memoizes a transient admission refusal.
This mirrors ``agent_utilities.orchestration.resource_reservation`` (RMDD-27)
field-for-field on purpose: RMDD-28 is explicitly specified as reusing that
transaction/auth/time/Raft/ReadIndex pattern for the development-lane domain.

Residual, disclosed rather than papered over: as of this lane's landing, the
epistemic-graph generated client (``epistemic_graph.client``) exposes named
per-operation methods for RMDD-27's ``work_items`` resource-reservation verbs
but does not yet expose an equivalent ``development_lanes`` namespace for the
eight RMDD-28 ``DevelopmentLane*`` methods, even though the Rust engine
already registers and services them (``crates/eg-capabilities/src/lib.rs``:
``ReserveDevelopmentLane``, ``RenewDevelopmentLane``, ``ObserveDevelopmentLane``,
``FinishDevelopmentLane``, ``CleanupDevelopmentLane``, ``QueryDevelopmentLane``,
``DevelopmentLaneStatus``, ``UpdateDevelopmentLaneQuota``). This transport is
written *against that future namespace* using the exact same
capability-negotiation and strict-name-allowlist shape RMDD-27 already
established, so it starts working the moment epistemic-graph adds
``client.development_lanes`` with no further AU-side change. Today, against
the real generated client, construction fails closed with
:class:`DevelopmentLaneAuthorityUnavailable` -- which is the accurate,
honest state of the world, not a local approximation standing in for it.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any


class DevelopmentLaneAuthorityUnavailable(RuntimeError):
    """The connected engine does not advertise a native development-lane method.

    Raised at construction time (missing ``development_lanes`` namespace) and
    at call time (capability not advertised by ``client.supports``). Never
    caught and silently replaced by a local record, a SQLite row, a process
    lock, or any other non-durable stand-in -- callers that catch this must
    refuse the operation, not degrade it.
    """

    code = "native_development_lane_authority_unavailable"


_METHODS: dict[str, str] = {
    "ReserveDevelopmentLane": "reserve",
    "RenewDevelopmentLane": "renew",
    "ObserveDevelopmentLane": "observe",
    "FinishDevelopmentLane": "finish",
    "CleanupDevelopmentLane": "cleanup_complete",
    "QueryDevelopmentLane": "query",
    "DevelopmentLaneStatus": "status",
    "UpdateDevelopmentLaneQuota": "update_quota",
}


def _bounded_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} result must be a mapping")
    if len(value) > 64:
        raise ValueError(f"{name} result has too many fields")
    return dict(value)


class EngineNativeDevelopmentLaneTransport:
    """Low-level sync/async-client-neutral native development-lane transport.

    The generated client is expected to perform the full request/result
    schema validation (see ``agent_utilities.protocols.epistemic_operations``
    for the strict ``DevelopmentLane*`` DTOs this transport's callers must
    build requests from and parse results into). This wrapper adds only
    capability negotiation and transport-shape checks; every lifecycle
    decision remains in the engine and Repository Manager's sole
    ``LaneAuthority`` adapter.
    """

    def __init__(self, client: Any) -> None:
        lanes = getattr(client, "development_lanes", None)
        if lanes is None:
            raise DevelopmentLaneAuthorityUnavailable(
                DevelopmentLaneAuthorityUnavailable.code
            )
        self._client = client
        self._lanes = lanes

    def _operation(self, name: str) -> Any:
        if name not in _METHODS:
            raise ValueError("unknown native development-lane method")
        # Keep a strict explicit map so a fake/old namespace cannot
        # accidentally route an unrelated operation.
        operation = getattr(self._lanes, _METHODS[name], None)
        if not callable(operation):
            raise DevelopmentLaneAuthorityUnavailable(
                DevelopmentLaneAuthorityUnavailable.code
            )
        return operation

    def _require_capability(self, name: str) -> None:
        supports = getattr(self._client, "supports", None)
        if not callable(supports):
            raise DevelopmentLaneAuthorityUnavailable(
                DevelopmentLaneAuthorityUnavailable.code
            )
        try:
            advertised = supports(name)
        except DevelopmentLaneAuthorityUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - malformed/old Health fails closed
            raise DevelopmentLaneAuthorityUnavailable(
                DevelopmentLaneAuthorityUnavailable.code
            ) from exc
        if inspect.isawaitable(advertised):
            close = getattr(advertised, "close", None)
            if callable(close):
                close()
            raise TypeError(
                "EngineNativeDevelopmentLaneTransport requires sync supports"
            )
        if advertised is not True:
            raise DevelopmentLaneAuthorityUnavailable(
                DevelopmentLaneAuthorityUnavailable.code
            )

    def _call(self, name: str, request: Mapping[str, Any]) -> dict[str, Any]:
        self._require_capability(name)
        operation = self._operation(name)
        try:
            value = operation(request=dict(request))
        except DevelopmentLaneAuthorityUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - preserve typed transport errors
            if getattr(exc, "code", None) == DevelopmentLaneAuthorityUnavailable.code:
                raise DevelopmentLaneAuthorityUnavailable(
                    DevelopmentLaneAuthorityUnavailable.code
                ) from exc
            raise
        if inspect.isawaitable(value):
            close = getattr(value, "close", None)
            if callable(close):
                close()
            raise TypeError(
                "EngineNativeDevelopmentLaneTransport requires a sync generated client"
            )
        return _bounded_mapping(value, name)

    def reserve(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Atomically reserve one branch/worktree/quota hold for a lane WorkItem."""

        return self._call("ReserveDevelopmentLane", request)

    def renew(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Renew the current hold in place, bound to the live WorkItem lease."""

        return self._call("RenewDevelopmentLane", request)

    def observe(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Replace the monotonic retained-footprint charge with a fresh observation."""

        return self._call("ObserveDevelopmentLane", request)

    def finish(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Release active-count charge on terminal WorkItem completion."""

        return self._call("FinishDevelopmentLane", request)

    def cleanup_complete(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Release retained disk/exclusivity after guarded local removal succeeds."""

        return self._call("CleanupDevelopmentLane", request)

    def query(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Read one native hold/tombstone; local mirrors are not authority."""

        return self._call("QueryDevelopmentLane", request)

    def status(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Return bounded tenant-scoped hold status with maintained counters."""

        result = self._call("DevelopmentLaneStatus", request)
        holds = result.get("holds")
        limit = request.get("limit")
        if isinstance(holds, list) and isinstance(limit, int) and len(holds) > limit:
            raise ValueError("native development-lane status exceeded requested limit")
        return result

    def update_quota(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Publish a controller/admin-only monotonic quota-policy update."""

        return self._call("UpdateDevelopmentLaneQuota", request)


__all__ = [
    "DevelopmentLaneAuthorityUnavailable",
    "EngineNativeDevelopmentLaneTransport",
]
