"""Narrow transport for the engine-native WorkItem reservation protocol.

Repository Manager owns the ``WorkItemReservationPort`` lifecycle mapping.  AU
only exposes a typed transport over the generated ``SyncEpistemicGraphClient``;
it never invents owner/fence/profile defaults, performs a local read-then-CAS,
or memoizes transient admission refusals.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any


class NativeReservationUnavailable(RuntimeError):
    """The connected engine does not advertise a native reservation method."""

    code = "native_resource_reservation_unavailable"


_METHODS = frozenset(
    {
        "ReserveWorkItemResources",
        "ReleaseWorkItemResources",
        "ReclaimWorkItemResources",
        "QueryWorkItemReservation",
        "ResourceReservationStatus",
        "UpdateResourceHost",
    }
)


def _bounded_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} result must be a mapping")
    if len(value) > 64:
        raise ValueError(f"{name} result has too many fields")
    return dict(value)


class EngineNativeReservationTransport:
    """Low-level sync/async-client-neutral native reservation transport.

    The generated client performs the full request/result schema validation.
    This wrapper adds only capability negotiation and transport-shape checks;
    all lifecycle decisions remain in the engine and RM's sole adapter.
    """

    def __init__(self, client: Any) -> None:
        work_items = getattr(client, "work_items", None)
        if work_items is None:
            raise NativeReservationUnavailable(NativeReservationUnavailable.code)
        self._client = client
        self._work_items = work_items

    def _operation(self, name: str) -> Any:
        if name not in _METHODS:
            raise ValueError("unknown native reservation method")
        # Generated clients expose the user-facing lower-case methods below;
        # keep a strict explicit map so a fake/old namespace cannot accidentally
        # route an unrelated operation.
        operation = {
            "ReserveWorkItemResources": getattr(self._work_items, "reserve", None),
            "ReleaseWorkItemResources": getattr(self._work_items, "release", None),
            "ReclaimWorkItemResources": getattr(self._work_items, "reclaim", None),
            "QueryWorkItemReservation": getattr(self._work_items, "query_reservation", None),
            "ResourceReservationStatus": getattr(self._work_items, "status", None),
            "UpdateResourceHost": getattr(self._work_items, "update_host", None),
        }[name]
        if not callable(operation):
            raise NativeReservationUnavailable(NativeReservationUnavailable.code)
        return operation

    def _require_capability(self, name: str) -> None:
        supports = getattr(self._client, "supports", None)
        if not callable(supports):
            raise NativeReservationUnavailable(NativeReservationUnavailable.code)
        try:
            advertised = supports(name)
        except NativeReservationUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - malformed/old Health fails closed
            if getattr(exc, "code", None) == NativeReservationUnavailable.code:
                raise NativeReservationUnavailable(
                    NativeReservationUnavailable.code
                ) from exc
            raise NativeReservationUnavailable(
                NativeReservationUnavailable.code
            ) from exc
        if inspect.isawaitable(advertised):
            close = getattr(advertised, "close", None)
            if callable(close):
                close()
            raise TypeError("EngineNativeReservationTransport requires sync supports")
        if advertised is not True:
            raise NativeReservationUnavailable(NativeReservationUnavailable.code)

    def _call(self, name: str, request: Mapping[str, Any]) -> dict[str, Any]:
        self._require_capability(name)
        operation = self._operation(name)
        try:
            value = operation(request=dict(request))
        except NativeReservationUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - preserve typed transport errors
            if getattr(exc, "code", None) == NativeReservationUnavailable.code:
                raise NativeReservationUnavailable(NativeReservationUnavailable.code) from exc
            raise
        if inspect.isawaitable(value):
            # A mistaken async fake/client may have already created a coroutine;
            # close it before failing so capability probing cannot leak work or
            # emit an unawaited-coroutine warning during shutdown.
            close = getattr(value, "close", None)
            if callable(close):
                close()
            raise TypeError("EngineNativeReservationTransport requires a sync generated client")
        return _bounded_mapping(value, name)

    def reserve(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return self._call("ReserveWorkItemResources", request)

    def release(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return self._call("ReleaseWorkItemResources", request)

    def reclaim(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return self._call("ReclaimWorkItemResources", request)

    def query(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return self._call("QueryWorkItemReservation", request)

    def status(self, request: Mapping[str, Any]) -> dict[str, Any]:
        result = self._call("ResourceReservationStatus", request)
        reservations = result.get("reservations")
        if isinstance(reservations, list) and len(reservations) > int(request.get("limit", 0)):
            raise ValueError("native reservation status exceeded requested limit")
        return result

    def update_host(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return self._call("UpdateResourceHost", request)


__all__ = ["EngineNativeReservationTransport", "NativeReservationUnavailable"]
