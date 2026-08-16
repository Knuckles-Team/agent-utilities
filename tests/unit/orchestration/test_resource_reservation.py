"""Focused tests for the narrow engine-native reservation transport."""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.orchestration.resource_reservation import (
    EngineNativeReservationTransport,
    NativeReservationUnavailable,
)


class _WorkItems:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def _result(self, name: str, *, request: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, request))
        return {"schema_version": "1", "decision": "accepted"}

    def reserve(self, *, request: dict[str, Any]) -> dict[str, Any]:
        return self._result("reserve", request=request)

    def release(self, *, request: dict[str, Any]) -> dict[str, Any]:
        return self._result("release", request=request)

    def reclaim(self, *, request: dict[str, Any]) -> dict[str, Any]:
        return self._result("reclaim", request=request)

    def query_reservation(self, *, request: dict[str, Any]) -> dict[str, Any]:
        return self._result("query", request=request)

    def status(self, *, request: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": "1",
            "complete": True,
            "next_cursor": None,
            "host_snapshot": None,
            "host_ref": None,
            "host_revision": 0,
            "held_cpu_weight": 0,
            "held_memory_mib": 0,
            "held_disk_mib": 0,
            "held_process_slots": 0,
            "fairness_debt": 0,
            "reservations": [],
            "orphan_count": 0,
            "superseded_count": 0,
        }

    def update_host(self, *, request: dict[str, Any]) -> dict[str, Any]:
        return self._result("update_host", request=request)


class _Client:
    def __init__(self) -> None:
        self.work_items = _WorkItems()

    def supports(self, operation: str) -> bool:
        return operation in {
            "ReserveWorkItemResources",
            "ReleaseWorkItemResources",
            "ReclaimWorkItemResources",
            "QueryWorkItemReservation",
            "ResourceReservationStatus",
            "UpdateResourceHost",
        }


def test_transport_maps_only_typed_native_operations() -> None:
    client = _Client()
    transport = EngineNativeReservationTransport(client)
    request = {"operation_id": "fresh-invocation", "now_ms": 10}

    assert transport.reserve(request)["decision"] == "accepted"
    assert transport.release(request)["decision"] == "accepted"
    assert transport.reclaim(request)["decision"] == "accepted"
    assert transport.query(request)["decision"] == "accepted"
    assert transport.update_host(request)["decision"] == "accepted"
    transport.status({**request, "limit": 1})
    assert [name for name, _ in client.work_items.calls] == [
        "reserve",
        "release",
        "reclaim",
        "query",
        "update_host",
    ]


def test_transport_rejects_async_generated_client() -> None:
    client = _Client()

    async def reserve(*, request: dict[str, Any]) -> dict[str, Any]:
        return {"request": request}

    client.work_items.reserve = reserve  # type: ignore[method-assign]
    with pytest.raises(TypeError, match="sync generated client"):
        EngineNativeReservationTransport(client).reserve({"now_ms": 1})


def test_status_result_cannot_exceed_requested_bound() -> None:
    class Overfull(_WorkItems):
        def status(self, *, request: dict[str, Any]) -> dict[str, Any]:
            result = super().status(request=request)
            result["reservations"] = [
                {"reservation_id": "one"},
                {"reservation_id": "two"},
            ]
            return result

    client = _Client()
    client.work_items = Overfull()
    with pytest.raises(ValueError, match="exceeded requested limit"):
        EngineNativeReservationTransport(client).status({"limit": 1})


def test_transport_fails_closed_without_work_item_namespace() -> None:
    with pytest.raises(NativeReservationUnavailable):
        EngineNativeReservationTransport(object())


def test_transport_fails_closed_when_old_engine_does_not_advertise_method() -> None:
    class OldEngine(_Client):
        def supports(self, _operation: str) -> bool:
            return False

    client = OldEngine()
    with pytest.raises(NativeReservationUnavailable):
        EngineNativeReservationTransport(client).reserve({"now_ms": 1})
    assert client.work_items.calls == []
