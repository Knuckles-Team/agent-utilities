"""Native batch wire contracts (CONCEPT:AU-KG.ingest.floor-codebase-admission-cap).

The native engine owns admission, depth, item, operation-count, and atomicity
limits.  These fixtures only prove that the AU routed client preserves the
``serde_bytes`` contract at the boundary: encoded nested payloads are
MessagePack ``bin`` values, never arrays of one integer per byte.
"""

from __future__ import annotations

import asyncio
from typing import Any

import msgpack
import pytest

from agent_utilities.knowledge_graph.core.graph_compute import (
    _CanonicalLifecycleClient,
    _encode_batch_operations,
    _encode_multi_graph_batches,
)

_MAX_WIRE_ITEMS = 500_000
_MAX_WIRE_DEPTH = 64
_MAX_WIRE_BYTES = 64 * 1024 * 1024
_MAX_BATCH_OPERATIONS = 50_000


class _Route:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def _send(self, method: str, params: dict[str, Any]) -> dict[str, str]:
        self.calls.append((method, params))
        return {"status": "accepted"}


class _Delegate:
    """A generated lifecycle namespace stand-in for non-batch delegation."""

    def __init__(self, client: _Route) -> None:
        self._client = client


def _invoke(
    method: str,
    value: list[dict[str, Any]] | dict[str, list[dict[str, Any]]],
) -> tuple[_Route, dict[str, Any]]:
    route = _Route()
    lifecycle = _CanonicalLifecycleClient(route, _Delegate(route))
    if method == "BatchUpdate":
        asyncio.run(lifecycle.batch_update(value))  # type: ignore[arg-type]
    else:
        asyncio.run(lifecycle.multi_graph_batch_update(value))  # type: ignore[arg-type]
    assert len(route.calls) == 1
    called_method, params = route.calls[0]
    assert called_method == method
    return route, params


def _outer_wire(params: dict[str, Any]) -> dict[str, Any]:
    """Round-trip through the native client's outer MessagePack serializer."""
    return msgpack.unpackb(msgpack.packb(params, use_bin_type=True), raw=False)


def _shape_items(value: Any, *, depth: int = 0) -> int:
    """Count structural items as the engine's allocation scanner does.

    Binary values are opaque and cost one item; arrays/maps charge their
    members.  This test-only fixture intentionally stays small and does not
    replace the engine validator.
    """
    if depth > _MAX_WIRE_DEPTH:
        raise ValueError("depth limit")
    count = 1
    if isinstance(value, dict):
        for key, child in value.items():
            count += _shape_items(key, depth=depth + 1)
            count += _shape_items(child, depth=depth + 1)
    elif isinstance(value, (list, tuple)):
        for child in value:
            count += _shape_items(child, depth=depth + 1)
    return count


def _assert_bounded_batch_wire(
    payload: bytes | list[int], *, expected_items: int | None = None
) -> None:
    """Apply the focused fixture gate without mutating or chunking a batch."""
    if not isinstance(payload, bytes):
        raise ValueError("batch payload must be MessagePack bin")
    if len(payload) > _MAX_WIRE_BYTES:
        raise ValueError("batch payload exceeds byte limit")
    decoded = msgpack.unpackb(payload, raw=False)
    items = _shape_items(decoded)
    if items > _MAX_WIRE_ITEMS:
        raise ValueError("batch payload exceeds item limit")
    if expected_items is not None:
        assert items == expected_items


def test_single_batch_uses_bin_for_operations_payload() -> None:
    operations = [{"op": "add_node", "id": "n-1", "properties": {}}]

    _route, params = _invoke("BatchUpdate", operations)

    payload = params["operations_msgpack"]
    assert isinstance(payload, bytes)
    assert _outer_wire(params)["operations_msgpack"] == payload
    assert msgpack.unpackb(payload, raw=False) == operations
    # The old generated-client shape is an array and would make every encoded
    # byte count as a collection item in the native outer preflight.
    assert isinstance(
        msgpack.unpackb(
            msgpack.packb({"operations_msgpack": list(payload)}, use_bin_type=True),
            raw=False,
        )["operations_msgpack"],
        list,
    )


def test_multi_graph_uses_bin_for_outer_and_inner_payloads() -> None:
    batches = {
        "tenant:a": [{"op": "add_node", "id": "a", "properties": {}}],
        "tenant:b": [{"op": "remove_node", "id": "b"}],
    }

    _route, params = _invoke("MultiGraphBatchUpdate", batches)

    payload = params["batches_msgpack"]
    assert isinstance(payload, bytes)
    assert _outer_wire(params)["batches_msgpack"] == payload
    decoded = msgpack.unpackb(payload, raw=False)
    assert [graph for graph, _operations in decoded] == list(batches)
    assert all(isinstance(operations, bytes) for _graph, operations in decoded)
    assert [msgpack.unpackb(operations, raw=False) for _graph, operations in decoded] == list(
        batches.values()
    )


@pytest.mark.parametrize("extra_items", [0, 1])
def test_exact_item_boundary_and_one_over_are_preserved_as_bin(extra_items: int) -> None:
    # The outer map/key/value consume three structural items.  The old list[int]
    # representation therefore reaches the boundary at MAX-3, while the binary
    # representation remains one opaque value regardless of payload length.
    encoded_length = _MAX_WIRE_ITEMS - 3 + extra_items
    # ``bin32`` has a five-byte header; the resulting payload is valid MessagePack
    # and carries no collection members for the structural scanner to enumerate.
    payload = msgpack.packb(
        bytes(encoded_length - 5), use_bin_type=True
    )
    assert len(payload) == encoded_length

    legacy_wire = msgpack.packb(
        {"operations_msgpack": list(payload)}, use_bin_type=True
    )
    legacy_decoded = msgpack.unpackb(legacy_wire, raw=False)
    legacy_items = _shape_items(legacy_decoded)
    if extra_items:
        assert legacy_items > _MAX_WIRE_ITEMS
    else:
        assert legacy_items == _MAX_WIRE_ITEMS

    canonical_wire = msgpack.packb(
        {"operations_msgpack": payload}, use_bin_type=True
    )
    canonical_decoded = msgpack.unpackb(canonical_wire, raw=False)
    assert _shape_items(canonical_decoded) == 3


def test_non_bin_and_depth_bomb_fixtures_fail_closed() -> None:
    with pytest.raises(ValueError, match="bin"):
        _assert_bounded_batch_wire([0x91, 0x80])

    depth_bomb: Any = None
    for _ in range(_MAX_WIRE_DEPTH + 1):
        depth_bomb = [depth_bomb]
    bomb = msgpack.packb(depth_bomb, use_bin_type=True)
    with pytest.raises(ValueError, match="depth"):
        _assert_bounded_batch_wire(bomb)


def test_multi_graph_encoder_retains_single_round_trip_atomic_shape() -> None:
    batches = {"tenant:one": [{"op": "remove_node", "id": "n"}]}
    payload = _encode_multi_graph_batches(batches)

    assert isinstance(payload, bytes)
    assert len(msgpack.unpackb(payload, raw=False)) == 1
    # No chunking/splitting occurs in the AU encoder; the engine receives one
    # opaque payload and remains responsible for transactional admission.
    inner = msgpack.unpackb(payload, raw=False)[0][1]
    _assert_bounded_batch_wire(inner)


@pytest.mark.parametrize(
    "operation_count", [_MAX_BATCH_OPERATIONS, _MAX_BATCH_OPERATIONS + 1]
)
def test_operation_count_boundary_is_forwarded_without_client_chunking(
    operation_count: int,
) -> None:
    operations = [
        {"op": "remove_node", "id": f"boundary-{index}"}
        for index in range(operation_count)
    ]

    payload = _encode_batch_operations(operations)

    # Both the exact boundary and one-over are sent as one opaque payload.  The
    # engine remains the authority that accepts/rejects the count atomically.
    assert msgpack.unpackb(payload, raw=False) == operations
