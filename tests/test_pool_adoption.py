"""Native request-pipelining adoption (CONCEPT:AU-KG.compute.when-exposes).

Independent engine operations share the one process-owned authenticated transport
and remain concurrently in flight. The generated client demultiplexes responses by
request id, so an auxiliary connection pool must not cache caller authority. Ops
inside one logical batch retain their input order.

These tests use an in-process async client stand-in; no socket or engine is started.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
from agent_utilities.knowledge_graph.enrichment.pipeline import _BatchedBackend


class _Probe:
    """Shared counters mutated on the single native-client event loop."""

    def __init__(self) -> None:
        self.inflight = 0
        self.max_inflight = 0
        self.transport_ids: set[int] = set()


class _FakeLifecycle:
    """One transport namespace that records concurrent pipelined RPCs."""

    def __init__(self, transport_id: int, probe: _Probe, op_delay: float) -> None:
        self._transport_id = transport_id
        self._probe = probe
        self._delay = op_delay

    async def batch_update(self, ops: list[dict[str, Any]]) -> dict[str, Any]:
        self._probe.inflight += 1
        self._probe.max_inflight = max(self._probe.max_inflight, self._probe.inflight)
        self._probe.transport_ids.add(self._transport_id)
        await asyncio.sleep(self._delay)
        self._probe.inflight -= 1
        return {
            "order": [operation["id"] for operation in ops],
            "transport": self._transport_id,
        }


class _FakeAsyncClient:
    def __init__(self, probe: _Probe, op_delay: float) -> None:
        self.lifecycle = _FakeLifecycle(id(self), probe, op_delay)


@contextlib.contextmanager
def _running_engine_loop() -> Iterator[asyncio.AbstractEventLoop]:
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)
        loop.close()


def _engine_with_client(
    loop: asyncio.AbstractEventLoop, client: Any
) -> GraphComputeEngine:
    """Wire a fake routed async client without constructing a live engine."""

    engine = GraphComputeEngine.__new__(GraphComputeEngine)
    engine.graph_name = "test_graph"
    engine._client = SimpleNamespace(_client=client, _loop=loop)
    return engine


def test_batch_update_concurrent_pipelines_on_one_transport() -> None:
    """Independent RPCs overlap while retaining one process transport."""

    op_delay = 0.05
    batch_count = 4
    probe = _Probe()
    with _running_engine_loop() as loop:
        engine = _engine_with_client(loop, _FakeAsyncClient(probe, op_delay))
        batches = [
            [
                {"op": "add_node", "id": f"b{batch}_n{node}"}
                for node in range(3)
            ]
            for batch in range(batch_count)
        ]

        started = time.monotonic()
        results = engine.batch_update_concurrent(batches)
        elapsed = time.monotonic() - started

    assert probe.max_inflight > 1
    assert len(probe.transport_ids) == 1
    assert elapsed < op_delay * batch_count * 0.75
    assert [result["order"][0] for result in results] == [
        f"b{batch}_n0" for batch in range(batch_count)
    ]


def test_ordering_within_one_logical_write_is_preserved() -> None:
    probe = _Probe()
    with _running_engine_loop() as loop:
        engine = _engine_with_client(loop, _FakeAsyncClient(probe, 0.0))
        ordered = [
            {"op": "add_node", "id": "n1"},
            {"op": "add_node", "id": "n2"},
            {"op": "add_edge", "id": "n1->n2"},
        ]
        [result] = engine.batch_update_concurrent([ordered])

    assert result["order"] == ["n1", "n2", "n1->n2"]
    assert probe.max_inflight == 1
    assert len(probe.transport_ids) == 1


def test_concurrent_batches_require_native_engine_loop() -> None:
    engine = GraphComputeEngine.__new__(GraphComputeEngine)
    engine.graph_name = "test_graph"
    engine._client = SimpleNamespace(_client=_FakeAsyncClient(_Probe(), 0.0))

    with pytest.raises(RuntimeError, match="no engine loop"):
        engine.batch_update_concurrent([[{"op": "add_node", "id": "a"}]])


def test_batched_backend_fans_node_flush_then_edges() -> None:
    """The enrichment seam fans chunks but drains every node before any edge."""

    calls: list[tuple[str, int]] = []

    class _Graph:
        def batch_update_concurrent(
            self, batches: list[list[dict[str, Any]]], graph: str | None = None
        ) -> list[dict[str, Any]]:
            del graph
            phase = batches[0][0]["op"]
            calls.append((phase, len(batches)))
            return [{"ok": True} for _ in batches]

    class _Backend:
        _graph = _Graph()

    backend = _BatchedBackend(_Backend(), batch_size=8)
    for index in range(8):
        backend.add_node(f"n{index}", type="Code")
    for index in range(7):
        backend.add_edge(f"n{index}", f"n{index + 1}", rel_type="CALLS")
    backend.flush()

    phases = [phase for phase, _ in calls]
    assert "add_node" in phases and "add_edge" in phases
    assert phases.index("add_edge") > max(
        index for index, phase in enumerate(phases) if phase == "add_node"
    )
    assert max(count for phase, count in calls if phase == "add_node") > 1
