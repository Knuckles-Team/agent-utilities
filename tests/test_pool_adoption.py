"""Hot-path adoption of the multiplexed engine connection pool (CONCEPT:KG-2.274).

Proves the consumer-side win of E's ``ShardRouter``/``ConnectionPool``: INDEPENDENT
engine ops fan out across the pool — each on its OWN connection — so wall-clock
collapses from the serial sum toward one op, while ordering WITHIN one logical write
(node-before-edge, a batch's op order) is preserved.

These run against an in-process fake router/engine (no real socket) so they stay fast
and deterministic: a fake connection records max simultaneous in-flight ops + which
connection serviced each, exactly the signal the real pool produces.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
from agent_utilities.knowledge_graph.enrichment.pipeline import _BatchedBackend


class _Probe:
    """Shared counters; mutated only on the single pool loop, so no lock needed."""

    def __init__(self) -> None:
        self.inflight = 0
        self.max_inflight = 0
        self.conns_used: set[int] = set()


class _FakeConn:
    """A pooled connection stand-in: ``client.lifecycle.batch_update`` simulates a
    round-trip and records concurrency on the shared probe."""

    def __init__(self, idx: int, probe: _Probe, op_delay: float) -> None:
        self._idx = idx
        self._probe = probe
        self._delay = op_delay
        self.lifecycle = self  # so ``client.lifecycle.batch_update`` resolves here

    async def batch_update(self, ops: list[dict[str, Any]]) -> dict[str, Any]:
        self._probe.inflight += 1
        self._probe.max_inflight = max(self._probe.max_inflight, self._probe.inflight)
        self._probe.conns_used.add(self._idx)
        await asyncio.sleep(self._delay)  # simulate the engine round-trip
        self._probe.inflight -= 1
        # Echo the op order so the caller can assert within-batch ordering held.
        return {"order": [o["id"] for o in ops], "conn": self._idx}


class _FakeRouter:
    """Mimics ``ShardRouter.map_concurrent``: each op gets its OWN connection and
    they run concurrently (the engine would ``tokio::spawn`` a task per connection)."""

    def __init__(self, probe: _Probe, op_delay: float, size: int = 8) -> None:
        self._probe = probe
        self._delay = op_delay
        self._free = list(range(size))

    async def map_concurrent(self, graph: str, ops: list[Any]) -> list[Any]:
        async def _run(op: Any) -> Any:
            idx = self._free.pop()  # a distinct connection per op
            try:
                return await op(_FakeConn(idx, self._probe, self._delay))
            finally:
                self._free.append(idx)

        return await asyncio.gather(*(_run(op) for op in ops))


def _pool_loop() -> asyncio.AbstractEventLoop:
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    return loop


def _engine_with_pool(
    loop: asyncio.AbstractEventLoop, router: Any
) -> GraphComputeEngine:
    """A GraphComputeEngine wired to a fake pool loop+router, bypassing __init__/connect."""
    eng = GraphComputeEngine.__new__(GraphComputeEngine)
    eng.graph_name = "test_graph"
    eng._pool_router = router  # cached → _ensure_pool() returns it, builds nothing
    eng._engine_loop = lambda: loop  # type: ignore[method-assign]
    return eng


def test_batch_update_concurrent_parallelizes_across_connections() -> None:
    """Independent batches run concurrently across DISTINCT pooled connections.

    Four batches, each a 0.05s round-trip. Serial would be ~0.20s on one connection;
    fanned across the pool it is ~0.05s. We assert >1 op in flight at once, >1 distinct
    connection used, and wall-clock well under the serial sum.
    """
    op_delay = 0.05
    n = 4
    probe = _Probe()
    loop = _pool_loop()
    eng = _engine_with_pool(loop, _FakeRouter(probe, op_delay, size=8))

    batches = [
        [{"op": "add_node", "id": f"b{i}_n{j}"} for j in range(3)] for i in range(n)
    ]
    t0 = time.monotonic()
    results = eng.batch_update_concurrent(batches)
    wall = time.monotonic() - t0

    assert probe.max_inflight > 1, (
        "ops did not overlap — still serialized on one connection"
    )
    assert len(probe.conns_used) > 1, "ops did not spread across multiple connections"
    assert wall < op_delay * n * 0.75, (
        f"no speedup: {wall:.3f}s vs serial ~{op_delay * n:.3f}s"
    )
    # Results keep input (batch) order.
    assert [r["order"][0] for r in results] == [f"b{i}_n0" for i in range(n)]


def test_ordering_within_one_logical_write_preserved() -> None:
    """A single logical write (ordered ops) stays in ONE batch == ONE connection.

    The node-before-edge ordering is expressed by keeping the ordered ops inside a
    single batch entry; the fake echoes the op order back unchanged, and the whole
    ordered batch is serviced by exactly one connection.
    """
    probe = _Probe()
    loop = _pool_loop()
    eng = _engine_with_pool(loop, _FakeRouter(probe, 0.0, size=8))

    ordered = [
        {"op": "add_node", "id": "n1"},
        {"op": "add_node", "id": "n2"},
        {"op": "add_edge", "id": "n1->n2"},
    ]
    [result] = eng.batch_update_concurrent([ordered])
    assert result["order"] == ["n1", "n2", "n1->n2"], (
        "within-batch op order was not preserved"
    )
    assert probe.max_inflight == 1, "one logical write must ride exactly one connection"


def test_degrades_to_single_connection_without_pool() -> None:
    """No pool ⇒ batches apply SEQUENTIALLY through the single shared client (still correct)."""
    eng = GraphComputeEngine.__new__(GraphComputeEngine)
    eng.graph_name = "test_graph"
    eng._pool_router = None  # pool unavailable
    eng._engine_loop = lambda: None  # type: ignore[method-assign]

    seen: list[list[str]] = []

    class _Client:
        class lifecycle:  # noqa: N801
            @staticmethod
            def batch_update(ops: list[dict[str, Any]]) -> dict[str, Any]:
                seen.append([o["id"] for o in ops])
                return {"ok": True}

    eng._client = _Client()
    out = eng.batch_update_concurrent(
        [[{"op": "add_node", "id": "a"}], [{"op": "add_node", "id": "b"}]]
    )
    assert len(out) == 2
    assert seen == [["a"], ["b"]], (
        "fallback must apply each batch in order on one connection"
    )


def test_batched_backend_fans_node_flush_then_edges() -> None:
    """The C-pipeline write seam (_BatchedBackend) fans a large flush across the pool,
    and still drains ALL nodes before ANY edge (CONCEPT:KG-2.274 + nodes-before-edges)."""
    calls: list[tuple[str, int]] = []  # (phase, n_chunks) in submission order

    class _Graph:
        def batch_update_concurrent(
            self, batches: list[list[dict[str, Any]]], graph: str | None = None
        ) -> list[dict[str, Any]]:
            phase = batches[0][0]["op"]  # "add_node" or "add_edge"
            calls.append((phase, len(batches)))
            return [{"ok": True} for _ in batches]

    class _Backend:
        _graph = _Graph()

    bb = _BatchedBackend(_Backend(), batch_size=8)  # sub_batch == 2
    # 8 nodes → one flush of 8 → split into 4 sub-batches (8 // 2) across the pool.
    for i in range(8):
        bb.add_node(f"n{i}", type="Code")
    for i in range(7):
        bb.add_edge(f"n{i}", f"n{i + 1}", rel_type="CALLS")
    bb.flush()

    phases = [c[0] for c in calls]
    assert "add_node" in phases and "add_edge" in phases
    # Every node submission precedes every edge submission.
    assert phases.index("add_edge") > max(
        i for i, p in enumerate(phases) if p == "add_node"
    ), "edges flushed before all nodes — ordering violated"
    # The node flush actually fanned into multiple concurrent sub-batches.
    node_chunks = max(c[1] for c in calls if c[0] == "add_node")
    assert node_chunks > 1, "node flush did not fan across multiple pooled connections"
