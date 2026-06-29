"""Non-blocking staged ingestion pipeline (CONCEPT:KG-2.267).

Roadmap item **C** of ``reports/north-star-agent-compute-architecture-2026-06-27.md``:
decompose the inline ``fetch → parse → enrich → write → reason`` ingestion job into
a chain of **separated, interconnected, never-dependent-locked stage queues**.

The operator principle (the law)
--------------------------------
*Separated, interconnected queues, never dependent-locked; non-blocking everywhere;
blocked time with nothing to do is wasted compute. A slow stage must NOT stall an
unrelated stage.*

Today (the block this removes)
------------------------------
``IngestionEngine.ingest`` runs the structural write (the adaptor handler) and then
the **LLM-bound enrichment** inline, in the same coroutine, before it returns. The
durable WRITE side therefore sits idle while a job enriches, and the worker that
claimed the job cannot pick up the next one until enrichment finishes. The lanes
(ingestion/connectors/worldview/…) separate *work items*, but a single item's
*stages* are coupled.

The model here
--------------
Each ``Stage`` is its **own async worker pool** consuming from a **bounded**
``asyncio.Queue`` and producing into the next stage's bounded queue::

    [items] -> (Stage A: pool=Na) -q-> (Stage B: pool=Nb) -q-> (Stage C: pool=Nc)

* **Stages run concurrently** — while repo *A* enriches, repo *B* parses and repo *C*
  writes. The pools are independent.
* **Backpressure, not locks** — a full downstream queue makes the *producer* stage
  ``await queue.put(...)`` (slow down); it never takes a lock another stage needs and
  never stalls an unrelated stage. The queue bound caps memory.
* **No dependent locks** — coupling is *only* via the bounded queues. A stage never
  holds a lock that a different stage must acquire. Reads/other stages never block on
  a writer. The durable write goes to the engine, which already group-commits.
* **Auto-sized per bottleneck** — fetch≈net-concurrency, parse≈cpu_count,
  enrich≈LLM/GPU slots, write≈engine batch writers. Sizes derive from ``(cpu, config)``
  via :func:`compute_ingest_worker_count`, honouring the Pi-OOM cap upstream.

Shutdown is via chained :meth:`asyncio.Queue.join` — drain stage *k* fully (every
``task_done`` accounted, which guarantees all of its puts into stage *k+1* landed)
before draining stage *k+1*. This is lock-free and deadlock-free for a linear DAG:
while we wait on an upstream ``join``, the downstream pools keep draining, so a
backpressured ``put`` always unblocks.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

#: A stage handler maps one input item to zero or more downstream items. Returning
#: ``None`` (or an empty iterable) forwards nothing — a terminal/sink stage. The
#: handler is awaited inside a worker; it must not hold any cross-stage lock.
StageHandler = Callable[[Any], Awaitable[Any]]

# Sentinel item meaning "no more work for this stage" — never used by callers; the
# pipeline drains via Queue.join(), not sentinels, but this object is reserved for
# future streaming sources that cannot pre-count their items.
_DONE = object()


@dataclass
class StageMetrics:
    """Live per-stage throughput + queue-depth counters (mirrors OS-5.55 lane profiling).

    Snapshotted by :meth:`StagedPipeline.metrics` so an operator can SEE where the
    bottleneck moved: a stage whose ``depth`` rides at ``capacity`` is the new
    bottleneck (its consumers are saturated); a stage whose ``depth`` stays ~0 is
    starved (its producer is the bottleneck).
    """

    name: str
    workers: int
    capacity: int
    #: items fully handled (handler returned) by this stage's pool.
    processed: int = 0
    #: items this stage emitted into the downstream queue.
    produced: int = 0
    #: handler exceptions (swallowed — a bad item never stalls the pool).
    errors: int = 0
    #: cumulative wall time the pool spent inside the handler (busy, not waiting).
    busy_s: float = 0.0
    #: high-water mark of the *input* queue depth observed by the pool.
    max_depth: int = 0
    #: live input-queue depth at snapshot time (filled in by the pipeline).
    depth: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "workers": self.workers,
            "capacity": self.capacity,
            "depth": self.depth,
            "max_depth": self.max_depth,
            "processed": self.processed,
            "produced": self.produced,
            "errors": self.errors,
            "busy_s": round(self.busy_s, 4),
            # Mean handler service time — the per-item cost at this stage.
            "mean_ms": round((self.busy_s / self.processed) * 1000, 2)
            if self.processed
            else 0.0,
        }


class Stage:
    """One stage = a bounded input queue + a fixed-size async worker pool.

    Workers loop forever (until cancelled at pipeline teardown): pull one item,
    run the handler, ``await`` each produced item into the *next* stage's queue
    (backpressure on this stage only), then ``task_done``. A handler exception is
    counted and swallowed so a single poison item can never stall the pool.
    """

    def __init__(
        self,
        name: str,
        handler: StageHandler,
        *,
        workers: int,
        capacity: int,
    ) -> None:
        self.name = name
        self._handler = handler
        self.workers = max(1, int(workers))
        self.capacity = max(1, int(capacity))
        #: bounded input queue — the backpressure point feeding this stage.
        self.in_q: asyncio.Queue[Any] = asyncio.Queue(maxsize=self.capacity)
        #: set by the pipeline when stages are linked; ``None`` => terminal sink.
        self.next: Stage | None = None
        self.metrics = StageMetrics(
            name=name, workers=self.workers, capacity=self.capacity
        )
        self._tasks: list[asyncio.Task[None]] = []

    def start(self) -> None:
        """Spawn the worker pool (idempotent)."""
        if self._tasks:
            return
        self._tasks = [
            asyncio.create_task(self._worker(i), name=f"stage:{self.name}:{i}")
            for i in range(self.workers)
        ]

    async def _worker(self, idx: int) -> None:
        in_q = self.in_q
        nxt = self.next
        m = self.metrics
        while True:
            item = await in_q.get()
            try:
                if item is _DONE:
                    continue
                depth = in_q.qsize()
                if depth > m.max_depth:
                    m.max_depth = depth
                t0 = time.monotonic()
                try:
                    produced = await self._handler(item)
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 — one bad item never stalls the pool
                    m.errors += 1
                    logger.debug("stage %s handler error", self.name, exc_info=True)
                    produced = None
                m.busy_s += time.monotonic() - t0
                m.processed += 1
                if produced is not None and nxt is not None:
                    for out in _as_iter(produced):
                        # Bounded put == backpressure on THIS stage only. Other
                        # stages keep draining; never a global stall, never a lock.
                        await nxt.in_q.put(out)
                        m.produced += 1
            finally:
                in_q.task_done()

    async def join(self) -> None:
        """Block until every item enqueued so far has been fully handled."""
        await self.in_q.join()

    async def stop(self) -> None:
        """Cancel the worker pool. Call only after this stage's queue has drained."""
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._tasks = []


def _as_iter(produced: Any) -> Iterable[Any]:
    """Normalize a handler's return into an iterable of downstream items.

    A handler may return a single item, a list/tuple/generator of items, or
    ``None``. Strings/bytes/dicts are treated as *one* item (not iterated).
    """
    if produced is None:
        return ()
    if isinstance(produced, str | bytes | dict):
        return (produced,)
    if isinstance(produced, list | tuple | set):
        return produced
    if isinstance(produced, Iterable):
        return produced
    return (produced,)


class StagedPipeline:
    """A linear chain of :class:`Stage` pools connected by bounded queues.

    Build with an ordered list of stages (first = head, last = sink). Feed source
    items into the head; :meth:`run` drains the whole chain stage-by-stage and tears
    the pools down. The pipeline holds **no lock**; the only coupling is the bounded
    inter-stage queues.
    """

    def __init__(self, stages: Sequence[Stage]) -> None:
        if not stages:
            raise ValueError("StagedPipeline needs at least one stage")
        self.stages: list[Stage] = list(stages)
        for a, b in zip(self.stages, self.stages[1:], strict=False):
            a.next = b
        self._started = False

    @property
    def head(self) -> Stage:
        return self.stages[0]

    def start(self) -> None:
        for s in self.stages:
            s.start()
        self._started = True

    async def feed(self, items: Iterable[Any]) -> None:
        """Push source items into the head queue (backpressured at the head bound)."""
        if not self._started:
            self.start()
        q = self.head.in_q
        for it in items:
            await q.put(it)

    async def drain(self) -> None:
        """Drain every stage in order, then tear the pools down.

        Chained ``join``: once stage *k* is fully drained, all of its puts into
        stage *k+1* have landed (the worker awaits the put before ``task_done``),
        so it is safe to ``join`` stage *k+1*. Lock-free and deadlock-free for the
        linear DAG — backpressured puts unblock because downstream pools stay live.
        """
        for s in self.stages:
            await s.join()
        # Drained: cancel pools from the head down (no item is in flight anywhere).
        for s in self.stages:
            await s.stop()
        self._started = False

    async def run(self, items: Iterable[Any]) -> None:
        """Convenience: ``start`` → ``feed`` → ``drain`` for a finite batch."""
        self.start()
        await self.feed(items)
        await self.drain()

    def metrics(self) -> dict[str, Any]:
        """Snapshot per-stage queue depth + throughput counters.

        Mirrors the OS-5.55 lane profiler shape so the same dashboards/tools can
        read it: a list of per-stage dicts plus a rollup.
        """
        stages: list[dict[str, Any]] = []
        for s in self.stages:
            s.metrics.depth = s.in_q.qsize()
            stages.append(s.metrics.as_dict())
        return {
            "stages": stages,
            "processed": sum(d["processed"] for d in stages),
            "errors": sum(d["errors"] for d in stages),
        }


def compute_stage_workers(kind: str, *, configured: int | None = None) -> int:
    """Size a stage's pool by its bottleneck, derived from ``(cpu, config)``.

    * ``fetch``  — net-bound: a small multiple of the cpu pool (I/O overlaps), capped.
    * ``parse``  — cpu-bound: the cpu-derived ingest worker count.
    * ``enrich`` — LLM/GPU-slot-bound: the local-inference concurrency MINUS the
      reserved interactive slot, so the background enrich pass can never starve the
      slot the messaging responder / spawned agents need.
    * ``write``  — engine-batch-bound: the cpu-derived ingest worker count (the K
      shard-writers already group-commit downstream).
    * ``reason`` — cheap/serial mirror/reason fan-out: small.

    All sizes share :func:`compute_ingest_worker_count` as the cpu/mem anchor so the
    Pi-OOM cap and the box's ~36%-of-cores budget are honoured automatically.
    """
    from agent_utilities.knowledge_graph.core.engine_tasks import (
        compute_ingest_worker_count,
    )

    base = compute_ingest_worker_count(configured)
    kind = (kind or "").lower()
    if kind == "fetch":
        return max(2, min(base * 2, 16))
    if kind in ("parse", "write"):
        return base
    if kind == "enrich":
        try:
            from agent_utilities.core.config import (
                RESERVED_INTERACTIVE_INSTANCES,
                setting,
            )

            cap = int(setting("KG_LLM_CONCURRENCY", 4))
            return max(1, cap - RESERVED_INTERACTIVE_INSTANCES)
        except Exception:  # noqa: BLE001 — config best-effort
            return max(1, base // 2)
    if kind == "reason":
        return max(1, base // 2)
    return base


__all__ = [
    "Stage",
    "StageMetrics",
    "StagedPipeline",
    "compute_stage_workers",
]
