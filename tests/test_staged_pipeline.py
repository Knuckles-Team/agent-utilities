"""Tests for the non-blocking staged ingestion pipeline (CONCEPT:KG-2.267).

The load-bearing test is :func:`test_slow_enrich_does_not_block_write` — it proves
the operator principle: a slow ENRICH stage does NOT stall the WRITE of other items.
The stages are separate worker pools coupled only by a bounded queue, so writes run
ahead of a slow enricher instead of waiting on it.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from agent_utilities.knowledge_graph.ingestion.staged_pipeline import (
    Stage,
    StagedPipeline,
    compute_stage_workers,
)

pytestmark = pytest.mark.asyncio


async def test_slow_enrich_does_not_block_write() -> None:
    """THE principle: a slow ENRICH must not block WRITE of unrelated items.

    WRITE is instant and emits one enrich item per input; ENRICH sleeps. If the two
    stages were coupled, the writer would idle waiting on the slow enricher. With
    separated bounded-queue pools, every WRITE completes before even the FIRST
    ENRICH finishes.
    """
    write_times: list[float] = []
    enrich_end_times: list[float] = []
    t0 = time.monotonic()
    enrich_sleep = 0.3
    n = 6

    async def write(item: int) -> dict[str, int]:
        write_times.append(time.monotonic() - t0)
        return {"id": item}

    async def enrich(item: dict[str, int]) -> None:
        await asyncio.sleep(enrich_sleep)
        enrich_end_times.append(time.monotonic() - t0)

    write_stage = Stage("write", write, workers=n, capacity=n)
    enrich_stage = Stage("enrich", enrich, workers=2, capacity=n)
    pipeline = StagedPipeline([write_stage, enrich_stage])

    await pipeline.run(range(n))

    assert len(write_times) == n, "every item written"
    assert len(enrich_end_times) == n, "every item enriched"
    # Non-blocking guarantee: all writes finished before the first enrich completed,
    # i.e. the WRITE pool ran ahead of the slow ENRICH pool instead of waiting on it.
    assert max(write_times) < min(enrich_end_times), (
        f"writes ({max(write_times):.3f}s) blocked on enrich "
        f"({min(enrich_end_times):.3f}s) — stages are coupled!"
    )


async def test_backpressure_caps_queue_depth() -> None:
    """A bounded downstream queue caps memory: depth never exceeds capacity.

    WRITE fans 24 items into an ENRICH queue bounded at 2 with a slow consumer. The
    backpressured ``await put`` on the WRITE side keeps the enrich queue from ever
    growing past its bound — backpressure, not an unbounded buffer.
    """
    cap = 2

    async def write(item: int) -> int:
        return item

    async def enrich(item: int) -> None:
        await asyncio.sleep(0.01)

    write_stage = Stage("write", write, workers=4, capacity=8)
    enrich_stage = Stage("enrich", enrich, workers=1, capacity=cap)
    pipeline = StagedPipeline([write_stage, enrich_stage])

    await pipeline.run(range(24))

    assert enrich_stage.in_q.maxsize == cap
    assert enrich_stage.metrics.max_depth <= cap, (
        f"enrich queue depth {enrich_stage.metrics.max_depth} exceeded bound {cap}"
    )
    assert enrich_stage.metrics.processed == 24


async def test_per_stage_counters_increment() -> None:
    """Per-stage throughput counters + the metrics snapshot reflect real work."""

    async def stage_a(item: int) -> list[int]:
        return [item, item]  # fan-out: each input produces two downstream items

    async def stage_b(item: int) -> None:
        return None

    a = Stage("a", stage_a, workers=2, capacity=8)
    b = Stage("b", stage_b, workers=2, capacity=16)
    pipeline = StagedPipeline([a, b])

    await pipeline.run(range(5))

    assert a.metrics.processed == 5
    assert a.metrics.produced == 10  # 5 inputs * 2
    assert b.metrics.processed == 10

    snap = pipeline.metrics()
    assert snap["processed"] == 15  # 5 + 10
    assert snap["errors"] == 0
    names = [s["name"] for s in snap["stages"]]
    assert names == ["a", "b"]
    assert all("depth" in s and "mean_ms" in s for s in snap["stages"])


async def test_end_to_end_correctness_through_stages() -> None:
    """A small batch flows correctly through all stages with order-independent fan-out."""
    collected: list[int] = []

    async def parse(item: int) -> int:
        return item * 10

    async def write(item: int) -> int:
        return item + 1

    async def sink(item: int) -> None:
        collected.append(item)

    pipeline = StagedPipeline(
        [
            Stage("parse", parse, workers=3, capacity=8),
            Stage("write", write, workers=3, capacity=8),
            Stage("sink", sink, workers=2, capacity=8),
        ]
    )
    await pipeline.run(range(5))

    assert sorted(collected) == [1, 11, 21, 31, 41]


async def test_handler_error_does_not_stall_pool() -> None:
    """A poison item is counted and swallowed; the rest of the batch still drains."""
    done: list[int] = []

    async def flaky(item: int) -> None:
        if item == 3:
            raise ValueError("boom")
        done.append(item)

    stage = Stage("flaky", flaky, workers=2, capacity=8)
    pipeline = StagedPipeline([stage])
    await pipeline.run(range(6))

    assert sorted(done) == [0, 1, 2, 4, 5]
    assert stage.metrics.errors == 1
    assert stage.metrics.processed == 6


async def test_compute_stage_workers_sizing() -> None:
    """Pool sizing is per-bottleneck and always a positive int."""
    for kind in ("fetch", "parse", "enrich", "write", "reason"):
        n = compute_stage_workers(kind)
        assert isinstance(n, int) and n >= 1
    # An explicit configured anchor flows through to cpu/write-bound stages.
    assert compute_stage_workers("write", configured=5) == 5


async def test_ingest_batch_staged_decouples_enrich_from_write() -> None:
    """Engine wiring: a slow ENRICH must not block the structural WRITE of other items.

    Exercises ``IngestionEngine.ingest_batch_staged`` with a fast structural write
    (``ingest(defer_enrich=True)``) and a slow ``_enrich_payload``. All N structural
    writes complete before the first enrichment finishes — the writer never idles on
    LLM work. Built via ``__new__`` so no real KG engine/backend is needed.
    """
    from agent_utilities.knowledge_graph.ingestion.engine import (
        ContentType,
        IngestionEngine,
        IngestionManifest,
        IngestionResult,
    )

    eng = IngestionEngine.__new__(IngestionEngine)  # bypass singleton/backend setup
    eng._history = []

    t0 = time.monotonic()
    write_times: list[float] = []
    enrich_end_times: list[float] = []
    n = 5

    async def fake_ingest(manifest, *, defer_enrich=False):  # noqa: ANN001
        assert defer_enrich is True  # staged path must defer enrichment
        write_times.append(time.monotonic() - t0)
        return IngestionResult(
            manifest=manifest,
            status="success",
            nodes_created=1,
            enrichable=[{"source_id": manifest.source_uri, "text": "x"}],
        )

    async def fake_enrich_payload(payload, default_source_type):  # noqa: ANN001
        await asyncio.sleep(0.25)
        enrich_end_times.append(time.monotonic() - t0)
        return {"concepts": 2, "facts": 1}

    eng.ingest = fake_ingest  # type: ignore[method-assign]
    eng._enrich_payload = fake_enrich_payload  # type: ignore[method-assign]

    manifests = [
        IngestionManifest(content_type=ContentType.DOCUMENT, source_uri=f"doc-{i}")
        for i in range(n)
    ]
    results = await eng.ingest_batch_staged(manifests)

    assert len(results) == n
    assert all(r.status == "success" for r in results)
    # Enrichment counts were folded back onto each owning result.
    assert all(r.nodes_created == 3 for r in results)  # 1 structural + 2 concepts
    assert all(r.edges_created == 1 for r in results)
    # The non-blocking guarantee at the engine level.
    assert len(write_times) == n and len(enrich_end_times) == n
    assert max(write_times) < min(enrich_end_times), (
        "structural writes blocked on enrichment — stages are coupled!"
    )
    # Observability counters are surfaced on the first result.
    pipe = results[0].details.get("pipeline")
    assert pipe and {s["name"] for s in pipe["stages"]} == {"write", "enrich"}
