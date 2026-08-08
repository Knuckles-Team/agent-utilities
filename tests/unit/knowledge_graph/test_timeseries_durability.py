"""Regression coverage for EngineTimeSeriesBackend's durability boundary."""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from agent_utilities.knowledge_graph.memory.timeseries.base import TimeSeriesDataPoint
from agent_utilities.knowledge_graph.memory.timeseries.engine_backend import (
    EngineTimeSeriesBackend,
)


def _point(symbol: str = "agent-telemetry") -> TimeSeriesDataPoint:
    return TimeSeriesDataPoint(
        symbol=symbol,
        timestamp=datetime(2026, 8, 2, tzinfo=UTC),
        metrics={"tokens": 12.0},
    )


class _SyncTimeSeries:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def register_series(
        self,
        series_id: str,
        *,
        entity_id: str,
        field_names: list[str],
        metadata: dict[str, str],
    ) -> None:
        self.calls.append("register")

    def append(
        self,
        series_id: str,
        points: list[tuple[int, list[float]]],
        *,
        field_names: list[str] | None,
    ) -> int:
        self.calls.append("append")
        return len(points)


class _BlockingSyncTimeSeries(_SyncTimeSeries):
    def __init__(self) -> None:
        super().__init__()
        self.register_started = threading.Event()
        self.allow_register = threading.Event()

    def register_series(
        self,
        series_id: str,
        *,
        entity_id: str,
        field_names: list[str],
        metadata: dict[str, str],
    ) -> None:
        self.register_started.set()
        assert self.allow_register.wait(1.0)
        super().register_series(
            series_id,
            entity_id=entity_id,
            field_names=field_names,
            metadata=metadata,
        )


class _AsyncTimeSeries:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.register_started = asyncio.Event()
        self.allow_register = asyncio.Event()
        self.registered = False

    async def register_series(
        self,
        series_id: str,
        *,
        entity_id: str,
        field_names: list[str],
        metadata: dict[str, str],
    ) -> None:
        self.calls.append("register-start")
        self.register_started.set()
        await self.allow_register.wait()
        self.registered = True
        self.calls.append("register-finish")

    async def append(
        self,
        series_id: str,
        points: list[tuple[int, list[float]]],
        *,
        field_names: list[str] | None,
    ) -> int:
        assert self.registered
        self.calls.append("append")
        return len(points)


class _FailingAsyncTimeSeries:
    def __init__(self) -> None:
        self.appended = False

    async def register_series(
        self,
        series_id: str,
        *,
        entity_id: str,
        field_names: list[str],
        metadata: dict[str, str],
    ) -> None:
        raise RuntimeError("very-secret-error-detail")

    async def append(
        self,
        series_id: str,
        points: list[tuple[int, list[float]]],
        *,
        field_names: list[str] | None,
    ) -> int:
        self.appended = True
        return len(points)


class _StrictTimeSeriesState:
    def __init__(self) -> None:
        self.fields: dict[str, list[str]] = {}
        self.batches: list[list[tuple[int, list[float]]]] = []

    def record_registration(self, series_id: str, field_names: list[str]) -> None:
        self.fields[series_id] = field_names

    def record_append(
        self,
        series_id: str,
        points: list[tuple[int, list[float]]],
        field_names: list[str] | None,
    ) -> int:
        assert field_names == self.fields[series_id]
        assert all(len(values) == len(field_names) for _, values in points)
        self.batches.append(points)
        return len(points)


class _StrictSyncTimeSeries(_StrictTimeSeriesState):
    """Engine-shaped client that rejects vectors outside the registered schema."""

    def register_series(
        self,
        series_id: str,
        *,
        entity_id: str,
        field_names: list[str],
        metadata: dict[str, str],
    ) -> None:
        self.record_registration(series_id, field_names)

    def append(
        self,
        series_id: str,
        points: list[tuple[int, list[float]]],
        *,
        field_names: list[str] | None,
    ) -> int:
        return self.record_append(series_id, points, field_names)


class _StrictAsyncTimeSeries(_StrictTimeSeriesState):
    async def register_series(
        self,
        series_id: str,
        *,
        entity_id: str,
        field_names: list[str],
        metadata: dict[str, str],
    ) -> None:
        self.record_registration(series_id, field_names)

    async def append(
        self,
        series_id: str,
        points: list[tuple[int, list[float]]],
        *,
        field_names: list[str] | None,
    ) -> int:
        return self.record_append(series_id, points, field_names)


def _client(timeseries: object) -> SimpleNamespace:
    return SimpleNamespace(timeseries=timeseries)


def _metrics_point(**metrics: float) -> TimeSeriesDataPoint:
    return TimeSeriesDataPoint(
        symbol="expanding-telemetry",
        timestamp=datetime(2026, 8, 2, tzinfo=UTC),
        metrics=metrics,
    )


def test_sync_batch_encodes_every_point_against_the_expanded_schema() -> None:
    timeseries = _StrictSyncTimeSeries()
    backend = EngineTimeSeriesBackend(client=_client(timeseries))

    backend.insert(
        [
            _metrics_point(tokens=12.0),
            _metrics_point(tokens=15.0, latency=3.0),
        ]
    )

    sid = "ts:expanding-telemetry"
    assert timeseries.fields[sid] == ["latency", "tokens"]
    assert [values for _, values in timeseries.batches[0]] == [
        [0.0, 12.0],
        [3.0, 15.0],
    ]


async def test_async_batch_preserves_known_order_and_widens_every_vector() -> None:
    timeseries = _StrictAsyncTimeSeries()
    backend = EngineTimeSeriesBackend(client=_client(timeseries))
    sid = "ts:expanding-telemetry"
    backend._fields[sid] = ["tokens"]

    backend.insert(
        [
            _metrics_point(tokens=12.0, latency=3.0),
            _metrics_point(tokens=15.0, errors=2.0),
        ]
    )
    tail = backend._pending_writes[sid]
    assert await asyncio.wait_for(tail, timeout=1.0) is True

    assert timeseries.fields[sid] == ["tokens", "errors", "latency"]
    assert [values for _, values in timeseries.batches[0]] == [
        [12.0, 0.0, 3.0],
        [15.0, 2.0, 0.0],
    ]


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_sync_client_keeps_register_before_append_without_pending_tasks() -> None:
    """The production sync facade preserves its immediate write contract."""
    timeseries = _SyncTimeSeries()
    backend = EngineTimeSeriesBackend(client=_client(timeseries))

    backend.insert([_point()])

    assert timeseries.calls == ["register", "append"]
    assert backend._pending_writes == {}
    assert backend._background_writes == set()


@pytest.mark.filterwarnings("error::RuntimeWarning")
async def test_sync_client_does_not_block_a_running_serving_loop() -> None:
    """The sync facade must use a worker rather than stalling GraphOS's loop."""
    timeseries = _BlockingSyncTimeSeries()
    backend = EngineTimeSeriesBackend(client=_client(timeseries))

    backend.insert([_point()])
    assert await asyncio.to_thread(timeseries.register_started.wait, 0.5)

    # This scheduler turn is the serving-loop liveness proof while the client
    # facade is waiting synchronously in its worker thread.
    await asyncio.sleep(0)
    assert timeseries.calls == []

    tail = next(iter(backend._pending_writes.values()))
    timeseries.allow_register.set()
    assert await asyncio.wait_for(tail, timeout=1.0) is True
    assert timeseries.calls == ["register", "append"]


@pytest.mark.filterwarnings("error::RuntimeWarning")
async def test_sync_write_allocates_no_extra_inner_task() -> None:
    """D-CDX-58: no redundant inner task per sync-facade register/append write.

    ``insert()`` legitimately creates TWO outer per-series fence tasks here —
    ``time-series-register`` and ``time-series-append`` (one per operation,
    via ``_queue_durability_write``) — that is correct fan-out, not the bug.
    Before the fix, ``_run_durability_write`` additionally routed each
    offload through ``run_blocking_ordered``, which created its OWN internal
    ``asyncio.create_task(asyncio.to_thread(...))`` UNDER each of those two
    outer tasks — four tasks and four thread-pool handoffs total for one
    insert, measured at ~1.0ms/insert versus ~19us/insert for the direct
    synchronous path. ``_run_offloaded`` waits on the bare ``asyncio.Future``
    ``loop.run_in_executor`` returns instead, which is never registered in
    ``asyncio.all_tasks()`` — so exactly the two legitimate fence tasks
    should appear, with no extra inner ones alongside them.
    """
    timeseries = _BlockingSyncTimeSeries()
    backend = EngineTimeSeriesBackend(client=_client(timeseries))

    before = asyncio.all_tasks()
    backend.insert([_point()])
    assert await asyncio.to_thread(timeseries.register_started.wait, 0.5)
    # Give the fence tasks' bodies a chance to actually run and — pre-fix —
    # create their nested run_blocking_ordered tasks too, before snapshotting.
    await asyncio.sleep(0)
    new_task_names = {task.get_name() for task in asyncio.all_tasks() - before}

    tail = next(iter(backend._pending_writes.values()))
    timeseries.allow_register.set()
    assert await asyncio.wait_for(tail, timeout=1.0) is True
    assert timeseries.calls == ["register", "append"]

    assert new_task_names == {"time-series-register", "time-series-append"}, (
        f"expected exactly the two per-operation fence tasks and no extra "
        f"inner ones, got: {new_task_names}"
    )


@pytest.mark.filterwarnings("error::RuntimeWarning")
async def test_async_client_is_nonblocking_and_orders_register_then_append() -> None:
    """A native async client has a per-series register-before-append fence."""
    timeseries = _AsyncTimeSeries()
    backend = EngineTimeSeriesBackend(client=_client(timeseries))

    backend.insert([_point()])
    assert await asyncio.wait_for(timeseries.register_started.wait(), timeout=0.5)
    tail = next(iter(backend._pending_writes.values()))

    # The serving loop still schedules work, but append cannot overtake register.
    await asyncio.sleep(0)
    assert timeseries.calls == ["register-start"]

    timeseries.allow_register.set()
    assert await asyncio.wait_for(tail, timeout=1.0) is True
    assert timeseries.calls == ["register-start", "register-finish", "append"]


@pytest.mark.filterwarnings("error::RuntimeWarning")
async def test_async_register_failure_is_sanitized_and_skips_append(caplog) -> None:
    """Async failures stay best-effort, observable, and free of payload details."""
    timeseries = _FailingAsyncTimeSeries()
    backend = EngineTimeSeriesBackend(client=_client(timeseries))
    raw_series_value = "sensitive-series-identifier"

    with caplog.at_level(
        logging.WARNING,
        logger="agent_utilities.knowledge_graph.memory.timeseries.engine_backend",
    ):
        backend.insert([_point(raw_series_value)])
        tail = next(iter(backend._pending_writes.values()))
        assert await asyncio.wait_for(tail, timeout=1.0) is False

    assert not timeseries.appended
    assert "operation=register" in caplog.text
    assert "error_type=RuntimeError" in caplog.text
    assert raw_series_value not in caplog.text
    assert "very-secret-error-detail" not in caplog.text


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_raw_async_client_without_a_running_loop_is_closed_and_reported(caplog) -> None:
    """A sync caller cannot leak or block on a raw async engine namespace."""
    timeseries = _AsyncTimeSeries()
    backend = EngineTimeSeriesBackend(client=_client(timeseries))
    raw_series_value = "sensitive-no-loop-series"

    with caplog.at_level(
        logging.WARNING,
        logger="agent_utilities.knowledge_graph.memory.timeseries.engine_backend",
    ):
        backend.insert([_point(raw_series_value)])

    assert timeseries.calls == []
    assert backend._pending_writes == {}
    assert backend._background_writes == set()
    assert "operation=register reason=no_running_loop" in caplog.text
    assert raw_series_value not in caplog.text


@pytest.mark.filterwarnings("error::RuntimeWarning")
async def test_close_cancels_all_queued_async_writes() -> None:
    """Closing a backend cannot leave a hidden register task alive behind append."""
    timeseries = _AsyncTimeSeries()
    backend = EngineTimeSeriesBackend(client=_client(timeseries))

    backend.insert([_point()])
    assert await asyncio.wait_for(timeseries.register_started.wait(), timeout=0.5)
    pending = tuple(backend._background_writes)
    assert len(pending) == 2

    backend.close()
    outcomes = await asyncio.wait_for(
        asyncio.gather(*pending, return_exceptions=True), timeout=1.0
    )

    assert all(isinstance(outcome, asyncio.CancelledError) for outcome in outcomes)
    assert timeseries.calls == ["register-start"]
    assert backend._background_writes == set()
    assert backend._pending_writes == {}


@pytest.mark.filterwarnings("error::RuntimeWarning")
async def test_close_finishes_started_sync_write_before_cancelling() -> None:
    """A cancelled sync facade write is not orphaned behind a skipped append."""
    timeseries = _BlockingSyncTimeSeries()
    backend = EngineTimeSeriesBackend(client=_client(timeseries))

    backend.insert([_point()])
    assert await asyncio.to_thread(timeseries.register_started.wait, 0.5)
    pending = {task.get_name(): task for task in backend._background_writes}
    assert set(pending) == {"time-series-register", "time-series-append"}

    backend.close()
    await asyncio.sleep(0)
    assert not pending["time-series-register"].done()

    timeseries.allow_register.set()
    outcomes = await asyncio.wait_for(
        asyncio.gather(*pending.values(), return_exceptions=True), timeout=1.0
    )

    assert all(isinstance(outcome, asyncio.CancelledError) for outcome in outcomes)
    assert timeseries.calls == ["register"]
