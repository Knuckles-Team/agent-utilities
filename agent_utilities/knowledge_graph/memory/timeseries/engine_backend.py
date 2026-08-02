"""Engine-backed time-series memory backend (CONCEPT:AU-KG.memory.time-series-lives-one).

Routes the time-series memory abstraction onto the **one epistemic-graph engine
authority** via its native ``client.timeseries.*`` namespace (eg-tsdb, CONCEPT:
KG-2.210/211) instead of a local ``timeseries.db`` SQLite file. The engine stores
each series as ``(ts_ns, [field0, field1, ...])`` points in its own durable
``series.redb``, so high-frequency points live beside the graph, not in a
straggler local DB.

LIVE CONSUMERS (CONCEPT:AU-KG.domains.ohlcv-gap-fill wired this from dead-on-arrival to in-flow):
``observability/token_tracker.py`` appends per-agent token telemetry here on every
``record()`` and reads trends via native range/window; ``domains/finance/
engine_series.py`` routes irregular-series gap-fill/asof through it. So this is no
longer a backend with no caller — it is the live time-series substrate.

Engine-only: this is the sole time-series backend (the local SQLite fallback was
removed). ``initialize()`` raises a clear error when the engine is genuinely
unreachable — the OS-5.63 resolver auto-starts the mandatory full engine artifact in prod and the
test fixture (CONCEPT:AU-KG.memory.provides-real-ephemeral-one) provides a real ephemeral one, so an unreachable
engine is a hard failure, never a silent degrade.

Mapping from the abstraction's ``TimeSeriesDataPoint`` (``symbol`` + ``datetime``
+ ``metrics: {name: float}`` + optional string ``tags``) onto the engine's
field-vector model:

* A point's ``metrics`` become the ordered field vector. Field order is fixed per
  series at registration (``field_names``) so reads decode back to the same names.
* ``tags`` (string key/values) cannot live in a numeric field vector, so each
  distinct ``(symbol, frozenset(tags))`` maps to its own ``series_id``; the tag map
  + field names are recorded on the ``:Series`` registry node. A tag-filtered query
  resolves the matching series ids and unions their points.
* ``datetime`` <-> integer nanoseconds since epoch.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import inspect
import json
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, cast

from agent_utilities.core.event_loop import run_blocking_ordered

from .base import TimeSeriesBackend, TimeSeriesDataPoint

logger = logging.getLogger(__name__)


_DurabilityCall = Callable[[], object]


def _cypher_string(value: str) -> str:
    """Render one bounded time-series symbol as a native Cypher string."""
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 256
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError("Time-series symbol is not safely representable")
    escaped = value.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def _to_ns(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1_000_000_000)


def _from_ns(ns: int) -> datetime:
    return datetime.fromtimestamp(ns / 1_000_000_000, tz=UTC)


def _series_id(symbol: str, tags: dict[str, str] | None) -> str:
    """Stable series id for a ``(symbol, tags)`` pair.

    Tags are folded into the id (sorted, hashed) so distinct tag-sets get distinct
    engine series while the same tag-set always resolves to the same id.
    """
    if not tags:
        return f"ts:{symbol}"
    canon = json.dumps(tags, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canon.encode("utf-8")).hexdigest()[:32]
    return f"ts:{symbol}:{digest}"


class EngineTimeSeriesBackend(TimeSeriesBackend):
    """Time-series backend served by the epistemic-graph engine's tsdb namespace.

    CONCEPT:AU-KG.memory.time-series-lives-one. Acquires a non-owning namespace view
    from the process :class:`GraphComputeEngine` authority at ``initialize()``;
    raises a clear error if the engine is genuinely unreachable. There is no
    SQLite fallback — the engine is the one time-series authority.
    """

    def __init__(self, client: Any = None):
        self._client = client
        # series_id -> ordered metric field names (decode key for reads)
        self._fields: dict[str, list[str]] = {}
        # A series's tail task is its durability fence: later writes await it
        # before invoking the engine.  Keep every task strongly referenced too;
        # otherwise a register task hidden behind a later append could disappear
        # before it has completed or been cancelled during ``close()``.
        self._pending_writes: dict[str, asyncio.Task[bool]] = {}
        self._background_writes: set[asyncio.Task[bool]] = set()

    def initialize(self) -> None:
        if self._client is not None:
            return
        from agent_utilities.knowledge_graph.core.graph_compute import (
            GraphComputeEngine,
        )

        try:
            self._client = GraphComputeEngine.get_or_create().client
        except Exception as exc:  # noqa: BLE001 — re-raise as a clear, typed error
            raise RuntimeError(
                "time-series memory requires the epistemic-graph engine, but no "
                "engine is reachable. The OS-5.63 resolver auto-starts the mandatory "
                "full engine artifact in prod and the test fixture (KG-2.238) provides one — "
                "there is no SQLite fallback (CONCEPT:AU-KG.memory.time-series-lives-one). "
                f"Underlying connect error: {exc}"
            ) from exc
        logger.debug("EngineTimeSeriesBackend initialized via engine tsdb")

    def _ensure_client(self) -> Any:
        if self._client is None:
            self.initialize()
        return self._client

    @staticmethod
    def _report_durability_failure(
        operation: str,
        reason: str,
        error: BaseException | None = None,
    ) -> None:
        """Make best-effort write loss observable without exposing payloads."""
        if error is None:
            logger.warning(
                "time-series durability write dropped: operation=%s reason=%s",
                operation,
                reason,
            )
            return
        logger.warning(
            "time-series durability write dropped: operation=%s reason=%s "
            "error_type=%s",
            operation,
            reason,
            type(error).__name__,
        )

    @staticmethod
    def _close_unowned_awaitable(value: Awaitable[object]) -> None:
        """Close a raw coroutine that cannot safely be scheduled from this thread."""
        if inspect.iscoroutine(value):
            value.close()

    def _write_without_running_loop(
        self,
        operation: str,
        call: _DurabilityCall,
    ) -> bool:
        """Preserve synchronous clients while rejecting raw async clients safely.

        The default GraphComputeEngine client is its synchronous facade, so this
        path retains its established synchronous success/error semantics.  A raw
        async client, however, has no owner loop in a synchronous caller.  Do
        not create a private blocking loop for best-effort telemetry: close the
        newly-created coroutine, make the drop observable, and let the caller
        continue without a warning leak.
        """
        try:
            result = call()
        except Exception as exc:  # noqa: BLE001 - preserve the sync client contract
            self._report_durability_failure(operation, "failed", exc)
            raise
        if not inspect.isawaitable(result):
            return True
        self._close_unowned_awaitable(cast(Awaitable[object], result))
        self._report_durability_failure(operation, "no_running_loop")
        return False

    async def _run_durability_write(
        self,
        previous: asyncio.Task[bool] | None,
        operation: str,
        call: _DurabilityCall,
        async_callable: bool,
    ) -> bool:
        """Run one write after its per-series predecessor without blocking the loop."""
        if previous is not None:
            try:
                previous_succeeded = await asyncio.shield(previous)
            except asyncio.CancelledError:
                current = asyncio.current_task()
                if previous.cancelled() and (
                    current is None or current.cancelling() == 0
                ):
                    self._report_durability_failure(operation, "prerequisite_cancelled")
                    return False
                self._report_durability_failure(operation, "cancelled")
                raise
            if not previous_succeeded:
                self._report_durability_failure(operation, "prerequisite_failed")
                return False

        try:
            # The default engine client exposes a synchronous facade that waits
            # on its private engine loop.  Calling it directly from GraphOS's
            # serving loop would stall that loop, so offload it while retaining
            # cancellation completion ordering.  Native async clients run on
            # the current loop directly.
            if async_callable:
                result = call()
            else:
                result = await run_blocking_ordered(call)
            if inspect.isawaitable(result):
                await cast(Awaitable[object], result)
        except asyncio.CancelledError:
            self._report_durability_failure(operation, "cancelled")
            raise
        except Exception as exc:  # noqa: BLE001 - telemetry is best-effort
            self._report_durability_failure(operation, "failed", exc)
            return False
        return True

    def _finish_durability_write(
        self,
        series_id: str,
        completed: asyncio.Task[bool],
    ) -> None:
        self._background_writes.discard(completed)
        if self._pending_writes.get(series_id) is completed:
            self._pending_writes.pop(series_id, None)

    def _queue_durability_write(
        self,
        series_id: str,
        operation: str,
        call: _DurabilityCall,
        *,
        async_callable: bool,
    ) -> bool:
        """Submit a write without letting time-series telemetry stall GraphOS.

        In a running event loop, all writes are represented by per-series tasks:
        the sync engine facade runs in a worker and native async clients are
        awaited directly.  The tail-task fence preserves register-before-append
        ordering even when callers submit multiple batches rapidly.  In a plain
        synchronous caller, preserve the existing sync-client behavior instead.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return self._write_without_running_loop(operation, call)

        previous = self._pending_writes.get(series_id)
        if previous is not None and previous.get_loop() is not loop:
            if previous.done():
                self._pending_writes.pop(series_id, None)
                previous = None
            else:
                self._report_durability_failure(operation, "different_event_loop")
                return False

        runner = self._run_durability_write(
            previous,
            operation,
            call,
            async_callable,
        )
        try:
            task = loop.create_task(runner, name=f"time-series-{operation}")
        except RuntimeError as exc:
            runner.close()
            self._report_durability_failure(operation, "scheduling_failed", exc)
            return False
        self._pending_writes[series_id] = task
        self._background_writes.add(task)
        task.add_done_callback(
            functools.partial(self._finish_durability_write, series_id)
        )
        return True

    def _register(
        self,
        series_id: str,
        symbol: str,
        fields: list[str],
        tags: dict[str, str] | None,
    ) -> bool:
        """Register the series + its field-name decode key (idempotent)."""
        client = self._ensure_client()
        field_names = list(fields)
        self._fields[series_id] = field_names
        register = client.timeseries.register_series
        return self._queue_durability_write(
            series_id,
            "register",
            functools.partial(
                register,
                series_id,
                entity_id=f"symbol:{symbol}",
                field_names=field_names,
                metadata={
                    "symbol": symbol,
                    "tags_json": json.dumps(tags) if tags else "",
                },
            ),
            async_callable=inspect.iscoroutinefunction(register),
        )

    def _field_names_for(self, series_id: str) -> list[str]:
        """Resolve the ordered field names for a series (cache, else registry node)."""
        if series_id in self._fields:
            return self._fields[series_id]
        client = self._ensure_client()
        props = client.nodes.properties(f"series:{series_id}") or {}
        fields = list(props.get("field_names") or [])
        self._fields[series_id] = fields
        return fields

    def insert(self, points: list[TimeSeriesDataPoint]) -> None:
        if not points:
            return
        client = self._ensure_client()
        # Group points by (symbol, tags) -> series, with a stable metric-field order.
        by_series: dict[str, list[tuple[int, list[float]]]] = {}
        meta: dict[str, tuple[str, list[str], dict[str, str] | None]] = {}
        for p in points:
            fields = sorted(p.metrics.keys())
            sid = _series_id(p.symbol, p.tags)
            known = self._fields.get(sid)
            if known is not None:
                fields = known + [f for f in fields if f not in known]
            meta[sid] = (p.symbol, fields, p.tags)
            vec = [float(p.metrics.get(f, 0.0)) for f in fields]
            by_series.setdefault(sid, []).append((_to_ns(p.timestamp), vec))
        registered: set[str] = set()
        for sid, (symbol, fields, tags) in meta.items():
            if self._register(sid, symbol, fields, tags):
                registered.add(sid)
        for sid, batch in by_series.items():
            if sid not in registered:
                continue
            append = client.timeseries.append
            self._queue_durability_write(
                sid,
                "append",
                functools.partial(
                    append,
                    sid,
                    batch,
                    field_names=self._fields.get(sid),
                ),
                async_callable=inspect.iscoroutinefunction(append),
            )

    def query(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        tags: dict[str, str] | None = None,
    ) -> list[TimeSeriesDataPoint]:
        client = self._ensure_client()
        from_ns, to_ns = _to_ns(start_time), _to_ns(end_time) + 1
        if tags:
            series_ids = [_series_id(symbol, tags)]
        else:
            series_ids = self._series_for_symbol(symbol)
        out: list[TimeSeriesDataPoint] = []
        for sid in series_ids:
            fields = self._field_names_for(sid)
            sid_tags = self._tags_for(sid)
            rows = client.timeseries.range(sid, from_ns, to_ns)
            for ts, vals in rows:
                metrics = {
                    name: vals[i] for i, name in enumerate(fields) if i < len(vals)
                }
                out.append(
                    TimeSeriesDataPoint(
                        symbol=symbol,
                        timestamp=_from_ns(ts),
                        metrics=metrics,
                        tags=sid_tags,
                    )
                )
        out.sort(key=lambda p: p.timestamp)
        return out

    def _series_for_symbol(self, symbol: str) -> list[str]:
        """All registered series ids for a symbol (engine registry query)."""
        client = self._ensure_client()
        try:
            rows = client.query.cypher_read(
                f"MATCH (s:Series) WHERE s.symbol = {_cypher_string(symbol)} "
                "RETURN s.series_id AS series_id"
            )
            ids = [r["series_id"] for r in rows if r.get("series_id")]
            if ids:
                return ids
        except Exception as exc:  # noqa: BLE001 - registry query best-effort
            logger.debug(
                "series-for-symbol query failed: error_type=%s", type(exc).__name__
            )
        local = [sid for sid in self._fields if sid.startswith(f"ts:{symbol}")]
        return local or [_series_id(symbol, None)]

    def _tags_for(self, series_id: str) -> dict[str, str] | None:
        client = self._ensure_client()
        props = client.nodes.properties(f"series:{series_id}") or {}
        raw = props.get("tags_json")
        if raw:
            try:
                return json.loads(raw)
            except (ValueError, json.JSONDecodeError):
                return None
        return None

    def close(self) -> None:
        """Release this backend's view; the process graph client remains open."""
        pending = tuple(self._background_writes)
        self._background_writes.clear()
        self._pending_writes.clear()
        for task in pending:
            loop = task.get_loop()
            if loop.is_closed() or task.done():
                continue
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError as exc:
                self._report_durability_failure("close", "cancellation_failed", exc)
        self._client = None
