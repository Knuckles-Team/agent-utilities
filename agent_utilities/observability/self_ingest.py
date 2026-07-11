#!/usr/bin/python
from __future__ import annotations

"""CONCEPT:AU-KG.ingest.attaching-this-root-logger — Self-ingest telemetry (dogfooding).

agent-utilities + graph-os ship their OWN telemetry (structured log records,
plus ``RunTrace`` / ``:ToolCall`` provenance events) INTO the epistemic-graph
engine's observability store, over the engine's OTLP/HTTP log-ingestion endpoint
(engine CONCEPT:AU-KG.ingest.self-ingest — ``EPISTEMIC_GRAPH_OBS_ADDR`` + ``POST /v1/logs`` OTLP,
or the ``_bulk`` endpoint). The engine becomes its own observability backend.

Design mirrors the Langfuse exporter (:mod:`langfuse_exporter`):

* **Opt-in, default-off.** Nothing happens unless ``AGENT_UTILITIES_SELF_INGEST``
  is truthy AND ``EPISTEMIC_GRAPH_OBS_ADDR`` is set. When disabled every method
  is a clean no-op, so the live path is never affected.
* **Non-blocking.** :meth:`SelfIngestSink.emit` only enqueues (``put_nowait``);
  a background daemon thread batches and ships records. The hot path never
  blocks on network I/O.
* **Durable + non-lossy once enabled.** CONCEPT:AU-OS.observability.durable-telemetry-pipeline — durable, non-lossy telemetry via bounded-retry requeue, durable spill-buffer backpressure, and per-tenant stamping.
  A failed drain is **requeued** (bounded retries, per-record attempt count)
  rather than dropped. When the in-process queue is under backpressure (full)
  or a record exhausts its retries, it **spills to a durable, crash-safe sqlite
  (WAL) buffer** (:class:`SpillBuffer`, mirroring the
  :class:`~agent_utilities.knowledge_graph.backends.outbox.GraphOutbox`
  pattern) instead of vanishing — the background worker opportunistically
  redeems that backlog once the endpoint recovers. The **only** remaining loss
  path is the durable buffer itself being unavailable or at its own bound; that
  case is counted (``dropped``) and logged at ``ERROR`` — loss is never silent.
* **Graceful degradation.** A missing/unreachable endpoint never raises: sends
  are wrapped, and repeated failures trip a cool-down so we stop hammering
  (the backoff half of "bounded retry + backoff").
* **Per-tenant.** Every record is stamped with the ambient actor's
  ``tenant.id`` / ``actor.id`` (:mod:`agent_utilities.security.brain_context`)
  at the single :meth:`SelfIngestSink.emit` choke-point, so log records,
  ``RunTrace``, and ``:ToolCall`` events are all attributable per tenant.
* **Zero new dependency.** Uses ``requests`` (already a core dependency),
  imported lazily, plus the stdlib ``sqlite3`` for the durable spill buffer.
  A transport callable can be injected for tests.

Wiring: :func:`install_self_ingest_logging` attaches a :class:`SelfIngestLogHandler`
to the root logger, so all agent-utilities + graph-os logs flow to the engine
when enabled. It is called from the process entrypoints' ``setup_logging``.
``RunTrace`` / tool-call events (already captured in the KG per KG-2.296) are
ADDED as a telemetry stream via :func:`emit_run_trace` / :func:`emit_tool_call`.

Out of scope (Codex guardrail): this remains high-volume time-series/OTLP
telemetry, not a semantic graph write — records are never materialized as
one KG node per log line. Only entities/incidents/patterns derived from this
stream get materialized elsewhere.
"""

import json
import logging
import queue
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_utilities.core.config import setting

logger = logging.getLogger(__name__)

# Transport: called with (url, payload_dict) → True on success, False otherwise.
Transport = Callable[[str, dict[str, Any]], bool]

# OTLP SeverityNumber mapping (opentelemetry-proto logs.proto). Python logging
# levels bucket onto the nearest OTLP severity.
_SEVERITY_NUMBER = {
    "TRACE": 1,
    "DEBUG": 5,
    "INFO": 9,
    "WARNING": 13,
    "WARN": 13,
    "ERROR": 17,
    "CRITICAL": 21,
    "FATAL": 21,
}


def _severity_number(level_name: str) -> int:
    """OTLP severity number for a Python level name (default INFO=9)."""
    return _SEVERITY_NUMBER.get(level_name.upper(), 9)


def _otlp_any_value(value: Any) -> dict[str, Any]:
    """Encode a scalar as an OTLP ``AnyValue`` (proto-JSON encoding)."""
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        # proto-JSON encodes 64-bit ints as strings.
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    return {"stringValue": "" if value is None else str(value)}


def _otlp_attributes(attrs: dict[str, Any]) -> list[dict[str, Any]]:
    """Encode a flat dict as an OTLP ``KeyValue`` list."""
    return [{"key": k, "value": _otlp_any_value(v)} for k, v in attrs.items()]


@dataclass
class SelfIngestConfig:
    """Resolved self-ingest configuration (read from env / config.json).

    All fields are read through :func:`~agent_utilities.core.config.setting`, so
    values injected by ``config.json`` (``_load_xdg_json_config``) or the
    environment both apply. :meth:`from_env` is the sanctioned constructor.
    """

    enabled: bool = False
    endpoint: str = ""
    mode: str = "otlp"  # "otlp" → /v1/logs ; "bulk" → /_bulk
    service_name: str = "agent-utilities"
    batch_size: int = 100
    flush_interval: float = 2.0
    queue_max: int = 10000
    min_level: int = logging.INFO
    timeout: float = 3.0
    headers: dict[str, str] = field(default_factory=dict)
    # Durability (CONCEPT:AU-OS.observability.durable-telemetry-pipeline). A failed send
    # is retried this many times (in-process requeue) before the record is diverted
    # to the durable spill buffer; ``spill_path`` defaults to the XDG data dir.
    max_retries: int = 3
    spill_path: str = ""
    spill_max_records: int = 50_000

    @classmethod
    def from_env(cls) -> SelfIngestConfig:
        """Build config from ``setting(...)``. Opt-in + endpoint gate the rest."""
        on = bool(setting("AGENT_UTILITIES_SELF_INGEST", False))
        endpoint = str(setting("EPISTEMIC_GRAPH_OBS_ADDR", "") or "")
        level_name = str(setting("AGENT_UTILITIES_SELF_INGEST_LEVEL", "INFO"))
        return cls(
            enabled=on and bool(endpoint),
            endpoint=endpoint,
            mode=str(setting("AGENT_UTILITIES_SELF_INGEST_MODE", "otlp")).lower(),
            service_name=str(
                setting("AGENT_UTILITIES_SELF_INGEST_SERVICE", "agent-utilities")
            ),
            batch_size=int(setting("AGENT_UTILITIES_SELF_INGEST_BATCH", 100)),
            flush_interval=float(setting("AGENT_UTILITIES_SELF_INGEST_INTERVAL", 2.0)),
            queue_max=int(setting("AGENT_UTILITIES_SELF_INGEST_QUEUE_MAX", 10000)),
            min_level=logging.getLevelName(level_name.upper())
            if isinstance(logging.getLevelName(level_name.upper()), int)
            else logging.INFO,
            timeout=float(setting("AGENT_UTILITIES_SELF_INGEST_TIMEOUT", 3.0)),
            max_retries=int(setting("AGENT_UTILITIES_SELF_INGEST_MAX_RETRIES", 3)),
            spill_path=str(setting("AGENT_UTILITIES_SELF_INGEST_SPILL_PATH", "") or ""),
            spill_max_records=int(
                setting("AGENT_UTILITIES_SELF_INGEST_SPILL_MAX", 50_000)
            ),
        )

    @property
    def url(self) -> str:
        """Full ingestion URL for the configured mode."""
        base = self.endpoint.rstrip("/")
        if self.mode == "bulk":
            return base if base.endswith("/_bulk") else f"{base}/_bulk"
        return base if base.endswith("/v1/logs") else f"{base}/v1/logs"


def _default_transport(timeout: float, headers: dict[str, str]) -> Transport:
    """Build a ``requests``-backed transport. Never raises; returns success bool."""

    def _send(url: str, payload: dict[str, Any]) -> bool:
        try:
            import requests  # lazy: core dep, but import off the hot path

            resp = requests.post(
                url,
                json=payload,
                timeout=timeout,
                headers={"Content-Type": "application/json", **headers},
            )
            return 200 <= resp.status_code < 300
        except Exception as exc:  # noqa: BLE001 — telemetry must never crash a run
            logger.debug("self-ingest send failed (%s): %s", url, exc)
            return False

    return _send


def _default_spill_path() -> str:
    """Default durable spill-buffer location: the XDG data dir (overridable)."""
    try:
        from agent_utilities.core.paths import data_dir

        return str(data_dir() / "observability" / "self_ingest_spill.db")
    except Exception:  # noqa: BLE001 — fall back to a relative path if paths is unavailable
        return "self_ingest_spill.db"


@dataclass
class _QueuedRecord:
    """A telemetry record plus its in-process retry attempt count."""

    record: dict[str, Any]
    attempts: int = 0


class SpillBuffer:
    """Durable, crash-safe overflow buffer for telemetry records (CONCEPT:AU-OS.observability.durable-telemetry-pipeline).

    Mirrors the design of
    :class:`~agent_utilities.knowledge_graph.backends.outbox.GraphOutbox`: one
    sqlite file in WAL mode, so a record that cannot be held in memory (queue
    under backpressure) or that exhausted its in-process retries survives a
    process crash/restart instead of vanishing. The background worker drains
    this backlog back out once the endpoint is healthy again
    (:meth:`SelfIngestSink._redeem_spill`).

    Bounded by ``max_records`` so a permanently-down endpoint cannot grow the
    buffer without limit. Once that bound is hit, :meth:`append` returns
    ``False`` — the *one* remaining true-loss case, which the caller counts
    (``SelfIngestSink.dropped``) and logs loudly. A sqlite/disk failure at
    construction time degrades the same way (``available`` is ``False``) so a
    bad path never raises into the emit hot path.
    """

    def __init__(self, path: str, max_records: int = 50_000) -> None:
        self._path = path
        self._max_records = max(1, max_records)
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        try:
            parent = Path(self._path).parent
            if str(parent):
                parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(
                self._path, check_same_thread=False, isolation_level=None, timeout=5.0
            )
            with self._lock:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS spill ("
                    "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
                    "  payload TEXT NOT NULL,"
                    "  created_at REAL NOT NULL"
                    ")"
                )
            self._conn = conn
        except Exception as exc:  # noqa: BLE001 — spill buffer must never crash the caller
            logger.error(
                "self-ingest durable spill buffer unavailable at %s: %s", path, exc
            )
            self._conn = None

    @property
    def available(self) -> bool:
        return self._conn is not None

    def append(self, record: dict[str, Any]) -> bool:
        """Durably persist one record. ``False`` only on true, final loss."""
        conn = self._conn
        if conn is None:
            return False
        try:
            with self._lock:
                row = conn.execute("SELECT COUNT(*) FROM spill").fetchone()
                if row and row[0] >= self._max_records:
                    return False
                conn.execute(
                    "INSERT INTO spill (payload, created_at) VALUES (?, ?)",
                    (json.dumps(record, default=str), time.time()),
                )
            return True
        except Exception as exc:  # noqa: BLE001 — never raise into the emit hot path
            logger.error("self-ingest spill append failed: %s", exc)
            return False

    def pop_batch(self, limit: int) -> list[dict[str, Any]]:
        """Remove and return up to ``limit`` of the oldest durable records."""
        conn = self._conn
        if conn is None or limit <= 0:
            return []
        try:
            with self._lock:
                rows = conn.execute(
                    "SELECT id, payload FROM spill ORDER BY id ASC LIMIT ?", (limit,)
                ).fetchall()
                if not rows:
                    return []
                ids = [r[0] for r in rows]
                conn.execute(
                    f"DELETE FROM spill WHERE id IN ({','.join('?' for _ in ids)})",
                    ids,
                )
            return [json.loads(r[1]) for r in rows]
        except Exception as exc:  # noqa: BLE001 — durable-backlog redemption is best-effort
            logger.error("self-ingest spill drain failed: %s", exc)
            return []

    def count(self) -> int:
        conn = self._conn
        if conn is None:
            return 0
        try:
            with self._lock:
                row = conn.execute("SELECT COUNT(*) FROM spill").fetchone()
            return int(row[0]) if row else 0
        except Exception:  # noqa: BLE001 — best-effort metric
            return 0

    def close(self) -> None:
        conn = self._conn
        if conn is not None:
            try:
                with self._lock:
                    conn.close()
            except Exception:  # noqa: BLE001 — best-effort teardown
                logger.debug("self-ingest spill buffer close failed", exc_info=True)


class SelfIngestSink:
    """Batching, non-blocking sink that ships records to the engine obs store.

    CONCEPT:AU-KG.ingest.attaching-this-root-logger. Records are plain dicts with keys ``timestamp_ns``,
    ``severity_text``, ``body``, ``attributes`` (flat dict) and ``event_type``
    (``log`` | ``run_trace`` | ``tool_call``). :meth:`format_otlp` / :meth:`format_bulk`
    render a batch into the wire shape the engine accepts.
    """

    def __init__(
        self,
        config: SelfIngestConfig | None = None,
        *,
        transport: Transport | None = None,
    ) -> None:
        self._config = config or SelfIngestConfig()
        self._transport = transport or _default_transport(
            self._config.timeout, self._config.headers
        )
        self._queue: queue.Queue[_QueuedRecord] = queue.Queue(
            maxsize=max(1, self._config.queue_max)
        )
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        # Counters expose activity for health widgets + tests.
        self.emitted = 0
        self.sent = 0
        self.dropped = 0
        self.failures = 0
        # Durability counters (CONCEPT:AU-OS.observability.durable-telemetry-pipeline):
        # ``requeued`` = in-process retry after a failed send; ``spilled`` = diverted
        # to the durable buffer (backpressure or exhausted retries). ``dropped``
        # stays reserved for the one true-loss case: the durable buffer itself is
        # unavailable/full — always logged loudly alongside the counter bump.
        self.requeued = 0
        self.spilled = 0
        # Backoff: after N consecutive failures, cool down before retrying.
        self._consecutive_failures = 0
        self._cooldown_until = 0.0
        # Durable spill buffer is constructed lazily (only once actually needed)
        # so a disabled/no-op sink never touches the filesystem.
        self._spill: SpillBuffer | None = None

    # ── config / lifecycle ────────────────────────────────────────────
    @property
    def config(self) -> SelfIngestConfig:
        return self._config

    @property
    def enabled(self) -> bool:
        return self._config.enabled and bool(self._config.endpoint)

    @property
    def _spill_buffer(self) -> SpillBuffer:
        """Lazily construct the durable spill buffer (first backpressure/failure)."""
        if self._spill is None:
            self._spill = SpillBuffer(
                self._config.spill_path or _default_spill_path(),
                max_records=self._config.spill_max_records,
            )
        return self._spill

    def spill_depth(self) -> int:
        """Records currently sitting in the durable buffer awaiting redelivery."""
        return self._spill.count() if self._spill is not None else 0

    def start(self) -> None:
        """Start the background flush worker (idempotent, no-op when disabled)."""
        if not self.enabled:
            return
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._stop.clear()
            self._worker = threading.Thread(
                target=self._run,
                name="au-self-ingest",
                daemon=True,
            )
            self._worker.start()

    def stop(self, *, flush: bool = True) -> None:
        """Signal the worker to stop and optionally flush what remains."""
        self._stop.set()
        worker = self._worker
        if worker is not None:
            worker.join(timeout=self._config.flush_interval + 1.0)
        if flush:
            self.flush()
        if self._spill is not None:
            self._spill.close()

    # ── identity stamping (per-tenant, CONCEPT:AU-OS.identity.authenticated-identity-enforcement) ──
    @staticmethod
    def _stamp_identity(record: dict[str, Any]) -> dict[str, Any]:
        """Stamp ``tenant.id`` / ``actor.id`` from the ambient actor onto a record.

        Applied at the single :meth:`emit` choke-point every record flows
        through (log handler, :func:`emit_run_trace`, :func:`emit_tool_call`),
        so every telemetry record — not just logs — is attributable per tenant.
        Best-effort: identity is optional context, never a reason to drop or
        block telemetry.
        """
        attrs = dict(record.get("attributes") or {})
        try:
            from agent_utilities.security.brain_context import current_actor

            actor = current_actor()
            attrs.setdefault("tenant.id", actor.tenant_id or "")
            attrs.setdefault("actor.id", actor.actor_id or "")
        except Exception:  # noqa: BLE001 — identity is best-effort context
            pass
        stamped = dict(record)
        stamped["attributes"] = attrs
        return stamped

    # ── emit (hot path — never blocks) ────────────────────────────────
    def emit(self, record: dict[str, Any]) -> None:
        """Enqueue one telemetry record. Non-blocking.

        Durability (CONCEPT:AU-OS.observability.durable-telemetry-pipeline): a full
        in-process queue is backpressure, not a reason to lose the record — it
        spills to the durable buffer instead. A record is only ever counted
        ``dropped`` (and logged at ERROR) when the durable buffer itself is
        unavailable or at its own bound.
        """
        if not self.enabled:
            return
        self.emitted += 1
        record = self._stamp_identity(record)
        try:
            self._queue.put_nowait(_QueuedRecord(record=record))
        except queue.Full:
            if self._spill_buffer.append(record):
                self.spilled += 1
                logger.warning(
                    "self-ingest queue at capacity (%d) — spilled record to "
                    "durable buffer instead of dropping",
                    self._config.queue_max,
                )
            else:
                self.dropped += 1
                logger.error(
                    "SELF-INGEST TELEMETRY DROPPED: queue full AND durable spill "
                    "buffer unavailable/at capacity — event_type=%s body=%r",
                    record.get("event_type"),
                    record.get("body"),
                )

    def emit_log(
        self,
        *,
        body: str,
        level: str = "INFO",
        timestamp_ns: int | None = None,
        attributes: dict[str, Any] | None = None,
        event_type: str = "log",
    ) -> None:
        """Convenience emit for a structured log-shaped record."""
        self.emit(
            {
                "timestamp_ns": timestamp_ns
                if timestamp_ns is not None
                else time.time_ns(),
                "severity_text": level.upper(),
                "body": body,
                "attributes": dict(attributes or {}),
                "event_type": event_type,
            }
        )

    # ── formatting (the wire shapes the engine expects) ───────────────
    def format_otlp(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        """Render a batch as an OTLP/HTTP JSON ``ExportLogsServiceRequest``."""
        log_records = []
        for rec in records:
            level = str(rec.get("severity_text", "INFO"))
            attrs = dict(rec.get("attributes") or {})
            attrs.setdefault("event.type", rec.get("event_type", "log"))
            log_records.append(
                {
                    "timeUnixNano": str(rec.get("timestamp_ns") or time.time_ns()),
                    "severityNumber": _severity_number(level),
                    "severityText": level,
                    "body": {"stringValue": str(rec.get("body", ""))},
                    "attributes": _otlp_attributes(attrs),
                }
            )
        return {
            "resourceLogs": [
                {
                    "resource": {
                        "attributes": _otlp_attributes(
                            {"service.name": self._config.service_name}
                        )
                    },
                    "scopeLogs": [
                        {
                            "scope": {"name": "agent_utilities"},
                            "logRecords": log_records,
                        }
                    ],
                }
            ]
        }

    def format_bulk(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        """Render a batch as the engine ``_bulk`` shape (flat record list)."""
        out = []
        for rec in records:
            item = {
                "timestamp_ns": rec.get("timestamp_ns") or time.time_ns(),
                "severity": str(rec.get("severity_text", "INFO")),
                "body": str(rec.get("body", "")),
                "event_type": rec.get("event_type", "log"),
                "service": self._config.service_name,
            }
            item.update(dict(rec.get("attributes") or {}))
            out.append(item)
        return {"records": out}

    def _format(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        if self._config.mode == "bulk":
            return self.format_bulk(records)
        return self.format_otlp(records)

    # ── flush / worker ────────────────────────────────────────────────
    def _drain(self, limit: int) -> list[_QueuedRecord]:
        batch: list[_QueuedRecord] = []
        while len(batch) < limit:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return batch

    def _handle_failed(self, items: list[_QueuedRecord]) -> None:
        """Requeue a failed batch (bounded retries), spilling on exhaustion.

        CONCEPT:AU-OS.observability.durable-telemetry-pipeline — the core durability fix: a
        failed drain is never simply dropped. Each record gets up to
        ``max_retries`` in-process resends; once exhausted (or the in-process
        queue itself is full) it moves to the durable spill buffer, which the
        worker redeems once the endpoint recovers (:meth:`_redeem_spill`).
        """
        for item in items:
            item.attempts += 1
            if item.attempts <= self._config.max_retries:
                try:
                    self._queue.put_nowait(item)
                    self.requeued += 1
                    continue
                except queue.Full:
                    pass  # queue saturated — fall through to the durable spill
            if self._spill_buffer.append(item.record):
                self.spilled += 1
            else:
                self.dropped += 1
                logger.error(
                    "SELF-INGEST TELEMETRY DROPPED: exhausted %d retries AND "
                    "durable spill buffer unavailable/at capacity — "
                    "event_type=%s body=%r",
                    item.attempts,
                    item.record.get("event_type"),
                    item.record.get("body"),
                )

    def _send_batch(self, items: list[_QueuedRecord]) -> bool:
        """Format + ship a batch, tracking counters and backoff. Never raises.

        On failure the batch is handed to :meth:`_handle_failed` (requeue /
        spill) rather than discarded — this is the non-lossy contract.
        """
        if not items:
            return True
        payload = self._format([item.record for item in items])
        ok = False
        try:
            ok = self._transport(self._config.url, payload)
        except Exception as exc:  # noqa: BLE001 — defensive: transport may misbehave
            logger.debug("self-ingest transport raised: %s", exc)
            ok = False
        if ok:
            self.sent += len(items)
            self._consecutive_failures = 0
        else:
            self.failures += 1
            self._consecutive_failures += 1
            # Exponential-ish cool-down, capped, to stop hammering a dead endpoint
            # (the "backoff" half of bounded retry + backoff).
            if self._consecutive_failures >= 3:
                backoff = min(
                    60.0, self._config.flush_interval * 2**self._consecutive_failures
                )
                self._cooldown_until = time.monotonic() + backoff
            self._handle_failed(items)
        return ok

    def flush(self) -> int:
        """Synchronously drain and ship everything queued. Returns records sent.

        Used at shutdown and by tests (deterministic, no worker required). A
        failed batch is requeued (up to ``max_retries``) or spilled to the
        durable buffer by :meth:`_send_batch`/:meth:`_handle_failed`, so this
        loop always terminates: every record either sends, exhausts its
        retries into the spill buffer, or (final bound) is loudly counted as
        dropped — it never disappears silently.
        """
        sent = 0
        while True:
            batch = self._drain(self._config.batch_size)
            if not batch:
                break
            if self._send_batch(batch):
                sent += len(batch)
        return sent

    def _redeem_spill(self, limit: int) -> None:
        """Opportunistically resend durable backlog once the endpoint is healthy.

        Gives the durable buffer priority over fresh traffic so it drains back
        down after an outage instead of growing forever. Records pulled here
        get a fresh attempt count — :meth:`_handle_failed` will re-spill them
        on failure, so this can never lose what it just popped.
        """
        if self._spill is None:
            return
        records = self._spill.pop_batch(limit)
        if not records:
            return
        self._send_batch([_QueuedRecord(record=r) for r in records])

    def _run(self) -> None:
        """Background loop: batch on interval / size, honoring the cool-down."""
        while not self._stop.is_set():
            self._stop.wait(self._config.flush_interval)
            if time.monotonic() < self._cooldown_until:
                continue
            self._redeem_spill(self._config.batch_size)
            batch = self._drain(self._config.batch_size)
            if batch:
                self._send_batch(batch)


# ── logging handler ───────────────────────────────────────────────────────────
class SelfIngestLogHandler(logging.Handler):
    """A ``logging.Handler`` that forwards records to a :class:`SelfIngestSink`.

    CONCEPT:AU-KG.ingest.attaching-this-root-logger. Attaching this to the root logger routes all
    agent-utilities + graph-os log output into the engine obs store when
    self-ingest is enabled.
    """

    def __init__(self, sink: SelfIngestSink, level: int = logging.INFO) -> None:
        super().__init__(level=level)
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        # A handler must never raise into the emitting code path.
        try:
            attrs: dict[str, Any] = {
                "logger.name": record.name,
                "code.filepath": record.pathname,
                "code.lineno": record.lineno,
                "thread.name": record.threadName or "",
            }
            try:
                from agent_utilities.observability.correlation import (
                    get_correlation_id,
                )

                cid = get_correlation_id()
                if cid:
                    attrs["correlation_id"] = cid
            except Exception:  # noqa: BLE001 — correlation must never break logging
                pass
            self._sink.emit_log(
                body=record.getMessage(),
                level=record.levelname,
                timestamp_ns=int(record.created * 1_000_000_000),
                attributes=attrs,
                event_type="log",
            )
        except Exception:  # noqa: BLE001 — defensive, per logging.Handler contract
            self.handleError(record)


# ── process-wide singleton (mirrors langfuse_exporter) ─────────────────────────
_SINK: SelfIngestSink | None = None
_SINK_BUILT = False
_HANDLER: SelfIngestLogHandler | None = None


def get_self_ingest_sink() -> SelfIngestSink | None:
    """Return the process-wide sink, or ``None`` when self-ingest is disabled.

    Default-off: when ``AGENT_UTILITIES_SELF_INGEST`` is truthy AND
    ``EPISTEMIC_GRAPH_OBS_ADDR`` is set, a live sink is returned. Otherwise
    ``None`` so callers skip emission with no overhead.
    """
    global _SINK, _SINK_BUILT
    if _SINK_BUILT:
        return _SINK
    _SINK_BUILT = True
    sink = SelfIngestSink(SelfIngestConfig.from_env())
    _SINK = sink if sink.enabled else None
    return _SINK


def set_self_ingest_sink(sink: SelfIngestSink | None) -> None:
    """Install a specific sink (used by tests to inject a fake transport)."""
    global _SINK, _SINK_BUILT
    _SINK = sink
    _SINK_BUILT = True


def reset_self_ingest_sink() -> None:
    """Reset the cached singleton so the next call re-probes the environment."""
    global _SINK, _SINK_BUILT, _HANDLER
    _SINK = None
    _SINK_BUILT = False
    _HANDLER = None


def install_self_ingest_logging(logger_obj: logging.Logger | None = None) -> bool:
    """Attach the self-ingest handler to ``logger_obj`` (root by default).

    Idempotent and safe to call unconditionally from process entrypoints: when
    self-ingest is disabled it is a clean no-op and returns ``False``. Returns
    ``True`` when a handler was installed and the background worker started.
    """
    global _HANDLER
    sink = get_self_ingest_sink()
    if sink is None:
        return False
    target = logger_obj or logging.getLogger()
    # Idempotent: never attach two self-ingest handlers to the same logger.
    for h in target.handlers:
        if isinstance(h, SelfIngestLogHandler):
            return True
    handler = SelfIngestLogHandler(sink, level=sink.config.min_level)
    target.addHandler(handler)
    sink.start()
    _HANDLER = handler
    logger.info(
        "self-ingest telemetry active → %s (mode=%s)",
        sink.config.url,
        sink.config.mode,
    )
    return True


# ── RunTrace / tool-call provenance stream (KG-2.296 already persists these; this
#    ADDS a telemetry stream, it does not replace) ──────────────────────────────
def emit_run_trace(
    *,
    run_id: str,
    status: str = "",
    agent_id: str = "",
    duration_ms: float = 0.0,
    query: str = "",
    attributes: dict[str, Any] | None = None,
) -> bool:
    """Emit a ``RunTrace`` telemetry record. No-op + ``False`` when disabled."""
    sink = get_self_ingest_sink()
    if sink is None:
        return False
    attrs: dict[str, Any] = {
        "run.id": run_id,
        "run.status": status,
        "agent.id": agent_id,
        "duration_ms": duration_ms,
    }
    if attributes:
        attrs.update(attributes)
    sink.emit_log(
        body=f"run_trace:{run_id} status={status}",
        level="ERROR" if status in {"error", "failed"} else "INFO",
        attributes=attrs,
        event_type="run_trace",
    )
    return True


def emit_tool_call(
    *,
    run_id: str,
    tool_name: str,
    status: str = "",
    duration_ms: float = 0.0,
    attributes: dict[str, Any] | None = None,
) -> bool:
    """Emit a ``:ToolCall`` telemetry record. No-op + ``False`` when disabled."""
    sink = get_self_ingest_sink()
    if sink is None:
        return False
    attrs: dict[str, Any] = {
        "run.id": run_id,
        "tool.name": tool_name,
        "tool.status": status,
        "duration_ms": duration_ms,
    }
    if attributes:
        attrs.update(attributes)
    sink.emit_log(
        body=f"tool_call:{tool_name} status={status}",
        level="ERROR" if status in {"error", "failed"} else "INFO",
        attributes=attrs,
        event_type="tool_call",
    )
    return True
