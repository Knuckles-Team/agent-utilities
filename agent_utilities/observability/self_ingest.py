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
* **Write-ahead, never network-blocking.** :meth:`SelfIngestSink.emit`
  synchronously appends a sanitized record to the local WAL before it becomes
  eligible to send; a background daemon thread batches network delivery. The
  hot path never blocks on network I/O, and an emitted record is not removed
  until the destination acknowledges its batch.
* **Durable + non-lossy once enabled.** CONCEPT:AU-OS.observability.durable-telemetry-pipeline — durable, non-lossy telemetry via bounded-retry requeue, durable spill-buffer backpressure, and per-tenant stamping.
  Every record first enters a durable, crash-safe SQLite WAL
  (:class:`SpillBuffer`, mirroring the
  :class:`~agent_utilities.knowledge_graph.backends.outbox.GraphOutbox`
  pattern). In-process queueing is only a delivery accelerator; backpressure or
  exhausted retries leave the WAL row pending for replay. The **only** loss
  path is the durable buffer itself being unavailable or at its own bound; that
  case is counted (``dropped``) and logged at ``ERROR`` — loss is never silent.
* **Graceful degradation.** A missing/unreachable endpoint never raises: sends
  are wrapped, and repeated failures trip a cool-down so we stop hammering
  (the backoff half of "bounded retry + backoff").
* **Per-tenant without persisted identity.** Every record is stamped with
  opaque ``tenant.ref`` / ``actor.ref`` values derived from the ambient actor;
  raw identity, endpoints, machine names, and filesystem paths are sanitized
  at the single :meth:`SelfIngestSink.emit` persistence choke-point.
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
from agent_utilities.security.persistence_privacy import (
    PersistencePrivacyGuard,
    persistence_reference,
)

logger = logging.getLogger(__name__)

# Internal delivery failures must never re-enter the root self-ingest handler.
# This deliberately isolated stderr channel emits only bounded, non-sensitive
# event codes/counts (never endpoints, paths, payloads, or exception reprs).
_emergency_logger = logging.getLogger(f"{__name__}.emergency")
_emergency_logger.propagate = False
if not _emergency_logger.handlers:
    _emergency_handler = logging.StreamHandler()
    _emergency_handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    _emergency_logger.addHandler(_emergency_handler)

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
    """Build an AgentConfig/TLS-profile-backed transport."""

    client: Any | None = None
    try:
        from agent_utilities.core.http_client import create_http_client
        from agent_utilities.core.transport_security import (
            resolve_configured_tls_profile,
        )

        trust = resolve_configured_tls_profile("observability-self-ingest")
        try:
            client = create_http_client(
                timeout=timeout,
                headers={"Content-Type": "application/json", **headers},
                **trust.httpx_kwargs(),
            )
        finally:
            trust.cleanup()
    except Exception:  # noqa: BLE001 — telemetry setup must not crash a run
        _emergency_logger.debug("self-ingest transport_initialization_failure")

    def _send(url: str, payload: dict[str, Any]) -> bool:
        if client is None:
            return False
        try:
            resp = client.post(url, json=payload)
            return 200 <= resp.status_code < 300
        except Exception:  # noqa: BLE001 — telemetry must never crash a run
            _emergency_logger.debug("self-ingest transport_failure")
            return False

    return _send


def _default_spill_path() -> str:
    """Default durable spill-buffer location: the XDG data dir (overridable)."""
    from agent_utilities.core.paths import data_dir

    return str(data_dir() / "observability" / "self_ingest_spill.db")


@dataclass
class _QueuedRecord:
    """A write-ahead telemetry record plus its retry attempt count."""

    record: dict[str, Any]
    durable_id: int
    attempts: int = 0


@dataclass(frozen=True)
class SpillRecord:
    """One unacknowledged durable telemetry row."""

    durable_id: int
    record: dict[str, Any]


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
    ``None`` — the *one* remaining true-loss case, which the caller counts
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
                # FULL is intentional: enqueue success is the write-ahead
                # durability boundary for telemetry that may otherwise exist
                # only in this process.
                conn.execute("PRAGMA synchronous=FULL")
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS spill ("
                    "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
                    "  payload TEXT NOT NULL,"
                    "  created_at REAL NOT NULL"
                    ")"
                )
            self._conn = conn
        except Exception:  # noqa: BLE001 — spill buffer must never crash the caller
            _emergency_logger.error("self-ingest spill_unavailable")
            self._conn = None

    @property
    def available(self) -> bool:
        return self._conn is not None

    def append(self, record: dict[str, Any]) -> int | None:
        """Durably persist one record and return its acknowledgment id.

        ``None`` is returned only on true, final loss. The row stays present
        until :meth:`ack` is called after a successful network send.
        """
        conn = self._conn
        if conn is None:
            return None
        try:
            with self._lock:
                row = conn.execute("SELECT COUNT(*) FROM spill").fetchone()
                if row and row[0] >= self._max_records:
                    return None
                cursor = conn.execute(
                    "INSERT INTO spill (payload, created_at) VALUES (?, ?)",
                    (json.dumps(record, default=str), time.time()),
                )
                durable_id = int(cursor.lastrowid)
            return durable_id
        except Exception:  # noqa: BLE001 — never raise into the emit hot path
            _emergency_logger.error("self-ingest spill_append_failed")
            return None

    def peek_batch(self, limit: int) -> list[SpillRecord]:
        """Read oldest unacknowledged records without deleting them."""
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
            return [
                SpillRecord(durable_id=int(row[0]), record=json.loads(row[1]))
                for row in rows
            ]
        except Exception:  # noqa: BLE001 — durable-backlog redemption is best-effort
            _emergency_logger.error("self-ingest spill_read_failed")
            return []

    def ack(self, durable_ids: list[int] | tuple[int, ...]) -> int:
        """Delete rows only after the destination acknowledged their batch."""
        conn = self._conn
        ids = [int(value) for value in durable_ids if int(value) > 0]
        if conn is None or not ids:
            return 0
        try:
            with self._lock:
                cursor = conn.execute(
                    f"DELETE FROM spill WHERE id IN ({','.join('?' for _ in ids)})",
                    ids,
                )
            return max(0, int(cursor.rowcount))
        except Exception:  # noqa: BLE001 — failed ack safely causes replay
            _emergency_logger.error("self-ingest spill_ack_failed")
            return 0

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
                _emergency_logger.debug("self-ingest spill_close_failed")


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
        self.persisted = 0
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

    # ── privacy/identity persistence boundary ─────────────────────────
    @staticmethod
    def _stamp_identity(record: dict[str, Any]) -> dict[str, Any]:
        """Stamp only opaque tenant/actor references onto a record.

        Raw ambient identity never crosses the persistence boundary. Existing
        caller-supplied ``tenant.id``/``actor.id`` values are removed rather
        than trusted.
        """
        attrs = dict(record.get("attributes") or {})
        attrs.pop("tenant.id", None)
        attrs.pop("actor.id", None)
        attrs.pop("tenant_id", None)
        attrs.pop("actor_id", None)
        try:
            from agent_utilities.security.brain_context import current_actor

            actor = current_actor()
            if actor.tenant_id:
                attrs["tenant.ref"] = persistence_reference(
                    "tenant", actor.tenant_id, namespace="self-ingest"
                )
            if actor.actor_id:
                attrs["actor.ref"] = persistence_reference(
                    "actor", actor.actor_id, namespace="self-ingest"
                )
        except Exception:  # noqa: BLE001 — identity is best-effort context
            pass
        stamped = dict(record)
        stamped["attributes"] = attrs
        return stamped

    @staticmethod
    def _sanitize_record(record: dict[str, Any]) -> dict[str, Any]:
        """Remove PII, secrets, endpoints, names, and local paths before WAL.

        Free-form log bodies are never retained. Pattern redaction cannot prove
        that an arbitrary person's name is absent, so the WAL stores only a
        keyed opaque body reference and its length alongside a generic event
        marker. Production requires the persistence HMAC key reference.
        """

        raw_body = str(record.get("body") or "")
        clean, report = PersistencePrivacyGuard().sanitize(record)
        assert isinstance(clean, dict)
        attrs = dict(clean.get("attributes") or {})
        event_type = str(clean.get("event_type") or "log")[:64]
        if raw_body:
            attrs["message.ref"] = persistence_reference(
                "telemetry_message", raw_body, namespace=event_type
            )
            attrs["message.length"] = len(raw_body)
        attrs["privacy.schema"] = "persistence-privacy-v1"
        attrs["privacy.redactions"] = report.redactions
        attrs["content.retention"] = "metadata"
        clean["body"] = f"event:{event_type}"
        clean["attributes"] = attrs
        return clean

    # ── emit (hot path — never blocks) ────────────────────────────────
    def emit(self, record: dict[str, Any]) -> None:
        """Enqueue one telemetry record. Non-blocking.

        Durability (CONCEPT:AU-OS.observability.durable-telemetry-pipeline): the
        sanitized record is written to the WAL before this method returns. The
        queue is only a delivery accelerator; if it is full, the durable row is
        left pending for replay. A record is counted ``dropped`` only when the
        WAL itself is unavailable or at its bound.
        """
        if not self.enabled:
            return
        self.emitted += 1
        record = self._sanitize_record(self._stamp_identity(record))
        durable_id = self._spill_buffer.append(record)
        if durable_id is None:
            self.dropped += 1
            _emergency_logger.error(
                "self-ingest write_ahead_rejected event_type=%s",
                str(record.get("event_type") or "unknown")[:32],
            )
            return
        self.persisted += 1
        try:
            self._queue.put_nowait(_QueuedRecord(record=record, durable_id=durable_id))
        except queue.Full:
            self.spilled += 1
            _emergency_logger.warning(
                "self-ingest queue_saturated durable_replay_pending"
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
        """Requeue a failed batch while retaining its write-ahead rows.

        CONCEPT:AU-OS.observability.durable-telemetry-pipeline — the core durability fix: a
        failed drain is never simply dropped. Each record gets bounded fast
        retries; once exhausted (or the queue is full), its existing WAL row is
        left for :meth:`_redeem_spill`.
        """
        for item in items:
            item.attempts += 1
            if item.attempts <= self._config.max_retries:
                try:
                    self._queue.put_nowait(item)
                    self.requeued += 1
                    continue
                except queue.Full:
                    pass  # row remains in the WAL for later redemption
            # The row was persisted before first send. Exhausting the fast
            # in-process retry leaves it pending for durable redemption.
            self.spilled += 1

    def _send_batch(
        self,
        items: list[_QueuedRecord],
        *,
        requeue_on_failure: bool = True,
    ) -> bool:
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
        except Exception:  # noqa: BLE001 — defensive: transport may misbehave
            _emergency_logger.debug("self-ingest transport_exception")
            ok = False
        if ok:
            self.sent += len(items)
            self._consecutive_failures = 0
            durable_ids = [item.durable_id for item in items]
            # Ack-after-send: a failed local delete is safe because it leaves
            # rows available for at-least-once replay.
            acknowledged = self._spill_buffer.ack(durable_ids)
            if acknowledged != len(durable_ids):
                _emergency_logger.error(
                    "self-ingest local_ack_incomplete count=%d",
                    len(durable_ids) - acknowledged,
                )
                return False
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
            if requeue_on_failure:
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
        # Rows that could not enter the accelerator queue, or that exhausted
        # its retry budget, remain in the WAL. Drain them only after queued
        # records have settled so no row can be sent down both paths at once.
        while self.spill_depth() > 0:
            redeemed = self._redeem_spill(self._config.batch_size)
            if redeemed <= 0:
                break
            sent += redeemed
        return sent

    def _redeem_spill(self, limit: int) -> int:
        """Opportunistically resend durable backlog once the endpoint is healthy.

        Rows are read without deletion. A successful network batch is followed
        by a local acknowledgment/delete; any failure leaves the original rows
        in place for at-least-once replay.
        """
        if self._spill is None:
            return 0
        records = self._spill.peek_batch(limit)
        if not records:
            return 0
        items = [
            _QueuedRecord(record=row.record, durable_id=row.durable_id)
            for row in records
        ]
        return len(items) if self._send_batch(items, requeue_on_failure=False) else 0

    def _run(self) -> None:
        """Background loop: batch on interval / size, honoring the cool-down."""
        while not self._stop.is_set():
            self._stop.wait(self._config.flush_interval)
            if time.monotonic() < self._cooldown_until:
                continue
            batch = self._drain(self._config.batch_size)
            if batch:
                self._send_batch(batch)
            else:
                self._redeem_spill(self._config.batch_size)


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
                # Never export a machine-local absolute source path. The
                # import namespace is stable and repo-neutral.
                "code.namespace": record.name,
                "code.lineno": record.lineno,
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
    logger.info("self-ingest telemetry active (mode=%s)", sink.config.mode)
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
    from agent_utilities.observability.trace_ontology import trace_id
    from agent_utilities.security.persistence_privacy import (
        PersistencePrivacyGuard,
        persistence_reference,
    )

    attrs: dict[str, Any] = {
        "run.ref": trace_id(run_id),
        "run.status": status,
        "agent.ref": persistence_reference(
            "agent", agent_id, namespace="external-observability"
        ),
        "duration_ms": duration_ms,
    }
    if attributes:
        attrs.update(attributes)
    attrs, _ = PersistencePrivacyGuard().sanitize(attrs)
    sink.emit_log(
        body=f"run_trace status={status}",
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
    from agent_utilities.observability.trace_ontology import trace_id
    from agent_utilities.security.persistence_privacy import (
        PersistencePrivacyGuard,
        persistence_reference,
    )

    attrs: dict[str, Any] = {
        "run.ref": trace_id(run_id),
        "tool.ref": persistence_reference(
            "tool", tool_name, namespace="external-observability"
        ),
        "tool.status": status,
        "duration_ms": duration_ms,
    }
    if attributes:
        attrs.update(attributes)
    attrs, _ = PersistencePrivacyGuard().sanitize(attrs)
    sink.emit_log(
        body=f"tool_call status={status}",
        level="ERROR" if status in {"error", "failed"} else "INFO",
        attributes=attrs,
        event_type="tool_call",
    )
    return True
