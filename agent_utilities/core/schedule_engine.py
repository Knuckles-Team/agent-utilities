"""CONCEPT:AU-OS.state.unified-scheduling-one-intelligent — Unified scheduling engine, one intelligent scheduler for all recurring work.

Collapses the four historical scheduling surfaces — fixed-interval maintenance
ticks, the static ``deploy/schedules.yml`` cron, the loop-cycle tick, and the
legacy OS-5.2 ``MaintenanceCron`` — into ONE durable, dynamic scheduler. The
scheduler is the sole *producer* of recurring work: when a schedule is due it
**enqueues** a ``scheduled_job`` WorkItem onto the native priority+
scheduled queue (KG-2.113), which the one worker pool drains under the
existing throttle and native lease hardening. Nothing recurring runs inline in
the scheduler thread anymore.

A schedule is a durable ``:Schedule`` graph node (survives restart and
leader-failover, and is editable at runtime — enable/disable/reprioritize/
set-interval/run-now). ``deploy/schedules.yml`` is the *seed* (desired state);
the node carries live state (last-run, next-run, failure backoff). Triggers:

  * ``cron``     — standard 5-field ``min hour dom month dow`` (no third-party dep)
  * ``interval`` — every ``interval_s`` seconds (the former maintenance ticks)
  * ``adaptive`` — interval that widens on repeated failure / can be re-tuned live

The payload describes WHAT to run (``kind``: skill / workflow / agent / maint /
loop / research_feed); :func:`run_scheduled_job` is the single dispatcher
the worker calls, so the routing lives in exactly one place.

Scheduled host-script execution is intentionally unsupported.  Schedule nodes
are graph data and therefore an untrusted control-plane boundary; treating a
``ref`` property as a local executable path would turn graph write access into
host code execution.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_SCHEDULE_LABEL = "Schedule"
# Adaptive backoff: a schedule whose job keeps failing widens its effective
# interval (interval_s * 2**failures) up to this multiple, so a broken job stops
# hammering the queue without an operator disabling it.
_ADAPTIVE_MAX_BACKOFF_MULT = 16

# A graph-carried schedule is untrusted input even when most schedules originate
# from the packaged desired-state file.  Keep dispatch envelopes small and
# restrict dynamic attribute lookup to the maintenance ticks registered by this
# release.  Without the explicit allowlist, ``kind=maint`` could invoke any
# future/private ``_tick_*`` method merely by writing a Schedule node.
_MAX_SCHEDULE_PAYLOAD_BYTES = 64 * 1024
_MAX_SCHEDULE_TEXT_BYTES = 32 * 1024
_MAX_SCHEDULE_IDENTIFIER_BYTES = 128
_SCHEDULE_KINDS = frozenset(
    {"maint", "research_feed", "feed_sweep", "skill", "workflow", "agent", "script"}
)
_MAINTENANCE_REF_ALLOWLIST = frozenset(
    {
        "anomaly_consumer",
        "compaction",
        "enrichment",
        "evolution",
        "failure_ingest",
        "file_watch",
        "fleet_autoscale_reactive",
        "fleet_autoscaler",
        "fleet_reconciler",
        "fuseki_publish",
        "goal_sla",
        "hygiene",
        "kg_analysis",
        "loop",
        "optimize_components",
        "package_install_ingest",
        "reconcile_mirrors",
        "sai_factory",
        "tenant_gc",
        "usage_log_sync",
        "usage_pricing_refresh",
        "warm_parent_reap",
    }
)


def _bounded_schedule_string(value: Any, *, allow_empty: bool = True) -> bool:
    """Return whether a control-plane string is bounded and contains no controls."""

    if not isinstance(value, str):
        return False
    encoded = value.encode("utf-8", errors="strict")
    if (not allow_empty and not encoded) or len(
        encoded
    ) > _MAX_SCHEDULE_IDENTIFIER_BYTES:
        return False
    return all(ord(char) >= 0x20 and char != "\x7f" for char in value)


def _validate_schedule_payload(payload: Any) -> str | None:
    """Validate a graph-carried dispatch envelope and return a stable reason code."""

    if not isinstance(payload, dict):
        return "invalid_payload"
    try:
        raw = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode(
            "utf-8"
        )
    except (TypeError, ValueError, UnicodeError):
        return "invalid_payload"
    if len(raw) > _MAX_SCHEDULE_PAYLOAD_BYTES:
        return "payload_too_large"

    kind = payload.get("kind", "skill")
    if (
        not _bounded_schedule_string(kind, allow_empty=False)
        or kind not in _SCHEDULE_KINDS
    ):
        return "unsupported_kind"
    for key in ("ref", "action", "name"):
        if key in payload and not _bounded_schedule_string(payload[key]):
            return "invalid_identifier"
    for key in ("task", "description"):
        value = payload.get(key)
        if value is not None:
            if (
                not isinstance(value, str)
                or len(value.encode("utf-8")) > _MAX_SCHEDULE_TEXT_BYTES
            ):
                return "invalid_text"
    kwargs = payload.get("kwargs", {})
    if not isinstance(kwargs, dict) or len(kwargs) > 64:
        return "invalid_kwargs"
    if any(
        not _bounded_schedule_string(key, allow_empty=False) or key.startswith("__")
        for key in kwargs
    ):
        return "invalid_kwargs"
    return None


def _control_backend(engine: Any) -> Any:
    """The engine's isolated control-plane backend (CONCEPT:AU-KG.backend.schedule-on-control-graph).

    The scheduler operates on :Schedule nodes and WorkItems — the CONTROL plane —
    through the configured native control authority. Missing control authority
    is a hard configuration error; scheduler state may never spill into the
    content backend.
    """
    control = getattr(engine, "control_backend", None)
    if control is None:
        raise RuntimeError("The scheduler requires the configured control authority")
    return control


def _registry_path() -> Path:
    """deploy/schedules.yml at the package repo root."""
    return Path(__file__).resolve().parents[2] / "deploy" / "schedules.yml"


def _enc(data: dict[str, Any]) -> str:
    return base64.b64encode(json.dumps(data).encode()).decode()


def _dec(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(base64.b64decode(raw).decode())
    except Exception:  # noqa: BLE001
        try:
            out = json.loads(raw)
            return out if isinstance(out, dict) else {}
        except Exception:  # noqa: BLE001
            return {}


# ── Cron matching (5-field; no third-party dep) ──────────────────────────────
def _field_match(field_expr: str, value: int) -> bool:
    for part in field_expr.split(","):
        part = part.strip()
        if part == "*":
            return True
        step = 1
        if "/" in part:
            base, step_s = part.split("/", 1)
            step = int(step_s)
            part = base or "*"
        if part == "*":
            if value % step == 0:
                return True
            continue
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            if lo <= value <= hi and (value - lo) % step == 0:
                return True
            continue
        if int(part) == value:
            return True
    return False


def cron_matches(expr: str, when: datetime) -> bool:
    """Does ``expr`` (``min hour dom month dow``) fire at ``when`` (to the minute)?"""
    fields = expr.split()
    if len(fields) != 5:
        raise ValueError(f"cron expr must have 5 fields, got {expr!r}")
    minute, hour, dom, month, dow = fields
    return (
        _field_match(minute, when.minute)
        and _field_match(hour, when.hour)
        and _field_match(dom, when.day)
        and _field_match(month, when.month)
        # cron dow: 0=Sunday..6=Saturday; datetime.weekday() is 0=Monday..6=Sunday
        and _field_match(dow, (when.weekday() + 1) % 7)
    )


# ── Schedule spec ────────────────────────────────────────────────────────────
@dataclass
class ScheduleSpec:
    """One recurring job. ``trigger`` is cron | interval | adaptive."""

    name: str
    payload: dict[str, Any]
    trigger: str = "cron"
    cron: str | None = None
    interval_s: float | None = None
    prio_bucket: int = 2
    enabled: bool = True
    last_minute: int = 0
    next_run_unix: float = 0.0
    consecutive_failures: int = 0
    backoff_until: float = 0.0
    description: str = ""
    # CONCEPT:AU-KG.ontology.capability-card-backfill-lane — the queue task type the scheduler enqueues for this
    # schedule, which selects the FUNCTIONAL LANE the tick runs in (see
    # :mod:`agent_utilities.knowledge_graph.core.task_lanes`). Defaults to
    # ``scheduled_job`` (the ``maint`` lane). A high-volume schedule whose work is
    # a throughput backfill (e.g. OWL card enrichment) overrides this so it runs in
    # its OWN lane instead of being capped at the best-effort maint floor. The
    # worker routes any of these types through the same ``run_scheduled_job``
    # dispatcher, so only the lane (and thus the worker share + model role) differs.
    task_type: str = "scheduled_job"

    def to_props(self) -> dict[str, Any]:
        return {
            "trigger": self.trigger,
            "cron": self.cron or "",
            "interval_s": float(self.interval_s or 0.0),
            "prio_bucket": int(self.prio_bucket),
            "enabled": bool(self.enabled),
            "last_minute": int(self.last_minute),
            "next_run_unix": float(self.next_run_unix),
            "consecutive_failures": int(self.consecutive_failures),
            "backoff_until": float(self.backoff_until),
            "description": self.description or "",
            "task_type": self.task_type or "scheduled_job",
            "payload": _enc(self.payload),
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> ScheduleSpec:
        return cls(
            name=row.get("id") or row.get("name") or "",
            payload=_dec(row.get("payload")),
            trigger=row.get("trigger") or "cron",
            cron=row.get("cron") or None,
            interval_s=row.get("interval_s") or None,
            prio_bucket=(
                int(row["prio_bucket"]) if row.get("prio_bucket") is not None else 2
            ),
            enabled=bool(row.get("enabled", True)),
            last_minute=int(row.get("last_minute", 0) or 0),
            next_run_unix=float(row.get("next_run_unix", 0.0) or 0.0),
            consecutive_failures=int(row.get("consecutive_failures", 0) or 0),
            backoff_until=float(row.get("backoff_until", 0.0) or 0.0),
            description=row.get("description") or "",
            task_type=row.get("task_type") or "scheduled_job",
        )


# ── Durable :Schedule node store ─────────────────────────────────────────────
def _upsert(engine: Any, spec: ScheduleSpec) -> None:
    """Upsert a ``:Schedule`` node via the engine-native O(1)-by-id ``add_node``.

    PERF (CONCEPT:AU-OS.state.unified-scheduling-one-intelligent): a write-Cypher ``MATCH (s:Schedule {id: $id}) SET …``
    forces the native engine to scan the whole graph to locate the node (no write-path
    id index) — ~5s per call on the live graph. The scheduler upserts ~27 schedules
    every boot, so that scan-per-upsert blocked the single-threaded maintenance loop
    for minutes (contending with ingestion on the engine write lock) and the
    scheduler/collapse tick effectively never ran. ``add_node`` is a direct,
    O(1)-by-id upsert that replaces the property blob — exactly what we want here,
    since ``spec.to_props()`` is the full desired state and callers
    (:func:`register_schedule`/:func:`seed_schedules`) already merge live runtime
    state into ``spec`` before upserting. So one fast write, no scan, no read.

    CONCEPT:AU-KG.backend.schedule-on-control-graph — :Schedule is CONTROL plane → write it to the isolated
    ``__control__`` graph (the control backend), never the content graph.
    """
    backend = _control_backend(engine)
    if backend is None:
        return
    backend.add_node(spec.name, node_type=_SCHEDULE_LABEL, **spec.to_props())


def _load_all(engine: Any) -> list[ScheduleSpec]:
    # CONCEPT:AU-KG.backend.schedule-on-control-graph — :Schedule lives on the control graph; read it from there.
    backend = _control_backend(engine)
    if backend is None:
        return []
    keys = (
        "trigger",
        "cron",
        "interval_s",
        "prio_bucket",
        "enabled",
        "last_minute",
        "next_run_unix",
        "consecutive_failures",
        "backoff_until",
        "description",
        "task_type",
        "payload",
    )
    proj = ", ".join(f"s.{k} as {k}" for k in keys)
    rows = backend.execute(f"MATCH (s:Schedule) RETURN s.id as id, {proj}")
    return [ScheduleSpec.from_row(r) for r in (rows or [])]


def _load_one(engine: Any, name: str) -> ScheduleSpec | None:
    # CONCEPT:AU-KG.backend.schedule-on-control-graph — :Schedule lives on the control graph; read it from there.
    backend = _control_backend(engine)
    if backend is None:
        return None
    keys = (
        "trigger",
        "cron",
        "interval_s",
        "prio_bucket",
        "enabled",
        "last_minute",
        "next_run_unix",
        "consecutive_failures",
        "backoff_until",
        "description",
        "task_type",
        "payload",
    )
    proj = ", ".join(f"s.{k} as {k}" for k in keys)
    rows = backend.execute(
        f"MATCH (s:Schedule {{id: $id}}) RETURN s.id as id, {proj}", {"id": name}
    )
    return ScheduleSpec.from_row(rows[0]) if rows else None


# ── Seeding from deploy/schedules.yml ────────────────────────────────────────
def seed_schedules(engine: Any) -> int:
    """Upsert every ``deploy/schedules.yml`` entry as a ``:Schedule`` node.

    YAML is the desired-state seed; the node is the live record. Re-seeding is
    idempotent: it refreshes trigger/payload/enabled/cron but preserves live
    runtime state (last_minute / next_run_unix / failure backoff).
    """
    path = _registry_path()
    if not path.exists():
        return 0
    doc = yaml.safe_load(path.read_text()) or {}
    seeded = 0
    for entry in doc.get("schedules") or []:
        name = entry.get("name")
        cron = entry.get("cron")
        if not name or not cron:
            continue
        payload = {
            "kind": entry.get("kind", "skill"),
            "ref": entry.get("ref", ""),
            "action": entry.get("action", ""),
            "args": entry.get("args", []),
            "task": entry.get("task", ""),
            "kwargs": entry.get("kwargs", {}),
            "timeout": entry.get("timeout", 600),
        }
        spec = ScheduleSpec(
            name=name,
            payload=payload,
            trigger="cron",
            cron=cron,
            prio_bucket=int(entry.get("prio_bucket", 2)),
            enabled=bool(entry.get("enabled", True)),
            description=entry.get("description", ""),
            # CONCEPT:AU-KG.ontology.capability-card-backfill-lane — YAML schedules may pick their own lane via task_type.
            task_type=entry.get("task_type") or "scheduled_job",
        )
        existing = _load_one(engine, name)
        if existing is not None:
            # Preserve live runtime state across re-seed.
            spec.last_minute = existing.last_minute
            spec.next_run_unix = existing.next_run_unix
            spec.consecutive_failures = existing.consecutive_failures
            spec.backoff_until = existing.backoff_until
            # Idempotent: skip the write when nothing changed (see register_schedule).
            if spec.to_props() == existing.to_props():
                seeded += 1
                continue
        _upsert(engine, spec)
        seeded += 1
    return seeded


def register_schedule(engine: Any, spec: ScheduleSpec) -> None:
    """Programmatically register a schedule (e.g. the former maintenance ticks).

    Idempotent and runtime-state-preserving like :func:`seed_schedules`.
    """
    existing = _load_one(engine, spec.name)
    if existing is not None:
        spec.last_minute = existing.last_minute
        spec.next_run_unix = existing.next_run_unix
        spec.consecutive_failures = existing.consecutive_failures
        spec.backoff_until = existing.backoff_until
        # Idempotent (CONCEPT:AU-OS.state.unified-scheduling-one-intelligent): once runtime state is merged in, if the
        # persisted node already equals the desired state there is NOTHING to
        # write. Re-upserting ~27 unchanged schedules on every boot needlessly
        # contended with ingestion on the engine write lock and blocked the
        # scheduler from ever ticking. A read is cheap and lock-free; skip the write.
        if spec.to_props() == existing.to_props():
            return
    _upsert(engine, spec)


# ── Stale-tick collapse (CONCEPT:AU-OS.state.stale-tick-collapse) ────────────────────────────────────
# The task types the scheduler enqueues for a due :Schedule. ``scheduled_job`` is
# the default (maint lane); a schedule may pick its own to land in a dedicated lane
# (CONCEPT:AU-KG.ontology.capability-card-backfill-lane, e.g. ``enrichment_backfill``). Both are interval ticks subject
# to stale-tick collapse.
_SCHEDULED_TICK_TYPES = ("scheduled_job", "enrichment_backfill")


def collapse_stale_ticks(engine: Any) -> dict[str, Any]:
    """Bulk-cancel duplicate ``scheduled_job`` ticks to ≤1 active per schedule.

    The per-schedule coalescer in :func:`run_scheduler_tick` stops NEW pileup, but a
    window where the consumer fell behind (an engine outage, an older build, or a
    transient coalescer-probe failure) can leave a backlog of duplicate interval
    ticks. A scheduled job is an interval tick, not a backlog item — running a stale
    missed tick adds no value (the same rationale as the coalescer) — and a backlog
    of them otherwise occupies the maint lane's workers re-running outdated sweeps.

    This collapses it: for every schedule with more than one ACTIVE
    (pending/scheduled/blocked) tick, all of that schedule's active ticks are
    cancelled in bulk (one UPDATE per status) — the normal due-evaluation that
    follows re-enqueues exactly one *fresh* tick when the schedule is next due, so a
    schedule never carries a stale tick and never a duplicate. ``running`` ticks are
    never touched. Idempotent and cheap in steady state: when every schedule already
    has ≤1 active tick it issues only the read probes and no writes. Best-effort —
    it must never raise into the scheduler tick.

    Operates only through ingestion WorkItems and native cancellation.
    """
    by_schedule: dict[str, list[str]] = {}
    # CONCEPT:AU-KG.ontology.capability-card-backfill-lane — collapse every scheduler-enqueued tick TYPE, not just
    # ``scheduled_job``: a schedule can now land its tick in a dedicated lane via a
    # custom task type (e.g. ``enrichment_backfill`` for OWL card backfill), and
    # those interval ticks must also never accumulate a stale backlog.
    try:
        work = engine._ingest_work_item_index()
    except Exception as exc:  # noqa: BLE001 — scheduler reports a closed failure
        logger.warning(
            "[OS-5.53] WorkItem collapse read failed (exception_type=%s)",
            type(exc).__name__,
        )
        return {"schedules_collapsed": 0, "cancelled": 0}
    for job_id, item in work.items():
        meta = item.get("metadata") or {}
        if meta.get("type") not in _SCHEDULED_TICK_TYPES:
            continue
        if item.get("status") not in {"submitted", "ready"}:
            continue
        name = meta.get("schedule")
        if name:
            by_schedule.setdefault(str(name), []).append(job_id)
    over = {name: ids for name, ids in by_schedule.items() if len(ids) > 1}
    logger.info(
        "[OS-5.53] collapse scan: active=%d schedules=%d over=%d",
        sum(len(v) for v in by_schedule.values()),
        len(by_schedule),
        len(over),
    )
    if not over:
        return {"schedules_collapsed": 0, "cancelled": 0}
    # Cancel every active tick of an over-subscribed schedule by id via the
    # engine-native O(1) compare-and-set (CONCEPT:AU-KG.compute.user-override-prompt-library) — NOT a write-Cypher
    # ``MATCH … SET`` (which forces an O(N) full-graph scan and, run per
    # (schedule, status), contended with ingestion on the engine write lock). The
    # due-evaluation that follows re-enqueues exactly one fresh tick per due
    # schedule, so a schedule keeps neither a stale tick nor a duplicate.
    cancelled = 0
    for ids in over.values():
        for tid in ids:
            try:
                if engine.cancel_task(tid).get("status") == "success":
                    cancelled += 1
            except Exception:  # noqa: BLE001 — best-effort per tick
                continue
    logger.info(
        "scheduler collapsed stale ticks: %d schedule(s), %d duplicate tick(s) cancelled",
        len(over),
        cancelled,
    )
    return {"schedules_collapsed": len(over), "cancelled": cancelled}


# ── The one scheduler tick: evaluate → enqueue ───────────────────────────────
def _is_due(spec: ScheduleSpec, now: datetime, now_unix: float) -> bool:
    if not spec.enabled:
        return False
    if spec.backoff_until and now_unix < spec.backoff_until:
        return False
    if spec.trigger == "cron":
        if not spec.cron:
            return False
        try:
            if not cron_matches(spec.cron, now):
                return False
        except ValueError as exc:
            logger.warning(
                "schedule cron validation failed (exception_type=%s)",
                type(exc).__name__,
            )
            return False
        minute_key = int(now.replace(second=0, microsecond=0).timestamp())
        return spec.last_minute < minute_key
    # interval / adaptive
    return now_unix >= spec.next_run_unix


def run_scheduler_tick(engine: Any, now: datetime | None = None) -> dict[str, Any]:
    """Evaluate every ``:Schedule`` and ENQUEUE a task for each that is due.

    The single generic scheduler tick (replaces ``run_due_schedules`` and the
    per-job maintenance-tick registrations). Idempotent within a minute via the
    node's ``last_minute``/``next_run_unix`` (advanced *before* enqueue, so a
    crash skips rather than double-fires) and the deterministic task id
    ``sched:<name>:<minute>``. Leader-gating happens in the caller.
    """
    logger.info("[OS-5.44] scheduler tick: begin")
    if not getattr(engine, "_schedules_seeded", False):
        try:
            seed_schedules(engine)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "schedule seed failed (exception_type=%s)", type(exc).__name__
            )
        engine._schedules_seeded = True

    # Curb/recover any duplicate interval-tick backlog before evaluating due
    # schedules (CONCEPT:AU-OS.state.stale-tick-collapse). Cheap no-op once every schedule has ≤1 active
    # tick; never raises into the tick.
    try:
        collapse_stale_ticks(engine)
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "collapse_stale_ticks failed (exception_type=%s)", type(exc).__name__
        )

    now = now or datetime.now()
    now_unix = time.time()
    minute_key = int(now.replace(second=0, microsecond=0).timestamp())
    fired: list[str] = []
    for spec in _load_all(engine):
        if not _is_due(spec, now, now_unix):
            continue
        # Advance run state FIRST (at-most-once on crash), then enqueue.
        if spec.trigger == "cron":
            spec.last_minute = minute_key
        else:
            interval = spec.interval_s or 60.0
            if spec.trigger == "adaptive" and spec.consecutive_failures:
                mult = min(2**spec.consecutive_failures, _ADAPTIVE_MAX_BACKOFF_MULT)
                interval *= mult
            spec.next_run_unix = now_unix + interval
        _upsert(engine, spec)
        # Coalesce: if the previous tick for this schedule hasn't been consumed
        # yet, do NOT pile another. A scheduled job is an interval tick, not a
        # backlog item — running a stale missed tick later adds no value. Without
        # this, a slow/stalled consumer (e.g. an engine outage) accumulates an
        # unbounded backlog of duplicate ticks (one per due-minute, per schedule).
        # Cheap top-level ``schedule``-property probe (not the O(N) metadata
        # dedupe scan). (CONCEPT:AU-OS.state.unified-scheduling-one-intelligent)
        work = engine._ingest_work_item_index()
        if any(
            (item.get("metadata") or {}).get("schedule") == spec.name
            and item.get("status") not in {"succeeded", "failed", "cancelled", "dead_letter"}
            for item in work.values()
        ):
            continue
        job_id = f"sched:{spec.name}:{minute_key}"
        try:
            engine.submit_task(
                target_path=f"schedule:{spec.name}",
                is_codebase=False,
                provenance={"schedule": spec.name},
                # CONCEPT:AU-KG.ontology.capability-card-backfill-lane — the task type selects the functional lane; most
                # schedules use ``scheduled_job`` (the maint lane), but a throughput
                # backfill (OWL card enrichment) overrides it to run in its own lane.
                task_type=spec.task_type or "scheduled_job",
                skip_dedupe=True,
                priority=spec.prio_bucket,
                job_id=job_id,
                extra_meta={"schedule": spec.name, "payload": spec.payload},
            )
            fired.append(spec.name)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "schedule enqueue failed (exception_type=%s)", type(exc).__name__
            )
    if fired:
        logger.info("scheduler fired schedule(s) (count=%d)", len(fired))
    logger.info("[OS-5.44] scheduler tick: end (fired=%d)", len(fired))
    return {"fired": fired, "count": len(fired)}


def _job_outcome(status: str | None, ok: bool) -> str:
    """Bounded outcome label (ok|failed|skipped) for the per-job metric (CONCEPT:AU-OS.observability.no-op-without-metrics)."""
    if status == "skipped":
        return "skipped"
    return "ok" if ok else "failed"


def _record_job_metrics(
    name: str, ok: bool, status: str | None, duration_s: float | None
) -> None:
    """Per-job outcome counter + duration histogram (Phase-0 daemon telemetry, CONCEPT:AU-OS.observability.no-op-without-metrics).

    Reuses the existing ``observability/gateway_metrics`` Prometheus registry —
    default-on where the optional ``metrics`` extra is configured, a no-op
    otherwise. Best-effort: telemetry must never break scheduling.
    """
    try:
        from agent_utilities.observability.gateway_metrics import (
            SCHEDULED_JOB_DURATION,
            SCHEDULED_JOB_RUNS,
        )

        outcome = _job_outcome(status, ok)
        SCHEDULED_JOB_RUNS.labels(schedule=name, outcome=outcome).inc()
        if duration_s is not None:
            SCHEDULED_JOB_DURATION.labels(schedule=name).observe(duration_s)
    except Exception as exc:  # noqa: BLE001 — telemetry is best-effort, never fatal
        logger.debug(
            "schedule metrics recording failed (exception_type=%s)",
            type(exc).__name__,
        )


def record_schedule_result(
    engine: Any,
    name: str,
    ok: bool,
    *,
    duration_s: float | None = None,
    status: str | None = None,
) -> None:
    """Update a schedule's failure backoff after its job ran (CONCEPT:AU-OS.state.unified-scheduling-one-intelligent).

    Called by the worker after ``run_scheduled_job`` so an ``adaptive`` schedule
    widens its interval on repeated failure and a failing job is throttled.
    Also emits the per-job outcome/duration telemetry (Phase-0 daemon
    telemetry, CONCEPT:AU-OS.observability.no-op-without-metrics) — ``duration_s``/``status`` are optional so
    existing callers are unaffected.
    """
    _record_job_metrics(name, ok, status, duration_s)
    spec = _load_one(engine, name)
    if spec is None:
        return
    if ok:
        spec.consecutive_failures = 0
        spec.backoff_until = 0.0
    else:
        spec.consecutive_failures += 1
        # Exponential backoff window (cron schedules use this too: they won't be
        # re-evaluated until backoff_until passes).
        base = spec.interval_s or 300.0
        spec.backoff_until = time.time() + min(
            base * (2**spec.consecutive_failures), base * _ADAPTIVE_MAX_BACKOFF_MULT
        )
    _upsert(engine, spec)


# ── Dispatch: the single place a scheduled job is executed ────────────────────
def _dispatch_liveness(engine: Any, payload: dict[str, Any]) -> dict[str, Any]:
    from agent_utilities.knowledge_graph.adaptation.code_health import (
        run_code_health_sweep,
    )

    return run_code_health_sweep(engine)


def _dispatch_memory_lifecycle(engine: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """CONCEPT:AU-KG.memory.drive-one-agent-native — drive one agent-native-memory lifecycle cycle.

    Selects a localized working set, summarises+consolidates a ripe episodic
    cluster, and runs decay+evict maintenance via the engine primitives. Gated
    off by ``AGENT_UTILITIES_MEMORY_LIFECYCLE`` (returns ``{"status":"disabled"}``
    otherwise), so the default-disabled schedule is inert until an operator opts in.
    """
    from agent_utilities.knowledge_graph.memory.lifecycle import run_memory_lifecycle

    return run_memory_lifecycle(engine)


# Deterministic skill actions runnable unattended on the daemon, keyed (ref, action).
_SKILL_HANDLERS: dict[
    tuple[str, str], Callable[[Any, dict[str, Any]], dict[str, Any]]
] = {
    ("code-enhancer", "liveness"): _dispatch_liveness,
    ("memory-lifecycle", "maintain"): _dispatch_memory_lifecycle,
}


def run_scheduled_job(engine: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Execute one scheduled job's payload — the worker calls this.

    The single dispatcher for every recurring job, routed by ``payload['kind']``
    so the routing lives in exactly one place (CONCEPT:AU-OS.state.unified-scheduling-one-intelligent). Times the
    dispatch and stamps the result with ``duration_s`` (Phase-0 daemon
    telemetry, CONCEPT:AU-OS.observability.no-op-without-metrics) so :func:`record_schedule_result` can emit
    per-job duration telemetry — purely additive, every other key in the
    returned dict is unchanged.
    """
    start = time.perf_counter()
    result = _dispatch_scheduled_job(engine, payload)
    result["duration_s"] = time.perf_counter() - start
    return result


def _dispatch_scheduled_job(engine: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """The routing body of :func:`run_scheduled_job` at the untrusted graph boundary."""
    invalid_reason = _validate_schedule_payload(payload)
    if invalid_reason is not None:
        return {"status": "error", "reason": invalid_reason}
    kind = payload.get("kind", "skill")
    if kind == "maint":
        # A former fixed-interval maintenance tick: an engine ``_tick_<ref>`` method.
        ref = payload.get("ref", "")
        if ref not in _MAINTENANCE_REF_ALLOWLIST:
            return {"status": "skipped", "reason": "maintenance_not_allowed"}
        tick = getattr(engine, f"_tick_{ref}", None)
        if not callable(tick):
            return {"status": "skipped", "reason": "maintenance_unavailable"}
        tick()
        return {"status": "ok"}
    if kind in ("research_feed", "feed_sweep"):
        # Unified feed sweep (CONCEPT:AU-KG.ingest.rss-feed-connector): native RSS + ScholarX arXiv through
        # the one world-model gate, plus the FreshRSS delta — all converge on the same
        # research/news routing. (Supersedes the scholarx-only run_rss_feed_screen.)
        from agent_utilities.knowledge_graph.core.source_sync import sync_source

        results = {
            "rss": sync_source(engine, "rss", mode="delta"),
            "freshrss": sync_source(engine, "freshrss", mode="delta"),
        }
        return {"status": "ok", "feeds": results}
    if kind == "skill":
        ref = payload.get("ref", "")
        action = payload.get("action", "")
        handler = _SKILL_HANDLERS.get((ref, action))
        if handler is not None:
            return handler(engine, payload)
        from agent_utilities.knowledge_graph.core.source_sync import (
            SYNC_ACTIONS,
            sync_source,
        )

        if action in SYNC_ACTIONS:
            return sync_source(engine, ref, mode=action)
        if action in ("writeback", "inventory"):
            from agent_utilities.knowledge_graph.enrichment.writeback import (
                push_inventory,
                run_writeback,
            )

            backend = getattr(engine, "backend", None)
            if action == "inventory":
                return push_inventory(
                    ref, backend=backend, engine=engine, dry_run=False
                )
            return run_writeback(ref, backend=backend, engine=engine, dry_run=False)
        return {"status": "skipped", "reason": "no_handler"}
    if kind == "script":
        # Retired security boundary: graph data must never select a host path to
        # execute.  Use a governed skill/workflow handler backed by an isolated
        # runtime instead.
        return {"status": "skipped", "reason": "host_script_execution_retired"}
    if kind in ("workflow", "agent"):
        try:
            import asyncio

            coro = engine.execute_workflow(
                workflow_id=payload.get("ref", payload.get("name", "")),
                task=payload.get("task", payload.get("description", "")),
                **(payload.get("kwargs") or {}),
            )
            asyncio.get_event_loop().run_until_complete(coro)
            return {"status": "ok"}
        except Exception as exc:  # noqa: BLE001
            from agent_utilities.security.error_surface import public_error_payload

            return public_error_payload(exc, logger=logger)
    return {"status": "skipped", "reason": "unsupported_kind"}


# ── Runtime control — enable/disable/reprioritize/retune, surfaced via MCP + REST (CONCEPT:AU-OS.state.unified-scheduling-one-intelligent)
def set_enabled(engine: Any, name: str, enabled: bool) -> dict[str, Any]:
    spec = _load_one(engine, name)
    if spec is None:
        return {"status": "error", "error": f"schedule {name} not found"}
    spec.enabled = enabled
    _upsert(engine, spec)
    return {"status": "success", "name": name, "enabled": enabled}


def set_priority(engine: Any, name: str, priority: int) -> dict[str, Any]:
    from agent_utilities.knowledge_graph.core.engine_tasks import _coerce_prio_bucket

    spec = _load_one(engine, name)
    if spec is None:
        return {"status": "error", "error": f"schedule {name} not found"}
    spec.prio_bucket = _coerce_prio_bucket(priority)
    _upsert(engine, spec)
    return {"status": "success", "name": name, "prio_bucket": spec.prio_bucket}


def set_interval(engine: Any, name: str, interval_s: float) -> dict[str, Any]:
    spec = _load_one(engine, name)
    if spec is None:
        return {"status": "error", "error": f"schedule {name} not found"}
    spec.trigger = "interval" if spec.trigger == "cron" else spec.trigger
    spec.interval_s = float(interval_s)
    spec.next_run_unix = time.time() + float(interval_s)
    _upsert(engine, spec)
    return {"status": "success", "name": name, "interval_s": interval_s}


def run_now(engine: Any, name: str) -> dict[str, Any]:
    """Force a schedule to fire on the next tick by clearing its run gate."""
    spec = _load_one(engine, name)
    if spec is None:
        return {"status": "error", "error": f"schedule {name} not found"}
    spec.last_minute = 0
    spec.next_run_unix = 0.0
    spec.backoff_until = 0.0
    _upsert(engine, spec)
    return {"status": "success", "name": name, "queued": "next_tick"}


def calendar(engine: Any) -> list[dict[str, Any]]:
    """Registry + live state for ``/cron calendar`` (real, node-backed)."""
    out = []
    for spec in _load_all(engine):
        last = (
            datetime.fromtimestamp(spec.last_minute).isoformat()
            if spec.last_minute
            else "never"
        )
        out.append(
            {
                "name": spec.name,
                "trigger": spec.trigger,
                "cron": spec.cron,
                "interval_s": spec.interval_s,
                "kind": spec.payload.get("kind", "skill"),
                "ref": spec.payload.get("ref"),
                "enabled": spec.enabled,
                "prio_bucket": spec.prio_bucket,
                "description": spec.description,
                "last_run": last,
                "consecutive_failures": spec.consecutive_failures,
            }
        )
    return out
