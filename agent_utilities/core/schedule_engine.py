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
import contextlib
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
# Hard backstop on a scheduled ``kind in (workflow, agent)`` dispatch
# (CONCEPT:AU-ORCH.scheduling.hard-io-deadline) -- generous enough for a
# multi-step workflow run, bounded so a stuck dispatch can never hang a
# scheduler tick indefinitely.
_WORKFLOW_DISPATCH_TIMEOUT_S = 1800.0
_SCHEDULE_KINDS = frozenset(
    {"maint", "research_feed", "feed_sweep", "skill", "workflow", "agent", "script"}
)
_MAINTENANCE_REF_ALLOWLIST = frozenset(
    {
        "anomaly_consumer",
        "compaction",
        "enrich_concepts",
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
        "placement_mining_reactive",
        "reasoning",
        "reconcile_mirrors",
        "runtime_reliability",
        "sai_factory",
        "tenant_gc",
        "trace_retention",
        "tms_revalidation",
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


def _validate_payload_shape(payload: Any) -> str | None:
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
    return None


def _validate_payload_kind(payload: dict[str, Any]) -> str | None:
    kind = payload.get("kind", "skill")
    if (
        not _bounded_schedule_string(kind, allow_empty=False)
        or kind not in _SCHEDULE_KINDS
    ):
        return "unsupported_kind"
    return None


def _validate_payload_identifiers(payload: dict[str, Any]) -> str | None:
    for key in ("ref", "action", "name"):
        if key in payload and not _bounded_schedule_string(payload[key]):
            return "invalid_identifier"
    return None


def _validate_payload_text_fields(payload: dict[str, Any]) -> str | None:
    for key in ("task", "description"):
        value = payload.get(key)
        if value is None:
            continue
        if (
            not isinstance(value, str)
            or len(value.encode("utf-8")) > _MAX_SCHEDULE_TEXT_BYTES
        ):
            return "invalid_text"
    return None


def _validate_payload_kwargs(payload: dict[str, Any]) -> str | None:
    kwargs = payload.get("kwargs", {})
    if not isinstance(kwargs, dict) or len(kwargs) > 64:
        return "invalid_kwargs"
    if any(
        not _bounded_schedule_string(key, allow_empty=False) or key.startswith("__")
        for key in kwargs
    ):
        return "invalid_kwargs"
    return None


_PAYLOAD_FIELD_VALIDATORS: tuple[Callable[[dict[str, Any]], str | None], ...] = (
    _validate_payload_kind,
    _validate_payload_identifiers,
    _validate_payload_text_fields,
    _validate_payload_kwargs,
)


def _validate_schedule_payload(payload: Any) -> str | None:
    """Validate a graph-carried dispatch envelope and return a stable reason code."""

    shape_reason = _validate_payload_shape(payload)
    if shape_reason is not None:
        return shape_reason
    # ``payload`` is confirmed a dict by ``_validate_payload_shape`` above.
    for validator in _PAYLOAD_FIELD_VALIDATORS:
        reason = validator(payload)
        if reason is not None:
            return reason
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


@contextlib.contextmanager
def _control_session_scope(backend: Any) -> Any:
    """Retarget the ambient verified ``GraphSession`` onto ``backend``'s own
    graph for the duration of one control-plane read/write.

    ``_control_backend(engine)`` (above) is a *graph-scoped view* pinned to
    ``__control__`` (``EpistemicGraphBackend.for_graph``,
    CONCEPT:AU-KG.backend.schedule-on-control-graph). The ambient session
    minted for a scheduler-tick daemon/request is bound to whatever tenant
    graph it actually runs under (e.g. ``homelab``), not ``__control__``.
    ``graph_compute._send_routed`` rejects any RPC where a fixed-graph view's
    target graph disagrees with the ambient session's graph
    (``PermissionError: "A graph-scoped view cannot retarget the verified
    GraphSession"``) — so every :Schedule read/write raised, `_load_all`
    always returned ``[]``, and the scheduler never fired a single one of
    ``deploy/schedules.yml``'s entries.

    This mirrors the sanctioned ``GraphSession.with_graph()`` +
    ``use_session()`` narrowing every other control-plane call site already
    uses for the identical shape of problem —
    ``TaskManagerMixin._control_session_scope`` /
    ``_ControlPlaneWorkItemEngine._control_session_scope``
    (``knowledge_graph/core/engine_tasks.py``). The pattern itself now lives
    in exactly one place, ``knowledge_graph.core.session.control_session_scope``
    (BUG-295's fix, generalized so other callers — e.g.
    ``knowledge_graph/core/tenant_registry.py`` — reuse it instead of
    reimplementing it): read the target graph off the backend itself
    (``graph_name``) rather than hardcoding the ``__control__`` literal, and
    retarget only the ``graph`` field of the ambient session (actor/tenant/
    scopes are untouched, so authorization is unchanged) for the scope of
    the call. No ambient session (an unauthenticated bootstrap context) or a
    session already scoped to the resolved control graph is a no-op.
    """
    from ..knowledge_graph.core.session import control_session_scope

    with control_session_scope(backend):
        yield


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
def _field_part_matches(part: str, value: int) -> bool:
    part = part.strip()
    if part == "*":
        return True
    step = 1
    if "/" in part:
        base, step_s = part.split("/", 1)
        step = int(step_s)
        part = base or "*"
    if part == "*":
        return value % step == 0
    if "-" in part:
        lo, hi = (int(x) for x in part.split("-", 1))
        return lo <= value <= hi and (value - lo) % step == 0
    return int(part) == value


def _field_match(field_expr: str, value: int) -> bool:
    return any(_field_part_matches(part, value) for part in field_expr.split(","))


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


# ── Row coercion helpers for ScheduleSpec.from_row — each collapses one
# unrolled ``or``/ternary fallback in a keyword-argument call into a single
# call, so the caller's own branch count stops growing with every field.
def _row_first_str(row: dict[str, Any], *keys: str) -> str:
    for k in keys:
        v = row.get(k)
        if v:
            return str(v)
    return ""


def _row_str_or(row: dict[str, Any], key: str, default: str) -> str:
    return row.get(key) or default


def _row_optional(row: dict[str, Any], key: str) -> Any | None:
    return row.get(key) or None


def _row_int_if_present(row: dict[str, Any], key: str, default: int) -> int:
    value = row.get(key)
    return int(value) if value is not None else default


def _row_int_or(row: dict[str, Any], key: str, default: int) -> int:
    return int(row.get(key, default) or default)


def _row_float_or(row: dict[str, Any], key: str, default: float) -> float:
    return float(row.get(key, default) or default)


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
            name=_row_first_str(row, "id", "name"),
            payload=_dec(row.get("payload")),
            trigger=_row_str_or(row, "trigger", "cron"),
            cron=_row_optional(row, "cron"),
            interval_s=_row_optional(row, "interval_s"),
            prio_bucket=_row_int_if_present(row, "prio_bucket", 2),
            enabled=bool(row.get("enabled", True)),
            last_minute=_row_int_or(row, "last_minute", 0),
            next_run_unix=_row_float_or(row, "next_run_unix", 0.0),
            consecutive_failures=_row_int_or(row, "consecutive_failures", 0),
            backoff_until=_row_float_or(row, "backoff_until", 0.0),
            description=_row_str_or(row, "description", ""),
            task_type=_row_str_or(row, "task_type", "scheduled_job"),
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
    with _control_session_scope(backend):
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
    with _control_session_scope(backend):
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
    with _control_session_scope(backend):
        rows = backend.execute(
            f"MATCH (s:Schedule {{id: $id}}) RETURN s.id as id, {proj}", {"id": name}
        )
    return ScheduleSpec.from_row(rows[0]) if rows else None


class ScheduleFileError(ValueError):
    """Raised when a seeded schedule doc (``deploy/schedules.yml``) contains a
    malformed entry (CA-28-P12).

    Deliberately a WHOLE-FILE failure, not a per-entry skip: a schedule file is
    desired STATE, not a best-effort batch of independent jobs, so silently
    dropping one malformed entry while seeding its siblings is indistinguishable
    from an operator's cron typo permanently and invisibly losing ONE recurring
    job — exactly the "built but not wired" failure shape this program keeps
    finding elsewhere. :func:`seed_schedules` validates every entry BEFORE
    writing any of them, and this carries the bad entry's name (or its 0-based
    position when even ``name`` is missing) in its message so the fix is a
    one-line diff, not a graph-wide hunt.
    """


def _validate_seed_entry(entry: Any, index: int) -> None:
    """Validate one desired-state schedule entry; raise :class:`ScheduleFileError`
    naming it on any defect. Called for every entry BEFORE :func:`seed_schedules`
    upserts any of them (see :class:`ScheduleFileError`)."""

    if not isinstance(entry, dict):
        raise ScheduleFileError(f"schedule entry #{index} is not a mapping")
    name = entry.get("name")
    if not name or not isinstance(name, str):
        raise ScheduleFileError(f"schedule entry #{index} is missing a 'name'")
    cron = entry.get("cron")
    if not cron or not isinstance(cron, str):
        raise ScheduleFileError(f"schedule {name!r}: missing or invalid 'cron'")
    try:
        # cron_matches raises ValueError on a field-count/token defect (e.g. the
        # 4-field "* * * *" P12 negative case) — reuse it as the load-time
        # validator so there is exactly one cron-shape check in this module.
        cron_matches(cron, datetime.now())
    except ValueError as exc:
        raise ScheduleFileError(
            f"schedule {name!r}: invalid cron {cron!r}: {exc}"
        ) from exc
    kind = entry.get("kind", "skill")
    if kind not in _SCHEDULE_KINDS:
        raise ScheduleFileError(f"schedule {name!r}: unsupported kind {kind!r}")


# ── Seeding from deploy/schedules.yml ────────────────────────────────────────
def seed_schedules(engine: Any) -> int:
    """Upsert every ``deploy/schedules.yml`` entry as a ``:Schedule`` node.

    YAML is the desired-state seed; the node is the live record. Re-seeding is
    idempotent: it refreshes trigger/payload/enabled/cron but preserves live
    runtime state (last_minute / next_run_unix / failure backoff).

    ALL-OR-NOTHING (CA-28-P12): every entry is validated (see
    :func:`_validate_seed_entry`) BEFORE any is written. One malformed entry
    raises :class:`ScheduleFileError` naming it and seeds NOTHING from this
    call — never a partial load that seeds the good entries and silently
    drops the bad one. :func:`run_scheduler_tick`'s caller already treats a
    seed failure as retry-next-tick (it never marks ``_schedules_seeded`` on
    an exception), so this fails closed without losing any due tick.
    """
    path = _registry_path()
    if not path.exists():
        return 0
    doc = yaml.safe_load(path.read_text()) or {}
    entries = doc.get("schedules") or []
    for i, entry in enumerate(entries):
        _validate_seed_entry(entry, i)
    seeded = 0
    for entry in entries:
        name = entry.get("name")
        cron = entry.get("cron")
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


def _group_active_ticks_by_schedule(
    work: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    by_schedule: dict[str, list[str]] = {}
    for job_id, item in work.items():
        meta = item.get("metadata") or {}
        if meta.get("type") not in _SCHEDULED_TICK_TYPES:
            continue
        if item.get("status") not in {"submitted", "ready"}:
            continue
        name = meta.get("schedule")
        if name:
            by_schedule.setdefault(str(name), []).append(job_id)
    return by_schedule


def _cancel_stale_ticks(engine: Any, over: dict[str, list[str]]) -> int:
    cancelled = 0
    for ids in over.values():
        for tid in ids:
            try:
                if engine.cancel_task(tid).get("status") == "success":
                    cancelled += 1
            except Exception:  # noqa: BLE001 — best-effort per tick
                continue
    return cancelled


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
    by_schedule = _group_active_ticks_by_schedule(work)
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
    cancelled = _cancel_stale_ticks(engine, over)
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
            logger.warning("schedule cron validation failed: %s", exc)
            return False
        minute_key = int(now.replace(second=0, microsecond=0).timestamp())
        return spec.last_minute < minute_key
    # interval / adaptive
    return now_unix >= spec.next_run_unix


def _seed_schedules_once(engine: Any) -> None:
    """Seed ``deploy/schedules.yml`` once per process, fail-closed on
    ``_schedules_seeded`` (AU-OS.governance.verified-write-state-advance):
    only a CONFIRMED seed may mark it, so a transient failure (e.g. the
    control-graph session not yet available at boot) retries on the NEXT
    tick instead of permanently disabling seeding for the process's life."""
    if getattr(engine, "_schedules_seeded", False):
        return
    try:
        seed_schedules(engine)
    except Exception as exc:  # noqa: BLE001 — schedule seeding is best-effort
        logger.warning("schedule seed failed, will retry next tick: %s", exc)
    else:
        engine._schedules_seeded = True


def _collapse_stale_ticks_best_effort(engine: Any) -> None:
    # Curb/recover any duplicate interval-tick backlog before evaluating due
    # schedules (CONCEPT:AU-OS.state.stale-tick-collapse). Cheap no-op once
    # every schedule has ≤1 active tick; never raises into the tick.
    try:
        collapse_stale_ticks(engine)
    except Exception as exc:  # noqa: BLE001 — stale-tick collapse is best-effort
        logger.debug("collapse_stale_ticks failed: %s", exc)


def _advance_run_state(
    spec: ScheduleSpec, minute_key: int, now_unix: float
) -> tuple[int, float]:
    """The (last_minute, next_run_unix) a DUE ``spec`` advances to — computed
    but, per the caller's contract, not persisted until the enqueue (or a
    legitimate coalesce-skip) is confirmed (CONCEPT:AU-OS.state.durable-schedule-outbox,
    GOC-22 gate 3)."""
    if spec.trigger == "cron":
        return minute_key, spec.next_run_unix
    interval = spec.interval_s or 60.0
    if spec.trigger == "adaptive" and spec.consecutive_failures:
        mult = min(2**spec.consecutive_failures, _ADAPTIVE_MAX_BACKOFF_MULT)
        interval *= mult
    return spec.last_minute, now_unix + interval


def _has_inflight_tick(engine: Any, spec: ScheduleSpec) -> bool:
    # Coalesce: if the previous tick for this schedule hasn't been consumed
    # yet, do NOT pile another (cheap top-level ``schedule``-property probe,
    # not the O(N) metadata dedupe scan).
    work = engine._ingest_work_item_index()
    return any(
        (item.get("metadata") or {}).get("schedule") == spec.name
        and item.get("status")
        not in {"succeeded", "failed", "cancelled", "dead_letter"}
        for item in work.values()
    )


def _process_due_schedule(
    engine: Any, spec: ScheduleSpec, minute_key: int, now_unix: float
) -> str:
    """Advance/enqueue one DUE schedule; returns ``"fired"``, ``"coalesced"``,
    or ``"reconciling"`` (enqueue failed, left due for the next tick's
    retry — state is never advanced/persisted on that path)."""
    advanced_last_minute, advanced_next_run_unix = _advance_run_state(
        spec, minute_key, now_unix
    )

    if _has_inflight_tick(engine, spec):
        # A legitimate reason to advance the run state: the interval is
        # genuinely covered by the still-in-flight prior tick, not lost.
        spec.last_minute = advanced_last_minute
        spec.next_run_unix = advanced_next_run_unix
        _upsert(engine, spec)
        return "coalesced"

    # Enqueue FIRST, using the deterministic ``sched:<name>:<minute>`` job id.
    # ``submit_task``/``ensure_ingest_task_work_item`` is an idempotent
    # create-or-reuse keyed on that id with a durable admission readback, so
    # re-attempting the SAME due tick on the next scheduler evaluation after a
    # failed/crashed attempt is safe — it can never double-fire. Only a
    # CONFIRMED enqueue may advance/persist the run state.
    job_id = f"sched:{spec.name}:{minute_key}"
    try:
        engine.submit_task(
            target_path=f"schedule:{spec.name}",
            is_codebase=False,
            provenance={"schedule": spec.name},
            # CONCEPT:AU-KG.ontology.capability-card-backfill-lane — the task type
            # selects the functional lane; most schedules use ``scheduled_job``
            # (the maint lane), but a throughput backfill overrides it.
            task_type=spec.task_type or "scheduled_job",
            skip_dedupe=True,
            priority=spec.prio_bucket,
            job_id=job_id,
            extra_meta={"schedule": spec.name, "payload": spec.payload},
        )
    except Exception as exc:  # noqa: BLE001 — one schedule's enqueue failure never blocks the tick
        logger.error(
            "schedule enqueue failed, tick left due for retry (name=%s): %s",
            spec.name,
            exc,
        )
        return "reconciling"

    spec.last_minute = advanced_last_minute
    spec.next_run_unix = advanced_next_run_unix
    _upsert(engine, spec)
    return "fired"


def run_scheduler_tick(engine: Any, now: datetime | None = None) -> dict[str, Any]:
    """Evaluate every ``:Schedule`` and ENQUEUE a task for each that is due.

    The single generic scheduler tick (replaces ``run_due_schedules`` and the
    per-job maintenance-tick registrations). Idempotent within a minute via the
    node's ``last_minute``/``next_run_unix`` (advanced *before* enqueue, so a
    crash skips rather than double-fires) and the deterministic task id
    ``sched:<name>:<minute>``. Leader-gating happens in the caller.
    """
    logger.info("[OS-5.44] scheduler tick: begin")
    _seed_schedules_once(engine)
    _collapse_stale_ticks_best_effort(engine)

    now = now or datetime.now()
    now_unix = time.time()
    minute_key = int(now.replace(second=0, microsecond=0).timestamp())
    fired: list[str] = []
    reconciling: list[str] = []
    for spec in _load_all(engine):
        if not _is_due(spec, now, now_unix):
            continue
        outcome = _process_due_schedule(engine, spec, minute_key, now_unix)
        if outcome == "fired":
            fired.append(spec.name)
        elif outcome == "reconciling":
            reconciling.append(spec.name)
    if fired:
        logger.info("scheduler fired schedule(s) (count=%d)", len(fired))
    if reconciling:
        logger.warning(
            "scheduler left due tick(s) for retry (count=%d, schedules=%s)",
            len(reconciling),
            reconciling,
        )
    logger.info("[OS-5.44] scheduler tick: end (fired=%d)", len(fired))
    return {"fired": fired, "count": len(fired), "reconciling": reconciling}


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


def _lakehouse_maintenance_dispatch(
    engine: Any,
    *,
    check: str,
    owner: str,
    module_path: str,
    finding_fn_name: str,
    source: str,
) -> dict[str, Any]:
    """The one body every ``lakehouse-maintenance`` dispatch target shares
    (CONCEPT:AU-OS.state.unified-scheduling-one-intelligent) — this IS the
    live wiring itself: it looks up ``finding_fn_name`` on
    ``module_path`` via ``getattr`` (never a static ``from X import Y``,
    which would be an unconditional mypy attr-defined error against a symbol
    that does not exist yet — the same rationale as
    ``intent_tools._lakehouse_status``), and — real callers only, no test
    reaches this line except by monkeypatching the looked-up module — when a
    lane (CA-21/24/25) lands that function and it returns a non-empty
    finding string, calls
    :func:`agent_utilities.mcp.tools.state_tools.propose_lakehouse_maintenance_gap`
    to file it as a canonical, reviewable :Gap (propose-only — see that
    function's docstring). Per this program's non-goals, CA-28 does not
    define what a check computes; it defines how a real finding, once
    computed, reaches the SAME gap lifecycle every other discovery track
    uses. Today none of ``kafka_adapter``/the OpenSearch indexer/
    ``etl.lineage`` exposes ``finding_fn_name`` yet, so this always returns
    the typed ``not_yet_implemented`` result — that is the CORRECT behavior
    for a check with nothing to report, not a bug to work around.
    """
    try:
        module = __import__(module_path, fromlist=["_"])
    except ImportError:
        return {"status": "not_yet_implemented", "check": check, "owner": owner}
    finding_fn = getattr(module, finding_fn_name, None)
    if finding_fn is None:
        return {"status": "not_yet_implemented", "check": check, "owner": owner}
    finding = finding_fn(engine)
    if not finding:
        return {"status": "ok", "check": check}
    from agent_utilities.mcp.tools.state_tools import (
        propose_lakehouse_maintenance_gap,
    )

    gap = propose_lakehouse_maintenance_gap(
        engine, source=source, statement=str(finding)
    )
    return {"status": "gap_proposed", "check": check, "gap": gap}


def _dispatch_debezium_lag_check(
    engine: Any, payload: dict[str, Any]
) -> dict[str, Any]:
    """CA-21's Debezium-consumer-lag check, wired via
    :func:`_lakehouse_maintenance_dispatch` — lag measurement itself lands in
    ``agent_utilities.knowledge_graph.streams.kafka_adapter`` (CA-21's to
    land); until it exposes ``lag_finding``, this returns the typed
    ``not_yet_implemented`` result."""

    return _lakehouse_maintenance_dispatch(
        engine,
        check="debezium_lag",
        owner="CA-21",
        module_path="agent_utilities.knowledge_graph.streams.kafka_adapter",
        finding_fn_name="lag_finding",
        source="lakehouse-maintenance:debezium_lag_check",
    )


def _dispatch_opensearch_reindex_staleness_check(
    engine: Any, payload: dict[str, Any]
) -> dict[str, Any]:
    """CA-24's OpenSearch CDC-vs-index staleness scan, wired via
    :func:`_lakehouse_maintenance_dispatch`. The rebuild ACTION itself
    already exists and is real (``graph_ingest`` ``action=opensearch_reindex``,
    CA-24) — this is the DETECTION half (when is a rebuild actually due),
    which lands in ``agent_utilities.knowledge_graph.search.indexer`` (CA-24's
    to land); until it exposes ``staleness_finding``, this returns the typed
    ``not_yet_implemented`` result."""

    return _lakehouse_maintenance_dispatch(
        engine,
        check="opensearch_reindex_staleness",
        owner="CA-24",
        module_path="agent_utilities.knowledge_graph.search.indexer",
        finding_fn_name="staleness_finding",
        source="lakehouse-maintenance:opensearch_reindex_staleness_check",
    )


def _dispatch_lineage_sweep(engine: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """CA-25's OpenLineage-vs-PROV-O backfill sweep, wired via
    :func:`_lakehouse_maintenance_dispatch` — real detection lands in
    ``agent_utilities.knowledge_graph.etl.lineage`` (CA-25's to land); until
    it exposes ``sweep_finding``, this returns the typed
    ``not_yet_implemented`` result."""

    return _lakehouse_maintenance_dispatch(
        engine,
        check="lineage_sweep",
        owner="CA-25",
        module_path="agent_utilities.knowledge_graph.etl.lineage",
        finding_fn_name="sweep_finding",
        source="lakehouse-maintenance:lineage_sweep",
    )


# Deterministic skill actions runnable unattended on the daemon, keyed (ref, action).
_SKILL_HANDLERS: dict[
    tuple[str, str], Callable[[Any, dict[str, Any]], dict[str, Any]]
] = {
    ("code-enhancer", "liveness"): _dispatch_liveness,
    ("memory-lifecycle", "maintain"): _dispatch_memory_lifecycle,
    ("lakehouse-maintenance", "debezium_lag_check"): _dispatch_debezium_lag_check,
    (
        "lakehouse-maintenance",
        "opensearch_reindex_staleness_check",
    ): _dispatch_opensearch_reindex_staleness_check,
    ("lakehouse-maintenance", "lineage_sweep"): _dispatch_lineage_sweep,
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


def _dispatch_maint(engine: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """A former fixed-interval maintenance tick: an engine ``_tick_<ref>`` method."""
    ref = payload.get("ref", "")
    if ref not in _MAINTENANCE_REF_ALLOWLIST:
        return {"status": "skipped", "reason": "maintenance_not_allowed"}
    tick = getattr(engine, f"_tick_{ref}", None)
    if not callable(tick):
        return {"status": "skipped", "reason": "maintenance_unavailable"}
    tick()
    return {"status": "ok"}


def _dispatch_research_feed(engine: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Unified feed sweep (CONCEPT:AU-KG.ingest.rss-feed-connector): native RSS
    + ScholarX arXiv through the one world-model gate, plus the FreshRSS delta
    and the native arXiv connector (CONCEPT:AU-KG.ingest.arxiv-feed-connector)
    — all converge on the same research/news routing. ``arxiv`` itself no-ops
    cleanly when ``KG_ARXIV_CATEGORIES`` is unset."""
    from agent_utilities.knowledge_graph.core.source_sync import sync_source

    results = {
        "rss": sync_source(engine, "rss", mode="delta"),
        "freshrss": sync_source(engine, "freshrss", mode="delta"),
        "arxiv": sync_source(engine, "arxiv", mode="delta"),
    }
    return {"status": "ok", "feeds": results}


def _dispatch_skill_writeback(engine: Any, ref: str, action: str) -> dict[str, Any]:
    from agent_utilities.knowledge_graph.enrichment.writeback import (
        push_inventory,
        run_writeback,
    )

    backend = getattr(engine, "backend", None)
    if action == "inventory":
        return push_inventory(ref, backend=backend, engine=engine, dry_run=False)
    return run_writeback(ref, backend=backend, engine=engine, dry_run=False)


def _dispatch_skill(engine: Any, payload: dict[str, Any]) -> dict[str, Any]:
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
        return _dispatch_skill_writeback(engine, ref, action)
    return {"status": "skipped", "reason": "no_handler"}


def _dispatch_workflow_or_agent(engine: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """``engine`` here is the raw IntelligenceGraphEngine (it has no
    ``execute_workflow`` of its own) -- the governed dispatch surface is
    Orchestrator.execute_workflow, which is also the D-WS-8 chokepoint that
    runs the SHACL+ACL gate (agent_utilities.knowledge_graph.core.workflow_gate)
    before any step runs, so scheduled workflow/agent jobs inherit the same
    governance as every other caller.

    The one production caller of this dispatcher
    (IntelligenceGraphEngine._run_background_task, engine_tasks.py) is itself
    an async method calling ``run_scheduled_job`` synchronously from inside an
    ALREADY-RUNNING event loop. ``_run_async`` (shared with
    ticket_playbooks._dispatch_workflow, the sibling D-WS-8 caller) runs the
    coroutine on a fresh loop in a worker thread whether or not a loop is
    already running, and bounds the wait so a stuck workflow can never hang
    the scheduler tick indefinitely.
    """
    try:
        from agent_utilities.orchestration.manager import Orchestrator
        from agent_utilities.protocols.source_connectors.connectors.mcp_package import (
            _run_async,
        )

        orchestrator = Orchestrator(engine)
        coro = orchestrator.execute_workflow(
            workflow_id=payload.get("ref", payload.get("name", "")),
            task=payload.get("task", payload.get("description", "")),
            **(payload.get("kwargs") or {}),
        )
        _run_async(coro, timeout=_WORKFLOW_DISPATCH_TIMEOUT_S)
        return {"status": "ok"}
    except Exception as exc:  # noqa: BLE001
        from agent_utilities.security.error_surface import public_error_payload

        return public_error_payload(exc, logger=logger)


def _dispatch_scheduled_job(engine: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """The routing body of :func:`run_scheduled_job` at the untrusted graph boundary."""
    invalid_reason = _validate_schedule_payload(payload)
    if invalid_reason is not None:
        return {"status": "error", "reason": invalid_reason}
    kind = payload.get("kind", "skill")
    if kind == "maint":
        return _dispatch_maint(engine, payload)
    if kind in ("research_feed", "feed_sweep"):
        return _dispatch_research_feed(engine, payload)
    if kind == "skill":
        return _dispatch_skill(engine, payload)
    if kind == "script":
        # Retired security boundary: graph data must never select a host path to
        # execute.  Use a governed skill/workflow handler backed by an isolated
        # runtime instead.
        return {"status": "skipped", "reason": "host_script_execution_retired"}
    if kind in ("workflow", "agent"):
        return _dispatch_workflow_or_agent(engine, payload)
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
