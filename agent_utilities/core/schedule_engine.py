"""CONCEPT:OS-5.44 — Unified scheduling engine, one intelligent scheduler for all recurring work.

Collapses the four historical scheduling surfaces — fixed-interval maintenance
ticks, the static ``deploy/schedules.yml`` cron, the loop-cycle tick, and the
legacy OS-5.2 ``MaintenanceCron`` — into ONE durable, dynamic scheduler. The
scheduler is the sole *producer* of recurring work: when a schedule is due it
**enqueues** a ``scheduled_job`` ``:Task`` onto the one hardened priority+
scheduled queue (KG-2.113), which the one worker pool drains under the
existing throttle / lease / reaper hardening. Nothing recurring runs inline in
the scheduler thread anymore.

A schedule is a durable ``:Schedule`` graph node (survives restart and
leader-failover, and is editable at runtime — enable/disable/reprioritize/
set-interval/run-now). ``deploy/schedules.yml`` is the *seed* (desired state);
the node carries live state (last-run, next-run, failure backoff). Triggers:

  * ``cron``     — standard 5-field ``min hour dom month dow`` (no third-party dep)
  * ``interval`` — every ``interval_s`` seconds (the former maintenance ticks)
  * ``adaptive`` — interval that widens on repeated failure / can be re-tuned live

The payload describes WHAT to run (``kind``: skill / script / workflow / agent /
maint / loop / research_feed); :func:`run_scheduled_job` is the single dispatcher
the worker calls, so the routing lives in exactly one place.
"""

from __future__ import annotations

import base64
import json
import logging
import subprocess
import sys
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
        )


# ── Durable :Schedule node store ─────────────────────────────────────────────
def _upsert(engine: Any, spec: ScheduleSpec) -> None:
    backend = getattr(engine, "backend", None)
    if backend is None:
        return
    props = spec.to_props()
    existing = backend.execute(
        "MATCH (s:Schedule {id: $id}) RETURN s.id as id", {"id": spec.name}
    )
    if existing:
        sets = ", ".join(f"s.{k} = ${k}" for k in props)
        backend.execute(
            f"MATCH (s:Schedule {{id: $id}}) SET {sets}",
            {"id": spec.name, **props},
        )
    else:
        backend.add_node(spec.name, node_type=_SCHEDULE_LABEL, **props)


def _load_all(engine: Any) -> list[ScheduleSpec]:
    backend = getattr(engine, "backend", None)
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
        "payload",
    )
    proj = ", ".join(f"s.{k} as {k}" for k in keys)
    rows = backend.execute(f"MATCH (s:Schedule) RETURN s.id as id, {proj}")
    return [ScheduleSpec.from_row(r) for r in (rows or [])]


def _load_one(engine: Any, name: str) -> ScheduleSpec | None:
    backend = getattr(engine, "backend", None)
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
        )
        existing = _load_one(engine, name)
        if existing is not None:
            # Preserve live runtime state across re-seed.
            spec.last_minute = existing.last_minute
            spec.next_run_unix = existing.next_run_unix
            spec.consecutive_failures = existing.consecutive_failures
            spec.backoff_until = existing.backoff_until
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
    _upsert(engine, spec)


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
        except ValueError as e:
            logger.warning("schedule %s: %s", spec.name, e)
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
    if not getattr(engine, "_schedules_seeded", False):
        try:
            seed_schedules(engine)
        except Exception as e:  # noqa: BLE001
            logger.warning("schedule seed failed: %s", e)
        engine._schedules_seeded = True

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
        job_id = f"sched:{spec.name}:{minute_key}"
        try:
            engine.submit_task(
                target_path=f"schedule:{spec.name}",
                is_codebase=False,
                provenance={"schedule": spec.name},
                task_type="scheduled_job",
                skip_dedupe=True,
                priority=spec.prio_bucket,
                job_id=job_id,
                extra_meta={"schedule": spec.name, "payload": spec.payload},
            )
            fired.append(spec.name)
        except Exception as e:  # noqa: BLE001
            logger.error("schedule %s enqueue error: %s", spec.name, e)
    if fired:
        logger.info("scheduler fired: %s", fired)
    return {"fired": fired, "count": len(fired)}


def record_schedule_result(engine: Any, name: str, ok: bool) -> None:
    """Update a schedule's failure backoff after its job ran (CONCEPT:OS-5.44).

    Called by the worker after ``run_scheduled_job`` so an ``adaptive`` schedule
    widens its interval on repeated failure and a failing job is throttled.
    """
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


# Deterministic skill actions runnable unattended on the daemon, keyed (ref, action).
_SKILL_HANDLERS: dict[
    tuple[str, str], Callable[[Any, dict[str, Any]], dict[str, Any]]
] = {
    ("code-enhancer", "liveness"): _dispatch_liveness,
}


def run_scheduled_job(engine: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Execute one scheduled job's payload — the worker calls this.

    The single dispatcher for every recurring job, routed by ``payload['kind']``
    so the routing lives in exactly one place (CONCEPT:OS-5.44).
    """
    kind = payload.get("kind", "skill")
    if kind == "maint":
        # A former fixed-interval maintenance tick: an engine ``_tick_<ref>`` method.
        ref = payload.get("ref", "")
        tick = getattr(engine, f"_tick_{ref}", None)
        if tick is None:
            return {"status": "skipped", "reason": f"no_maint_tick:{ref}"}
        tick()
        return {"status": "ok"}
    if kind in ("research_feed", "feed_sweep"):
        # Unified feed sweep (CONCEPT:KG-2.121): native RSS + ScholarX arXiv through
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
        ref = payload.get("ref", "")
        res = subprocess.run(  # noqa: S603
            [sys.executable, ref, *map(str, payload.get("args", []))],
            capture_output=True,
            text=True,
            timeout=payload.get("timeout", 600),
        )
        return {
            "status": "ok" if res.returncode == 0 else "error",
            "rc": res.returncode,
        }
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
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "reason": str(e)}
    return {"status": "skipped", "reason": f"unknown_kind:{kind}"}


# ── Runtime control — enable/disable/reprioritize/retune, surfaced via MCP + REST (CONCEPT:OS-5.44)
def set_enabled(engine: Any, name: str, enabled: bool) -> dict[str, Any]:
    spec = _load_one(engine, name)
    if spec is None:
        return {"status": "error", "error": f"schedule {name} not found"}
    spec.enabled = enabled
    _upsert(engine, spec)
    return {"status": "success", "name": name, "enabled": enabled}


def set_priority(engine: Any, name: str, priority: int | str) -> dict[str, Any]:
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
