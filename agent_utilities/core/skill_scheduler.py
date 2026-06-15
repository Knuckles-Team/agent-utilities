"""CONCEPT:OS-5.30 — Declarative skill / skill-workflow scheduler.

The daemon must NOT hardcode a tick per recurring job. Instead, every scheduled
skill or skill-workflow is **declared** in ``deploy/schedules.yml`` with a cron
expression, and ONE generic tick (:func:`run_due_schedules`) reads that registry
and dispatches whatever is due. Adding a scheduled job = a YAML entry, not daemon
code. ``/cron calendar`` reads this same registry (replacing the old hardcoded
placeholder text).

Dispatch kinds (how the scheduled item is invoked):
  * ``skill``    — a deterministic code-enhancer-style skill action; routed through
    :data:`_SKILL_HANDLERS` (no LLM, safe to run unattended on the daemon).
  * ``script``   — a plain script path, run via subprocess.
  * ``workflow`` / ``agent`` — an LLM path: a dynamic agent workflow via the engine
    (``execute_workflow`` / ``run_agent``). Kept opt-in per entry because LLM work
    has cost; detection-style jobs stay deterministic.

Cron support is the standard 5-field ``min hour dom month dow`` with ``*``,
``*/N``, ``N``, ``N,M`` lists and ``N-M`` ranges (no third-party dep).
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from agent_utilities.core.paths import data_dir

logger = logging.getLogger(__name__)

_STATE_FILE = data_dir() / "schedule_state.json"


def _registry_path() -> Path:
    # deploy/schedules.yml lives at the package repo root's deploy/ dir.
    return Path(__file__).resolve().parents[2] / "deploy" / "schedules.yml"


# ── Cron matching (5-field) ──────────────────────────────────────────────────
def _field_match(field: str, value: int) -> bool:
    for part in field.split(","):
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


# ── Registry + state ─────────────────────────────────────────────────────────
def load_schedules() -> list[dict[str, Any]]:
    path = _registry_path()
    if not path.exists():
        return []
    doc = yaml.safe_load(path.read_text()) or {}
    return [s for s in (doc.get("schedules") or []) if s.get("enabled", True)]


def _load_state() -> dict[str, float]:
    try:
        return json.loads(_STATE_FILE.read_text())
    except Exception:  # noqa: BLE001
        return {}


def _save_state(state: dict[str, float]) -> None:
    try:
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _STATE_FILE.write_text(json.dumps(state, indent=2))
    except Exception as e:  # noqa: BLE001
        logger.warning("schedule state save failed: %s", e)


# ── Dispatch ─────────────────────────────────────────────────────────────────
def _dispatch_liveness(engine: Any, entry: dict[str, Any]) -> dict[str, Any]:
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


def dispatch(entry: dict[str, Any], engine: Any) -> dict[str, Any]:
    kind = entry.get("kind", "skill")
    if kind == "skill":
        ref = entry.get("ref", "")
        action = entry.get("action", "")
        handler = _SKILL_HANDLERS.get((ref, action))
        if handler is not None:
            return handler(engine, entry)
        # Generic source-sync dispatch: any source registered in the hydration
        # capability registry syncs through the one entrypoint (delta/full/reconcile).
        from agent_utilities.knowledge_graph.core.source_sync import (
            SYNC_ACTIONS,
            sync_source,
        )

        if action in SYNC_ACTIONS:
            return sync_source(engine, ref, mode=action)
        return {"status": "skipped", "reason": "no_handler"}
    if kind == "script":
        ref = entry.get("ref", "")
        res = subprocess.run(
            [sys.executable, ref, *map(str, entry.get("args", []))],
            capture_output=True,
            text=True,
            timeout=entry.get("timeout", 600),
        )
        return {
            "status": "ok" if res.returncode == 0 else "error",
            "rc": res.returncode,
        }
    if kind in ("workflow", "agent"):
        # LLM path — dispatch a dynamic agent workflow through the engine.
        try:
            import asyncio

            coro = engine.execute_workflow(
                workflow_id=entry.get("ref", entry["name"]),
                task=entry.get("task", entry.get("description", "")),
                **(entry.get("kwargs") or {}),
            )
            asyncio.get_event_loop().run_until_complete(coro)
            return {"status": "ok"}
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "reason": str(e)}
    return {"status": "skipped", "reason": f"unknown_kind:{kind}"}


# ── The one generic tick ─────────────────────────────────────────────────────
def run_due_schedules(engine: Any, now: datetime | None = None) -> dict[str, Any]:
    """Dispatch every schedule whose cron fires this minute and that has not already
    run this minute. Called once per minute by the single generic daemon tick — the
    ONLY scheduling code in the daemon; all jobs are declared in the registry."""
    now = now or datetime.now()
    minute_key = int(now.replace(second=0, microsecond=0).timestamp())
    state = _load_state()
    fired: list[str] = []
    for entry in load_schedules():
        name = entry.get("name")
        cron = entry.get("cron")
        if not name or not cron:
            continue
        try:
            if not cron_matches(cron, now):
                continue
        except ValueError as e:
            logger.warning("schedule %s: %s", name, e)
            continue
        if state.get(name, 0) >= minute_key:  # already fired this minute
            continue
        state[name] = minute_key
        try:
            result = dispatch(entry, engine)
            logger.info("schedule %s fired: %s", name, result.get("status"))
            fired.append(name)
        except Exception as e:  # noqa: BLE001
            logger.error("schedule %s dispatch error: %s", name, e)
    if fired:
        _save_state(state)
    return {"fired": fired, "count": len(fired)}


def calendar() -> list[dict[str, Any]]:
    """Registry + last-run, for the ``/cron calendar`` command (real, not stubbed)."""
    state = _load_state()
    out = []
    for s in load_schedules():
        last = state.get(s.get("name", ""))
        out.append(
            {
                "name": s.get("name"),
                "cron": s.get("cron"),
                "kind": s.get("kind", "skill"),
                "ref": s.get("ref"),
                "description": s.get("description", ""),
                "last_run": (
                    datetime.fromtimestamp(last).isoformat() if last else "never"
                ),
            }
        )
    return out
