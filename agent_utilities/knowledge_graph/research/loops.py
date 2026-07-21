#!/usr/bin/python
from __future__ import annotations

"""The Loop — a long-running objective the loop engine advances (CONCEPT:AU-KG.research.these-properties-carry).

A **Loop** is the single unit of long-running work. Goals, research topics, failure
gaps, and skill executions all collapse into this one node — distinguished only by
``kind``:

- ``research`` — acquire knowledge addressing the objective; *done* when the topic
  has ``ADDRESSED_BY`` sources.
- ``develop`` — iterate act→validate until ``validation_cmd`` / ``end_state`` holds.
- ``skill``   — run a skill / skill-workflow to its completion state.

The ``LoopController`` advances **every active Loop** through ONE hot path, so there
is one engine and one entrypoint for "make progress on a long-running objective",
whatever its kind. This generalizes the old separate Concept-topic intake
(``topic_resolver.unresolved_topics``), the ``failure_gap`` topic
(``failure_analyzer.file_gap_topic``), and the ``GoalNode`` execution spec.

A Loop definition is projected as a ``Concept`` node (so dedup / gap / synergy /
ConceptMatcher still apply), while its deterministic WorkItem is the only
writable lifecycle/claim/lease authority.

Concept: loop
"""

import logging
import re
import time
from enum import StrEnum
from typing import Any, Literal

logger = logging.getLogger(__name__)

LoopKind = Literal["research", "develop", "skill", "external_event"]


class LoopStatus(StrEnum):
    """Every lifecycle status a Loop (and its Goal adapter) can hold.

    Replaces the former bare frozenset of terminal strings with an authoritative
    enum so an illegal/misspelled status can no longer silently pass a
    ``status not in TERMINAL_STATUS`` test as "keep looping" (:func:`is_terminal`
    / :func:`to_status` validate against these members).

    Each of the eight harness-enforced loop exits transitions to a DISTINCT
    terminal status so the reason a loop stopped is diagnosable, never a generic
    ``failed``. Each terminal status maps cleanly onto one guarded transition in a
    future explicit loop state machine.
    """

    # --- non-terminal ---
    SUBMITTED = "submitted"
    RUNNING = "running"
    PENDING = "pending"
    VALIDATING = "validating"
    PAUSED = "paused"
    ORPHANED = "orphaned"
    # --- terminal: canonical outcomes (pre-existing vocabulary) ---
    COMPLETED = "completed"  # exit 1 GOAL MET — a real measured pass
    FAILED = "failed"
    CANCELLED = "cancelled"  # exit 6 HUMAN INTERRUPT — kill
    REJECTED = "rejected"
    # --- terminal: the harness-enforced loop exits (each distinct/diagnosable) ---
    MAX_ITERATIONS_EXCEEDED = "max_iterations_exceeded"  # exit 2 TURN CAP
    BUDGET_EXCEEDED = "budget_exceeded"  # exit 3 BUDGET CAP
    WALL_CLOCK_EXCEEDED = "wall_clock_exceeded"  # exit 4 WALL CLOCK
    STALLED = "stalled"  # exit 5 NO PROGRESS
    ERROR_THRESHOLD_EXCEEDED = "error_threshold_exceeded"  # exit 7 ERROR THRESHOLD
    EXTERNAL_EVENT_SATISFIED = "external_event_satisfied"  # exit 8 EXTERNAL EVENT


#: The terminal ``LoopStatus`` members — a Loop in one of these never needs more
#: work. Success terminals (``COMPLETED``, ``EXTERNAL_EVENT_SATISFIED``) are kept
#: separate from the abnormal/exhaustion terminals for downstream reporting.
_SUCCESS_TERMINALS: frozenset[LoopStatus] = frozenset(
    {LoopStatus.COMPLETED, LoopStatus.EXTERNAL_EVENT_SATISFIED}
)
_TERMINAL_MEMBERS: frozenset[LoopStatus] = _SUCCESS_TERMINALS | frozenset(
    {
        LoopStatus.FAILED,
        LoopStatus.CANCELLED,
        LoopStatus.REJECTED,
        LoopStatus.MAX_ITERATIONS_EXCEEDED,
        LoopStatus.BUDGET_EXCEEDED,
        LoopStatus.WALL_CLOCK_EXCEEDED,
        LoopStatus.STALLED,
        LoopStatus.ERROR_THRESHOLD_EXCEEDED,
    }
)

#: Backwards-compatible plain-string frozenset (``status not in TERMINAL_STATUS``
#: checks throughout the codebase keep working) — now derived from the enum and
#: covering ALL ten terminal statuses, so the six new harness-enforced exits are
#: correctly recognised as terminal by every existing caller.
TERMINAL_STATUS: frozenset[str] = frozenset(s.value for s in _TERMINAL_MEMBERS)
#: Preferred alias (enum-typed name) for new code.
TERMINAL_STATUSES = TERMINAL_STATUS

#: Known aliases folded onto canonical members by :func:`to_status`.
_STATUS_ALIASES: dict[str, LoopStatus] = {
    "succeeded": LoopStatus.COMPLETED,
    "success": LoopStatus.COMPLETED,
    "canceled": LoopStatus.CANCELLED,
    "": LoopStatus.PENDING,
}


def to_status(
    raw: str | LoopStatus, *, default: LoopStatus = LoopStatus.FAILED
) -> LoopStatus:
    """Coerce a raw status to a :class:`LoopStatus` member (fail-closed).

    An unrecognised status is mapped to ``default`` (``FAILED``) with a warning
    rather than silently flowing through as a non-terminal "keep looping" string —
    closing the audit gap where a typo'd status would neither terminate the loop
    nor be noticed.
    """
    if isinstance(raw, LoopStatus):
        return raw
    key = str(raw or "").strip().lower()
    if key in _STATUS_ALIASES:
        return _STATUS_ALIASES[key]
    try:
        return LoopStatus(key)
    except ValueError:
        logger.warning(
            "unknown Loop status %r -> treating as %s (fail-closed)",
            raw,
            default.value,
        )
        return default


def is_terminal(status: str | LoopStatus) -> bool:
    """True when ``status`` is a terminal Loop status.

    Unlike a bare ``status in TERMINAL_STATUS`` set-membership test, ``status`` is
    first validated against :class:`LoopStatus` (via :func:`to_status`), so an
    unknown/misspelled status fails closed (treated as ``FAILED``, which IS
    terminal) instead of silently reading as non-terminal.
    """
    return to_status(status) in _TERMINAL_MEMBERS


def is_success(status: str | LoopStatus) -> bool:
    """True when ``status`` is a *successful* terminal status."""
    return to_status(status) in _SUCCESS_TERMINALS


def _prio_bucket(value: Any, default: int = 2) -> int:
    """Normalize a priority spec to the ONE 0..3 claim bucket (CONCEPT:AU-KG.ingest.hardened-priority-scheduled-task).

    Thin lazy-import wrapper over ``engine_tasks._coerce_prio_bucket`` — the
    single priority normalizer shared by tasks / dispatch / schedules / loops.
    Lazy because ``engine_tasks`` pulls in the engine, and this module is
    imported on that path (avoids an import cycle, mirroring how ``bus.py`` /
    ``state_tools.py`` / ``schedule_engine.py`` reach the same normalizer).
    """
    from agent_utilities.knowledge_graph.core.engine_tasks import _coerce_prio_bucket

    return _coerce_prio_bucket(value, default)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _slug(text: str, *, limit: int = 60) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return (s[:limit] or "loop").rstrip("-")


def submit_loop(
    engine: Any,
    objective: str,
    *,
    kind: LoopKind = "research",
    end_state: str = "",
    validation_cmd: str = "",
    skill_ref: str = "",
    source: str = "user",
    loop_id: str = "",
    max_iterations: int = 20,
    prio_bucket: int = 2,
) -> dict[str, Any] | None:
    """Materialize a long-running objective as a Loop node (CONCEPT:AU-KG.research.these-properties-carry).

    The single shared creation path for goals, research topics, failure gaps, and
    skill executions. Idempotent (re-submitting the same id upserts). Returns the
    loop dict the ``LoopController`` intake consumes, or ``None`` if persist failed.
    """
    objective = (objective or "").strip()
    if not objective and not skill_ref:
        return None
    oid = (loop_id or f"loop:{kind}:{_slug(objective or skill_ref)}").strip()
    props: dict[str, Any] = {
        "name": objective or skill_ref,
        "objective": objective,
        "loop_kind": kind,
        "source": source,
        "max_iterations": int(max_iterations),
        # Claim/intake priority bucket (0=critical .. 3=background); active_loops
        # emits in ascending-bucket order so a hot loop is advanced first, and a
        # loop-spawned child task inherits this. Coerced through the ONE shared
        # normalizer so a loop bucket is the same 0..3 value as a task's.
        # (CONCEPT:AU-KG.ingest.hardened-priority-scheduled-task)
        "prio_bucket": _prio_bucket(prio_bucket),
        "timestamp": _now_iso(),
    }
    if end_state:
        props["end_state"] = end_state
    if validation_cmd:
        props["validation_cmd"] = validation_cmd
    if skill_ref:
        props["skill_ref"] = skill_ref
    from agent_utilities.orchestration.work_item import ensure_loop_work_item

    ensure_loop_work_item(
        engine,
        oid,
        priority=_prio_bucket(prio_bucket),
        max_attempts=max(1, int(max_iterations)),
    )
    # Concept is immutable lifecycle-neutral definition/content. The WorkItem
    # created above is the only place scheduling and lifecycle state exist.
    try:
        engine.add_node(oid, "Concept", properties=props)
    except Exception as e:  # noqa: BLE001 — definition persistence is reported
        logger.debug("submit_loop definition persist failed: %s", e)
    return {**_loop_dict(oid, props), "status": "submitted"}


def _loop_dict(oid: str, data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": oid,
        "name": data.get("name") or data.get("objective") or oid,
        "kind": data.get("loop_kind") or "research",
        # Validated through the ONE shared bucket normalizer (default 2); it
        # preserves bucket 0, which is falsy.
        # (CONCEPT:AU-KG.ingest.hardened-priority-scheduled-task)
        "prio_bucket": _prio_bucket(data.get("prio_bucket")),
    }
    for k in (
        "objective",
        "end_state",
        "validation_cmd",
        "skill_ref",
        "spec_id",
    ):
        v = data.get(k)
        if v:
            out[k] = v
    return out


def mark_loop_status(
    engine: Any,
    loop_id: str,
    status: str,
    *,
    iteration: int | None = None,
    output: str = "",
    source: str = "loop_engine",
) -> bool:
    """Advance a Loop's lifecycle state (CONCEPT:AU-KG.research.these-properties-carry).

    The controller's develop/skill stages and the
    ``graph_loops(action="cancel")`` entrypoint all call this. Definition nodes
    are never mutated with lifecycle state.
    """
    from agent_utilities.orchestration.work_item import transition_loop_work_item

    # Stamp the PRECISE terminal reason onto the WorkItem's result/error ref so
    # each harness-enforced exit is durably diagnosable (not collapsed to a bare
    # 'failed'). Success terminals carry a result_ref; abnormal/exhaustion
    # terminals carry an error_ref naming the exact exit.
    normalized = str(status).strip().lower()
    result_ref = (
        f"loop:{loop_id}:{normalized}"
        if normalized in {"completed", "succeeded", "external_event_satisfied"}
        else None
    )
    error_ref = (
        f"loop:{loop_id}:{normalized}"
        if normalized
        in {
            "failed",
            "rejected",
            "cancelled",
            "max_iterations_exceeded",
            "budget_exceeded",
            "wall_clock_exceeded",
            "stalled",
            "error_threshold_exceeded",
        }
        else None
    )
    transitioned = transition_loop_work_item(
        engine,
        loop_id,
        status,
        result_ref=result_ref,
        error_ref=error_ref,
    )
    del iteration, output, source
    return transitioned


def prioritize_loop(engine: Any, loop_id: str, prio_bucket: int) -> bool:
    """Set a Loop's intake/claim priority bucket (CONCEPT:AU-KG.ingest.hardened-priority-scheduled-task).

    ``active_loops`` emits loops in ascending-bucket order, so bumping a loop to
    bucket 0/1 advances it ahead of background loops on the next cycle.
    Best-effort: a failed persist returns ``False``, never raises.
    """
    from agent_utilities.orchestration import work_item as _wi

    item_id = _wi.loop_work_item_id(loop_id)
    item = _wi.get_work_item(engine, item_id)
    if item is None or item.get("kind") != "goal_loop":
        return False
    bucket = _prio_bucket(prio_bucket)
    return _wi.set_work_item_priority(engine, item_id, bucket)


def claim_loop(engine: Any, loop_id: str) -> bool:
    """Acquire the existing Loop's native WorkItem lease.

    A negative native claim is final; no definition status can substitute for
    the engine-issued lease.
    """
    from agent_utilities.orchestration.work_item import claim_loop_work_item

    return claim_loop_work_item(engine, loop_id) is not None


def active_loops(engine: Any, limit: int = 10) -> list[dict[str, Any]]:
    """Every Loop still needing work — the LoopController's intake (CONCEPT:AU-KG.research.these-properties-carry).

    Generalizes ``unresolved_topics``: a Loop is *active* when

    - it is a ``research`` Loop with
      no ``ADDRESSED_BY`` source, OR
    - it is a ``develop`` / ``skill`` Loop whose ``status`` is not terminal.

    Each returned dict carries ``kind`` so the controller dispatches by stage.
    Computed with SUPPORTED query shapes only (positive single-hop + plain node
    scan, then subtract) — same constraint as ``unresolved_topics``.
    """
    addressed: set[str] = set()
    try:
        rows = engine.query_cypher(
            "MATCH (c:Concept)-[:ADDRESSED_BY]->(s) RETURN c.id AS id"
        )
        addressed = {
            r["id"] for r in (rows or []) if isinstance(r, dict) and r.get("id")
        }
    except Exception as e:  # noqa: BLE001
        logger.debug("active_loops: addressed query failed: %s", e)

    try:
        rows = engine.query_cypher(
            "MATCH (c:Concept) RETURN c.id AS id, c.name AS name, "
            "c.loop_kind AS loop_kind, "
            "c.objective AS objective, c.validation_cmd AS validation_cmd, "
            "c.skill_ref AS skill_ref, c.end_state AS end_state, "
            "c.spec_id AS spec_id, c.prio_bucket AS prio_bucket LIMIT $limit",
            {"limit": int(limit) * 20},
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("active_loops: concept query failed: %s", e)
        return []

    from agent_utilities.orchestration.work_item import (
        TERMINAL_WORK_ITEM_STATUSES,
        WorkItemStatus,
        claim_loop_work_item,
        get_work_item,
        loop_work_item_id,
        transition_loop_work_item,
    )

    out: list[dict[str, Any]] = []
    for r in rows or []:
        if not isinstance(r, dict) or not r.get("id"):
            continue
        cid = r["id"]
        kind = str(r.get("loop_kind") or "")
        if kind not in {"research", "develop", "skill"}:
            continue
        item_id = loop_work_item_id(cid)
        item = get_work_item(engine, item_id)
        if item is None or item.get("kind") != "goal_loop":
            continue
        status = str(item.get("status") or "")
        if status in TERMINAL_WORK_ITEM_STATUSES:
            continue
        if kind in ("develop", "skill") and status in {
            WorkItemStatus.LEASED.value,
            WorkItemStatus.RUNNING.value,
        }:
            # In-flight: a run_loop / goal driver owns it. Excluding it from intake
            # keeps the daemon cycle from double-driving the same iteration; a crash
            # leaves it 'orphaned' (rehydrated, re-intakeable). (CONCEPT:AU-KG.research.these-properties-carry)
            continue
        if kind == "research" and cid in addressed:
            # Addressed evidence is the research completion condition. Settle
            # the WorkItem before dropping the Concept from intake.
            claim = claim_loop_work_item(engine, cid)
            if claim is not None:
                transition_loop_work_item(
                    engine,
                    cid,
                    "completed",
                    result_ref=f"loop:{cid}:addressed",
                )
            continue
        out.append({**_loop_dict(cid, r), "status": status})
    # Priority-ordered intake is normalized once after the bounded native read,
    # so every query backend shares the same integer bucket semantics.
    out.sort(key=lambda d: _prio_bucket(d.get("prio_bucket")))
    return out[:limit]


__all__ = [
    "LoopKind",
    "LoopStatus",
    "TERMINAL_STATUS",
    "TERMINAL_STATUSES",
    "is_success",
    "is_terminal",
    "to_status",
    "submit_loop",
    "active_loops",
    "mark_loop_status",
    "prioritize_loop",
    "claim_loop",
]
