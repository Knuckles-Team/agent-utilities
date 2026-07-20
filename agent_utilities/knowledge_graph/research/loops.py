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
from typing import Any, Literal

logger = logging.getLogger(__name__)

LoopKind = Literal["research", "develop", "skill"]

#: statuses from which a Loop never needs more work.
TERMINAL_STATUS = frozenset({"completed", "failed", "cancelled", "rejected"})


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

    result_ref = (
        f"loop:{loop_id}:completed" if status in {"completed", "succeeded"} else None
    )
    error_ref = (
        f"loop:{loop_id}:{status}"
        if status in {"failed", "rejected", "cancelled"}
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
    "TERMINAL_STATUS",
    "submit_loop",
    "active_loops",
    "mark_loop_status",
    "prioritize_loop",
    "claim_loop",
]
