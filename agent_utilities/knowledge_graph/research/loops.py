#!/usr/bin/python
from __future__ import annotations

"""The Loop — a long-running objective the loop engine advances (CONCEPT:KG-2.78).

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

A Loop is stored as a ``Concept`` node (so the existing graph-compute middle —
dedup / gap / synergy / ConceptMatcher — applies unchanged) carrying ``loop_kind``,
``status``, ``objective`` and the kind-specific completion fields.

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
) -> dict[str, Any] | None:
    """Materialize a long-running objective as a Loop node (CONCEPT:KG-2.78).

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
        "status": "pending",
        "source": source,
        "max_iterations": int(max_iterations),
        "timestamp": _now_iso(),
    }
    if end_state:
        props["end_state"] = end_state
    if validation_cmd:
        props["validation_cmd"] = validation_cmd
    if skill_ref:
        props["skill_ref"] = skill_ref
    try:
        engine.add_node(oid, "Concept", properties=props)
    except Exception as e:  # noqa: BLE001 — best-effort persist
        logger.debug("submit_loop persist failed: %s", e)
        return None
    return _loop_dict(oid, props)


def _loop_dict(oid: str, data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": oid,
        "name": data.get("name") or data.get("objective") or oid,
        "kind": data.get("loop_kind") or "research",
    }
    for k in ("objective", "end_state", "validation_cmd", "skill_ref", "status"):
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
    """Advance a Loop's lifecycle state (CONCEPT:KG-2.78).

    The single shared status-transition path for every kind — the controller's
    develop/skill stages and the ``graph_loops(action="cancel")`` entrypoint all
    call this. Upserts the status (+ optional iteration / last output) onto the
    Loop node; best-effort (a failed persist returns ``False``, never raises).
    """
    props: dict[str, Any] = {
        "status": status,
        "timestamp": _now_iso(),
        "last_source": source,
    }
    if iteration is not None:
        props["iteration"] = int(iteration)
    if output:
        props["last_output"] = output[:2000]
    try:
        engine.add_node(loop_id, "Concept", properties=props)
        return True
    except Exception as e:  # noqa: BLE001 — best-effort
        logger.debug("mark_loop_status persist failed: %s", e)
        return False


def active_loops(engine: Any, limit: int = 10) -> list[dict[str, Any]]:
    """Every Loop still needing work — the LoopController's intake (CONCEPT:KG-2.78).

    Generalizes ``unresolved_topics``: a Loop is *active* when

    - it is a ``research`` Loop (or a legacy bare Concept topic / failure_gap) with
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
            "c.loop_kind AS loop_kind, c.status AS status, "
            "c.objective AS objective, c.validation_cmd AS validation_cmd, "
            "c.skill_ref AS skill_ref, c.end_state AS end_state LIMIT $limit",
            {"limit": int(limit) * 20},
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("active_loops: concept query failed: %s", e)
        return []

    out: list[dict[str, Any]] = []
    for r in rows or []:
        if not isinstance(r, dict) or not r.get("id"):
            continue
        cid = r["id"]
        kind = (r.get("loop_kind") or "research") or "research"
        status = (r.get("status") or "").lower()
        if status in TERMINAL_STATUS:
            continue
        if kind in ("develop", "skill") and status == "running":
            # In-flight: a run_loop / goal driver owns it. Excluding it from intake
            # keeps the daemon cycle from double-driving the same iteration; a crash
            # leaves it 'orphaned' (rehydrated, re-intakeable). (CONCEPT:KG-2.78)
            continue
        if kind == "research" and cid in addressed:
            continue  # research loop already addressed → resolved
        out.append(_loop_dict(cid, r))
        if len(out) >= limit:
            break
    return out


__all__ = [
    "LoopKind",
    "TERMINAL_STATUS",
    "submit_loop",
    "active_loops",
    "mark_loop_status",
]
