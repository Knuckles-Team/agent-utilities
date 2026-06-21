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

#: Loop statuses from which a develop/skill Loop may be CLAIMED for advancement.
#: ``orphaned`` is a Loop whose previous driver crashed mid-run (the durable
#: rehydration marks it so) — re-claimable. ``""``/missing reads as a fresh
#: pending Loop. A ``running`` Loop is owned by a live driver and is NOT
#: claimable. (CONCEPT:KG-2.141)
CLAIMABLE_STATUS = frozenset({"", "pending", "orphaned"})


def _prio_bucket(value: Any, default: int = 2) -> int:
    """Normalize a priority spec to the ONE 0..3 claim bucket (CONCEPT:KG-2.113).

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
        # Claim/intake priority bucket (0=critical .. 3=background); active_loops
        # emits in ascending-bucket order so a hot loop is advanced first, and a
        # loop-spawned child task inherits this. Coerced through the ONE shared
        # normalizer so a loop bucket is the same 0..3 value as a task's.
        # (CONCEPT:KG-2.113)
        "prio_bucket": _prio_bucket(prio_bucket),
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
        # Normalized through the ONE shared bucket normalizer (default 2); it
        # preserves bucket 0 (critical, which is falsy) and maps any legacy
        # ``priority`` string a Concept might carry. (CONCEPT:KG-2.113)
        "prio_bucket": _prio_bucket(data.get("prio_bucket")),
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


def prioritize_loop(engine: Any, loop_id: str, prio_bucket: int) -> bool:
    """Set a Loop's intake/claim priority bucket (CONCEPT:KG-2.113).

    ``active_loops`` emits loops in ascending-bucket order, so bumping a loop to
    bucket 0/1 advances it ahead of background loops on the next cycle.
    Best-effort: a failed persist returns ``False``, never raises.
    """
    try:
        engine.add_node(
            loop_id, "Concept", properties={"prio_bucket": _prio_bucket(prio_bucket)}
        )
        return True
    except Exception as e:  # noqa: BLE001 — best-effort
        logger.debug("prioritize_loop persist failed: %s", e)
        return False


def claim_loop(engine: Any, loop_id: str, *, current_status: str = "") -> bool:
    """Atomically claim a develop/skill Loop for advancement (CONCEPT:KG-2.141).

    The single cross-host-safe Loop claim, mirroring the engine ``:Task`` claim
    (``_claim_next_task``): flip the Loop's ``status`` from a *claimable* state
    (``pending``/``orphaned``/unset) to ``running`` via the engine's
    compare-and-set, which holds the graph write lock for the flip. Returns
    ``True`` only if THIS caller won the flip; ``False`` means a concurrent
    driver (another daemon cycle, a ``graph_loops`` run, or a peer host) already
    claimed it — the caller must then skip the Loop instead of double-driving it.

    Replaces the former non-atomic "claim" (a blind ``mark_loop_status(...,
    'running')`` after an intake-time status read), which had a TOCTOU window:
    two cycles could both intake the same pending Loop and both flip it to
    running. ``intake`` (``active_loops``) still pre-filters running loops; this
    CAS is the authoritative arbiter that closes the race.

    The CAS conditions a *single* expected status (the engine primitive matches
    equality, ``missing ≡ null``). We try the caller's observed status first
    (typically from intake), then the other claimable states, so a Loop is
    claimable whether it was seen as ``pending``, freshly created (unset), or
    left ``orphaned`` by a crashed driver. Best-effort: if the backend lacks the
    CAS primitive (older engine) we fall back to the legacy blind flip so the
    single-host path keeps working.
    """
    backend = getattr(engine, "backend", None)
    cas = getattr(backend, "compare_and_set_node_fields", None)
    if not callable(cas):
        # Older engine without the CAS primitive: preserve the single-host
        # behavior (blind flip). No cross-host guarantee, but no regression.
        return mark_loop_status(engine, loop_id, "running")

    # Try the caller's observed status first (the common, cheap win), then the
    # remaining claimable states. Each is ONE equality CAS; ``""`` matches a
    # node with no ``status`` field (the engine reads missing as null).
    candidates: list[str] = []
    seen = (current_status or "").lower()
    if seen in CLAIMABLE_STATUS:
        candidates.append(seen)
    candidates.extend(s for s in CLAIMABLE_STATUS if s != seen)

    updates = {
        "status": "running",
        "timestamp": _now_iso(),
        "last_source": "loop_engine",
    }
    for status in candidates:
        try:
            if cas(loop_id, {"status": status}, updates):
                return True
        except Exception as e:  # noqa: BLE001 — one failed CAS never blocks the rest
            logger.debug(
                "claim_loop CAS (%s→running) failed for %s: %s", status, loop_id, e
            )
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
            "c.skill_ref AS skill_ref, c.end_state AS end_state, "
            "c.prio_bucket AS prio_bucket LIMIT $limit",
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
    # Priority-ordered intake: the L1 interpreter strips ORDER BY, so we sort the
    # already-fetched candidate set in-memory by claim bucket (0 first) before
    # the limit cutoff — a hot loop is advanced ahead of background ones. Each
    # dict's ``prio_bucket`` is already normalized by ``_loop_dict``.
    out.sort(key=lambda d: _prio_bucket(d.get("prio_bucket")))
    return out[:limit]


__all__ = [
    "LoopKind",
    "TERMINAL_STATUS",
    "CLAIMABLE_STATUS",
    "submit_loop",
    "active_loops",
    "mark_loop_status",
    "prioritize_loop",
    "claim_loop",
]
