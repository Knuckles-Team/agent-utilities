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


# ── eg-statechart definition (W2.5 control-plane migration) ──────────────────
#
# Wires the Loop lifecycle onto a real, engine-native ``eg-statechart`` state
# machine (via the existing ``Method::Statechart`` wire surface: define /
# instantiate / send_event / get_state / list — no protocol change) instead of
# ``run_loop``'s ad-hoc Python if/elif chain. See
# ``reports/w2_5-statechart-migration-design.md`` §1 for the full derivation;
# this is a direct transcription of that table WITH three corrections applied
# that the Rust side (``epistemic-graph`` ``src/loop_statechart.rs``, commit
# ``4bcf07a`` on ``impl/w25-statechart``) already carries and this def must
# agree with:
#
# 1. The legacy-trust ``callee_terminal`` mirror (design doc row 13) is 9
#    concrete transitions — every final ``LoopStatus`` EXCEPT ``completed``
#    (which gets its own dedicated transition, row 13 below) — not 5.
# 2. That mirror fires from ALL THREE active states (``running``/``pending``/
#    ``validating``), not ``running`` only: ``run_loop``'s actual legacy-trust
#    check runs unconditionally on the freshly computed ``decided`` value
#    every iteration, regardless of which active state ``status`` was in
#    going into that iteration.
# 3. The error-threshold (row 10) and turn-cap (row 15) guards read plain
#    booleans the CALLER precomputes and sends in the event payload
#    (``error_threshold_tripped``, ``turn_cap_reached``) rather than numeric
#    ``Guard::Ge`` comparisons over event fields — ``eg-statechart``'s
#    ``Ge``/``Gt``/``Lt``/``Le`` guards read persistent machine CONTEXT, never
#    the event payload (only ``EventEq`` reads the payload), and this chart's
#    context is always empty (every signal is computed fresh by ``run_loop``
#    each tick and passed as the event payload).
#
# Two more transitions exist ONLY so every declared state is reachable
# (``eg_statechart::check::validate`` rejects an unreachable state): the
# ``heartbeat_target`` guard (row 17, makes ``pending``/``validating``
# reachable) and the ``lease_lost`` event (row 18, makes ``orphaned``
# reachable — mirrors WorkItem's ``lease_reclaim``, fired by a future
# lease-expiry reaper; ``run_loop`` itself does not send it today).

#: The three non-final states ``run_loop`` actually iterates in.
ACTIVE_LOOP_STATES: tuple[str, ...] = (
    LoopStatus.RUNNING.value,
    LoopStatus.PENDING.value,
    LoopStatus.VALIDATING.value,
)

#: All ten final ``LoopStatus`` values, in a fixed, reviewable order.
_FINAL_LOOP_STATES: tuple[str, ...] = (
    LoopStatus.COMPLETED.value,
    LoopStatus.FAILED.value,
    LoopStatus.CANCELLED.value,
    LoopStatus.REJECTED.value,
    LoopStatus.MAX_ITERATIONS_EXCEEDED.value,
    LoopStatus.BUDGET_EXCEEDED.value,
    LoopStatus.WALL_CLOCK_EXCEEDED.value,
    LoopStatus.STALLED.value,
    LoopStatus.ERROR_THRESHOLD_EXCEEDED.value,
    LoopStatus.EXTERNAL_EVENT_SATISFIED.value,
)
#: The 9 finals a callee can self-declare and have legacy-trusted verbatim
#: (every final except ``completed``, which is its own dedicated transition).
_CALLEE_TERMINAL_MIRROR_STATES: tuple[str, ...] = tuple(
    s for s in _FINAL_LOOP_STATES if s != LoopStatus.COMPLETED.value
)


def _always() -> dict[str, Any]:
    return {"op": "always"}


def _event_eq(key: str, value: Any) -> dict[str, Any]:
    return {"op": "event_eq", "key": key, "value": value}


def _all_guards(*guards: dict[str, Any]) -> dict[str, Any]:
    return {"op": "all", "guards": list(guards)}


def _any_guards(*guards: dict[str, Any]) -> dict[str, Any]:
    return {"op": "any", "guards": list(guards)}


def _build_loop_statechart_def() -> dict[str, Any]:
    """Author the Loop lifecycle as a plain dict shaped exactly like Rust's
    ``eg_statechart::model::StatechartDef`` (fields: ``name``,
    ``schema_version``, ``states``, ``alphabet``, ``transitions``, ``initial``,
    ``finals``, ``meta``; guards externally-tagged on ``op``, matching Rust's
    ``#[serde(tag="op")]``)."""
    states = [
        {"id": s}
        for s in (
            LoopStatus.SUBMITTED.value,
            LoopStatus.RUNNING.value,
            LoopStatus.PENDING.value,
            LoopStatus.VALIDATING.value,
            LoopStatus.PAUSED.value,
            LoopStatus.ORPHANED.value,
            *_FINAL_LOOP_STATES,
        )
    ]

    transitions: list[dict[str, Any]] = []

    # 1 / 1b: claim -> running (fresh submit, or re-claim after an orphaned lease)
    transitions.append(
        {"from": LoopStatus.SUBMITTED.value, "event": "claim", "guard": _always(), "to": LoopStatus.RUNNING.value}
    )
    transitions.append(
        {"from": LoopStatus.ORPHANED.value, "event": "claim", "guard": _always(), "to": LoopStatus.RUNNING.value}
    )
    # 2: intake refusal
    transitions.append(
        {"from": LoopStatus.SUBMITTED.value, "event": "reject", "guard": _always(), "to": LoopStatus.REJECTED.value}
    )

    # 3: pretick human interrupt — pause
    for frm in ACTIVE_LOOP_STATES:
        transitions.append(
            {
                "from": frm,
                "event": "pretick",
                "guard": _event_eq("human_signal", "pause"),
                "to": LoopStatus.PAUSED.value,
                "actions": [{"do": "log", "message": "human interrupt: pause"}],
            }
        )
    # 4: pretick human interrupt — kill/cancel/stop
    for frm in ACTIVE_LOOP_STATES:
        transitions.append(
            {
                "from": frm,
                "event": "pretick",
                "guard": _any_guards(
                    _event_eq("human_signal", "kill"),
                    _event_eq("human_signal", "cancel"),
                    _event_eq("human_signal", "stop"),
                ),
                "to": LoopStatus.CANCELLED.value,
                "actions": [{"do": "log", "message": "human interrupt: kill"}],
            }
        )
    # 5: pretick budget cap
    for frm in ACTIVE_LOOP_STATES:
        transitions.append(
            {
                "from": frm,
                "event": "pretick",
                "guard": _event_eq("budget_exceeded", True),
                "to": LoopStatus.BUDGET_EXCEEDED.value,
            }
        )
    # 6: pretick external event
    for frm in ACTIVE_LOOP_STATES:
        transitions.append(
            {
                "from": frm,
                "event": "pretick",
                "guard": _event_eq("external_event_fired", True),
                "to": LoopStatus.EXTERNAL_EVENT_SATISFIED.value,
            }
        )
    # 7: operator resume
    transitions.append(
        {"from": LoopStatus.PAUSED.value, "event": "resume", "guard": _always(), "to": LoopStatus.RUNNING.value}
    )
    # 8: posttick goal met
    for frm in ACTIVE_LOOP_STATES:
        transitions.append(
            {
                "from": frm,
                "event": "posttick",
                "guard": _event_eq("measured_pass", True),
                "to": LoopStatus.COMPLETED.value,
            }
        )
    # 9: posttick error threshold (precomputed boolean — correction #3)
    for frm in ACTIVE_LOOP_STATES:
        transitions.append(
            {
                "from": frm,
                "event": "posttick",
                "guard": _all_guards(
                    _event_eq("retryable_failure", True),
                    _event_eq("error_threshold_tripped", True),
                ),
                "to": LoopStatus.ERROR_THRESHOLD_EXCEEDED.value,
            }
        )
    # 10: posttick stalled
    for frm in ACTIVE_LOOP_STATES:
        transitions.append(
            {
                "from": frm,
                "event": "posttick",
                "guard": _event_eq("stalled", True),
                "to": LoopStatus.STALLED.value,
            }
        )
    # 11: posttick retryable-failure self-loop heartbeat (only reached if #9 didn't fire)
    for frm in ACTIVE_LOOP_STATES:
        transitions.append(
            {
                "from": frm,
                "event": "posttick",
                "guard": _event_eq("retryable_failure", True),
                "to": frm,
            }
        )
    # 12: posttick legacy-trust — callee_terminal == "completed"
    for frm in ACTIVE_LOOP_STATES:
        transitions.append(
            {
                "from": frm,
                "event": "posttick",
                "guard": _event_eq("callee_terminal", LoopStatus.COMPLETED.value),
                "to": LoopStatus.COMPLETED.value,
            }
        )
    # 13: posttick legacy-trust — callee_terminal == <every other final> (9 x 3 = 27 rows)
    for frm in ACTIVE_LOOP_STATES:
        for terminal in _CALLEE_TERMINAL_MIRROR_STATES:
            transitions.append(
                {
                    "from": frm,
                    "event": "posttick",
                    "guard": _event_eq("callee_terminal", terminal),
                    "to": terminal,
                }
            )
    # 14: posttick turn cap (precomputed boolean — correction #3)
    for frm in ACTIVE_LOOP_STATES:
        transitions.append(
            {
                "from": frm,
                "event": "posttick",
                "guard": _event_eq("turn_cap_reached", True),
                "to": LoopStatus.MAX_ITERATIONS_EXCEEDED.value,
            }
        )
    # 15: posttick wall clock
    for frm in ACTIVE_LOOP_STATES:
        transitions.append(
            {
                "from": frm,
                "event": "posttick",
                "guard": _event_eq("deadline_passed", True),
                "to": LoopStatus.WALL_CLOCK_EXCEEDED.value,
            }
        )
    # 16: posttick ordinary heartbeat continuation — declared LAST (only fires
    # once nothing terminal/self-loop above it did).
    for frm in ACTIVE_LOOP_STATES:
        for target in ACTIVE_LOOP_STATES:
            transitions.append(
                {
                    "from": frm,
                    "event": "posttick",
                    "guard": _event_eq("heartbeat_target", target),
                    "to": target,
                }
            )
    # 17: internal lease-loss -> orphaned (not sent by run_loop today; makes
    # ``orphaned`` reachable, mirrors a future lease-expiry reaper).
    for frm in ACTIVE_LOOP_STATES:
        transitions.append(
            {
                "from": frm,
                "event": "lease_lost",
                "guard": _always(),
                "to": LoopStatus.ORPHANED.value,
            }
        )

    return {
        "name": "loop_lifecycle",
        "schema_version": "1",
        "states": states,
        "alphabet": ["claim", "pretick", "resume", "posttick", "reject", "lease_lost"],
        "transitions": transitions,
        "initial": LoopStatus.SUBMITTED.value,
        "finals": list(_FINAL_LOOP_STATES),
        "meta": {
            "concept": "AU-AHE.harness.loop-exit-conditions",
            "source": "agent_utilities.knowledge_graph.research.loops.LOOP_STATECHART_DEF",
        },
    }


#: The Loop lifecycle chart — a module-level constant, safe to build at import
#: time (pure data, no engine call). ``loop_def_id`` registers it with a live
#: engine lazily (see below), never at import.
LOOP_STATECHART_DEF: dict[str, Any] = _build_loop_statechart_def()


def loop_def_id(engine: Any) -> str:
    """Register the Loop lifecycle chart, returning its content-addressed id.

    ``engine.statechart.define`` is content-addressed/idempotent on the
    server, so calling it defensively every time this is needed is correct
    (just an extra round trip) — no caching is attempted here (W2.5 design
    guidance: start simple; add caching later only if it proves trivial and
    safe).
    """
    return engine.statechart.define(LOOP_STATECHART_DEF)


def ensure_loop_statechart_instance(engine: Any, loop_id: str) -> str | None:
    """Resolve (creating if needed) the Loop's ``eg-statechart`` instance id.

    ``Method::Statechart``'s ``Instantiate`` op server-generates the
    ``instance_id`` (unlike ``submit_work_item``'s caller-supplied
    ``work_item_id``) — extending that wire op to accept a caller-supplied id
    is out of scope for W2.5. So the server-returned id is stored back onto
    the Loop's backing WorkItem (``loop_statechart_instance_id``) right after
    instantiation, and read back from there on every subsequent call — a
    disclosed, reasonable deviation from the design doc's deterministic-
    ``instance_id`` proposal. Returns ``None`` if the backing WorkItem doesn't
    exist yet or the instantiate call didn't return an id.
    """
    from agent_utilities.orchestration import work_item as _wi

    item_id = _wi.loop_work_item_id(loop_id)
    item = _wi.get_work_item(engine, item_id)
    if item is None:
        return None
    existing = item.get("loop_statechart_instance_id")
    if existing:
        return str(existing)
    def_id = loop_def_id(engine)
    result = engine.statechart.instantiate(def_id, context={})
    instance_id = result.get("instance_id") if isinstance(result, dict) else None
    if not instance_id:
        return None
    _wi.set_loop_statechart_instance_id(engine, item_id, str(instance_id))
    return str(instance_id)


def send_loop_statechart_event(
    engine: Any, loop_id: str, event: str, payload: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """Send one event to the Loop's ``eg-statechart`` instance.

    Resolves (creating if needed) the instance first. Returns ``None`` only
    when the instance itself can't be resolved (e.g. no backing WorkItem) —
    the WorkItem stays the hard lease/fencing authority (§3.1, unchanged); the
    statechart adds a durable "why" on top of it.
    """
    instance_id = ensure_loop_statechart_instance(engine, loop_id)
    if not instance_id:
        return None
    return engine.statechart.send_event(instance_id, event, payload=payload)


def statechart_active_state(result: dict[str, Any]) -> str:
    """Extract the single active-state id from a ``send_event``/``get_state``
    response (the Loop chart has no parallel regions, so exactly one)."""
    return result["instance"]["configuration"]["active"][0]


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
    # The Loop's durable "why" state machine (W2.5): register the chart and
    # instantiate this Loop's instance, stored back on its backing WorkItem.
    ensure_loop_statechart_instance(engine, oid)
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
    the engine-issued lease. On a successful claim, additionally advances the
    Loop's ``eg-statechart`` instance with a ``claim`` event (W2.5) — a no-op
    if the instance is already past ``submitted``/``orphaned`` (e.g. a
    resumed driver re-claiming an already-``running`` instance).
    """
    from agent_utilities.orchestration.work_item import claim_loop_work_item

    won = claim_loop_work_item(engine, loop_id) is not None
    if won:
        send_loop_statechart_event(engine, loop_id, "claim")
    return won


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
    "LOOP_STATECHART_DEF",
    "ACTIVE_LOOP_STATES",
    "loop_def_id",
    "ensure_loop_statechart_instance",
    "send_loop_statechart_event",
    "statechart_active_state",
]
