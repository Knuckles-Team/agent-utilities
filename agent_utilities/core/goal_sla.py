#!/usr/bin/python
from __future__ import annotations

"""Goals-as-contracts: SLA evaluation + escalation (CONCEPT:ORCH-1.78).

Turns a goal from "a loop that runs" into a durable *contract*: an objective with a
deadline (``sla_seconds``) and an escalation target. A maintenance tick
(:func:`evaluate_goal_slas`, wired into the scheduler) checks every open goal
against its SLA, and on breach records the outcome (AHE-3.62) and escalates to the
owner — closing sense→act→**verify**→learn for goals so "triage every P1 within 1h"
is enforced, not just attempted.

``assess_goal_sla`` is a pure predicate (unit-tested without an engine);
``evaluate_goal_slas`` reads the live open goals and acts, best-effort, never raising.
"""

import logging
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

#: Fraction of the SLA window past which a still-open goal is "at risk".
AT_RISK_FRACTION = 0.8
_OPEN_STATUSES = ("pending", "running", "validating", "paused")


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    # Goal nodes store created_at as a unix float (time.time()); accept that too.
    if isinstance(value, int | float):
        try:
            return datetime.fromtimestamp(float(value), tz=UTC)
        except (ValueError, OverflowError, OSError):
            return None
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return ts if ts.tzinfo else ts.replace(tzinfo=UTC)
    except (ValueError, TypeError):
        return None


def assess_goal_sla(
    *,
    created_at: Any,
    sla_seconds: float | int | None,
    status: str = "running",
    now: datetime | None = None,
) -> dict[str, Any]:
    """Classify a goal against its SLA: on_track | at_risk | breached | no_sla.

    Returns ``{state, age_seconds, sla_seconds, breach}``. A goal with no
    ``sla_seconds`` (or a terminal/unknown status) is ``no_sla`` (never escalated).
    """
    now = now or datetime.now(UTC)
    start = _parse_ts(created_at)
    try:
        sla = float(sla_seconds) if sla_seconds else 0.0
    except (TypeError, ValueError):
        sla = 0.0
    if not sla or start is None:
        return {
            "state": "no_sla",
            "age_seconds": None,
            "sla_seconds": sla,
            "breach": False,
        }
    age = (now - start).total_seconds()
    if age >= sla:
        state = "breached"
    elif age >= sla * AT_RISK_FRACTION:
        state = "at_risk"
    else:
        state = "on_track"
    return {
        "state": state,
        "age_seconds": round(age, 1),
        "sla_seconds": sla,
        "breach": state == "breached",
    }


def _open_goals(engine: Any) -> list[dict[str, Any]]:
    from agent_utilities.knowledge_graph.retrieval.context_plane import read_rows

    statuses = list(_OPEN_STATUSES)
    return read_rows(
        engine,
        "MATCH (g:Concept) WHERE g.loop_kind = 'develop' AND g.status IN $st "
        "RETURN g.id AS id, g.objective AS objective, g.status AS status, "
        "g.created_at AS created_at, g.sla_seconds AS sla_seconds, "
        "g.escalate_to AS escalate_to",
        {"st": statuses},
    )


def _escalate(engine: Any, goal: dict[str, Any], verdict: dict[str, Any]) -> None:
    """Best-effort breach escalation: record the outcome + notify the owner."""
    goal_id = goal.get("id", "")
    # 1. Record a (failed) action outcome so the goal's reward-EMA reflects the miss.
    try:
        from agent_utilities.knowledge_graph.adaptation.feedback import FeedbackService

        FeedbackService.from_engine(engine).record_action_outcome(
            f"goal:{goal_id}",
            success=False,
            reason=f"SLA breach: open {verdict['age_seconds']}s > {verdict['sla_seconds']}s",
        )
    except Exception as exc:  # pragma: no cover - best-effort
        logger.debug("goal SLA outcome failed: %s", exc)
    # 2. Notify the goal's owner / operators.
    target = goal.get("escalate_to") or ""
    msg = (
        f"⏰ Goal SLA breach: '{goal.get('objective', goal_id)}' has been open "
        f"{int(verdict['age_seconds'])}s (SLA {int(verdict['sla_seconds'])}s)."
    )
    try:
        from agent_utilities.messaging.service import MessagingService

        svc = MessagingService.instance(engine)
        sync = getattr(svc, "reach_user_sync", None)
        if callable(sync):
            sync(msg, user_id=target or None, source="goal_sla", reason="sla_breach")
        else:  # pragma: no cover - messaging shape varies
            logger.info("[ORCH-1.78] %s", msg)
    except Exception as exc:  # pragma: no cover - messaging optional
        logger.info("[ORCH-1.78] %s (notify unavailable: %s)", msg, exc)


def evaluate_goal_slas(engine: Any, *, now: datetime | None = None) -> dict[str, Any]:
    """Check every open goal against its SLA; escalate breaches (best-effort)."""
    now = now or datetime.now(UTC)
    breached: list[dict[str, Any]] = []
    at_risk: list[dict[str, Any]] = []
    goals = _open_goals(engine)
    for g in goals:
        verdict = assess_goal_sla(
            created_at=g.get("created_at"),
            sla_seconds=g.get("sla_seconds"),
            status=str(g.get("status") or "running"),
            now=now,
        )
        if verdict["state"] == "breached":
            breached.append({"goal": g.get("id"), **verdict})
            _escalate(engine, g, verdict)
        elif verdict["state"] == "at_risk":
            at_risk.append({"goal": g.get("id"), **verdict})
    return {"checked": len(goals), "breached": breached, "at_risk": at_risk}
