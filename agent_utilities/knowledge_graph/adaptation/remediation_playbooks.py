#!/usr/bin/python
from __future__ import annotations

"""Remediation playbooks on the fleet-event triage seam.

CONCEPT:OS-5.26 — Remediation playbooks with stepwise verification and policy-gated actuation plugged into the fleet-event triage seam, recording every step on the originating FleetEvent node

OS-5.15 shipped the dispatch seam ("full remediation playbooks are a later
task"); this module is that task. Three playbooks, each a small explicit step
list — observe → confirm → policy gate → actuate → verify → escalate — with
every step outcome recorded on the originating ``FleetEvent`` node
(``remediation_log`` JSON trail + ``remediation_status``):

* **service_down** — confirm via the FleetObserver (an already-recovered
  service ends the playbook), then a ``restart_service`` proposal through the
  ActionPolicy gate (CONCEPT:OS-5.24). Allowed ⇒ actuate + schedule an
  OS-5.27 health watch (which itself escalates/rolls back on failure);
  queued/denied ⇒ escalate to the approvals flow + operator notification.
* **service_flapping** — ≥ ``FLAP_THRESHOLD`` down-events inside the observer
  window means restarting again is just feeding the flap: back off (no
  actuation) and escalate a restart *proposal* for a human.
* **resource_pressure** — disk/memory/CPU pressure events NEVER auto-act:
  notify + queue an ``investigate_resource_pressure`` proposal.

Severity *warning/info* events keep the OS-5.15 default playbook (correlate +
failure-gap escalation), which every remediation playbook also runs first so
the golden-loop intake still sees recurring incidents.

Registration is idempotent via :func:`ensure_registered`, called by the
engine's ``fleet_event_triage`` task dispatch — so wherever triage runs, the
playbooks are live.
"""

import json
import logging
import time
from typing import Any

from .fleet_event_triage import default_playbook, register_playbook

logger = logging.getLogger(__name__)

# A subject with this many down-events inside the observer window is flapping.
FLAP_THRESHOLD = 3

# Substrings that classify an event as resource pressure, not service-down.
_PRESSURE_MARKERS = (
    "disk",
    "filesystem",
    "memory",
    "oom",
    "cpu",
    "load average",
    "inode",
    "swap",
    "volume full",
    "no space",
)

# Sources whose critical/error events get the remediation dispatcher.
_SOURCES = ("alertmanager", "uptime-kuma", "portainer", "generic")
_SEVERITIES = ("critical", "error")

_registered = False


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _record_step(
    engine: Any, event: dict[str, Any], step: str, outcome: str, **extra: Any
) -> dict[str, Any]:
    """Append one step outcome to the FleetEvent's ``remediation_log`` trail."""
    entry = {"step": step, "outcome": outcome, "at": _now_iso(), **extra}
    event.setdefault("_remediation_log", []).append(entry)
    event_id = event.get("id")
    if event_id is not None and engine is not None:
        try:
            engine.backend.execute(
                "MATCH (e:FleetEvent {id: $id}) "
                "SET e.remediation_log = $log, e.remediation_status = $status",
                {
                    "id": event_id,
                    "log": json.dumps(event["_remediation_log"], default=str)[:4000],
                    "status": f"{step}:{outcome}",
                },
            )
        except Exception as e:  # noqa: BLE001 — the trail is best-effort
            logger.debug("remediation step record failed: %s", e)
    return entry


def _notify(message: str) -> None:
    try:
        from agent_utilities.knowledge_graph.actions.dispatch import send_notification
        from agent_utilities.knowledge_graph.actions.models import NotificationSpec

        send_notification(
            NotificationSpec(channel="fleet", recipient="operators"), message
        )
    except Exception as e:  # noqa: BLE001
        logger.info("remediation notification (fallback log): %s (%s)", message, e)


def _escalate(
    engine: Any,
    event: dict[str, Any],
    request: Any,
    reason: str,
) -> dict[str, Any]:
    """Escalation = approval-queue entry + operator notification."""
    from agent_utilities.orchestration.action_policy import get_action_policy

    approval_id = get_action_policy(engine).queue_approval(request, reason=reason)
    _notify(
        f"[fleet-remediation] escalation for {event.get('subject')}: {reason} "
        f"(approval={approval_id or 'unavailable'})"
    )
    return _record_step(
        engine, event, "escalate", "queued", reason=reason, approval_id=approval_id
    )


def _classify(event: dict[str, Any], flap_count: int) -> str:
    summary = str(event.get("summary") or "").lower()
    if any(marker in summary for marker in _PRESSURE_MARKERS):
        return "resource_pressure"
    if flap_count >= FLAP_THRESHOLD:
        return "service_flapping"
    return "service_down"


def service_down_playbook(
    engine: Any, event: dict[str, Any], observation: Any
) -> dict[str, Any]:
    """observe → confirm → policy gate → restart → schedule verification."""
    from agent_utilities.orchestration.action_policy import (
        ActionRequest,
        get_action_policy,
    )
    from agent_utilities.orchestration.fleet_actuation import execute_action

    subject = str(event.get("subject") or "unknown")
    out: dict[str, Any] = {"playbook": "service_down", "subject": subject}

    # Confirm: monitoring said down, does the observer still agree?
    if observation.status == "up":
        _record_step(engine, event, "confirm", "recovered", detail=observation.detail)
        out["resolution"] = "already_recovered"
        return out
    _record_step(
        engine,
        event,
        "confirm",
        "still_down",
        detail=observation.detail or "no counter-evidence",
    )

    request = ActionRequest(
        kind="restart_service",
        target=subject,
        source="playbook",
        reason=f"fleet event {event.get('id')}: {str(event.get('summary'))[:120]}",
    )
    decision = get_action_policy(engine).decide(request)
    _record_step(
        engine, event, "policy", decision.decision, approval_id=decision.approval_id
    )
    out["decision"] = decision.decision

    if not decision.allowed:
        if decision.decision == "deny":
            _escalate(engine, event, request, f"restart denied: {decision.reason}")
        else:
            _notify(
                f"[fleet-remediation] restart of {subject} awaits approval "
                f"({decision.approval_id})"
            )
        return out

    execution = execute_action(engine, request)
    _record_step(
        engine,
        event,
        "actuate",
        "ok" if execution.get("ok") else "failed",
        detail=str(execution.get("detail", ""))[:200],
        dry_run=bool(execution.get("dry_run")),
    )
    out["execution"] = execution
    if not execution.get("ok"):
        _escalate(engine, event, request, "restart actuation failed")
        return out

    # Verify within a timeout: the durable OS-5.27 watch probes the observer
    # and escalates (policy-gated rollback + notification) on failure.
    from agent_utilities.orchestration.deploy_watch import watch_deploy

    watch_job = watch_deploy(engine, subject, source="playbook")
    _record_step(engine, event, "verify", "scheduled", watch_job=watch_job)
    out["watch_job"] = watch_job
    return out


def service_flapping_playbook(
    engine: Any, event: dict[str, Any], observation: Any
) -> dict[str, Any]:
    """Back off — another restart feeds the flap — and hand a human the call."""
    from agent_utilities.orchestration.action_policy import ActionRequest

    subject = str(event.get("subject") or "unknown")
    _record_step(
        engine, event, "confirm", "flapping", flap_count=observation.flap_count
    )
    request = ActionRequest(
        kind="restart_service",
        target=subject,
        params={"flapping": True},
        source="playbook",
        reason=f"flapping ({observation.flap_count} down-events in window)",
    )
    _escalate(
        engine,
        event,
        request,
        f"{subject} is flapping ({observation.flap_count} down-events) — "
        "backing off, human decision required",
    )
    return {
        "playbook": "service_flapping",
        "subject": subject,
        "flap_count": observation.flap_count,
        "resolution": "backed_off",
    }


def resource_pressure_playbook(
    engine: Any, event: dict[str, Any], observation: Any
) -> dict[str, Any]:
    """Notify + propose; resource pressure is NEVER auto-acted on."""
    from agent_utilities.orchestration.action_policy import ActionRequest

    subject = str(event.get("subject") or "unknown")
    _record_step(engine, event, "classify", "resource_pressure")
    _escalate(
        engine,
        event,
        ActionRequest(
            kind="investigate_resource_pressure",
            target=subject,
            source="playbook",
            reason=str(event.get("summary") or "")[:200],
        ),
        f"resource pressure on {subject} — proposal only, no autonomous action",
    )
    return {
        "playbook": "resource_pressure",
        "subject": subject,
        "resolution": "proposed_only",
    }


def remediation_playbook(engine: Any, event: dict[str, Any]) -> dict[str, Any]:
    """Dispatcher for critical/error events: classify, then run the playbook.

    Runs the OS-5.15 default playbook first so correlation + failure-gap
    escalation are preserved, then layers the remediation steps on top.
    """
    base = default_playbook(engine, event) or {}

    from agent_utilities.orchestration.fleet_observation import get_fleet_observer

    subject = str(event.get("subject") or "unknown")
    try:
        observation = get_fleet_observer(engine).service_status(subject)
    except Exception as e:  # noqa: BLE001 — a blind observer degrades to unknown
        logger.debug("remediation observer failed: %s", e)
        from agent_utilities.orchestration.fleet_observation import ServiceObservation

        observation = ServiceObservation(name=subject)

    kind = _classify(event, observation.flap_count)
    _record_step(engine, event, "observe", observation.status, classified=kind)

    if kind == "resource_pressure":
        result = resource_pressure_playbook(engine, event, observation)
    elif kind == "service_flapping":
        result = service_flapping_playbook(engine, event, observation)
    else:
        result = service_down_playbook(engine, event, observation)
    return {**base, **result}


def ensure_registered() -> None:
    """Idempotently register the remediation dispatcher on the OS-5.15 seam."""
    global _registered
    if _registered:
        return
    for source in _SOURCES:
        for severity in _SEVERITIES:
            register_playbook(f"{source}:{severity}", remediation_playbook)
    _registered = True
    logger.info("remediation playbooks registered for %s × %s", _SOURCES, _SEVERITIES)
