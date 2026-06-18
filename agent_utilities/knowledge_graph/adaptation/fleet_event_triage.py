#!/usr/bin/python
from __future__ import annotations

"""Fleet-event triage (CONCEPT:OS-5.15 — Fleet Event Ingress).

The acting half of the fleet-events ingress: the gateway's
``POST /api/fleet/events`` receiver (:mod:`agent_utilities.gateway.fleet_events`)
persists each monitoring event as a ``FleetEvent`` KG node and enqueues a
durable ``fleet_event_triage`` task; the engine's task workers dispatch that
task here.

Triage today is deliberately minimal:

* **correlate** — find KG entities (servers/services/sessions/tools) whose name
  matches the event subject, and link them with ``OBSERVED_ON`` edges so the
  incident is graph-queryable next to the things it hit;
* **escalate** — when severity warrants (critical/error, or a firing/down
  status), file a ``failure_gap`` ``Concept`` topic through the failure
  analyzer's shared gap-topic path, so the golden loop's existing intake
  synthesizes a remediation proposal for it (propose-only, AHE-3.18 chain).

Full remediation playbooks are a later task: :data:`PLAYBOOKS` is the clean
dispatch seam where they plug in — keyed ``"<source>:<severity>"`` then
``"<source>"`` then ``"default"`` — via :func:`register_playbook`.
"""

import logging
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# Severity/status that escalate an event into a failure_gap remediation topic.
GAP_SEVERITIES = {"critical", "error"}
GAP_STATUSES = {"firing", "down"}

# A playbook takes (engine, event_props) and returns a JSON-able report.
PlaybookFn = Callable[[Any, dict[str, Any]], dict[str, Any]]


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def correlate_event(engine: Any, event: dict[str, Any]) -> list[dict[str, Any]]:
    """Best-effort lookup of known KG entities matching the event subject."""
    subject = str(event.get("subject") or "").strip()
    if not subject or subject == "unknown":
        return []
    try:
        rows = engine.query_cypher(
            "MATCH (n) WHERE (n:Server OR n:Session OR n:Resource OR n:Tool) "
            "AND toLower(n.name) CONTAINS toLower($subject) "
            "RETURN n.id AS id, n.name AS name LIMIT 10",
            {"subject": subject},
        )
    except Exception as e:  # noqa: BLE001 — correlation is best-effort
        logger.debug("fleet event correlation failed: %s", e)
        return []
    return [r for r in rows or [] if isinstance(r, dict) and r.get("id")]


def _file_gap(engine: Any, event: dict[str, Any]) -> dict[str, Any] | None:
    """File a failure_gap topic for the event via the failure-analyzer path."""
    from .failure_analyzer import (
        ANOMALY_ERROR,
        FailurePattern,
        _normalize_detail,
        _sig,
        file_gap_topic,
    )

    subject = str(event.get("subject") or "unknown")
    summary = str(event.get("summary") or "")
    pattern = FailurePattern(
        signature=_sig(subject, "fleet_event", _normalize_detail(summary)),
        name=subject,
        kind="fleet_event",
        anomaly_type=ANOMALY_ERROR,
        count=1,
        sample_detail=summary,
    )
    return file_gap_topic(
        engine, pattern, anomaly_id=event.get("id"), source="fleet_event_triage"
    )


def default_playbook(engine: Any, event: dict[str, Any]) -> dict[str, Any]:
    """Correlate the event, escalating to a failure_gap topic when warranted."""
    correlated = correlate_event(engine, event)
    out: dict[str, Any] = {
        "playbook": "default",
        "correlated": [c.get("id") for c in correlated],
    }

    event_id = event.get("id")
    if event_id:
        for c in correlated:
            try:
                engine.link_nodes(
                    source_id=event_id,
                    target_id=c["id"],
                    rel_type="OBSERVED_ON",
                    properties={"source": "fleet_event_triage"},
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("OBSERVED_ON edge failed: %s", e)

    severity = str(event.get("severity") or "info")
    status = str(event.get("status") or "unknown")
    if severity in GAP_SEVERITIES or status in GAP_STATUSES:
        gap = _file_gap(engine, event)
        if gap:
            out["gap_topic"] = gap["id"]
    return out


#: Remediation playbook dispatch table — the seam where real playbooks plug in.
#: Resolution order: ``"<source>:<severity>"`` → ``"<source>"`` → ``"default"``.
PLAYBOOKS: dict[str, PlaybookFn] = {"default": default_playbook}


def register_playbook(key: str, playbook: PlaybookFn) -> None:
    """Register (or replace) a triage playbook under ``key``."""
    PLAYBOOKS[key] = playbook


def _resolve_playbook(source: str, severity: str) -> PlaybookFn:
    return (
        PLAYBOOKS.get(f"{source}:{severity}")
        or PLAYBOOKS.get(source)
        or PLAYBOOKS["default"]
    )


def _load_event(engine: Any, event_node_id: str) -> dict[str, Any] | None:
    try:
        rows = engine.query_cypher(
            "MATCH (e:FleetEvent {id: $id}) RETURN e", {"id": event_node_id}
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("fleet event load failed: %s", e)
        return None
    if not rows:
        return None
    props = rows[0].get("e") if isinstance(rows[0], dict) else None
    if not isinstance(props, dict):
        return None
    props.setdefault("id", event_node_id)
    return props


def triage_fleet_event(engine: Any, event_node_id: str) -> dict[str, Any]:
    """Triage one persisted ``FleetEvent`` node (the daemon task handler).

    Loads the event, dispatches the matching playbook (correlate + escalate),
    and stamps the node ``triage_status='triaged'``. Always returns a JSON-able
    report; a missing/unreadable event reports ``triaged=False`` rather than
    raising (the durable task is then completed, not retried forever).
    """
    event = _load_event(engine, event_node_id)
    if event is None:
        return {
            "triaged": False,
            "event_id": event_node_id,
            "reason": "event not found",
        }

    playbook = _resolve_playbook(
        str(event.get("source") or "generic"), str(event.get("severity") or "info")
    )
    try:
        report = playbook(engine, event) or {}
    except Exception as e:  # noqa: BLE001 — a playbook bug never kills the worker
        logger.warning("fleet event playbook failed for %s: %s", event_node_id, e)
        report = {"playbook_error": str(e)}

    try:
        engine.backend.execute(
            "MATCH (e:FleetEvent {id: $id}) "
            "SET e.triage_status = 'triaged', e.triaged_at = $ts",
            {"id": event_node_id, "ts": _now_iso()},
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("fleet event triage stamp failed: %s", e)

    return {"triaged": True, "event_id": event_node_id, **report}


# Register the jira/plane ticket-driven workflow playbooks (CONCEPT:ORCH-1.60) when the
# triage system loads — the import side-effect calls register_playbook("jira"/"plane").
from . import ticket_playbooks  # noqa: E402,F401 — registration side-effect
