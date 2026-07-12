#!/usr/bin/python
from __future__ import annotations

"""Incident → ticket routing — the second half of the Phase D/E payoff
(``reports/unified-infra-intelligence-plan.md``): backfeed a correlated
``:Incident`` (see :mod:`.incidents`) into a ticketing system-of-record.

CONCEPT:AU-KG.enrichment.incident-ticket-routing. A small pluggable-adapter
interface (:class:`TicketAdapter`) mirrors the ``kg_writeback`` sink pattern
(:mod:`agent_utilities.knowledge_graph.enrichment.writeback.core` — fail-closed,
dry-run-first, one target selected by config) without going through its
``run_writeback``/``WritebackResult`` machinery directly for every backend,
because the writeback sinks for ``servicenow``/``erpnext`` already own a
*different* domain (CMDB CIs / Items·Assets) — an incident ticket is a
different table (ServiceNow ``incident``, ERPNext ``Issue``). Where a writeback
sink already does exactly this job (the ``jira``/``plane`` **issue** sinks in
``.writeback.sinks.issue_tracker``), the adapters below call ``run_writeback``
directly rather than re-implementing issue creation.

Backend selection is config-driven (``INCIDENT_TICKET_BACKEND`` —
``servicenow|erpnext|jira|plane|none``, default ``none`` = graph-only — never a
bare ``os.environ`` read, per this package's Configuration discipline).
**Fail-closed + dry-run-first + report-only by default**: ``INCIDENT_TICKET_ENABLE``
gates an actual live ticket create/update; otherwise (or for the default
``none`` backend) only the INTENDED ticket is recorded. Every call — live or
proposed — writes a ``:Ticket`` node + ``:hasTicket`` edge on the incident, so
the graph tracks asset ↔ incident ↔ ticket end to end even when no
system-of-record is reachable.
"""

import logging
from typing import Any, Protocol, runtime_checkable

from agent_utilities.core.config import setting

logger = logging.getLogger("agent_utilities.observability.incident_router")

_SOURCE = "agent-utilities-health"


@runtime_checkable
class TicketAdapter(Protocol):
    """A ticketing system-of-record adapter."""

    name: str

    def create_ticket(self, incident: dict[str, Any]) -> dict[str, Any]:
        """Create (or, report-only, propose) a ticket for ``incident``.
        Returns ``{"ticket_id", "url", "status"}``."""
        ...

    def update_ticket(self, ref: str, status: str) -> dict[str, Any]:
        """Update an existing ticket's status. Returns ``{"ticket_id", "status"}``."""
        ...


def _write_enabled() -> bool:
    return bool(setting("INCIDENT_TICKET_ENABLE", False, cast=bool))


def _summary_text(incident: dict[str, Any]) -> str:
    return str(incident.get("summary") or f"incident {incident.get('id')}")


def _incident_body(incident: dict[str, Any]) -> str:
    return (
        f"entity: {incident.get('entity')}\n"
        f"layers: {', '.join(incident.get('layers') or [])}\n"
        f"signals: {', '.join(incident.get('signals') or [])}\n"
        f"root_cause_layer: {incident.get('root_cause_layer')}\n"
        f"severity: {incident.get('severity')}\n"
        f"opened_at: {incident.get('opened_at')}\n\n"
        f"{incident.get('summary') or ''}"
    )


def _proposed(incident: dict[str, Any], *, status: str = "proposed") -> dict[str, Any]:
    return {"ticket_id": f"proposed:{incident.get('id')}", "url": "", "status": status}


class GraphOnlyAdapter:
    """Default backend (``none``) — records the INTENDED ticket, calls no SoR.

    Guarantees ``route_incident`` always tracks asset↔incident↔ticket in the
    graph even when no ticketing system is configured/reachable.
    """

    name = "none"

    def create_ticket(self, incident: dict[str, Any]) -> dict[str, Any]:
        return _proposed(incident)

    def update_ticket(self, ref: str, status: str) -> dict[str, Any]:
        return {"ticket_id": ref, "status": status}


class _WritebackIssueAdapter:
    """jira/plane — reuses the existing ``kg_writeback`` issue-tracker sinks
    (``.writeback.sinks.issue_tracker``) via ``run_writeback`` rather than
    re-implementing issue creation."""

    name = ""
    _target = ""
    _project_setting = ""

    def _project(self) -> str | None:
        val = str(setting(self._project_setting, "") or "").strip()
        return val or None

    def create_ticket(self, incident: dict[str, Any]) -> dict[str, Any]:
        from agent_utilities.knowledge_graph.enrichment.writeback.core import (
            run_writeback,
        )

        project = self._project()
        dry_run = not _write_enabled()
        creations = [
            {
                "title": f"[incident] {_summary_text(incident)}",
                "body": _incident_body(incident),
                "project": project,
                "project_id": project,
            }
        ]
        out = run_writeback(self._target, dry_run=dry_run, creations=creations)
        if out.get("status") == "refused" or not out.get("created"):
            return _proposed(incident, status="proposed" if dry_run else "failed")
        return {
            "ticket_id": f"{self._target}:{incident.get('id')}",
            "url": "",
            "status": "created",
        }

    def update_ticket(self, ref: str, status: str) -> dict[str, Any]:
        # The writeback transition sinks (jira_transition/plane_state) need the
        # tracker's own key/work-item id, which this router does not resolve —
        # report the requested status without a live call (report-only parity
        # with the other adapters' dry-run path).
        return {"ticket_id": ref, "status": status}


class JiraAdapter(_WritebackIssueAdapter):
    name = "jira"
    _target = "jira"
    _project_setting = "INCIDENT_JIRA_PROJECT"


class PlaneAdapter(_WritebackIssueAdapter):
    name = "plane"
    _target = "plane"
    _project_setting = "INCIDENT_PLANE_PROJECT"


def _sn_severity(severity: Any) -> str:
    return {"critical": "1", "warning": "2"}.get(str(severity or "").lower(), "3")


def _sn_state(status: str) -> str:
    return "6" if str(status).lower() in ("resolved", "closed") else "2"


class ServiceNowAdapter:
    """Files a real ServiceNow ``incident`` table record — distinct from the
    ``kg_writeback`` ServiceNow sink, which only manages CMDB CIs. Uses the
    ``servicenow-api`` connector's own client (``create_incident`` /
    generic ``patch_table_record`` for updates), fail-closed + dry-run-first."""

    name = "servicenow"

    def _client(self) -> Any | None:
        try:
            from servicenow_api.auth import get_client

            return get_client()
        except Exception:  # noqa: BLE001 — connector absent/unconfigured
            logger.debug("servicenow incident client unavailable", exc_info=True)
            return None

    def create_ticket(self, incident: dict[str, Any]) -> dict[str, Any]:
        if not _write_enabled():
            return _proposed(incident)
        client = self._client()
        if client is None:
            return _proposed(incident, status="failed")
        try:
            resp = client.create_incident(
                data={
                    "short_description": _summary_text(incident)[:160],
                    "description": _incident_body(incident),
                    "severity": _sn_severity(incident.get("severity")),
                    "urgency": _sn_severity(incident.get("severity")),
                }
            )
            result = getattr(resp, "result", None)
            number = getattr(result, "number", None) or getattr(result, "sys_id", None)
            return {
                "ticket_id": str(number or incident.get("id")),
                "url": "",
                "status": "created",
            }
        except Exception as e:  # noqa: BLE001 — a broken client must not break the pass
            logger.debug("servicenow create_incident failed: %s", e)
            return _proposed(incident, status="failed")

    def update_ticket(self, ref: str, status: str) -> dict[str, Any]:
        if not _write_enabled():
            return {"ticket_id": ref, "status": status}
        client = self._client()
        if client is not None:
            try:
                client.patch_table_record(
                    table="incident",
                    table_record_sys_id=ref,
                    data={"state": _sn_state(status)},
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("servicenow incident update failed: %s", e)
        return {"ticket_id": ref, "status": status}


class ErpNextAdapter:
    """Files a report-only ERPNext ``Issue`` (helpdesk ticket) via the generic
    ``create_document`` resource client — the ``kg_writeback`` ERPNext sink
    only manages Items/Assets, not tickets."""

    name = "erpnext"

    def _client(self) -> Any | None:
        try:
            from erpnext_agent.auth import get_client

            return get_client()
        except Exception:  # noqa: BLE001 — connector absent/unconfigured
            logger.debug("erpnext ticket client unavailable", exc_info=True)
            return None

    def create_ticket(self, incident: dict[str, Any]) -> dict[str, Any]:
        if not _write_enabled():
            return _proposed(incident)
        client = self._client()
        if client is None:
            return _proposed(incident, status="failed")
        try:
            doc = client.create_document(
                "Issue",
                {
                    "subject": _summary_text(incident)[:140],
                    "description": _incident_body(incident),
                },
            )
            name = (doc or {}).get("name") or incident.get("id")
            return {"ticket_id": str(name), "url": "", "status": "created"}
        except Exception as e:  # noqa: BLE001
            logger.debug("erpnext create_document (Issue) failed: %s", e)
            return _proposed(incident, status="failed")

    def update_ticket(self, ref: str, status: str) -> dict[str, Any]:
        if not _write_enabled():
            return {"ticket_id": ref, "status": status}
        client = self._client()
        if client is not None:
            try:
                client.update_document("Issue", ref, {"status": status})
            except Exception as e:  # noqa: BLE001
                logger.debug("erpnext Issue update failed: %s", e)
        return {"ticket_id": ref, "status": status}


_ADAPTERS: dict[str, TicketAdapter] = {
    "none": GraphOnlyAdapter(),
    "jira": JiraAdapter(),
    "plane": PlaneAdapter(),
    "servicenow": ServiceNowAdapter(),
    "erpnext": ErpNextAdapter(),
}


def get_adapter(name: str | None = None) -> TicketAdapter:
    """Resolve the configured :class:`TicketAdapter` (``INCIDENT_TICKET_BACKEND``,
    default ``none``); an unknown name falls back to the graph-only adapter."""
    backend = (
        (name or str(setting("INCIDENT_TICKET_BACKEND", "none") or "none"))
        .strip()
        .lower()
    )
    return _ADAPTERS.get(backend, _ADAPTERS["none"])


def route_incident(
    incident: dict[str, Any], *, adapter: TicketAdapter | None = None
) -> dict[str, Any]:
    """Route ``incident`` to the configured ticketing system-of-record.

    Selects the backend via ``INCIDENT_TICKET_BACKEND`` (default ``none`` =
    graph-only). Fail-closed + dry-run-first: a non-``none`` backend only
    places a LIVE call when ``INCIDENT_TICKET_ENABLE`` is set — otherwise (and
    always for ``none``) it records the INTENDED ticket. Always writes a
    ``:Ticket`` node + ``:hasTicket`` edge on the incident (via
    :func:`_write_ticket_node`), so asset↔incident↔ticket is tracked
    end-to-end regardless of whether the SoR is reachable. Best-effort: a
    broken adapter degrades to a failed/proposed ticket rather than raising.
    """
    adapter = adapter or get_adapter()
    try:
        ticket = adapter.create_ticket(incident)
    except Exception as e:  # noqa: BLE001 — a broken adapter must not break correlation
        logger.warning("incident ticket routing failed (%s): %s", adapter.name, e)
        ticket = _proposed(incident, status="failed")

    out = {
        "backend": adapter.name,
        "ticket_id": ticket.get("ticket_id"),
        "ticket_url": ticket.get("url") or "",
        "ticket_status": ticket.get("status") or "proposed",
    }
    _write_ticket_node(incident, out)
    return out


def close_ticket(
    incident: dict[str, Any],
    ticket_ref: str,
    *,
    status: str = "resolved",
    adapter: TicketAdapter | None = None,
) -> dict[str, Any]:
    """Update/close a previously-routed ticket when its incident resolves.

    ``ticket_ref`` is the ``ticket_id`` :func:`route_incident` returned for
    this incident. Same fail-closed/dry-run-first gating as ``create_ticket``.
    """
    adapter = adapter or get_adapter()
    try:
        result = adapter.update_ticket(ticket_ref, status)
    except Exception as e:  # noqa: BLE001
        logger.warning("incident ticket update failed (%s): %s", adapter.name, e)
        result = {"ticket_id": ticket_ref, "status": "failed"}

    out = {
        "backend": adapter.name,
        "ticket_id": result.get("ticket_id", ticket_ref),
        "ticket_url": "",
        "ticket_status": result.get("status", status),
    }
    _write_ticket_node(incident, out)
    return out


def _write_ticket_node(
    incident: dict[str, Any], ticket: dict[str, Any]
) -> dict[str, int] | None:
    from agent_utilities.knowledge_graph.memory.native_ingest import ingest_entities

    incident_id = str(incident.get("id") or "")
    if not incident_id:
        return None
    tid = f"health:ticket:{incident_id}"
    entities = [
        {
            "id": tid,
            "type": "Ticket",
            "incident": incident_id,
            "backend": ticket.get("backend"),
            "ticketRef": ticket.get("ticket_id"),
            "ticketUrl": ticket.get("ticket_url") or "",
            "ticketStatus": ticket.get("ticket_status"),
        }
    ]
    relationships = [{"source": incident_id, "target": tid, "type": "hasTicket"}]
    return ingest_entities(entities, relationships, source=_SOURCE, domain="health")
