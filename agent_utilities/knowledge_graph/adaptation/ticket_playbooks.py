#!/usr/bin/python
from __future__ import annotations

"""Ticket-driven workflow playbooks.

CONCEPT:AU-ORCH.scheduling.ticket-workflow-playbook — Ticket-driven workflow playbook

The event-driven half of Jira/Plane enablement. A tracker webhook posts to the
gateway's ``POST /api/fleet/events?source=jira`` (or ``?source=plane``); the fleet
ingress persists a ``FleetEvent`` and enqueues triage, which dispatches the playbook
registered here for that source. On a ticket create/update the playbook, generically
and the same way across Jira and *both* Plane instances:

1. **Ingests the change** — a watermark-narrowed ``source_sync(... ids=[ticket])`` so
   the KG reflects the new/changed ticket immediately (not just on the next sweep).
2. **Dispatches a workflow** — when a workflow name is configured
   (``JIRA_TICKET_WORKFLOW`` / ``PLANE_TICKET_WORKFLOW``), runs it via the orchestrator
   with the ticket as context. This is the seam where a concrete ticket→PR pipeline
   (intake-gate → coder → reviewer → CI, defined as a graph_orchestrate workflow) plugs
   in — the playbook stays generic; the pipeline is data.
3. **Notifies the operator** — a Telegram (last-active channel) message via
   ``MessagingService.reach_user`` — the cockpit for the loop.

Registered as the ``jira``/``plane`` triage playbooks; wired on import from
``fleet_event_triage`` (see :func:`register_ticket_playbooks`).
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _ticket_ref(event: dict[str, Any]) -> str | None:
    """The changed ticket id/key from the event (webhook subject or metadata)."""
    for key in ("ticket", "issue_key", "key", "work_item_id", "subject"):
        val = event.get(key)
        if val and str(val).strip() and str(val) != "unknown":
            return str(val).strip()
    return None


def _run_async(coro: Any) -> Any:
    from ...protocols.source_connectors.connectors.mcp_package import _run_async as run

    return run(coro)


def _notify(engine: Any, text: str) -> None:
    """Best-effort Telegram/last-channel notification (never raises into triage)."""
    try:
        from ...messaging.service import MessagingService

        _run_async(
            MessagingService(engine=engine).reach_user(
                text, source="ticket_playbook", reason="ticket workflow"
            )
        )
    except Exception as exc:  # noqa: BLE001 — notification is best-effort
        logger.debug("ticket playbook notify failed: %s", exc)


def _dispatch_workflow(
    engine: Any, workflow: str, ticket: str, source: str
) -> str | None:
    """Run the configured graph_orchestrate workflow for this ticket (or None)."""
    if not workflow:
        return None
    try:
        from ...orchestration.manager import Orchestrator

        result = _run_async(
            Orchestrator(engine).execute_workflow(
                workflow_id=workflow,
                task=f"{source} ticket {ticket}",
            )
        )
        return workflow if result is not None else None
    except Exception as exc:  # noqa: BLE001 — a missing/invalid workflow must not kill triage
        logger.warning("ticket workflow %r dispatch failed: %s", workflow, exc)
        return None


def _ticket_playbook(source: str, workflow_setting: str) -> Any:
    """Build the triage playbook for a tracker ``source`` (jira/plane)."""
    from .fleet_event_triage import default_playbook

    def playbook(engine: Any, event: dict[str, Any]) -> dict[str, Any]:
        out = default_playbook(engine, event)
        out["playbook"] = f"{source}_ticket"
        ticket = _ticket_ref(event)
        out["ticket"] = ticket
        if not ticket:
            return out

        # 1. Narrowed ingest so the KG reflects the change now.
        try:
            from ..core.source_sync import sync_source

            sync_source(engine, source, mode="delta", ids=[ticket])
            out["ingested"] = True
        except Exception as exc:  # noqa: BLE001 — ingest is best-effort
            logger.debug("ticket ingest failed for %s: %s", ticket, exc)

        # 2. Optional configurable workflow dispatch.
        from ...core.config import setting

        workflow = (setting(workflow_setting, default="") or "").strip()
        if dispatched := _dispatch_workflow(engine, workflow, ticket, source):
            out["workflow"] = dispatched

        # 3. Operator notification (the Telegram cockpit).
        _notify(
            engine,
            f"🎫 {source} {ticket}: workflow triggered ({out.get('workflow') or 'ingest only'}).",
        )
        return out

    return playbook


def register_ticket_playbooks() -> None:
    """Register the jira/plane ticket playbooks (idempotent)."""
    from .fleet_event_triage import register_playbook

    register_playbook("jira", _ticket_playbook("jira", "JIRA_TICKET_WORKFLOW"))
    register_playbook("plane", _ticket_playbook("plane", "PLANE_TICKET_WORKFLOW"))


register_ticket_playbooks()
