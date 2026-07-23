"""graph_incident — Incident Brain MCP tool (CONCEPT:AU-KG.enrichment.cross-layer-incident-correlation follow-up).

Cross-layer ``:HealthAnomaly`` correlation into deduplicated ``:Incident``s
already exists and runs (:mod:`agent_utilities.observability.incidents`), but
was reachable only via a CronJob / ``graph_loops`` tick — no MCP tool exposed
it for interactive browse/approve, a gap both
``reports/surpass-6mo/03-au-orchestration-ops.md`` (#5) and
``04-five-intersections.md`` (§4 top build #2) name explicitly.

This module is a THIN wrapper over the existing correlation engine:

* ``correlate`` -> :func:`agent_utilities.observability.incidents.correlate_incidents`
  — one correlation pass (groups recent cross-layer ``:HealthAnomaly`` rows into
  ``:Incident``s; idempotent — an already-open incident is deduped, not
  re-written).
* ``list``       -> reads ``:Incident`` nodes via the SAME engine accessor
  :mod:`.incidents` itself uses (``health_ingest._engine()`` +
  ``get_nodes_by_label``), optionally filtered by ``status``/``severity``.
* ``get``        -> one ``:Incident`` node by id, PLUS its correlated
  ``:HealthAnomaly`` group and affected entity ids
  (:func:`agent_utilities.observability.incidents.get_incident_evidence`).
* ``timeline``   -> the incident's anomaly/event sequence: the same
  correlated-evidence read, oldest first, interleaved with the incident's own
  opened/acknowledged/resolved lifecycle events.
* ``ack``/``resolve`` -> state transitions on the ``:Incident`` node ITSELF
  (:func:`agent_utilities.observability.incidents.set_incident_status`),
  gated FIRST by the fail-closed
  :class:`~agent_utilities.orchestration.action_policy.ActionPolicy`
  (kind ``incident.ack``/``incident.resolve`` — mirrors ``graph_claims``'s
  ``_gate`` pattern in ``claim_tools.py``) and provenance-stamped (who/when).
  A denied/queued decision blocks the mutation without ever touching the
  incident.

Remediation stays REPORT-ONLY, exactly as this module always specified:
``ack``/``resolve`` mutate ONLY the ``:Incident`` node's own status/provenance
fields — never a ticket, a ``:RemediationProposal``, or any infrastructure
action. ``propose_remediation``/``run_incident_correlation``'s ticket-routing
and remediation-proposal steps stay report-only and CronJob-driven; this tool
does not reach them.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import Field

from agent_utilities.mcp import kg_server
from agent_utilities.security.error_surface import public_error_payload


def _incident_engine() -> Any | None:
    """The SAME engine accessor :mod:`agent_utilities.observability.incidents`
    uses for its own ``:Incident`` reads (``health_ingest._engine()``) — reused
    here rather than a second engine-resolution path, so `list`/`get` see
    exactly what `correlate` (and the CronJob) wrote."""
    from agent_utilities.observability import health_ingest

    return health_ingest._engine()


def _gate(kind: str, target: str, reason: str, actor_id: str = "") -> tuple[bool, dict]:
    """Run the fail-closed ActionPolicy gate for one incident-state mutation.

    Mirrors ``claim_tools._gate``/``secret_tools._gate`` exactly — a thin call
    into the shared ``get_action_policy(engine).decide(...)`` decision point,
    never a reimplementation of its rules. Uses ``kg_server._get_engine()``
    (the same MCP-layer engine handle every other tool's gate uses for policy
    audit/rate-limit accounting) — a DIFFERENT handle than
    ``_incident_engine()``'s ``GraphComputeEngine``, which is what actually
    reads/writes the ``:Incident`` node; the two concerns (policy audit vs.
    domain data) are intentionally kept on their own already-established
    engine paths.
    """
    from agent_utilities.orchestration.action_policy import (
        ActionRequest,
        get_action_policy,
    )

    decision = get_action_policy(kg_server._get_engine()).decide(
        ActionRequest(
            kind=kind,
            target=target,
            source="mcp",
            reason=reason or f"{kind} on {target}",
            actor_id=actor_id,
        )
    )
    info = {
        "decision": decision.decision,
        "tier": decision.tier,
        "reason": decision.reason,
        "approval_id": decision.approval_id,
        "audit_id": decision.audit_id,
    }
    return decision.allowed, info


def _list_incidents(*, status: str, severity: str, limit: int) -> dict[str, Any]:
    engine = _incident_engine()
    if engine is None:
        return {
            "surface": "incident",
            "action": "list",
            "error": "no reachable engine",
            "incidents": [],
        }
    try:
        rows = engine.get_nodes_by_label("Incident", 0) or []
    except Exception as exc:  # noqa: BLE001 — surface engine errors as data
        return {
            "surface": "incident",
            "action": "list",
            "incidents": [],
            **public_error_payload(exc),
        }
    status_filter = (status or "").strip()
    severity_filter = (severity or "").strip()
    items: list[dict[str, Any]] = []
    for node_id, props in rows:
        if not isinstance(props, dict):
            continue
        if status_filter and str(props.get("status") or "") != status_filter:
            continue
        if severity_filter and str(props.get("severity") or "") != severity_filter:
            continue
        items.append({"id": node_id, **props})
    items.sort(
        key=lambda i: str(i.get("observedAt") or i.get("opened_at") or ""), reverse=True
    )
    bounded = items[: max(0, int(limit))]
    return {
        "surface": "incident",
        "action": "list",
        "count": len(bounded),
        "total_matched": len(items),
        "incidents": bounded,
    }


def _get_incident(incident_id: str) -> dict[str, Any]:
    from agent_utilities.observability.incidents import get_incident_evidence

    engine = _incident_engine()
    if engine is None:
        return {
            "surface": "incident",
            "action": "get",
            "incident_id": incident_id,
            "error": "no reachable engine",
        }
    try:
        rows = engine.get_nodes_by_label("Incident", 0) or []
    except Exception as exc:  # noqa: BLE001
        return {
            "surface": "incident",
            "action": "get",
            "incident_id": incident_id,
            **public_error_payload(exc),
        }
    for node_id, props in rows:
        if node_id == incident_id:
            evidence = get_incident_evidence(incident_id, engine=engine) or {
                "anomalies": [],
                "entities": [],
            }
            return {
                "surface": "incident",
                "action": "get",
                "incident_id": incident_id,
                "incident": {
                    "id": node_id,
                    **(props if isinstance(props, dict) else {}),
                },
                "anomalies": evidence["anomalies"],
                "entities": evidence["entities"],
            }
    return {
        "surface": "incident",
        "action": "get",
        "incident_id": incident_id,
        "error": "not found",
    }


def _incident_timeline(incident_id: str) -> dict[str, Any]:
    """The incident's anomaly/event sequence: its own opened/acknowledged/
    resolved lifecycle events interleaved with its correlated
    ``:HealthAnomaly`` group (:func:`_get_incident`'s ``anomalies``), oldest
    first."""
    got = _get_incident(incident_id)
    if "error" in got:
        return {
            "surface": "incident",
            "action": "timeline",
            "incident_id": incident_id,
            "error": got["error"],
            "timeline": [],
        }
    incident = got["incident"]
    events: list[dict[str, Any]] = []
    if incident.get("observedAt"):
        events.append(
            {
                "kind": "incident_opened",
                "at": incident["observedAt"],
                "detail": incident.get("summary"),
            }
        )
    for anomaly in got["anomalies"]:
        events.append(
            {
                "kind": "anomaly",
                "at": anomaly.get("observedAt"),
                "id": anomaly.get("id"),
                "entity": anomaly.get("entity"),
                "signal": anomaly.get("signal"),
                "detail": anomaly.get("kind"),
            }
        )
    if incident.get("ackedAt"):
        events.append(
            {
                "kind": "incident_acknowledged",
                "at": incident["ackedAt"],
                "by": incident.get("ackedBy"),
            }
        )
    if incident.get("resolvedAt"):
        events.append(
            {
                "kind": "incident_resolved",
                "at": incident["resolvedAt"],
                "by": incident.get("resolvedBy"),
            }
        )
    events.sort(key=lambda e: str(e.get("at") or ""))
    return {
        "surface": "incident",
        "action": "timeline",
        "incident_id": incident_id,
        "count": len(events),
        "timeline": events,
    }


def register_incident_tools(mcp: Any) -> None:
    """Register the ``graph_incident`` group on the given FastMCP server."""

    @mcp.tool(
        name="graph_incident",
        description=(
            "Incident Brain (CONCEPT:AU-KG.enrichment.cross-layer-incident-correlation): browse and drive the "
            "cross-layer :HealthAnomaly -> :Incident correlation the incidents "
            "CronJob already runs. Actions: 'correlate' (run one correlation "
            "pass now — groups recent anomalies within window_s/days into "
            ":Incident nodes; idempotent, an already-open incident is deduped "
            "not re-written), 'list' (recent :Incident nodes, optionally "
            "filtered by status e.g. status='open' and/or severity e.g. "
            "severity='critical', newest first), 'get' (incident_id -> the "
            "full :Incident node PLUS its correlated :HealthAnomaly group and "
            "affected entity ids), 'timeline' (incident_id -> its anomaly/"
            "event sequence, oldest first), 'ack'/'resolve' (incident_id -> "
            "transition the incident's OWN status to acknowledged/resolved — "
            "FIRST passes the fail-closed ActionPolicy gate, kind "
            "incident.ack/incident.resolve; a denied/queued decision blocks "
            "the mutation and is returned under `policy`). ack/resolve are "
            "REPORT-ONLY: they mutate only the :Incident node's own status/"
            "provenance — never a ticket or infrastructure action; "
            "propose_remediation stays report-only/CronJob-driven."
        ),
        tags=["graph-os", "incident", "observability", "aiops"],
    )
    def graph_incident(
        action: str = Field(
            default="list",
            description="correlate | list | get | timeline | ack | resolve",
        ),
        incident_id: str = Field(
            default="",
            description="Incident id (required for get/timeline/ack/resolve).",
        ),
        window_s: int = Field(
            default=300,
            description="Correlation clustering window, seconds (correlate).",
        ),
        days: int = Field(
            default=1, description="Lookback window in days (correlate)."
        ),
        status: str = Field(
            default="",
            description="Filter by status, e.g. 'open' (list). Empty = no filter.",
        ),
        severity: str = Field(
            default="",
            description="Filter by severity, e.g. 'critical' (list). Empty = no filter.",
        ),
        limit: int = Field(default=50, description="Max incidents returned (list)."),
        reason: str = Field(
            default="",
            description="Why this transition is happening — the audit trail (ack/resolve).",
        ),
        actor_id: str = Field(
            default="",
            description="Optional caller identity stamped onto the transition (ack/resolve).",
        ),
    ) -> str:
        """Incident Brain: correlate / list / get / timeline / ack / resolve."""
        action_key = (action or "list").strip().lower()

        if action_key == "correlate":
            from agent_utilities.observability.incidents import correlate_incidents

            incidents = correlate_incidents(window_s=int(window_s), days=int(days))
            return json.dumps(
                {
                    "surface": "incident",
                    "action": action_key,
                    "count": len(incidents),
                    "incidents": incidents,
                },
                default=str,
            )
        if action_key == "list":
            return json.dumps(
                _list_incidents(status=status, severity=severity, limit=limit),
                default=str,
            )
        if action_key == "get":
            if not incident_id:
                return json.dumps(
                    {
                        "surface": "incident",
                        "action": action_key,
                        "error": "incident_id required for get",
                    }
                )
            return json.dumps(_get_incident(incident_id), default=str)
        if action_key == "timeline":
            if not incident_id:
                return json.dumps(
                    {
                        "surface": "incident",
                        "action": action_key,
                        "error": "incident_id required for timeline",
                    }
                )
            return json.dumps(_incident_timeline(incident_id), default=str)
        if action_key in ("ack", "resolve"):
            if not incident_id:
                return json.dumps(
                    {
                        "surface": "incident",
                        "action": action_key,
                        "error": f"incident_id required for {action_key}",
                    }
                )
            from agent_utilities.observability.incidents import (
                STATUS_ACKNOWLEDGED,
                STATUS_RESOLVED,
                set_incident_status,
            )

            status_value = (
                STATUS_ACKNOWLEDGED if action_key == "ack" else STATUS_RESOLVED
            )
            allowed, policy = _gate(
                f"incident.{action_key}", incident_id, reason, actor_id
            )
            if not allowed:
                return json.dumps(
                    {
                        "surface": "incident",
                        "action": action_key,
                        "incident_id": incident_id,
                        "error": "policy_denied",
                        "policy": policy,
                    }
                )
            result = set_incident_status(incident_id, status_value, actor=actor_id)
            if result is None:
                return json.dumps(
                    {
                        "surface": "incident",
                        "action": action_key,
                        "incident_id": incident_id,
                        "policy": policy,
                        "error": "not found or engine unreachable",
                    }
                )
            updated = _get_incident(incident_id)
            return json.dumps(
                {
                    "surface": "incident",
                    "action": action_key,
                    "incident_id": incident_id,
                    "status": status_value,
                    "policy": policy,
                    "incident": updated.get("incident"),
                },
                default=str,
            )
        return json.dumps(
            {
                "surface": "incident",
                "action": action_key,
                "error": f"unknown action {action_key!r}",
            }
        )

    kg_server.REGISTERED_TOOLS["graph_incident"] = graph_incident
    # No bespoke endpoint needed — the generic REST-twin factory in
    # kg_server._build_server mounts POST /incident for every ACTION_TOOL_ROUTES
    # entry without a bespoke handler, dispatching through the SAME
    # _execute_tool core.
    kg_server.ACTION_TOOL_ROUTES["graph_incident"] = "/incident"
