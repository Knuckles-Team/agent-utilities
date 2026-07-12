"""graph_incident — Incident Brain MCP tool (CONCEPT:AU-KG.enrichment.cross-layer-incident-correlation follow-up).

Cross-layer ``:HealthAnomaly`` correlation into deduplicated ``:Incident``s
already exists and runs (:mod:`agent_utilities.observability.incidents`), but
was reachable only via a CronJob / ``graph_loops`` tick — no MCP tool exposed
it for interactive browse/approve, a gap both
``reports/surpass-6mo/03-au-orchestration-ops.md`` (#5) and
``04-five-intersections.md`` (§4 top build #2) name explicitly.

This module is a THIN, read-only wrapper over the existing correlation engine:

* ``correlate`` -> :func:`agent_utilities.observability.incidents.correlate_incidents`
  — one correlation pass (groups recent cross-layer ``:HealthAnomaly`` rows into
  ``:Incident``s; idempotent — an already-open incident is deduped, not
  re-written).
* ``list``       -> reads ``:Incident`` nodes via the SAME engine accessor
  :mod:`.incidents` itself uses (``health_ingest._engine()`` +
  ``get_nodes_by_label``), optionally filtered by ``status``.
* ``get``        -> one ``:Incident`` node by id, from the same read.

No remediation is exposed or executed here — ``propose_remediation`` /
``run_incident_correlation``'s ticket-routing + remediation-proposal steps stay
report-only and CronJob-driven exactly as the module docstring specifies; this
tool only makes the READ-ONLY half (correlate/list/get) interactively
reachable.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import Field

from agent_utilities.mcp import kg_server


def _incident_engine() -> Any | None:
    """The SAME engine accessor :mod:`agent_utilities.observability.incidents`
    uses for its own ``:Incident`` reads (``health_ingest._engine()``) — reused
    here rather than a second engine-resolution path, so `list`/`get` see
    exactly what `correlate` (and the CronJob) wrote."""
    from agent_utilities.observability import health_ingest

    return health_ingest._engine()


def _list_incidents(*, status: str, limit: int) -> dict[str, Any]:
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
            "error": str(exc),
            "incidents": [],
        }
    status_filter = (status or "").strip()
    items: list[dict[str, Any]] = []
    for node_id, props in rows:
        if not isinstance(props, dict):
            continue
        if status_filter and str(props.get("status") or "") != status_filter:
            continue
        items.append({"id": node_id, **props})
    items.sort(key=lambda i: str(i.get("opened_at") or ""), reverse=True)
    bounded = items[: max(0, int(limit))]
    return {
        "surface": "incident",
        "action": "list",
        "count": len(bounded),
        "total_matched": len(items),
        "incidents": bounded,
    }


def _get_incident(incident_id: str) -> dict[str, Any]:
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
            "error": str(exc),
        }
    for node_id, props in rows:
        if node_id == incident_id:
            return {
                "surface": "incident",
                "action": "get",
                "incident_id": incident_id,
                "incident": {
                    "id": node_id,
                    **(props if isinstance(props, dict) else {}),
                },
            }
    return {
        "surface": "incident",
        "action": "get",
        "incident_id": incident_id,
        "error": "not found",
    }


def register_incident_tools(mcp: Any) -> None:
    """Register the ``graph_incident`` group on the given FastMCP server."""

    @mcp.tool(
        name="graph_incident",
        description=(
            "Incident Brain (CONCEPT:AU-KG.enrichment.cross-layer-incident-correlation): browse the "
            "cross-layer :HealthAnomaly -> :Incident correlation the incidents "
            "CronJob already runs. Actions: 'correlate' (run one correlation "
            "pass now — groups recent anomalies within window_s/days into "
            ":Incident nodes; idempotent, an already-open incident is deduped "
            "not re-written), 'list' (recent :Incident nodes, optionally "
            "status-filtered e.g. status='open', newest first), 'get' "
            "(incident_id -> the full :Incident node). Read-only: no "
            "remediation is proposed or executed by this tool — "
            "propose_remediation stays report-only/CronJob-driven."
        ),
        tags=["graph-os", "incident", "observability", "aiops"],
    )
    def graph_incident(
        action: str = Field(default="list", description="correlate | list | get"),
        incident_id: str = Field(
            default="", description="Incident id (required for get)."
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
        limit: int = Field(default=50, description="Max incidents returned (list)."),
    ) -> str:
        """Incident Brain: correlate / list / get (read-only)."""
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
            return json.dumps(_list_incidents(status=status, limit=limit), default=str)
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
