"""graph_compliance — compliance posture + redacted bulk export MCP tool.

CONCEPT:AU-KG.enrichment.compliance-posture-rollup

``reports/surpass-6mo/04-five-intersections.md`` (§2) found the primitives for
an auditor-facing compliance view already deep and real — but scattered:
the tamper-evident hash-chained audit ledger (``graph_audit``), CISO
Assistant / TRM governance data already in the graph (``:Control``/
``:ComplianceRequirement``/``:ComplianceGate``/``:Regulation``/``:Assessment``/
``:Incident``/...), and per-node policy-aware redaction
(``ExplainBelief``'s ``disclosure_level`` — Full/Skeleton/ExistenceOnly). No
tool joined them into one rollup, and there was no bulk "export this subgraph,
redacted, for an auditor" primitive (only a per-node ``explain_belief`` call).

This module ADDS two thin actions, reusing those primitives directly (no new
compliance/redaction logic):

* ``posture`` — one rollup: the audit ledger's ``verify()`` report (reused
  from :mod:`.audit_tools`) + node-count / status-breakdown of the governance
  labels already ingested into the KG (CISO Assistant extractor + the TRM
  portfolio-intelligence engine).
* ``export``  — bulk redacted export: given an explicit id list or a read-only
  Cypher query selecting ids, calls the engine's OWN ``explain_belief(node_id,
  disclosure_level)`` (via the SAME generic dispatcher ``engine_tools._dispatch``
  uses) for each id and collects the results — the bulk sibling of the
  existing per-node redaction primitive, bounded by ``limit``.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import Field

from agent_utilities.mcp import kg_server

#: Governance/compliance node labels already ingested by the CISO Assistant
#: extractor (agent_utilities/knowledge_graph/enrichment/extractors/ciso_assistant.py)
#: and the TRM portfolio-intelligence engine (agent_utilities/observability/
#: portfolio_intelligence.py) — read-only rollup, no new label taxonomy.
_POSTURE_LABELS: tuple[str, ...] = (
    "Control",
    "Policy",
    "Risk",
    "ComplianceRequirement",
    "ComplianceGate",
    "Regulation",
    "ComplianceAssessment",
    "Assessment",
    "Incident",
    "RemediationProposal",
    "Finding",
    "SecurityException",
)


def _nodes_by_label(engine: Any, label: str) -> list[tuple[str, dict[str, Any]]]:
    """Best-effort ``get_nodes_by_label`` read — the SAME primitive
    ``observability/portfolio_intelligence.py`` and ``observability/incidents.py``
    already use for this kind of rollup."""
    fn = getattr(engine, "get_nodes_by_label", None)
    if not callable(fn):
        return []
    try:
        return fn(label, 0) or []
    except Exception:  # noqa: BLE001 — one label's read failure must not break the rollup
        return []


def _posture() -> dict[str, Any]:
    engine = kg_server._get_engine()
    if engine is None:
        return {
            "surface": "compliance",
            "action": "posture",
            "error": "IntelligenceGraphEngine not active",
        }

    from agent_utilities.mcp.tools.audit_tools import _verify

    audit = _verify()

    node_counts: dict[str, int] = {}
    status_breakdown: dict[str, dict[str, int]] = {}
    read_engine = getattr(engine, "graph", engine)
    for label in _POSTURE_LABELS:
        rows = _nodes_by_label(read_engine, label)
        node_counts[label] = len(rows)
        statuses: dict[str, int] = {}
        for _node_id, props in rows:
            if not isinstance(props, dict):
                continue
            status = str(props.get("status") or props.get("requestState") or "unknown")
            statuses[status] = statuses.get(status, 0) + 1
        if statuses:
            status_breakdown[label] = statuses

    return {
        "surface": "compliance",
        "action": "posture",
        "audit_ledger": audit,
        "node_counts": node_counts,
        "status_breakdown": status_breakdown,
    }


def _export(
    *, cypher: str, node_ids_json: str, disclosure_level: str, as_of: str, limit: int
) -> dict[str, Any]:
    engine = kg_server._get_engine()
    if engine is None:
        return {
            "surface": "compliance",
            "action": "export",
            "error": "IntelligenceGraphEngine not active",
        }

    try:
        raw_ids = json.loads(node_ids_json) if node_ids_json else []
    except (TypeError, ValueError) as exc:
        return {
            "surface": "compliance",
            "action": "export",
            "error": f"invalid node_ids: {exc}",
        }
    if not isinstance(raw_ids, list):
        return {
            "surface": "compliance",
            "action": "export",
            "error": "node_ids must decode to a JSON array",
        }

    ids: list[str] = [str(i) for i in raw_ids]
    if not ids and cypher.strip():
        try:
            rows = engine.query_cypher(cypher, as_of=as_of or None)
        except Exception as exc:  # noqa: BLE001 — surface as data, not a 500
            return {
                "surface": "compliance",
                "action": "export",
                "error": f"cypher selection failed: {exc}",
            }
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            rid = row.get("id")
            if isinstance(rid, str) and rid:
                ids.append(rid)

    if not ids:
        return {
            "surface": "compliance",
            "action": "export",
            "error": "supply node_ids (JSON array) or cypher (a query selecting an 'id' column)",
        }

    bounded_ids = ids[: max(1, int(limit))]

    from agent_utilities.mcp.tools.engine_tools import _dispatch

    level = (disclosure_level or "Full").strip()
    entries: list[dict[str, Any]] = []
    for node_id in bounded_ids:
        params = {"node_id": node_id}
        if level:
            params["disclosure_level"] = level
        raw = _dispatch(
            "query", {"explain_belief"}, "explain_belief", json.dumps(params), ""
        )
        try:
            belief = json.loads(raw)
        except (TypeError, ValueError):
            belief = {"error": "unparseable engine response"}
        entries.append({"node_id": node_id, "belief": belief})

    return {
        "surface": "compliance",
        "action": "export",
        "disclosure_level": level,
        "as_of": as_of or None,
        "requested": len(ids),
        "exported": len(entries),
        "truncated": len(ids) > len(bounded_ids),
        "entries": entries,
    }


def register_compliance_tools(mcp: Any) -> None:
    """Register the ``graph_compliance`` group on the given FastMCP server."""

    @mcp.tool(
        name="graph_compliance",
        description=(
            "Compliance posture rollup + redacted bulk export — an aggregation "
            "layer over primitives that already exist (no new compliance/"
            "redaction logic). Actions: 'posture' (join the tamper-evident "
            "hash-chained audit-ledger verify() report with node-count + "
            "status-breakdown of the governance labels already ingested by the "
            "CISO Assistant extractor + TRM portfolio-intelligence engine — "
            "Control/Policy/Risk/ComplianceRequirement/ComplianceGate/"
            "Regulation/ComplianceAssessment/Assessment/Incident/"
            "RemediationProposal/Finding/SecurityException), 'export' (bulk "
            "policy-redacted subgraph export: given node_ids (JSON array) or a "
            "read-only cypher query selecting an 'id' column, calls the "
            "engine's own explain_belief(node_id, disclosure_level) per id — "
            "the SAME per-node redaction primitive graph_epistemic's 'why' "
            "action uses — bounded by limit, honoring an optional as_of instant)."
        ),
        tags=["graph-os", "compliance", "audit", "governance", "redaction"],
    )
    def graph_compliance(
        action: str = Field(default="posture", description="posture | export"),
        cypher: str = Field(
            default="",
            description="Read-only Cypher selecting an 'id' column of nodes to "
            "export (export action; ignored when node_ids is non-empty). "
            'E.g. "MATCH (n:Control) RETURN n.id AS id LIMIT 100".',
        ),
        node_ids: str = Field(
            default="[]",
            description="JSON array of explicit node ids to export (export action).",
        ),
        disclosure_level: str = Field(
            default="Full",
            description="Full | Skeleton | ExistenceOnly — policy-aware redaction "
            "applied to every exported node (export action).",
        ),
        as_of: str = Field(
            default="",
            description="Optional ISO-8601 instant — bitemporal cutoff for the "
            "cypher id selection (export action).",
        ),
        limit: int = Field(
            default=200, description="Max nodes exported in one call (export action)."
        ),
    ) -> str:
        """Compliance posture rollup + redacted bulk subgraph export."""
        action_key = (action or "posture").strip().lower()
        if action_key == "posture":
            return json.dumps(_posture(), default=str)
        if action_key == "export":
            return json.dumps(
                _export(
                    cypher=cypher,
                    node_ids_json=node_ids,
                    disclosure_level=disclosure_level,
                    as_of=as_of,
                    limit=limit,
                ),
                default=str,
            )
        return json.dumps(
            {
                "surface": "compliance",
                "action": action_key,
                "error": f"unknown action {action_key!r}",
            }
        )

    kg_server.REGISTERED_TOOLS["graph_compliance"] = graph_compliance
    # No bespoke endpoint needed — the generic REST-twin factory in
    # kg_server._build_server mounts POST /compliance for every
    # ACTION_TOOL_ROUTES entry without a bespoke handler, dispatching through
    # the SAME _execute_tool core.
    kg_server.ACTION_TOOL_ROUTES["graph_compliance"] = "/compliance"
