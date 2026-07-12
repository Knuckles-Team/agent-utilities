"""graph_epistemic — dedicated epistemic-answer MCP tool (CONCEPT:AU-KB-CURRENCY, Seam 1 follow-up).

The epistemic read layer (`EpistemicRow`/`EvidenceSpan`, `include_epistemic` on
`graph_query`/`graph_ask`/`KnowledgeGraph.query`) surfaces PER-ROW currency
upgrades, but the deeper per-claim diagnostics — "why do we believe this",
"what changed between two points in time", "resolve this contradiction" — were
only reachable via the generic `engine_query` 1:1 passthrough (see the
`kg-epistemic-answer` skill), never as a purpose-named tool.

This module is a THIN, purpose-named wrapper over four existing
``client.query.*`` methods on the connected epistemic-graph engine (discovered
the same way ``engine_tools.py`` discovers every ``engine_<domain>`` action —
this module reuses that SAME generic dispatcher, ``engine_tools._dispatch``,
rather than re-implementing client resolution / connection pooling / the
ADMIN-scope gate):

* ``status``          -> ``client.query.epistemic_status(node_id)`` — the
  acceptance capstone (believed? since when? on what evidence? what would
  invalidate it?). Opt-in engine ``epistemic-tms`` feature.
* ``why``              -> ``client.query.explain_belief(node_id,
  disclosure_level)`` — the justification tree (Asserted / DerivedSupport /
  DerivedContradiction / BayesianUpdate), optionally policy-redacted via
  ``disclosure_level`` (Full/Skeleton/ExistenceOnly). In the default engine
  build.
* ``what_changed``     -> ``client.query.what_changed(tx_from, tx_to)`` — a
  whole-graph bitemporal diff between two transaction times. Opt-in engine
  ``epistemic-tms`` feature.
* ``resolve_conflict`` -> ``client.query.resolve_conflict(node_ids,
  semantics)`` — argumentation-based conflict resolution over a set of
  contradicting claim ids.

No epistemic logic is reimplemented here — every action is a direct call into
the engine client method the ``kg-epistemic-answer`` skill already documents;
this tool only adds the ergonomic action-name mapping + REST twin.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import Field

from agent_utilities.mcp import kg_server

#: Friendly action name -> underlying ``client.query.<method>`` name.
_ACTION_TO_METHOD: dict[str, str] = {
    "status": "epistemic_status",
    "why": "explain_belief",
    "what_changed": "what_changed",
    "resolve_conflict": "resolve_conflict",
}


def register_epistemic_tools(mcp: Any) -> None:
    """Register the ``graph_epistemic`` group on the given FastMCP server."""

    @mcp.tool(
        name="graph_epistemic",
        description=(
            "Purpose-named epistemic-answer surface over the engine's belief/"
            "provenance primitives (CONCEPT:AU-KB-CURRENCY) — 'why do we believe "
            "this', 'what changed between two points in time', 'resolve this "
            "contradiction'. Actions: 'status' (node_id -> acceptance capstone: "
            "believed? since when? on what evidence? what would invalidate it? "
            "requires the opt-in epistemic-tms engine feature), 'why' (node_id "
            "[+ optional disclosure_level='Full'|'Skeleton'|'ExistenceOnly'] -> "
            "the justification tree, policy-redacted when disclosure_level is "
            "set), 'what_changed' (tx_from + tx_to -> whole-graph bitemporal "
            "diff between two transaction times; requires epistemic-tms), "
            "'resolve_conflict' (node_ids + optional semantics='grounded' -> "
            "argumentation-based resolution over a set of contradicting claims). "
            "Thin wrapper over the engine's own client.query methods — reuses "
            "the same dispatcher as engine_query, adds no new belief logic. A "
            "build/config that doesn't expose an action degrades to a clear "
            "error rather than raising."
        ),
        tags=["graph-os", "epistemic", "belief", "provenance", "engine"],
    )
    def graph_epistemic(
        action: str = Field(
            default="why", description="status | why | what_changed | resolve_conflict"
        ),
        node_id: str = Field(default="", description="Claim/node id (status, why)."),
        node_ids: str = Field(
            default="[]",
            description="JSON array of contradicting node ids (resolve_conflict).",
        ),
        disclosure_level: str = Field(
            default="",
            description="Full | Skeleton | ExistenceOnly — policy-aware redaction "
            "(why). Empty = engine default (full disclosure).",
        ),
        tx_from: int = Field(
            default=0, description="Lower transaction-time bound (what_changed)."
        ),
        tx_to: int = Field(
            default=0, description="Upper transaction-time bound (what_changed)."
        ),
        semantics: str = Field(
            default="grounded",
            description="Argumentation semantics for resolve_conflict "
            "(e.g. 'grounded').",
        ),
        graph: str = Field(
            default="", description="Target graph name (empty = deployment default)."
        ),
    ) -> str:
        """Epistemic-answer surface: status / why / what_changed / resolve_conflict."""
        from agent_utilities.mcp.tools.engine_tools import _dispatch

        action_key = (action or "why").strip().lower()
        method = _ACTION_TO_METHOD.get(action_key)
        if method is None:
            return json.dumps(
                {
                    "surface": "epistemic",
                    "action": action_key,
                    "error": f"unknown action {action_key!r}",
                    "actions": sorted(_ACTION_TO_METHOD),
                }
            )

        if method in ("epistemic_status", "explain_belief"):
            if not node_id:
                return json.dumps(
                    {
                        "surface": "epistemic",
                        "action": action_key,
                        "error": f"node_id required for {action_key!r}",
                    }
                )
            params: dict[str, Any] = {"node_id": node_id}
            if method == "explain_belief" and disclosure_level:
                params["disclosure_level"] = disclosure_level
        elif method == "what_changed":
            params = {"tx_from": int(tx_from), "tx_to": int(tx_to)}
        else:  # resolve_conflict
            try:
                parsed_ids = json.loads(node_ids) if node_ids else []
            except (TypeError, ValueError) as exc:
                return json.dumps(
                    {
                        "surface": "epistemic",
                        "action": action_key,
                        "error": f"invalid node_ids: {exc}",
                    }
                )
            if not isinstance(parsed_ids, list) or not parsed_ids:
                return json.dumps(
                    {
                        "surface": "epistemic",
                        "action": action_key,
                        "error": "node_ids must be a non-empty JSON array for resolve_conflict",
                    }
                )
            params = {
                "node_ids": [str(i) for i in parsed_ids],
                "semantics": semantics or "grounded",
            }

        raw = _dispatch(
            "query", set(_ACTION_TO_METHOD.values()), method, json.dumps(params), graph
        )
        try:
            result = json.loads(raw)
        except (TypeError, ValueError):
            result = {"raw": raw}
        return json.dumps(
            {
                "surface": "epistemic",
                "action": action_key,
                "engine_method": method,
                "result": result,
            },
            default=str,
        )

    kg_server.REGISTERED_TOOLS["graph_epistemic"] = graph_epistemic
    # No bespoke endpoint needed — the generic REST-twin factory in
    # kg_server._build_server mounts POST /epistemic for every ACTION_TOOL_ROUTES
    # entry without a bespoke handler, dispatching through the SAME
    # _execute_tool core.
    kg_server.ACTION_TOOL_ROUTES["graph_epistemic"] = "/epistemic"
