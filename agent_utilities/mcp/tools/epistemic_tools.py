"""graph_epistemic — dedicated epistemic-answer MCP tool (CONCEPT:AU-KB-CURRENCY, Seam 1 follow-up).

The epistemic read layer (`EpistemicRow`/`EvidenceSpan`, `include_epistemic` on
`graph_query`/`graph_ask`/`KnowledgeGraph.query`) surfaces PER-ROW currency
upgrades, but the deeper per-claim diagnostics — "why do we believe this",
"why NOT", "what would change my mind", "what changed between two points in
time", "resolve this contradiction" — were only reachable via the generic
`engine_query` 1:1 passthrough (see the "Epistemic answers" section of the
`graph-query-and-explanation` skill), and even there "why not"/"what would
invalidate" had no dedicated action at all — only as fields buried inside
`epistemic_status`'s response a caller had to reach into by hand. None of the
above was a purpose-named tool.

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

Two more actions, ``why_not`` and ``what_would_invalidate``, are also
purpose-named entry points — but NOT a fifth/sixth ``client.query`` method.
``why_not``/``what_would_invalidate`` (``eg_epistemic::query``) have no
standalone wire ``Method``: grepping ``crates/eg-types/src/protocol.rs`` and
``src/server/handlers/query.rs`` in the epistemic-graph engine repo shows they
are only reachable as the ``why_not``/``what_would_invalidate`` FIELDS of
``Method::EpistemicStatus``'s response (``EpistemicStatusResult.status``,
already the same call ``status`` makes). So these two actions call
``client.query.epistemic_status(node_id)`` exactly like ``status`` does, then
PROJECT the response down to just that one field — a clean, first-class name
for a diagnostic that would otherwise require a caller to fetch the whole
acceptance capstone and reach into it by hand. Opt-in engine ``epistemic-tms``
feature, same as ``status``.

No epistemic logic is reimplemented here — every action is a direct call into
the engine client method the ``graph-query-and-explanation`` skill already
documents (plus, for the two projected actions, a field-select over that
call's own result); this tool only adds the ergonomic action-name mapping +
REST twin.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import Field

from agent_utilities.mcp import kg_server
from agent_utilities.security.error_surface import public_error_json

#: Friendly action name -> underlying ``client.query.<method>`` name.
#: ``why_not``/``what_would_invalidate`` deliberately map to the SAME
#: ``epistemic_status`` method as ``status`` — see the module docstring: they
#: have no standalone wire ``Method``, only a field on ``EpistemicStatus``'s
#: response, which :data:`_PROJECTED_STATUS_FIELDS` selects out below.
_ACTION_TO_METHOD: dict[str, str] = {
    "status": "epistemic_status",
    "why": "explain_belief",
    "what_changed": "what_changed",
    "resolve_conflict": "resolve_conflict",
    "why_not": "epistemic_status",
    "what_would_invalidate": "epistemic_status",
}

#: Actions whose engine call is ``epistemic_status`` but whose RESULT is one
#: named field of ``EpistemicStatusResult.status`` (CONCEPT:EPI-P3-5), not the
#: whole acceptance capstone — the action name IS the wire field name, so no
#: separate name->field map is needed.
_PROJECTED_STATUS_FIELDS: frozenset[str] = frozenset(
    {"why_not", "what_would_invalidate"}
)


def _epistemic_status_view(result: Any) -> dict[str, Any]:
    """The flat ``EpistemicStatusWire`` fields out of an ``epistemic_status`` call.

    The real wire payload (``EpistemicStatusResult``, `crates/eg-types/src/
    protocol.rs`) nests every field one level down under a ``status`` key
    (``{"status": {"claim":..., "why_not":..., "what_would_invalidate":...}}``)
    — confirmed against the engine's own round-trip test
    (``result.status.why_not`` in `tests/advanced_crossmodal_roundtrip.rs`).
    Accepts that real shape OR an already-flattened dict (e.g. a test stub),
    so callers never need to know which one they got.
    """
    if isinstance(result, dict) and isinstance(result.get("status"), dict):
        return result["status"]
    return result if isinstance(result, dict) else {}


def register_epistemic_tools(mcp: Any) -> None:
    """Register the ``graph_epistemic`` group on the given FastMCP server."""

    @mcp.tool(
        name="graph_epistemic",
        description=(
            "Purpose-named epistemic-answer surface over the engine's belief/"
            "provenance primitives (CONCEPT:AU-KB-CURRENCY) — 'why do we believe "
            "this', 'why NOT', 'what would change my mind', 'what changed between "
            "two points in time', 'resolve this contradiction'. Actions: 'status' "
            "(node_id -> acceptance capstone: believed? since when? on what "
            "evidence? what would invalidate it? requires the opt-in epistemic-tms "
            "engine feature), 'why' (node_id [+ optional "
            "disclosure_level='Full'|'Skeleton'|'ExistenceOnly'] -> the "
            "justification tree, policy-redacted when disclosure_level is set), "
            "'why_not' (node_id -> the diagnostic for a claim that is NOT "
            "believed: reason is one of 'Unknown' (claim absent from the graph), "
            "'InsufficientConfidence' (unattacked but too weak), 'Contradicted' "
            "(names the blocking 'blockers'), or 'Undecided' (a symmetric conflict "
            "— names the 'competing' claims), plus the claim's 'confidence'; null "
            "when the claim IS believed; requires epistemic-tms), "
            "'what_would_invalidate' (node_id -> the minimal evidence-id set whose "
            "retraction would flip belief — 'evidence_ids' + 'believed_now'/"
            "'believed_after'; null when no such flip exists within the search "
            "bound; requires epistemic-tms), 'what_changed' (tx_from + tx_to -> "
            "whole-graph bitemporal diff between two transaction times; requires "
            "epistemic-tms), 'resolve_conflict' (node_ids + optional "
            "semantics='grounded' -> argumentation-based resolution over a set of "
            "contradicting claims). Thin wrapper over the engine's own "
            "client.query methods — reuses the same dispatcher as engine_query, "
            "adds no new belief logic. 'why_not'/'what_would_invalidate' have no "
            "standalone engine method: both call the SAME client.query."
            "epistemic_status(node_id) 'status' does, then project the response "
            "down to just that one field (EpistemicStatusResult.status.why_not / "
            ".what_would_invalidate on the wire) — a purpose-named shortcut, not a "
            "new capability. A build/config that doesn't expose an action "
            "degrades to a clear error rather than raising."
        ),
        tags=["graph-os", "epistemic", "belief", "provenance", "engine"],
    )
    def graph_epistemic(
        action: str = Field(
            default="why",
            description="status | why | why_not | what_would_invalidate | "
            "what_changed | resolve_conflict",
        ),
        node_id: str = Field(
            default="",
            description="Claim/node id (status, why, why_not, what_would_invalidate).",
        ),
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
        """Epistemic-answer surface: status / why / why_not / what_would_invalidate /
        what_changed / resolve_conflict."""
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
                return public_error_json(
                    exc,
                    code="invalid_request",
                    context={"surface": "epistemic", "action": action_key},
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
        # why_not/what_would_invalidate: project epistemic_status's response down
        # to the one field they're named after (see _epistemic_status_view) —
        # but never swallow a dispatch-level failure (unbuilt epistemic-tms, a
        # bad node_id, a transport error, ...), which surfaces as {"error": ...}
        # and must pass through unprojected.
        if (
            action_key in _PROJECTED_STATUS_FIELDS
            and isinstance(result, dict)
            and "error" not in result
        ):
            result = _epistemic_status_view(result).get(action_key)
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
