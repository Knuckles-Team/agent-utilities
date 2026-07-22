"""graph_ops_causal — Enterprise Operations Causal Graph MCP tool (Codex X-2).

CONCEPT:AU-KG.enrichment.ops-causal-graph

One coherent, action-routed surface over
:mod:`agent_utilities.knowledge_graph.enrichment.ops_causal_graph` — the join
layer that links Langfuse traces -> agent/tool/model -> service ->
deployment -> commit/merge-request -> incident/change -> capability/owner ->
policy/control/evidence, plus the analyses built on top of the causal-
reasoning engine already shipped
(:mod:`agent_utilities.knowledge_graph.core.formal_reasoning_core`).

Mirrors the ``graph_mine``/``graph_code`` action-router shape (single
``@mcp.tool``, an ``action`` enum, JSON payload fields, registered into
``kg_server.REGISTERED_TOOLS``) rather than inventing a new tool convention.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import Field

from agent_utilities.mcp import kg_server
from agent_utilities.security.error_surface import public_error_json


def _parse_links(links_json: str) -> list[Any]:
    from agent_utilities.knowledge_graph.enrichment.ops_causal_graph import (
        OpsCausalLink,
    )

    if not links_json:
        return []
    try:
        raw = json.loads(links_json)
    except (TypeError, ValueError):
        raise ValueError("invalid links_json") from None
    if not isinstance(raw, list):
        raise ValueError("links_json must decode to a JSON array")
    links: list[OpsCausalLink] = []
    for item in raw:
        if isinstance(item, dict):
            links.append(
                OpsCausalLink(
                    source=str(item["source"]),
                    target=str(item["target"]),
                    rel_type=str(item.get("rel_type", "related_to")),
                    strength=float(item.get("strength", 1.0)),
                    observed_at=item.get("observed_at"),
                    mechanism=str(item.get("mechanism", "")),
                )
            )
        elif isinstance(item, list | tuple) and len(item) >= 3:
            links.append(
                OpsCausalLink(
                    source=str(item[0]), rel_type=str(item[1]), target=str(item[2])
                )
            )
        else:
            raise ValueError(f"unrecognized link entry: {item!r}")
    return links


def _materialize_root_cause_claims(
    engine: Any, failure_node: str, ranked: list[dict[str, Any]]
) -> tuple[list[str], list[str]]:
    """Persist each above-floor ``root_cause_rank`` finding as a reviewable ``:Claim``.

    W2 wire-up #3 (CONCEPT:AU-KG.enrichment.ops-causal-graph) — the ops-causal → claim loop.
    Reuses the SAME ``CandidateInsight`` → ``ClaimNode`` → ``register_claim_materialization``
    pipeline the mining flywheel already uses for association-rule/anomaly/predicted-edge/
    sequential-pattern findings (``knowledge_graph/research/candidate_insight.py``,
    ``loop_controller._run_insight_validation``): a finding below
    ``CONFIDENCE_FLOOR`` is skipped (never a low-confidence Claim); every claim persists
    with ``status="proposal"``/``is_verified=False`` — ALWAYS, never self-verifying,
    same propose-only floor the mining flywheel guarantees. Actual promotion review/
    veto is the SAME downstream governance gate (``action_policy.decide``, the loop
    engine's flywheel review surface) other mined claims go through — this function
    only proposes, exactly like every other ``CandidateInsight`` producer.

    Best-effort: a single finding's write failure is recorded in the returned
    error list and does not stop the rest — mirrors every other best-effort
    persist loop in this module family. Returns ``(claim_ids_written, errors)``.
    """
    from agent_utilities.knowledge_graph.research.candidate_insight import (
        candidates_from_ops_causal_root_cause,
        register_claim_materialization,
    )

    claim_ids: list[str] = []
    errors: list[str] = []
    for cand in candidates_from_ops_causal_root_cause(failure_node, ranked):
        if not cand.clears_floor:
            continue
        claim = cand.to_claim_node()
        bundle = cand.to_evidence_bundle()
        try:
            engine.add_node(
                claim.id,
                "Claim",
                properties={
                    **claim.model_dump(mode="json", exclude={"type"}),
                    "status": "proposal",
                    "evidence_bundle_json": bundle.model_dump_json(),
                },
            )
        except Exception as e:  # noqa: BLE001 — persistence is best-effort
            errors.append(f"ops_causal:persist {claim.id}: {e}")
            continue
        claim_ids.append(claim.id)
        register_claim_materialization(engine, claim, errors, context="ops_causal")
    return claim_ids, errors


def register_ops_causal_tools(mcp: Any) -> None:
    """Register the ``graph_ops_causal`` group on the given FastMCP server."""

    @mcp.tool(
        name="graph_ops_causal",
        description=(
            "Enterprise operations causal graph (Codex X-2): joins Langfuse traces -> "
            "agent/tool/model -> service -> deployment/container -> commit/merge-"
            "request -> incident/change -> capability/owner -> policy/control/"
            "evidence into one causal chain, and runs root-cause/blast-radius/"
            "change-risk/control-evidence analyses on it. Reuses the causal-"
            "reasoning engine already shipped (StructuralCausalModel + "
            "CausalVerifier + SpuriousnessDetector) — no new traversal algorithm. "
            "Actions: 'root_cause' (rank probable root-cause changes/services for "
            "a failure node_id, upstream), 'blast_radius' (downstream impact of a "
            "change node_id), 'change_risk' (predict risk of a proposed change "
            "node_id from its blast radius + incident_history_json), "
            "'control_evidence' (gather + verify the evidence chain for a control "
            "node_id), 'join' (materialize links_json as real graph edges via the "
            "shared enrichment writer — no new nodes, only edges between ids that "
            "already exist). Supply the causal edges explicitly via links_json "
            "([{source,target,rel_type,strength,observed_at}, ...] or "
            "[[source,rel_type,target], ...]) for an offline/test-friendly model, "
            "or omit it with an active engine + node_id to load the neighborhood "
            "live from the KG. 'root_cause' additionally MATERIALIZES each "
            "above-floor finding as a reviewable ``:Claim`` (status='proposal', "
            "never auto-verified — CONCEPT:AU-KG.enrichment.ops-causal-graph) "
            "when an engine is active, so a root-cause finding becomes queryable/"
            "citable instead of a one-off JSON answer; set materialize_claims=false "
            "to skip the write for a pure read."
        ),
        tags=["graph-os", "ops", "causal", "root-cause", "blast-radius"],
    )
    def graph_ops_causal(
        action: str = Field(
            default="root_cause",
            description="root_cause | blast_radius | change_risk | control_evidence | join",
        ),
        node_id: str = Field(
            default="",
            description="Seed node id: the failure/trace (root_cause), the "
            "change/commit (blast_radius, change_risk), or the control "
            "(control_evidence).",
        ),
        links_json: str = Field(
            default="[]",
            description="JSON array of ops-causal edges: "
            '[{"source":..,"target":..,"rel_type":..,"strength":1.0,'
            '"observed_at":null}, ...] or [[source,rel_type,target], ...]. '
            "Empty + an active engine ⇒ load the neighborhood live from the KG "
            "around node_id (join, root_cause, blast_radius, control_evidence).",
        ),
        depth: int = Field(default=6, description="Traversal depth bound."),
        max_results: int = Field(
            default=10, description="Result cap (root_cause / blast_radius)."
        ),
        incident_history_json: str = Field(
            default="[]",
            description='JSON array of {"node_id":..,"severity":0..1} historical '
            "incidents (change_risk).",
        ),
        now: float = Field(
            default=0.0,
            description="Unix seconds 'current time' for recency weighting "
            "(root_cause); 0 ⇒ no recency weighting.",
        ),
        materialize_claims: bool = Field(
            default=True,
            description="root_cause only: persist each above-floor finding as a "
            "reviewable :Claim (CONCEPT:AU-KG.enrichment.ops-causal-graph). Default-on when an "
            "engine is active; set False for a pure read with no graph write.",
        ),
    ) -> str:
        """Ops causal graph: join + root-cause/blast-radius/change-risk/control-evidence."""
        from agent_utilities.knowledge_graph.enrichment.ops_causal_graph import (
            blast_radius_analysis,
            build_causal_model,
            change_risk_score,
            control_evidence_chain,
            load_ops_causal_neighborhood,
            materialize_ops_causal_links,
            root_cause_rank,
        )

        action = (action or "root_cause").strip().lower()
        engine = kg_server._get_engine()

        try:
            links = _parse_links(links_json)
        except ValueError as exc:
            return public_error_json(
                exc,
                code="invalid_request",
                context={"surface": "ops_causal", "action": action},
            )

        if not links and node_id and engine is not None:
            links = load_ops_causal_neighborhood(engine, node_id, depth=depth)

        if action == "join":
            backend = getattr(engine, "backend", None) if engine else None
            if backend is None:
                return json.dumps(
                    {
                        "surface": "ops_causal",
                        "action": action,
                        "error": "no engine backend available to materialize links",
                    }
                )
            nodes_written, edges_written = materialize_ops_causal_links(backend, links)
            return json.dumps(
                {
                    "surface": "ops_causal",
                    "action": action,
                    "result": {
                        "nodes_written": nodes_written,
                        "edges_written": edges_written,
                    },
                }
            )

        model = build_causal_model(links)
        claims_materialized: list[str] = []
        claim_errors: list[str] = []

        try:
            if action == "root_cause":
                result: Any
                if not node_id:
                    raise ValueError("node_id required for root_cause")
                result = root_cause_rank(
                    model,
                    node_id,
                    max_results=max_results,
                    now=now or None,
                )
                if materialize_claims and engine is not None:
                    claims_materialized, claim_errors = _materialize_root_cause_claims(
                        engine, node_id, result
                    )
            elif action == "blast_radius":
                if not node_id:
                    raise ValueError("node_id required for blast_radius")
                result = blast_radius_analysis(
                    model, node_id, depth=depth, max_results=max_results
                )
            elif action == "change_risk":
                if not node_id:
                    raise ValueError("node_id required for change_risk")
                history = (
                    json.loads(incident_history_json) if incident_history_json else []
                )
                if not isinstance(history, list):
                    raise ValueError(
                        "incident_history_json must decode to a JSON array"
                    )
                result = change_risk_score(model, node_id, incident_history=history)
            elif action == "control_evidence":
                if not node_id:
                    raise ValueError("node_id required for control_evidence")
                result = control_evidence_chain(model, node_id)
            else:
                return json.dumps(
                    {
                        "surface": "ops_causal",
                        "action": action,
                        "error": f"unknown action {action!r}",
                    }
                )
        except (ValueError, TypeError) as exc:
            return public_error_json(
                exc,
                code="invalid_request",
                context={"surface": "ops_causal", "action": action},
            )

        response: dict[str, Any] = {
            "surface": "ops_causal",
            "action": action,
            "result": result,
        }
        if action == "root_cause":
            response["claims_materialized"] = claims_materialized
            if claim_errors:
                response["claim_errors"] = claim_errors
        return json.dumps(response, default=str)

    kg_server.REGISTERED_TOOLS["graph_ops_causal"] = graph_ops_causal
    # No bespoke endpoint needed — the generic REST-twin factory in
    # kg_server._build_server (CONCEPT:AU-KG.coordination.engine-message-broker)
    # mounts POST /ops/causal for every ACTION_TOOL_ROUTES entry without a
    # bespoke handler, dispatching through the SAME _execute_tool core.
    kg_server.ACTION_TOOL_ROUTES["graph_ops_causal"] = "/ops/causal"
