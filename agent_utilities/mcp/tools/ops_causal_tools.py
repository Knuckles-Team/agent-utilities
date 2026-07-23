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

W3.5 — ``as_claim`` (CONCEPT:AU-KG.enrichment.ops-causal-graph,
CONCEPT:AU-KG.evolution.mining-flywheel): ``materialize_claims`` above
(W2 wire-up #3) already writes root-cause findings as raw ``:Claim`` nodes,
unconditionally, with no governance and no lifecycle audit trail — an
ops-causal finding could not be challenged/validated/promoted through the
epistemic machinery the rest of the codebase already has for mined claims.
The opt-in ``as_claim=true`` parameter on ``root_cause``/``blast_radius``
closes that gap: it proposes ONE structured, citable, revisable Claim
summarizing the call's analysis through the SAME governed path
``graph_claims action="propose"`` uses — the SAME fail-closed
``ActionPolicy`` gate (``kind="claim.propose"``, the ``_gate`` pattern
``claim_tools.py``/``secret_tools.py`` already use) and the SAME
:class:`~agent_utilities.knowledge_graph.research.claim_flywheel.ClaimFlywheel`
state machine — never a second lifecycle. A denied gate never blocks the
read-only analysis itself; it only adds a ``claim_denied`` note. See
:func:`_propose_ops_causal_claim`.
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


def _gate(kind: str, target: str, reason: str) -> tuple[bool, dict]:
    """Run the fail-closed ActionPolicy gate for one ``as_claim`` proposal.

    Mirrors ``claim_tools._gate``/``secret_tools._gate`` exactly (each MCP
    tool module ships its own copy of this same thin wrapper, per those
    modules' docstrings) — a thin call into the shared
    ``get_action_policy(engine).decide(...)`` decision point, never a
    reimplementation of its rules. Using the SAME ``kind="claim.propose"``
    :func:`_propose_ops_causal_claim` passes here means one governance rule
    covers a proposal regardless of whether it came from ``graph_claims`` or
    ``graph_ops_causal``.
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


#: Documented conservative default confidence for an ``as_claim`` blast-radius
#: finding. ``blast_radius_analysis`` is a pure structural-reachability walk
#: with no per-node ranking signal (unlike ``root_cause_rank``'s
#: ``path_strength``), so there is no real tool score to report — never
#: fabricate one. Deliberately kept below
#: ``candidate_insight.CONFIDENCE_FLOOR`` (0.6) so a blast-radius claim never
#: reads as highly confident: it asserts REACHABILITY, not likelihood.
_BLAST_RADIUS_CLAIM_CONFIDENCE = 0.4


def _touched_node_ids(seed_node_id: str, rows: list[dict[str, Any]]) -> list[str]:
    """Every node id a causal walk actually visited: the seed plus each result
    row's own ``node_id`` and every hop on its ``path`` (when present) — the
    full, honest evidence-ref set for an ``as_claim`` proposal, never just
    the two endpoints of one path."""
    touched = {seed_node_id}
    for row in rows:
        node_id = row.get("node_id")
        if node_id:
            touched.add(str(node_id))
        for hop in row.get("path") or []:
            if hop:
                touched.add(str(hop))
    return sorted(touched)


def _root_cause_claim_finding(
    seed_node_id: str, ranked: list[dict[str, Any]]
) -> tuple[str, list[str], float] | None:
    """Build the ``as_claim`` finding for a ``root_cause`` result: one
    structured one-line statement about the TOP-ranked candidate (``ranked``
    is already sorted root-ness-first by ``root_cause_rank``), evidence refs
    spanning every node the ranked walk touched, confidence from the top
    candidate's own ``path_strength`` — NEVER ``score``, which
    ``root_cause_rank``'s own docstring documents as a tie-breaker among
    nodes of the SAME root-ness, not a confidence signal (the identical
    choice :func:`~agent_utilities.knowledge_graph.research.candidate_insight.
    candidates_from_ops_causal_root_cause` already makes, for the same
    reason). Returns ``None`` when there is nothing to claim (no ranked
    ancestors) — never a fabricated claim with no content.
    """
    if not ranked:
        return None
    top = ranked[0]
    statement = (
        f"Ops-causal root-cause analysis of {seed_node_id}: {top.get('node_id')} "
        f"is {'a ROOT cause' if top.get('is_root') else 'an upstream contributor'} "
        f"(path_strength={top.get('path_strength')}, hops={top.get('hops')}, "
        f"stage={top.get('stage')}) — {len(ranked)} candidate(s) ranked."
    )
    confidence = max(0.0, min(1.0, float(top.get("path_strength") or 0.0)))
    return statement, _touched_node_ids(seed_node_id, ranked), confidence


def _blast_radius_claim_finding(
    seed_node_id: str, downstream: list[dict[str, Any]]
) -> tuple[str, list[str], float] | None:
    """Build the ``as_claim`` finding for a ``blast_radius`` result: one
    structured one-line statement about the change's downstream impact,
    evidence refs spanning every node the reachability walk touched,
    confidence = :data:`_BLAST_RADIUS_CLAIM_CONFIDENCE` (a documented
    conservative default — see that constant's docstring). Returns ``None``
    when nothing is downstream — no blast radius to claim.
    """
    if not downstream:
        return None
    nearest = min(int(row.get("depth") or 0) for row in downstream)
    statement = (
        f"Ops-causal blast-radius analysis: change {seed_node_id} affects "
        f"{len(downstream)} downstream node(s), nearest at depth {nearest}."
    )
    return (
        statement,
        _touched_node_ids(seed_node_id, downstream),
        _BLAST_RADIUS_CLAIM_CONFIDENCE,
    )


def _propose_ops_causal_claim(
    engine: Any,
    *,
    action: str,
    seed_node_id: str,
    finding: tuple[str, list[str], float],
) -> dict[str, Any]:
    """Propose one ops-causal finding through the SAME governed ClaimFlywheel
    lifecycle path ``graph_claims action="propose"`` uses (X-3, CONCEPT:AU-KG.
    evolution.mining-flywheel) — the ``as_claim=true`` opt-in on
    ``root_cause``/``blast_radius`` (W3.5, CONCEPT:AU-KG.enrichment.
    ops-causal-graph).

    Thin reuse, not a second lifecycle:

    * The ActionPolicy gate is the SAME ``kind="claim.propose"`` decision
      ``graph_claims`` gates its own ``propose`` action on (:func:`_gate`) —
      one governance rule covers both entry points.
    * The claim is the SAME :class:`~agent_utilities.models.knowledge_graph.
      ClaimNode` shape every other claim producer in this codebase writes
      (``status="proposal"``, ``is_verified=False`` — ALWAYS, never
      self-verifying).
    * Evidence refs are persisted as REAL ``DERIVED_FROM`` edges via the SAME
      shared seam every mining family already uses
      (:func:`~agent_utilities.knowledge_graph.research.candidate_insight.
      register_claim_materialization`) — not a JSON-only echo.
    * The lifecycle transition itself is the SAME
      :meth:`~agent_utilities.knowledge_graph.research.claim_flywheel.
      ClaimFlywheel.propose` ``graph_claims action="propose"`` calls.

    None of the above is reimplemented here.

    ``finding`` is ``(statement, evidence_ids, confidence)`` — see
    :func:`_root_cause_claim_finding`/:func:`_blast_radius_claim_finding`. The
    claim id is content-addressed on ``(action, seed_node_id)`` so a repeat
    call over the SAME analysis converges on the SAME claim (idempotent
    upsert, matching every other claim id in this codebase) rather than
    minting a duplicate on every call.

    Returns the fields to fold into the tool's JSON response:
    ``{"claim_denied": {...}}`` when the ActionPolicy gate denies (the
    read-only analysis is NEVER blocked by this — the caller merges these
    fields onto the answer regardless of the outcome); otherwise
    ``{"claim_id": ..., "claim_transition": {...} | None}`` (plus
    ``"claim_write_errors"`` only when a best-effort provenance write
    failed).
    """
    from datetime import UTC, datetime

    from agent_utilities.knowledge_graph.research.candidate_insight import (
        register_claim_materialization,
    )
    from agent_utilities.knowledge_graph.research.claim_flywheel import ClaimFlywheel
    from agent_utilities.models.knowledge_graph import ClaimNode, RegistryNodeType

    statement, evidence_ids, confidence = finding
    claim_id = f"claim:ops_causal:{action}:{seed_node_id}"
    reason = f"ops_causal {action} analysis of {seed_node_id}"

    allowed, policy = _gate("claim.propose", claim_id, reason)
    if not allowed:
        return {"claim_denied": policy}

    claim = ClaimNode(
        id=claim_id,
        type=RegistryNodeType.CLAIM,
        name=statement[:120],
        claim_text=statement,
        confidence=confidence,
        claim_type="finding",
        source_ids=evidence_ids,
        extracted_from=seed_node_id,
        domain="ops_causal",
        is_verified=False,
        metadata={
            "finding_type": "OpsCausalFinding",
            "ops_causal_action": action,
            # PROV-O generator tagging (docs/pillars/2_epistemic_knowledge_graph/
            # company_brain/provenance.md's wasGeneratedBy/generatedAtTime
            # alignment; the same property names owlready2_backend's PROV-O
            # edge-alias table already recognizes). Kept as plain Claim
            # metadata rather than a separate prov:Activity node/edge —
            # thin tagging, not a new provenance subsystem.
            "was_generated_by": "mcp:graph_ops_causal",
            "generated_at_time": datetime.now(UTC).isoformat(),
        },
        importance_score=confidence,
    )

    errors: list[str] = []
    try:
        engine.add_node(
            claim.id,
            "Claim",
            properties={
                **claim.model_dump(mode="json", exclude={"type"}),
                "status": "proposal",
            },
        )
    except Exception as e:  # noqa: BLE001 — persistence is best-effort
        errors.append(f"ops_causal_as_claim:persist {claim.id}: {e}")
        return {"claim_id": claim.id, "claim_write_errors": errors}

    register_claim_materialization(engine, claim, errors, context="ops_causal_as_claim")

    flywheel = ClaimFlywheel(engine)
    transition = flywheel.propose(claim.id, reason=reason)

    out: dict[str, Any] = {
        "claim_id": claim.id,
        "claim_transition": transition.to_dict() if transition else None,
    }
    if errors:
        out["claim_write_errors"] = errors
    return out


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
            "to skip the write for a pure read. 'root_cause'/'blast_radius' also "
            "accept as_claim=true: propose ONE structured finding through the SAME "
            "governed ClaimFlywheel lifecycle `graph_claims propose` uses, gated by "
            "the SAME ActionPolicy claim.propose decision — a denial never blocks "
            "this read-only answer, it only adds a claim_denied note; on success "
            "the minted claim_id is returned alongside the analysis."
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
        as_claim: bool = Field(
            default=False,
            description="root_cause/blast_radius only: propose this call's finding "
            "through the SAME governed ClaimFlywheel lifecycle `graph_claims "
            "propose` uses (CONCEPT:AU-KG.evolution.mining-flywheel), gated by the "
            "SAME fail-closed ActionPolicy claim.propose decision. Off by default "
            "— the read-only analysis is returned either way; a denied gate adds "
            "a claim_denied note instead of blocking the answer. On success, "
            "returns claim_id (+ claim_transition) alongside the analysis.",
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
        as_claim_fields: dict[str, Any] = {}

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
                if as_claim and engine is not None:
                    finding = _root_cause_claim_finding(node_id, result)
                    if finding is not None:
                        try:
                            as_claim_fields = _propose_ops_causal_claim(
                                engine,
                                action="root_cause",
                                seed_node_id=node_id,
                                finding=finding,
                            )
                        except Exception as e:  # noqa: BLE001 — never blocks the read-only answer
                            as_claim_fields = {"claim_error": str(e)}
            elif action == "blast_radius":
                if not node_id:
                    raise ValueError("node_id required for blast_radius")
                result = blast_radius_analysis(
                    model, node_id, depth=depth, max_results=max_results
                )
                if as_claim and engine is not None:
                    finding = _blast_radius_claim_finding(node_id, result)
                    if finding is not None:
                        try:
                            as_claim_fields = _propose_ops_causal_claim(
                                engine,
                                action="blast_radius",
                                seed_node_id=node_id,
                                finding=finding,
                            )
                        except Exception as e:  # noqa: BLE001 — never blocks the read-only answer
                            as_claim_fields = {"claim_error": str(e)}
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
        response.update(as_claim_fields)
        return json.dumps(response, default=str)

    kg_server.REGISTERED_TOOLS["graph_ops_causal"] = graph_ops_causal
    # No bespoke endpoint needed — the generic REST-twin factory in
    # kg_server._build_server (CONCEPT:AU-KG.coordination.engine-message-broker)
    # mounts POST /ops/causal for every ACTION_TOOL_ROUTES entry without a
    # bespoke handler, dispatching through the SAME _execute_tool core.
    kg_server.ACTION_TOOL_ROUTES["graph_ops_causal"] = "/ops/causal"
