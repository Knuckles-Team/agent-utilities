"""graph_claims MCP tool — the X-3 epistemic mining flywheel's governed claim
lifecycle (CONCEPT:AU-KG.evolution.mining-flywheel), exposed as a standalone,
directly-callable surface.

Until now the five-state ``proposed -> validated -> accepted -> deprecated ->
retracted`` lifecycle (``knowledge_graph/research/claim_flywheel.py::
ClaimFlywheel``) only ever fired as a side effect of a mining pass
(``loop_controller._run_insight_validation``/``_run_trace_mining``) — a caller
could not directly propose/validate/accept/deprecate/retract one claim through
either the MCP or REST surface. This tool closes that gap.

**Thin dispatch only** — this module never reimplements the lifecycle's
transition rules (the ``ClaimLifecycleState`` machine, terminal/sticky
retraction) or its governance semantics. Every action is either a straight
read off :class:`~agent_utilities.knowledge_graph.research.claim_flywheel.
ClaimFlywheel` (``get``/``list``), or — for the five state-changing actions —
FIRST a fail-closed :class:`~agent_utilities.orchestration.action_policy.
ActionPolicy` decision (mirroring ``graph_secret``'s ``_gate`` pattern in
``secret_tools.py``) and only THEN, if allowed, the matching
``ClaimFlywheel`` method call. A denied/queued decision returns
``{"error": "policy_denied", ...}`` without ever calling the flywheel.

Provenance is inherited, never re-written here: ``ActionPolicy.decide()``
persists its own ``ActionDecision`` audit node, and every real
``ClaimFlywheel`` transition persists its own append-only
``ClaimLifecycleEvent`` node — both writes already exist upstream of this
tool.

**The universal-ingestion program's governed promotion** (CONCEPT:
AU-KG.ingest.governed-claim-promotion, ``knowledge_graph/ingestion/
promotion.py``) hooks into this SAME ``accept``/``retract`` dispatch rather
than a separate tool: ``accept`` calls ``promotion.materialize_on_claim_accepted``
(the ONLY place an ingestion candidate claim's proposed fact gets written —
so the fail-closed ``ActionPolicy`` approval-queue hold on ``claim.accept``
IS that program's steward-review hold, not a second gate), and ``retract``
calls ``promotion.supersede_materialized_claim`` (tombstones an already-
materialized fact rather than deleting it). Both hooks are no-ops for any
claim without a ``proposed_envelope`` (every non-ingestion claim family).
"""

from __future__ import annotations

import json

from pydantic import Field

from agent_utilities.mcp import kg_server
from agent_utilities.security.error_surface import public_error_json


def _gate(kind: str, target: str, reason: str) -> tuple[bool, dict]:
    """Run the fail-closed ActionPolicy gate for one claim-lifecycle mutation.

    Mirrors ``secret_tools._gate`` exactly — a thin call into the shared
    ``get_action_policy(engine).decide(...)`` decision point, never a
    reimplementation of its rules.
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


def register_claim_tools(mcp):
    """Register the ``graph_claims`` tool onto the MCP server."""

    @mcp.tool(
        name="graph_claims",
        description=(
            "Drive the X-3 epistemic mining flywheel's governed claim lifecycle "
            "(CONCEPT:AU-KG.evolution.mining-flywheel) directly: proposed -> "
            "validated -> accepted -> deprecated -> retracted (any pre-terminal "
            "state may also be retracted directly; RETRACTED is TERMINAL and "
            "STICKY — 'propose' refuses to reopen a retracted claim). Every "
            "state-changing action ('propose', 'validate', 'accept', 'deprecate', "
            "'retract') FIRST passes the fail-closed ActionPolicy gate "
            "(orchestration/action_policy.py, kind='claim.<action>') — a deny/"
            "queue verdict blocks the mutation and is returned under `policy` — "
            "and only then is routed through the SAME ClaimFlywheel the mining "
            "pipeline uses; this tool never reimplements the transition rules or "
            "recomputes governance validity itself. For an ingestion candidate "
            "claim (knowledge_graph/ingestion/promotion.py) this IS the steward-"
            "review hold: 'accept' is the only action that materializes the "
            "claim's proposed fact (returned under `materialization`), and "
            "'retract' supersedes an already-materialized fact rather than "
            "deleting it (returned under `superseded_fact`) — both are no-ops "
            "(null) for every other claim family. 'get' returns one claim's "
            "current state + full transition history; 'list' enumerates claims "
            "by their current state."
        ),
        tags=["graph-os", "epistemic", "claims", "governance"],
    )
    async def graph_claims(
        action: str = Field(
            default="get",
            description="propose|validate|accept|deprecate|retract|get|list",
        ),
        claim_id: str = Field(
            default="", description="Target claim id (every action but 'list')."
        ),
        reason: str = Field(
            default="",
            description="Why this transition is happening — the audit trail.",
        ),
        valid: bool = Field(
            default=True,
            description=(
                "'validate' only: the caller's ALREADY-COMPUTED governance verdict "
                "(e.g. from PromotionGovernanceValidator) — this tool never "
                "recomputes validity itself, it only records it. False HOLDS the "
                "claim at its current state (not a retraction — validate may be "
                "called again later once the hold clears)."
            ),
        ),
        state: str = Field(
            default="",
            description=(
                "'list' only: filter to claims whose CURRENT state equals this "
                "(proposed|validated|accepted|deprecated|retracted)."
            ),
        ),
        limit: int = Field(default=50, description="'list' only: max rows."),
    ) -> str:
        """Propose / validate / accept / deprecate / retract / get / list claims
        through the governed ClaimFlywheel lifecycle, ActionPolicy-gated."""

        from agent_utilities.knowledge_graph.research.claim_flywheel import (
            ClaimFlywheel,
            IllegalTransition,
            list_claims,
        )

        try:
            engine = kg_server._get_engine()
            flywheel = ClaimFlywheel(engine)

            if action == "get":
                if not claim_id:
                    return json.dumps({"error": "get requires claim_id"})
                return json.dumps(
                    {
                        "action": "get",
                        "claim_id": claim_id,
                        "current_state": flywheel.current_state(claim_id).value,
                        "history": flywheel.history(claim_id),
                    },
                    default=str,
                )
            if action == "list":
                return json.dumps(
                    {
                        "action": "list",
                        "claims": list_claims(
                            engine, state=(state or None), limit=limit
                        ),
                    },
                    default=str,
                )
            if action not in ("propose", "validate", "accept", "deprecate", "retract"):
                return json.dumps({"error": f"unknown action {action!r}"})
            if not claim_id:
                return json.dumps({"error": f"{action} requires claim_id"})

            allowed, policy = _gate(f"claim.{action}", claim_id, reason)
            if not allowed:
                return json.dumps(
                    {
                        "action": action,
                        "claim_id": claim_id,
                        "error": "policy_denied",
                        "policy": policy,
                    }
                )

            materialization = None
            superseded_fact = None
            try:
                if action == "propose":
                    transition = flywheel.propose(
                        claim_id, reason=reason or "mined finding"
                    )
                elif action == "validate":
                    transition = flywheel.validate(claim_id, valid, reason=reason)
                elif action == "accept":
                    transition = flywheel.accept(
                        claim_id,
                        reason=reason or "action-gated promotion",
                        action_decision=policy["decision"],
                    )
                    # Ingestion candidate-claim materialization hook
                    # (CONCEPT:AU-KG.ingest.governed-claim-promotion): a no-op
                    # (returns None) for every OTHER claim family (mining,
                    # ops-causal, ...) — only a claim carrying a
                    # ``proposed_envelope`` (this program's promotion.py)
                    # writes the real fact here. Never swallowed: a failed
                    # materialization is returned to the caller, not hidden
                    # behind an "accepted" that silently wrote nothing.
                    from agent_utilities.knowledge_graph.ingestion.promotion import (
                        materialize_on_claim_accepted,
                    )

                    materialization = materialize_on_claim_accepted(engine, claim_id)
                elif action == "deprecate":
                    transition = flywheel.deprecate(
                        claim_id, reason=reason or "superseded or drifted"
                    )
                else:  # retract
                    transition = flywheel.retract(
                        claim_id, reason=reason or "retracted"
                    )
                    from agent_utilities.knowledge_graph.ingestion.promotion import (
                        supersede_materialized_claim,
                    )

                    superseded_fact = supersede_materialized_claim(
                        engine, claim_id, reason=reason or "retracted"
                    )
            except IllegalTransition as e:
                return json.dumps(
                    {
                        "action": action,
                        "claim_id": claim_id,
                        "error": "illegal_transition",
                        "detail": str(e),
                        "policy": policy,
                    }
                )

            return json.dumps(
                {
                    "action": action,
                    "claim_id": claim_id,
                    "policy": policy,
                    "transition": transition.to_dict() if transition else None,
                    "materialization": materialization,
                    "superseded_fact": superseded_fact,
                },
                default=str,
            )
        except Exception as e:  # noqa: BLE001
            return public_error_json(e)

    kg_server.REGISTERED_TOOLS["graph_claims"] = graph_claims
