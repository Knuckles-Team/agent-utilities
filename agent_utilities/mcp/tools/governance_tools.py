"""Focused graph-os governance operations."""

from __future__ import annotations

import json
import uuid
from typing import Any

from pydantic import Field

from agent_utilities.mcp import kg_server
from agent_utilities.security.error_surface import public_error_text


def register_governance_tools(mcp: Any) -> None:
    """Register approval decisions, risk vetoes, and ActionPolicy verification."""

    @mcp.tool(
        name="graph_governance",
        description=(
            "Govern orchestration actions. Actions: 'grant_approval' atomically decides a "
            "pending ActionApproval; 'submit_risk_veto' records a governed veto against an "
            "existing target; 'verify_action' evaluates an ActionPolicy request without "
            "writing a decision or approval."
        ),
        tags=["graph-os", "governance", "approval", "policy"],
    )
    def graph_governance(
        action: str = Field(
            default="verify_action",
            description="grant_approval | submit_risk_veto | verify_action",
        ),
        approval_id: str = Field(
            default="", description="Pending action_approval:* id (grant_approval)."
        ),
        decision: str = Field(
            default="approved", description="approved | denied (grant_approval)."
        ),
        target_id: str = Field(
            default="",
            description="Existing target id for a risk veto or policy check.",
        ),
        reason: str = Field(default="", description="Risk-veto or policy reason."),
        kind: str = Field(
            default="", description="ActionPolicy action kind (verify_action)."
        ),
        params_json: str = Field(
            default="{}", description="JSON object of ActionPolicy parameters."
        ),
        source: str = Field(default="manual", description="Policy request source."),
        actor_id: str = Field(default="", description="Optional policy actor id."),
    ) -> str:
        engine = kg_server._get_engine()
        if engine is None:
            return "Error: IntelligenceGraphEngine not active."
        try:
            if action == "grant_approval":
                from agent_utilities.orchestration.approval import (
                    decide_action_approval,
                )

                return json.dumps(
                    decide_action_approval(engine, approval_id, decision), default=str
                )

            if action == "submit_risk_veto":
                if not target_id:
                    raise ValueError("target_id is required for submit_risk_veto")
                from agent_utilities.knowledge_graph.core.session import resolve_session

                session = resolve_session(required_scope="kg:write")
                veto_id = f"risk_veto:{uuid.uuid4().hex}"
                engine.add_node(
                    veto_id,
                    "RiskVeto",
                    {"reason": reason, "target_id": target_id, "status": "submitted"},
                    session=session,
                )
                engine.add_edge(
                    veto_id,
                    target_id,
                    "CONTRADICTS_BELIEF_PROP",
                    session=session,
                )
                return json.dumps(
                    {"veto_id": veto_id, "target_id": target_id, "status": "submitted"}
                )

            if action == "verify_action":
                from agent_utilities.orchestration.action_policy import (
                    ActionPolicy,
                    ActionRequest,
                )

                if not kind:
                    raise ValueError("kind is required for verify_action")
                params = json.loads(params_json) if params_json else {}
                if not isinstance(params, dict):
                    raise ValueError("params_json must decode to an object")
                verdict = ActionPolicy(engine=engine).evaluate(
                    ActionRequest(
                        kind=kind,
                        target=target_id or "*",
                        params=params,
                        source=source,
                        reason=reason,
                        actor_id=actor_id,
                    )
                )
                return json.dumps(
                    {
                        "decision": verdict.decision,
                        "allowed": verdict.allowed,
                        "tier": verdict.tier,
                        "reason": verdict.reason,
                        "invariant": verdict.invariant,
                        "verify_ms": verdict.verify_ms,
                    },
                    default=str,
                )
            return f"Error: Unknown graph_governance action '{action}'"
        except PermissionError:
            raise
        except Exception as exc:
            return public_error_text(exc)

    kg_server.REGISTERED_TOOLS["graph_governance"] = graph_governance
    kg_server.ACTION_TOOL_ROUTES["graph_governance"] = "/graph/governance"
