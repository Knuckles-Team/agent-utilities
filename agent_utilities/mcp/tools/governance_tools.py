"""Focused graph-os governance operations."""

from __future__ import annotations

import json
import uuid
from typing import Any

from pydantic import Field

from agent_utilities.core.event_loop import run_blocking_ordered
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
            "writing a decision or approval; 'ownership_report' runs the W2.8 per-graph "
            "RBAC-ownership disposition pass (read-only) over the live engine catalog; "
            "'ownership_apply' previews the UNAMBIGUOUS grant plan + its rollback for that "
            "same report (always a dry-run preview here — the actual live mutating apply "
            "stays the HG-4 human-gated `scripts/apply_graph_ownership_grants.py "
            "--apply-unambiguous` CLI, never this network surface); 'claim_ownership' "
            "(GOC-61, per decisions/GOC-61-unowned-node-disposition.md) admin-only "
            "assumption of ownership over currently-UNOWNED per-node '_owner_id' — never "
            "an already-owned node — selectable by node_type/node_ids (never blind), "
            "dry-run by DEFAULT (pass apply=true to mutate), grouped-by-node_type report "
            "showing owner-to-be per group. owner_id is ALWAYS required and explicit — "
            "never defaults to the calling admin. 'Concept' cannot be swept by node_type "
            "(dual origin: ontology vs chat-extracted, indistinguishable by type alone) "
            "but may be claimed via an explicit, individually reviewed node_ids entry; "
            "'policy_status' reports the live PermissionPolicy/context_policy + KG-native "
            "audit-sink configuration (read-only, no engine required)."
        ),
        tags=["graph-os", "governance", "approval", "policy"],
    )
    async def graph_governance(
        action: str = Field(
            default="verify_action",
            description=(
                "grant_approval | submit_risk_veto | verify_action | ownership_report | "
                "ownership_apply | claim_ownership | policy_status"
            ),
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
        node_types_json: str = Field(
            default="[]",
            description=(
                "claim_ownership: JSON list of node_type labels to claim (or preview) "
                'ownership over, e.g. \'["RuntimeSignal","WorkItem"]\'. At least one of '
                "node_types_json / node_ids_json is required — there is no blind "
                "full-graph mode."
            ),
        ),
        node_ids_json: str = Field(
            default="[]",
            description=(
                "claim_ownership: JSON list of explicit node ids to claim (or preview) "
                "ownership over, as an alternative or addition to node_types_json."
            ),
        ),
        owner_id: str = Field(
            default="",
            description=(
                "claim_ownership: REQUIRED (both preview and apply) — the new owner id "
                "to stamp. NEVER defaults to the calling admin actor; the caller must "
                "state explicitly whose data this is (e.g. the real human user for "
                "private conversational content, or a system/service identity for "
                "operational data) — silent self-assignment is refused."
            ),
        ),
        shared_scope: str = Field(
            default="org",
            description=(
                "claim_ownership: '_shared_scope' to stamp alongside the new owner — "
                "'private' | 'org' | 'commons'. Defaults to 'org'."
            ),
        ),
        apply: bool = Field(
            default=False,
            description=(
                "claim_ownership only: false (DEFAULT) = dry-run preview, grouped by "
                "node_type, mutates nothing. true = actually claim — required explicitly, "
                "never implied."
            ),
        ),
    ) -> str:
        if action == "policy_status":
            # No engine needed — this is static config + a live-wiring confirmation,
            # not a graph read.
            return json.dumps(_policy_status(), indent=2, default=str)

        engine = kg_server._get_engine()
        if engine is None:
            return "Error: IntelligenceGraphEngine not active."

        def _execute() -> str:
            try:
                if action == "ownership_report":
                    return json.dumps(_ownership_report(), default=str)

                if action == "ownership_apply":
                    return json.dumps(_ownership_apply(), default=str)

                if action == "claim_ownership":
                    return json.dumps(
                        _claim_ownership(
                            engine,
                            node_types_json=node_types_json,
                            node_ids_json=node_ids_json,
                            owner_id=owner_id,
                            shared_scope=shared_scope,
                            apply=apply,
                        ),
                        default=str,
                    )

                if action == "grant_approval":
                    from agent_utilities.orchestration.approval import (
                        decide_action_approval,
                    )

                    return json.dumps(
                        decide_action_approval(engine, approval_id, decision),
                        default=str,
                    )

                if action == "submit_risk_veto":
                    if not target_id:
                        raise ValueError("target_id is required for submit_risk_veto")
                    from agent_utilities.knowledge_graph.core.session import (
                        resolve_session,
                    )

                    session = resolve_session(required_scope="kg:write")
                    veto_id = f"risk_veto:{uuid.uuid4().hex}"
                    engine.add_node(
                        veto_id,
                        "RiskVeto",
                        {
                            "reason": reason,
                            "target_id": target_id,
                            "status": "submitted",
                        },
                        session=session,
                    )
                    engine.add_edge(
                        veto_id,
                        target_id,
                        "CONTRADICTS_BELIEF_PROP",
                        session=session,
                    )
                    return json.dumps(
                        {
                            "veto_id": veto_id,
                            "target_id": target_id,
                            "status": "submitted",
                        }
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

        return await run_blocking_ordered(_execute)

    kg_server.REGISTERED_TOOLS["graph_governance"] = graph_governance
    kg_server.ACTION_TOOL_ROUTES["graph_governance"] = "/graph/governance"


def _ownership_report() -> dict[str, Any]:
    """The W2.8 per-graph RBAC-ownership disposition report (read-only).

    CONCEPT:AU-KG.audit.graph-ownership-disposition. Runs the SAME
    ``build_ownership_report`` pass ``scripts/graph_ownership_report.py`` uses,
    against the live engine catalog reached through the process-authority engine
    (:class:`~agent_utilities.knowledge_graph.maintenance.graph_ownership.
    LiveEngineCatalogClient` — the same one every graph-os tool shares). Never
    fabricates a report against an unreachable catalog: an
    :class:`EngineUnreachableError` surfaces as a structured error, not a fake
    TEMPLATE stand-in (the TEMPLATE fixture stays a script/test-only device).
    """
    from agent_utilities.knowledge_graph.maintenance.graph_ownership import (
        EngineUnreachableError,
        build_ownership_report,
        check_invariant,
        render_markdown,
        resolve_catalog_client,
    )

    try:
        client = resolve_catalog_client()
        report = build_ownership_report(
            client,
            mode="live",
            source_note="live epistemic-graph engine catalog (graph_governance surface)",
        )
    except EngineUnreachableError as exc:
        return {
            "surface": "governance",
            "action": "ownership_report",
            "error": type(exc).__name__,
        }
    violations = check_invariant(report.dispositions, enforced=False)
    return {
        "surface": "governance",
        "action": "ownership_report",
        "mode": report.mode,
        "counts": report.counts,
        "invariant_violations_report_only": len(violations),
        "markdown": render_markdown(report),
    }


def _ownership_apply() -> dict[str, Any]:
    """Preview (never apply) the UNAMBIGUOUS ownership grant plan + its rollback.

    CONCEPT:AU-KG.audit.graph-ownership-disposition (HG-4). This network surface
    ALWAYS runs the dry-run half of :func:`~agent_utilities.knowledge_graph.
    maintenance.graph_ownership_apply.apply_plan` — it never issues a single
    mutating RBAC call. The actual live apply is a human-gated CLI
    (``scripts/apply_graph_ownership_grants.py --apply-unambiguous``), deliberately
    not reachable from MCP/REST, matching the module's own PREPARE-ONLY discipline.
    """
    from dataclasses import asdict

    from agent_utilities.knowledge_graph.maintenance.graph_ownership import (
        EngineUnreachableError,
        build_ownership_report,
        resolve_catalog_client,
    )
    from agent_utilities.knowledge_graph.maintenance.graph_ownership_apply import (
        FixtureRbacAdminClient,
        apply_plan,
        plan_grants,
        rollback_to_json,
    )

    try:
        client = resolve_catalog_client()
        report = build_ownership_report(
            client,
            mode="live",
            source_note="live epistemic-graph engine catalog (graph_governance surface)",
        )
    except EngineUnreachableError as exc:
        return {
            "surface": "governance",
            "action": "ownership_apply",
            "error": type(exc).__name__,
        }

    # UNAMBIGUOUS + not-already-covered only — the program's auto-apply-UNAMBIGUOUS /
    # hold-ambiguous decision; never ambiguous rows (see plan_grants' own docstring).
    plan = plan_grants(report)
    # dry_run=True never calls a single method on the client — an empty fixture
    # satisfies the RbacAdminClient parameter without touching anything real.
    preview, rollback = apply_plan(plan, FixtureRbacAdminClient(), dry_run=True)
    return {
        "surface": "governance",
        "action": "ownership_apply",
        "mode": "dry_run",
        "report_mode": report.mode,
        "plan": [asdict(p) for p in plan],
        "preview": [asdict(p) for p in preview],
        "rollback": rollback_to_json(rollback),
        "hint": (
            "preview only; the live mutating apply is HG-4 human-gated via "
            "scripts/apply_graph_ownership_grants.py --apply-unambiguous"
        ),
    }


def _claim_ownership(
    engine: Any,
    *,
    node_types_json: str,
    node_ids_json: str,
    owner_id: str,
    shared_scope: str,
    apply: bool,
) -> dict[str, Any]:
    """GOC-61 admin ownership-claim — dispatches into the ONE shared core
    (:mod:`agent_utilities.knowledge_graph.core.ownership_claim`) that both this
    MCP tool and its automatic REST twin (``/graph/governance``,
    ``kg_server.ACTION_TOOL_ROUTES``) run through. ``apply=False`` (the
    default) is a read-only dry-run preview grouped by node_type, showing the
    owner-to-be per group; ``apply=True`` requires kg:admin and actually
    mutates, claiming ONLY nodes verified unowned immediately before the
    write. ``owner_id`` is required for BOTH modes — see that module's
    docstring for the full invariant list (never reassigns an already-owned
    node, owner_id always explicit and never a default, ``Concept`` cannot be
    swept by node_type, never blind/unselective, real counts, idempotent) and
    ``decisions/GOC-61-unowned-node-disposition.md`` for the authorized
    disposition this implements.
    """
    from dataclasses import asdict

    from agent_utilities.knowledge_graph.core.ownership_claim import (
        OwnershipClaimError,
        apply_claim,
        preview_claim,
    )
    from agent_utilities.knowledge_graph.core.session import resolve_session
    from agent_utilities.security.brain_context import use_actor

    node_types = json.loads(node_types_json) if node_types_json else []
    node_ids = json.loads(node_ids_json) if node_ids_json else []
    if not isinstance(node_types, list) or not isinstance(node_ids, list):
        return {
            "surface": "governance",
            "action": "claim_ownership",
            "error": "node_types_json and node_ids_json must each decode to a JSON list",
        }

    # Fail-closed BEFORE touching the core: the session boundary requires
    # kg:admin explicitly (this action is not gated by the tool's other
    # actions, several of which need only kg:write or no scope at all).
    try:
        session = resolve_session(required_scope="kg:admin")
    except Exception as exc:  # noqa: BLE001 — surfaced as a structured denial, never a bare 500
        return {
            "surface": "governance",
            "action": "claim_ownership",
            "error": f"kg:admin required: {type(exc).__name__}",
        }

    with use_actor(session.actor):
        try:
            if apply:
                result = apply_claim(
                    engine,
                    owner_id=owner_id,
                    node_types=node_types,
                    node_ids=node_ids,
                    shared_scope=shared_scope or "org",
                    actor=session.actor,
                )
                return {
                    "surface": "governance",
                    "action": "claim_ownership",
                    "mode": "apply",
                    **asdict(result),
                }
            preview = preview_claim(
                engine,
                owner_id=owner_id,
                node_types=node_types,
                node_ids=node_ids,
                actor=session.actor,
            )
            return {
                "surface": "governance",
                "action": "claim_ownership",
                "mode": "dry_run",
                **asdict(preview),
                "hint": "preview only; pass apply=true to actually claim",
            }
        except OwnershipClaimError as exc:
            return {
                "surface": "governance",
                "action": "claim_ownership",
                "error": type(exc).__name__,
            }


def _policy_status() -> dict[str, Any]:
    """Live PermissionPolicy/context_policy + KG-native audit-sink status (PA-R1).

    CONCEPT:AU-OS.identity.identity-policy-check / AU-KG.audit.kg-native-audit-sink.
    Both are already default-on, native-by-default capabilities wired into every
    governed agent (``security/tool_guard.py``'s ``flag_mcp_tool_definitions`` /
    ``capabilities/composition.py``'s ``default_runtime_capabilities`` respectively)
    — this action is read-only introspection over their live configuration, not a
    control surface (there is nothing to set here; tune via the same config
    fields every other setting uses).
    """
    from agent_utilities.core.config import config

    return {
        "surface": "governance",
        "action": "policy_status",
        "tool_guard_mode": config.tool_guard_mode,
        "sensitive_tool_pattern_count": len(config.sensitive_tool_patterns),
        "permission_policy": {
            "merge_semantics": "most-restrictive-wins (rule -> ontological-guardrail -> context_policy)",
            "default_verdict": "deny",
            "context_policy_channel": "identity policy (PermissionsKernel.authorize_tool)",
            "wired_at": [
                "agent/factory.py",
                "graph/_router_impl.py",
                "graph/executor.py",
            ],
        },
        "kg_audit_sink": {
            "enabled_by_default": True,
            "capability": "capabilities.kg_audit_sink.AuditLog",
            "wired_via": "capabilities/composition.py:default_runtime_capabilities(kg_audit=True)",
        },
    }
