#!/usr/bin/python
from __future__ import annotations

"""Ontology Action System — executor (CONCEPT:KG-2.25).

The :class:`ActionExecutor` is the single governed entry point for invoking a
verb over the ontology. For every invocation it:

  (a) looks the action up in the registry,
  (b) AUTHORIZES the actor against the action's ``required_capability`` using the
      existing :class:`~agent_utilities.security.permissions_kernel.PermissionsKernel`
      (deny → audited, denied :class:`ActionInvocation` returned, handler never runs),
  (c) validates params against the action's parameter schema,
  (d) runs the registered handler,
  (e) emits an :class:`~agent_utilities.observability.audit_logger.AuditLogger`
      entry (actor / action / resource / outcome),
  (f) persists the invocation as a KG ``action_invocation`` node with
      ``INVOKED_BY`` → actor and ``ACTS_ON`` → target edges (lazy/optional
      backend — skipped cleanly when no engine is reachable),
  (g) returns the :class:`ActionInvocation` carrying the result.

Reuses the existing governance fabric — it does NOT reinvent permissions, audit,
or KG persistence. The KG backend is probed lazily and degrades gracefully
offline, mirroring ``domains/finance/forensic_screener.py``.
"""

import logging
from typing import TYPE_CHECKING, Any

from agent_utilities.observability.audit_logger import (
    RESOURCE_TOOL,
    AuditLogger,
)
from agent_utilities.security.permissions_kernel import (
    AgentIdentity,
    AuthDecision,
    PermissionsKernel,
)

from .models import ActionInvocation, ActionStatus, OntologyAction
from .registry import ActionRegistry

if TYPE_CHECKING:
    from agent_utilities.security.brain_context import ActorContext

logger = logging.getLogger(__name__)

ACTION_AUDIT = "ontology_action.invoke"

# Lazy, cached KG facade for invocation persistence. Probed once; ``None`` when
# no backend is reachable so importing/using the executor never requires a
# running engine (mirrors forensic_screener's lazy engine pattern).
_KG_PROBED = False
_KG_FACADE: Any = None


def _persistence_facade() -> Any:
    """Return a KnowledgeGraph facade with a live store, or ``None`` if offline."""
    global _KG_PROBED, _KG_FACADE
    if _KG_PROBED:
        return _KG_FACADE
    _KG_PROBED = True
    try:
        from agent_utilities.knowledge_graph.facade import KnowledgeGraph

        kg = KnowledgeGraph()
        if kg.store is None:  # backend unavailable — degrade cleanly
            _KG_FACADE = None
        else:
            _KG_FACADE = kg
    except Exception as exc:  # noqa: BLE001 — degrade gracefully, never raise
        logger.debug("KG persistence unavailable for action invocations: %s", exc)
        _KG_FACADE = None
    return _KG_FACADE


def reset_persistence_cache() -> None:
    """Reset the cached KG probe (used by tests to re-probe)."""
    global _KG_PROBED, _KG_FACADE
    _KG_PROBED = False
    _KG_FACADE = None


def _actor_fields(actor: ActorContext | AgentIdentity) -> tuple[str, list[str]]:
    """Extract ``(actor_id, capabilities)`` from either actor representation."""
    if isinstance(actor, AgentIdentity):
        return actor.agent_id, list(actor.capabilities)
    # ActorContext (duck-typed): roles act as the capability set it carries.
    actor_id = getattr(actor, "actor_id", "system")
    caps = list(getattr(actor, "roles", ()) or ())
    extra = getattr(actor, "capabilities", None)
    if extra:
        caps.extend(extra)
    return actor_id, caps


class ActionExecutor:
    """Governed executor for ontology actions. CONCEPT:KG-2.25.

    Args:
        registry: The :class:`ActionRegistry` to resolve actions from.
        kernel: A :class:`PermissionsKernel` for capability authorization. A
            fresh kernel (built-in default policies) is created when omitted.
        audit: An :class:`AuditLogger`; a fresh in-memory logger is created when
            omitted.
        persist: When ``True`` (default), invocations are persisted to the KG if
            a backend is reachable. Set ``False`` to disable persistence.
        escalation_gate: The HITL :class:`EscalationGate` (CONCEPT:OS-5.12)
            consulted after authorization to decide whether the verb requires
            human approval before its handler runs. Defaults to a gate built on
            the conservative default :class:`EscalationMatrix`.
    """

    def __init__(
        self,
        registry: ActionRegistry,
        kernel: PermissionsKernel | None = None,
        audit: AuditLogger | None = None,
        persist: bool = True,
        escalation_gate: Any = None,
    ) -> None:
        self.registry = registry
        self.kernel = kernel or PermissionsKernel()
        self.audit = audit or AuditLogger()
        self.persist = persist
        if escalation_gate is None:
            from agent_utilities.observability.escalation_matrix import EscalationGate

            escalation_gate = EscalationGate(audit=self.audit, persist=persist)
        self.escalation_gate = escalation_gate

    # ── Authorization ──────────────────────────────────────────────────

    def _authorize(
        self,
        action: OntologyAction,
        actor: ActorContext | AgentIdentity,
    ) -> bool:
        """Return True if ``actor`` may invoke ``action``.

        Capability-first: an actor that holds the action's ``required_capability``
        (in its capability/role set) is authorized — this is the OWL-substrate
        eligibility (Agent providesCapability == Action requiresCapability). For a
        signed :class:`AgentIdentity` we additionally defer to the
        :class:`PermissionsKernel` so role policies (DENY > APPROVAL > ALLOW) can
        still veto the verb by name.
        """
        _actor_id, caps = _actor_fields(actor)
        has_capability = action.required_capability in caps or "admin" in caps

        if isinstance(actor, AgentIdentity):
            decision = self.kernel.authorize_tool(actor, action.name)
            if decision == AuthDecision.DENY:
                return False
            # Identity verified + role policy allows; capability still required
            # unless the policy explicitly granted the verb (ALLOW).
            return has_capability or decision == AuthDecision.ALLOW

        return has_capability

    # ── Execution ──────────────────────────────────────────────────────

    def execute(
        self,
        action_name: str,
        actor: ActorContext | AgentIdentity,
        params: dict[str, Any] | None = None,
        target_id: str = "",
        *,
        value: Any = None,
        decision_provider: Any = None,
    ) -> ActionInvocation:
        """Authorize, escalate (HITL), validate, run, audit, and persist.

        ``value`` is an optional value/business-impact hint (ValueTier, int, or
        "low".."critical") that — together with the action's risk tier — drives
        the HITL :class:`EscalationGate` (CONCEPT:OS-5.12). When the matrix says
        a human must approve, ``decision_provider`` (a callable returning the
        approval payload) is consulted; without one, the rule's safe fallback
        (auto-deny by default) applies so a high-risk verb never silently runs.
        """
        params = dict(params or {})
        actor_id, _caps = _actor_fields(actor)

        action = self.registry.get(action_name)
        if action is None:
            inv = ActionInvocation(
                action_name=action_name,
                actor_id=actor_id,
                params=params,
                target_id=target_id,
                status=ActionStatus.ERROR,
                error=f"unknown action: {action_name!r}",
            )
            self._audit(inv, action=None)
            return inv

        # (b) Authorize — DENY short-circuits before the handler runs.
        if not self._authorize(action, actor):
            inv = ActionInvocation(
                action_name=action_name,
                actor_id=actor_id,
                params=params,
                target_id=target_id,
                status=ActionStatus.DENIED,
                result_summary=(
                    f"denied: actor lacks required capability "
                    f"{action.required_capability!r}"
                ),
            )
            self._audit(inv, action=action)
            self._persist(inv, action)
            return inv

        # (b2) HITL escalation — consult the matrix (CONCEPT:OS-5.12). A
        # high-risk/high-value verb pauses for a human (via the gate's
        # decision_provider / ApprovalManager) before its handler runs; a
        # low-tier verb is NOT_REQUIRED and proceeds on the fast path.
        decision = self.escalation_gate.evaluate_sync(
            action_name=action_name,
            actor_id=actor_id,
            effect=str(action.produces_effect),
            idempotent=action.idempotent,
            value=value if value is not None else action.value_tier,
            explicit_risk=action.risk_tier,
            decision_provider=decision_provider,
            target_id=target_id,
        )
        if not decision.allowed:
            inv = ActionInvocation(
                action_name=action_name,
                actor_id=actor_id,
                params=params,
                target_id=target_id,
                status=ActionStatus.DENIED,
                result_summary=(
                    f"escalation {decision.outcome}: risk={decision.risk_tier.name} "
                    f"value={decision.value_tier.name}"
                    + (f" — {decision.reason}" if decision.reason else "")
                ),
            )
            self._audit(inv, action=action)
            self._persist(inv, action)
            inv._escalation = decision  # type: ignore[attr-defined]
            return inv

        # (c) Validate params against the schema.
        errors = action.validate_params(params)
        if errors:
            inv = ActionInvocation(
                action_name=action_name,
                actor_id=actor_id,
                params=params,
                target_id=target_id,
                status=ActionStatus.ERROR,
                error="; ".join(errors),
                result_summary=f"parameter validation failed: {'; '.join(errors)}",
            )
            self._audit(inv, action=action)
            self._persist(inv, action)
            return inv

        # (d) Run the handler.
        handler = self.registry.get_handler(action_name)
        try:
            result = handler(params) if handler else None
            inv = ActionInvocation(
                action_name=action_name,
                actor_id=actor_id,
                params=params,
                target_id=target_id,
                status=ActionStatus.SUCCESS,
                result_summary=_summarize(result),
            )
            inv_result = result
        except Exception as exc:  # noqa: BLE001 — surface as ERROR, never crash
            logger.warning("Action handler %s failed: %s", action_name, exc)
            inv = ActionInvocation(
                action_name=action_name,
                actor_id=actor_id,
                params=params,
                target_id=target_id,
                status=ActionStatus.ERROR,
                error=str(exc),
                result_summary=f"handler error: {exc}",
            )
            inv_result = None

        # (e) Audit + (f) persist.
        self._audit(inv, action=action)
        self._persist(inv, action)
        # Carry the live result object out-of-band for callers that need it.
        inv._result = inv_result  # type: ignore[attr-defined]
        return inv

    # ── Audit + persistence helpers ────────────────────────────────────

    def _audit(self, inv: ActionInvocation, action: OntologyAction | None) -> None:
        """Emit an AuditLog entry for an invocation (never raises)."""
        record = self.audit.log(
            actor=inv.actor_id,
            action=ACTION_AUDIT,
            resource_type=RESOURCE_TOOL,
            resource_id=inv.action_name,
            details={
                "status": str(inv.status),
                "verb": action.verb if action else "",
                "effect": str(action.produces_effect) if action else "",
                "required_capability": action.required_capability if action else "",
                "target_id": inv.target_id,
                "error": inv.error,
            },
        )
        if record is not None:
            inv.audit_ref = record.id

    def _persist(self, inv: ActionInvocation, action: OntologyAction) -> None:
        """Persist the invocation as a KG node + edges (lazy/optional backend)."""
        if not self.persist:
            return
        kg = _persistence_facade()
        if kg is None:
            return  # offline — skip cleanly
        store = kg.store
        if store is None:
            return
        try:
            store.execute(
                "MERGE (n {id: $id}) SET n.type = 'action_invocation', "
                "n.action_name = $action_name, n.actor_id = $actor_id, "
                "n.status = $status, n.result_summary = $summary, "
                "n.audit_ref = $audit_ref, n.timestamp = $timestamp",
                {
                    "id": inv.id,
                    "action_name": inv.action_name,
                    "actor_id": inv.actor_id,
                    "status": str(inv.status),
                    "summary": inv.result_summary,
                    "audit_ref": inv.audit_ref,
                    "timestamp": inv.timestamp,
                },
            )
            # INVOKED_BY → actor
            store.execute(
                "MATCH (n {id: $id}) MERGE (a {id: $actor_id}) "
                "MERGE (n)-[:INVOKED_BY]->(a)",
                {"id": inv.id, "actor_id": inv.actor_id},
            )
            # ACTS_ON → concrete target object (when supplied)
            if inv.target_id:
                store.execute(
                    "MATCH (n {id: $id}) MERGE (t {id: $target_id}) "
                    "MERGE (n)-[:ACTS_ON]->(t)",
                    {"id": inv.id, "target_id": inv.target_id},
                )
            inv.persisted = True
        except Exception as exc:  # noqa: BLE001 — persistence is best-effort
            logger.debug("Failed to persist action invocation %s: %s", inv.id, exc)


def _summarize(result: Any) -> str:
    """Produce a short, log-safe summary string for a handler result."""
    if result is None:
        return "ok"
    if isinstance(result, str):
        return result[:240]
    if isinstance(result, list | tuple | set):
        return f"{type(result).__name__} of {len(result)} item(s)"
    if isinstance(result, dict):
        return f"dict with keys {sorted(result)[:8]}"
    text = str(result)
    return text[:240]
