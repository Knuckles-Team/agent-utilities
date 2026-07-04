#!/usr/bin/python
from __future__ import annotations

"""Ontology Action System — executor (CONCEPT:AU-KG.ontology.ontology-action-system).

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

from .dispatch import send_notification, send_webhook
from .effects import (
    apply_side_effects,
    evaluate_submission_criteria,
    resolve_template,
)
from .models import ActionInvocation, ActionStatus, OntologyAction
from .registry import ActionRegistry

if TYPE_CHECKING:
    from agent_utilities.knowledge_graph.ontology.edits import EditLedger
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
    """Governed executor for ontology actions. CONCEPT:AU-KG.ontology.ontology-action-system.

    Args:
        registry: The :class:`ActionRegistry` to resolve actions from.
        kernel: A :class:`PermissionsKernel` for capability authorization. A
            fresh kernel (built-in default policies) is created when omitted.
        audit: An :class:`AuditLogger`; a fresh in-memory logger is created when
            omitted.
        persist: When ``True`` (default), invocations are persisted to the KG if
            a backend is reachable. Set ``False`` to disable persistence.
        escalation_gate: The HITL :class:`EscalationGate` (CONCEPT:AU-OS.observability.empty-derive-from-effect)
            consulted after authorization to decide whether the verb requires
            human approval before its handler runs. Defaults to a gate built on
            the conservative default :class:`EscalationMatrix`.
        ledger: The C1 :class:`~agent_utilities.knowledge_graph.ontology.edits.EditLedger`
            through which an action's typed side-effects are applied + journaled
            (CONCEPT:AU-KG.ontology.batch-actions-executor). A fresh ledger (sharing this executor's audit log)
            is created when omitted so actions are revertible out of the box.
        notifier: A registerable
            :class:`~agent_utilities.knowledge_graph.actions.dispatch.Notifier`
            for action notifications; the process default is used when omitted.
    """

    def __init__(
        self,
        registry: ActionRegistry,
        kernel: PermissionsKernel | None = None,
        audit: AuditLogger | None = None,
        persist: bool = True,
        escalation_gate: Any = None,
        ledger: EditLedger | None = None,
        notifier: Any = None,
    ) -> None:
        self.registry = registry
        self.kernel = kernel or PermissionsKernel()
        self.audit = audit or AuditLogger()
        self.persist = persist
        if escalation_gate is None:
            from agent_utilities.observability.escalation_matrix import EscalationGate

            escalation_gate = EscalationGate(audit=self.audit, persist=persist)
        self.escalation_gate = escalation_gate
        if ledger is None:
            from agent_utilities.knowledge_graph.ontology.edits import EditLedger as _EL

            ledger = _EL(audit=self.audit)
        self.ledger = ledger
        self.notifier = notifier

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
        the HITL :class:`EscalationGate` (CONCEPT:AU-OS.observability.empty-derive-from-effect). When the matrix says
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

        # (b2) HITL escalation — consult the matrix (CONCEPT:AU-OS.observability.empty-derive-from-effect). A
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

        # (c2) Submission criteria — Palantir submission-rules gate (KG-2.42).
        # An action with no criteria always passes (preserves KG-2.25 semantics).
        crit_failures = evaluate_submission_criteria(action, params, actor_id, _caps)
        if crit_failures:
            inv = ActionInvocation(
                action_name=action_name,
                actor_id=actor_id,
                params=params,
                target_id=target_id,
                status=ActionStatus.DENIED,
                error="; ".join(crit_failures),
                result_summary=(
                    f"submission criteria failed: {'; '.join(crit_failures)}"
                ),
            )
            self._audit(inv, action=action)
            self._persist(inv, action)
            return inv

        # (d) Batch fan-out (KG-2.42): apply the action over a list of targets.
        # Defaults preserve single-target behaviour when ``batch`` is False.
        if action.batch:
            return self._execute_batch(action, actor_id, params, target_id)

        return self._execute_one(action, actor_id, params, target_id)

    def _execute_one(
        self,
        action: OntologyAction,
        actor_id: str,
        params: dict[str, Any],
        target_id: str,
    ) -> ActionInvocation:
        """Run one fully-validated/authorized invocation against a single target.

        Runs the handler (or the function-backed ref), applies each declared
        side-effect through the C1 EditLedger (recording one Edit per effect),
        fires notifications/webhooks, then audits + persists. CONCEPT:AU-KG.ontology.batch-actions-executor.
        """
        action_name = action.name
        # (d) Run the handler — a function_ref backs the action when declared,
        # otherwise the registered handler runs.
        handler = self.registry.get_handler(action_name)
        try:
            if action.function_ref is not None:
                result = self._invoke_function(action, params, actor_id)
            else:
                result = handler(params) if handler else None
            status = ActionStatus.SUCCESS
            err = ""
        except Exception as exc:  # noqa: BLE001 — surface as ERROR, never crash
            logger.warning("Action handler %s failed: %s", action_name, exc)
            result = None
            status = ActionStatus.ERROR
            err = str(exc)

        inv = ActionInvocation(
            action_name=action_name,
            actor_id=actor_id,
            params=params,
            target_id=target_id,
            status=status,
            error=err,
            result_summary=(
                _summarize(result)
                if status == ActionStatus.SUCCESS
                else f"handler error: {err}"
            ),
        )

        # (e) Apply typed side-effects through the C1 EditLedger (revertible).
        if status == ActionStatus.SUCCESS and action.side_effects:
            try:
                edits = apply_side_effects(
                    self.ledger,
                    action,
                    params,
                    actor=actor_id,
                    invocation_ref=inv.id,
                )
                inv.edit_ids = [e.id for e in edits]
            except Exception as exc:  # noqa: BLE001 — record, never crash the call
                logger.warning("Side-effects for %s failed: %s", action_name, exc)
                inv.status = ActionStatus.ERROR
                inv.error = f"side-effect error: {exc}"

        # (f) Dispatch notifications + webhooks (real, never a silent no-op).
        if inv.status == ActionStatus.SUCCESS:
            inv.dispatches = self._dispatch(action, params)

        # (g) Audit + persist.
        self._audit(inv, action=action)
        self._persist(inv, action)
        inv._result = result  # type: ignore[attr-defined]
        return inv

    def _execute_batch(
        self,
        action: OntologyAction,
        actor_id: str,
        params: dict[str, Any],
        target_id: str,
    ) -> ActionInvocation:
        """Apply a batch action across ``params['targets']`` (a list of target ids).

        Each target produces its own :class:`_execute_one` sub-invocation (its own
        edits + dispatches); the returned envelope aggregates their ids/statuses.
        ``params`` for each target is the action params with ``target`` bound to
        the per-item id (so templated side-effects resolve per target).
        CONCEPT:AU-KG.ontology.batch-actions-executor — Palantir batch actions.
        """
        targets = params.get("targets") or []
        if not isinstance(targets, list | tuple):
            targets = [targets]
        envelope = ActionInvocation(
            action_name=action.name,
            actor_id=actor_id,
            params=params,
            target_id=target_id,
            status=ActionStatus.SUCCESS,
            result_summary=f"batch over {len(targets)} target(s)",
        )
        for tgt in targets:
            sub_params = {k: v for k, v in params.items() if k != "targets"}
            sub_params["target"] = tgt
            sub = self._execute_one(action, actor_id, sub_params, str(tgt))
            envelope.edit_ids.extend(sub.edit_ids)
            envelope.dispatches.extend(sub.dispatches)
            envelope.batch_results.append(
                {
                    "target": tgt,
                    "invocation_id": sub.id,
                    "status": str(sub.status),
                    "edit_ids": list(sub.edit_ids),
                }
            )
            if sub.status != ActionStatus.SUCCESS:
                envelope.status = ActionStatus.ERROR
        self._audit(envelope, action=action)
        self._persist(envelope, action)
        return envelope

    def _invoke_function(
        self,
        action: OntologyAction,
        params: dict[str, Any],
        actor_id: str,
    ) -> Any:
        """Resolve + run the action's function_ref via the Wave-1 functions runtime.

        Soft import: a function-backed action only hard-depends on the functions
        runtime at *call* time, so authoring such an action never requires the
        runtime to be importable. CONCEPT:AU-KG.ontology.batch-actions-executor.
        """
        from agent_utilities.knowledge_graph.ontology.functions import FunctionRuntime

        ref = action.function_ref
        if ref is None:
            raise RuntimeError("action has no function_ref to invoke")
        runtime = FunctionRuntime(graph=getattr(self.ledger, "_graph", None))
        fn_result = runtime.invoke(
            ref.name,
            params,
            version=ref.version or None,
            actor_id=actor_id,
        )
        if not getattr(fn_result, "ok", False):
            raise RuntimeError(
                f"function {ref.name!r} failed: {getattr(fn_result, 'error', '')}"
            )
        return getattr(fn_result, "value", None)

    def _dispatch(
        self, action: OntologyAction, params: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Fire all notifications + webhooks for an action; return outcome records."""
        records: list[dict[str, Any]] = []
        for note in action.notifications:
            message = resolve_template(note.template, params) if note.template else ""
            rec = send_notification(note, message, self.notifier)
            records.append(rec)
        for hook in action.webhooks:
            rec = send_webhook(hook, {"action": action.name, "params": params})
            records.append(rec)
        for rec in records:
            try:
                self.audit.log(
                    actor=str(params.get("actor", "system")),
                    action=ACTION_AUDIT,
                    resource_type=RESOURCE_TOOL,
                    resource_id=action.name,
                    details={
                        "dispatch": rec.get("kind", ""),
                        "transport": rec.get("transport", ""),
                    },
                )
            except Exception as exc:  # noqa: BLE001 — audit of dispatch is best-effort
                logger.debug("Dispatch audit failed: %s", exc)
        return records

    # ── Undo / revert ──────────────────────────────────────────────────────
    def undo(
        self,
        invocation: ActionInvocation,
        *,
        actor: str = "system",
    ) -> list[Any]:
        """Revert every edit an invocation produced, via the C1 revert path.

        Reverts the invocation's recorded ``edit_ids`` newest-first through the
        EditLedger, recording compensating edits so the action's effects are
        cleanly undone and the trail stays append-only. Returns the compensating
        edits. CONCEPT:AU-KG.ontology.batch-actions-executor (Palantir action undo/revert).
        """
        from agent_utilities.knowledge_graph.ontology.edits import revert_edits

        if not invocation.edit_ids:
            return []
        return revert_edits(self.ledger, list(invocation.edit_ids), actor=actor)

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
