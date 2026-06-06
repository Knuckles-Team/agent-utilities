#!/usr/bin/python
from __future__ import annotations

"""Formal HITL Escalation Matrix (CONCEPT:OS-5.12).

Palantir AIP gates high-impact Ontology Actions behind a formal
human-in-the-loop *escalation matrix*: a value/impact-tier × risk policy that
decides, per action, whether a human must approve, which role may approve, how
long to wait, and what happens on timeout. ``approval_manager.py`` already gives
us the async pause/resume *mechanism* (futures the UI resolves) — what was
missing is the *policy* that decides **when** to pause and **who** must answer.

This module adds that policy. An :class:`EscalationMatrix` maps an action's
``(risk_tier, value_tier)`` to an :class:`EscalationRule` carrying the required
approver role(s), a timeout, and a fallback (auto-deny / auto-approve). The
:class:`EscalationGate` consults the matrix, and — when approval is required —
drives the existing :class:`~agent_utilities.observability.approval_manager.ApprovalManager`
to actually block for a decision, recording the escalation + outcome to the
:class:`~agent_utilities.observability.audit_logger.AuditLogger` (and, lazily,
to the KG).

Reuses the existing governance fabric — it does NOT reinvent approval futures,
audit, or KG persistence. It is wired into the live Ontology Action System
executor (``knowledge_graph/actions/executor.py``) so a high-risk/high-value
verb genuinely pauses for a human before its handler runs.
"""

import logging
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any

from agent_utilities.observability.audit_logger import (
    RESOURCE_TOOL,
    AuditLogger,
)

logger = logging.getLogger(__name__)

AUDIT_ESCALATION = "hitl.escalation"


class RiskTier(IntEnum):
    """Risk tier of an action (ordered low → critical).

    Derived from the action's effect class and explicit annotations. Higher
    tiers demand a more privileged approver and a stricter fallback.
    """

    LOW = 0  # read-only / idempotent
    MEDIUM = 1  # ontology mutation
    HIGH = 2  # external side effect / non-idempotent mutation
    CRITICAL = 3  # destructive / irreversible


class ValueTier(IntEnum):
    """Value / business-impact tier of an action invocation."""

    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


class Fallback(StrEnum):
    """What to do when no human responds within the timeout."""

    AUTO_DENY = "auto_deny"  # safe default — deny on timeout
    AUTO_APPROVE = "auto_approve"  # only for low-stakes, explicitly chosen rules


class EscalationOutcome(StrEnum):
    """Result of consulting + (optionally) resolving an escalation."""

    NOT_REQUIRED = "not_required"  # policy said no human needed
    APPROVED = "approved"
    DENIED = "denied"
    TIMEOUT = "timeout"  # resolved by the fallback (deny/approve)


@dataclass(frozen=True)
class EscalationRule:
    """The policy for one (risk, value) cell of the matrix.

    Attributes:
        require_approval: When True a human decision is required before the
            action proceeds.
        approver_roles: Roles permitted to satisfy the approval (any one).
            Empty + ``require_approval`` means "any human".
        timeout_seconds: Max seconds to block for a decision (0 = wait forever).
        fallback: What to do if the timeout elapses with no decision.
    """

    require_approval: bool
    approver_roles: tuple[str, ...] = ()
    timeout_seconds: float = 0.0
    fallback: Fallback = Fallback.AUTO_DENY


@dataclass
class EscalationDecision:
    """The recorded result of an escalation consult/resolve. CONCEPT:OS-5.12."""

    action_name: str
    actor_id: str
    risk_tier: RiskTier
    value_tier: ValueTier
    rule: EscalationRule
    outcome: EscalationOutcome
    approver: str = ""
    reason: str = ""
    audit_ref: str = ""
    request_id: str = ""
    timestamp: float = field(default_factory=time.time)

    @property
    def allowed(self) -> bool:
        """True when the action may proceed (approved, or not required)."""
        return self.outcome in (
            EscalationOutcome.NOT_REQUIRED,
            EscalationOutcome.APPROVED,
        )


class EscalationMatrix:
    """A risk × value → :class:`EscalationRule` policy table. CONCEPT:OS-5.12.

    The matrix is keyed by ``(RiskTier, ValueTier)``; lookup falls back to the
    highest rule whose tiers are *<=* the requested ones, so an unspecified cell
    inherits the strictest applicable lower-bound rule. The conservative default
    (built by :meth:`default`) requires approval for everything HIGH-risk or
    HIGH-value and auto-denies on timeout.
    """

    def __init__(self, rules: dict[tuple[RiskTier, ValueTier], EscalationRule]) -> None:
        self._rules = dict(rules)

    def rule_for(self, risk: RiskTier, value: ValueTier) -> EscalationRule:
        """Return the rule governing ``(risk, value)``.

        Exact match wins; otherwise the most-restrictive rule at or below both
        tiers is used (a request strictly more severe than any defined cell
        inherits the highest defined rule).
        """
        exact = self._rules.get((risk, value))
        if exact is not None:
            return exact
        # Inherit from the strongest rule whose tiers don't exceed the request.
        best: EscalationRule | None = None
        best_key = (-1, -1)
        for (r, v), rule in self._rules.items():
            if r <= risk and v <= value and (r, v) >= best_key:
                best, best_key = rule, (r, v)
        if best is not None:
            return best
        # No applicable rule defined → conservative: require approval, auto-deny.
        return EscalationRule(
            require_approval=True,
            approver_roles=("admin",),
            timeout_seconds=0.0,
            fallback=Fallback.AUTO_DENY,
        )

    # ── default policy ─────────────────────────────────────────────────
    @classmethod
    def default(cls) -> EscalationMatrix:
        """Build the real, conservative default matrix. CONCEPT:OS-5.12.

        - LOW risk & LOW/MEDIUM value → no approval (fast path).
        - MEDIUM risk → approval by operator+ when value is HIGH+.
        - HIGH risk → operator approval required, 5-minute timeout, auto-deny.
        - CRITICAL risk → admin approval required, wait forever, auto-deny.
        """
        timeout = float(os.getenv("HITL_ESCALATION_TIMEOUT", "300"))
        no = EscalationRule(require_approval=False)
        return cls(
            {
                (RiskTier.LOW, ValueTier.LOW): no,
                (RiskTier.LOW, ValueTier.MEDIUM): no,
                (RiskTier.LOW, ValueTier.HIGH): EscalationRule(
                    require_approval=True,
                    approver_roles=("operator", "admin"),
                    timeout_seconds=timeout,
                    fallback=Fallback.AUTO_DENY,
                ),
                (RiskTier.MEDIUM, ValueTier.LOW): no,
                (RiskTier.MEDIUM, ValueTier.MEDIUM): EscalationRule(
                    require_approval=True,
                    approver_roles=("operator", "admin"),
                    timeout_seconds=timeout,
                    fallback=Fallback.AUTO_DENY,
                ),
                (RiskTier.HIGH, ValueTier.LOW): EscalationRule(
                    require_approval=True,
                    approver_roles=("operator", "admin"),
                    timeout_seconds=timeout,
                    fallback=Fallback.AUTO_DENY,
                ),
                (RiskTier.CRITICAL, ValueTier.LOW): EscalationRule(
                    require_approval=True,
                    approver_roles=("admin",),
                    timeout_seconds=0.0,
                    fallback=Fallback.AUTO_DENY,
                ),
            }
        )


def classify_risk_tier(effect: str, idempotent: bool, explicit: str = "") -> RiskTier:
    """Map an action's ``produces_effect`` + idempotency to a :class:`RiskTier`.

    An explicit annotation (the action's ``risk_tier`` metadata, if present)
    overrides the derived tier.
    """
    if explicit:
        try:
            return RiskTier[explicit.strip().upper()]
        except KeyError:
            logger.debug("unknown explicit risk tier %r; deriving", explicit)
    eff = (effect or "").lower()
    if eff == "external":
        # A non-idempotent external call has a real side effect → HIGH. An
        # idempotent external call (a read-style API screen) is MEDIUM, so it is
        # not gated behind approval at LOW value (keeps read-style verbs flowing
        # while still escalating genuine external side effects).
        return RiskTier.MEDIUM if idempotent else RiskTier.HIGH
    if eff == "mutation":
        return RiskTier.HIGH if not idempotent else RiskTier.MEDIUM
    return RiskTier.LOW  # read


def classify_value_tier(value: Any) -> ValueTier:
    """Coerce a caller-supplied value/impact hint into a :class:`ValueTier`.

    Accepts a ValueTier, an int (0–3), or a string ("low".."critical"). Unknown
    inputs default to MEDIUM (never silently LOW for an unrecognized hint).
    """
    if isinstance(value, ValueTier):
        return value
    if isinstance(value, bool):  # guard: bool is an int subclass
        return ValueTier.HIGH if value else ValueTier.LOW
    if isinstance(value, int):
        return ValueTier(max(0, min(3, value)))
    if isinstance(value, str):
        try:
            return ValueTier[value.strip().upper()]
        except KeyError:
            return ValueTier.MEDIUM
    return ValueTier.MEDIUM


class EscalationGate:
    """Consults an :class:`EscalationMatrix` and enforces the decision. CONCEPT:OS-5.12.

    The gate is the bridge between the *policy* (matrix) and the *mechanism*
    (``ApprovalManager`` futures). It:

      1. classifies the action's risk + value tiers,
      2. looks up the governing rule,
      3. if no approval is required, returns ``NOT_REQUIRED`` immediately,
      4. otherwise blocks on the :class:`ApprovalManager` for a decision,
         applying the rule's timeout + fallback,
      5. audits the escalation + outcome (and persists it lazily to the KG).

    It exposes both an async path (:meth:`evaluate`) that genuinely pauses for a
    human, and a sync path (:meth:`evaluate_sync`) for non-async executors that
    short-circuits to the rule's fallback when no decision provider is supplied
    — so a high-risk verb can never *silently* proceed without governance.
    """

    def __init__(
        self,
        matrix: EscalationMatrix | None = None,
        *,
        approval_manager: Any = None,
        audit: AuditLogger | None = None,
        persist: bool = True,
    ) -> None:
        self.matrix = matrix or EscalationMatrix.default()
        self._approval_manager = approval_manager
        self.audit = audit or AuditLogger()
        self.persist = persist

    # ── async (live HITL) path ─────────────────────────────────────────
    async def evaluate(
        self,
        *,
        action_name: str,
        actor_id: str,
        effect: str,
        idempotent: bool = True,
        value: Any = ValueTier.LOW,
        explicit_risk: str = "",
        event_queue: Any = None,
        target_id: str = "",
    ) -> EscalationDecision:
        """Decide + (if required) block for human approval. CONCEPT:OS-5.12."""
        risk = classify_risk_tier(effect, idempotent, explicit_risk)
        vtier = classify_value_tier(value)
        rule = self.matrix.rule_for(risk, vtier)

        if not rule.require_approval:
            return self._finalize(
                action_name,
                actor_id,
                risk,
                vtier,
                rule,
                EscalationOutcome.NOT_REQUIRED,
                target_id=target_id,
            )

        mgr = self._approval_manager
        if mgr is None:
            from agent_utilities.observability.approval_manager import (
                elicitation_manager,
            )

            mgr = elicitation_manager

        import uuid

        request_id = f"escalation:{action_name}:{uuid.uuid4().hex[:10]}"
        if event_queue is not None:
            await event_queue.put(
                {
                    "type": "escalation_required",
                    "request_id": request_id,
                    "action": action_name,
                    "risk_tier": risk.name,
                    "value_tier": vtier.name,
                    "approver_roles": list(rule.approver_roles),
                    "timeout_seconds": rule.timeout_seconds,
                }
            )
        try:
            resolution = await mgr.wait_for_approval(
                request_id, timeout=rule.timeout_seconds
            )
        except TimeoutError:
            outcome = (
                EscalationOutcome.APPROVED
                if rule.fallback == Fallback.AUTO_APPROVE
                else EscalationOutcome.DENIED
            )
            return self._finalize(
                action_name,
                actor_id,
                risk,
                vtier,
                rule,
                EscalationOutcome.TIMEOUT
                if outcome == EscalationOutcome.DENIED
                else EscalationOutcome.APPROVED,
                reason=f"timeout → {rule.fallback}",
                request_id=request_id,
                target_id=target_id,
            )

        return self._apply_resolution(
            resolution,
            action_name,
            actor_id,
            risk,
            vtier,
            rule,
            request_id=request_id,
            target_id=target_id,
        )

    # ── sync path (non-async executors) ────────────────────────────────
    def evaluate_sync(
        self,
        *,
        action_name: str,
        actor_id: str,
        effect: str,
        idempotent: bool = True,
        value: Any = ValueTier.LOW,
        explicit_risk: str = "",
        decision_provider: Any = None,
        target_id: str = "",
    ) -> EscalationDecision:
        """Synchronous consult. CONCEPT:OS-5.12.

        ``decision_provider`` is an optional callable
        ``(request: dict) -> dict | None`` that returns a resolution payload
        (``{"approved": bool, "approver": str, "reason": str}``) — e.g. a
        pre-seeded policy, a queued CLI answer, or a test stub. When it is
        ``None`` (no human reachable on a sync path), the rule's *fallback* is
        applied so the action is never silently allowed.
        """
        risk = classify_risk_tier(effect, idempotent, explicit_risk)
        vtier = classify_value_tier(value)
        rule = self.matrix.rule_for(risk, vtier)

        if not rule.require_approval:
            return self._finalize(
                action_name,
                actor_id,
                risk,
                vtier,
                rule,
                EscalationOutcome.NOT_REQUIRED,
                target_id=target_id,
            )

        request = {
            "action": action_name,
            "actor_id": actor_id,
            "risk_tier": risk.name,
            "value_tier": vtier.name,
            "approver_roles": list(rule.approver_roles),
        }
        resolution = None
        if decision_provider is not None:
            try:
                resolution = decision_provider(request)
            except Exception as exc:  # noqa: BLE001 — fall through to fallback
                logger.warning("escalation decision_provider failed: %s", exc)
                resolution = None

        if resolution is None:
            outcome = (
                EscalationOutcome.APPROVED
                if rule.fallback == Fallback.AUTO_APPROVE
                else EscalationOutcome.TIMEOUT
            )
            return self._finalize(
                action_name,
                actor_id,
                risk,
                vtier,
                rule,
                outcome,
                reason=f"no decision provider → {rule.fallback}",
                target_id=target_id,
            )

        return self._apply_resolution(
            resolution,
            action_name,
            actor_id,
            risk,
            vtier,
            rule,
            target_id=target_id,
        )

    # ── helpers ────────────────────────────────────────────────────────
    def _apply_resolution(
        self,
        resolution: dict[str, Any],
        action_name: str,
        actor_id: str,
        risk: RiskTier,
        vtier: ValueTier,
        rule: EscalationRule,
        *,
        request_id: str = "",
        target_id: str = "",
    ) -> EscalationDecision:
        approved = bool(
            resolution.get("approved")
            or resolution.get("decision") in ("approve", "accept", "allow", True)
        )
        approver = str(resolution.get("approver", ""))
        if rule.approver_roles and approver:
            approver_role = str(resolution.get("approver_role", approver))
            if (
                approver_role not in rule.approver_roles
                and approver not in rule.approver_roles
            ):
                # Approver not authorized for this tier → deny.
                return self._finalize(
                    action_name,
                    actor_id,
                    risk,
                    vtier,
                    rule,
                    EscalationOutcome.DENIED,
                    approver=approver,
                    reason=f"approver role {approver_role!r} not in {rule.approver_roles}",
                    request_id=request_id,
                    target_id=target_id,
                )
        outcome = EscalationOutcome.APPROVED if approved else EscalationOutcome.DENIED
        return self._finalize(
            action_name,
            actor_id,
            risk,
            vtier,
            rule,
            outcome,
            approver=approver,
            reason=str(resolution.get("reason", "")),
            request_id=request_id,
            target_id=target_id,
        )

    def _finalize(
        self,
        action_name: str,
        actor_id: str,
        risk: RiskTier,
        vtier: ValueTier,
        rule: EscalationRule,
        outcome: EscalationOutcome,
        *,
        approver: str = "",
        reason: str = "",
        request_id: str = "",
        target_id: str = "",
    ) -> EscalationDecision:
        decision = EscalationDecision(
            action_name=action_name,
            actor_id=actor_id,
            risk_tier=risk,
            value_tier=vtier,
            rule=rule,
            outcome=outcome,
            approver=approver,
            reason=reason,
            request_id=request_id,
        )
        record = self.audit.log(
            actor=actor_id,
            action=AUDIT_ESCALATION,
            resource_type=RESOURCE_TOOL,
            resource_id=action_name,
            details={
                "risk_tier": risk.name,
                "value_tier": vtier.name,
                "require_approval": rule.require_approval,
                "approver_roles": list(rule.approver_roles),
                "outcome": str(outcome),
                "approver": approver,
                "reason": reason,
                "fallback": str(rule.fallback),
                "target_id": target_id,
            },
        )
        if record is not None:
            decision.audit_ref = record.id
        self._persist(decision, target_id)
        return decision

    def _persist(self, decision: EscalationDecision, target_id: str) -> None:
        """Persist the escalation as a KG node (lazy/optional backend)."""
        if not self.persist:
            return
        try:
            from agent_utilities.knowledge_graph.actions.executor import (
                _persistence_facade,
            )

            kg = _persistence_facade()
            if kg is None or kg.store is None:
                return
            kg.store.execute(
                "MERGE (n {id: $id}) SET n.type = 'escalation_decision', "
                "n.action_name = $action_name, n.actor_id = $actor_id, "
                "n.risk_tier = $risk, n.value_tier = $value, n.outcome = $outcome, "
                "n.approver = $approver, n.audit_ref = $audit_ref, "
                "n.timestamp = $timestamp",
                {
                    "id": f"escalation:{decision.action_name}:{decision.timestamp}",
                    "action_name": decision.action_name,
                    "actor_id": decision.actor_id,
                    "risk": decision.risk_tier.name,
                    "value": decision.value_tier.name,
                    "outcome": str(decision.outcome),
                    "approver": decision.approver,
                    "audit_ref": decision.audit_ref,
                    "timestamp": decision.timestamp,
                },
            )
        except Exception as exc:  # noqa: BLE001 — persistence is best-effort
            logger.debug("escalation persist skipped: %s", exc)


def make_decision_provider(decisions: dict[str, Any]) -> Any:
    """Build a static ``decision_provider`` from an ``{action_name: payload}`` map.

    Convenience for tests / pre-seeded policy: returns the payload registered for
    the request's action (or ``None`` to trigger the rule fallback).
    """

    def _provider(request: dict[str, Any]) -> Any:
        return decisions.get(request.get("action", ""))

    return _provider


def sequence_decision_provider(answers: Sequence[dict[str, Any]]) -> Any:
    """Build a ``decision_provider`` that returns queued ``answers`` in order."""
    it = iter(answers)

    def _provider(_request: dict[str, Any]) -> Any:
        return next(it, None)

    return _provider
