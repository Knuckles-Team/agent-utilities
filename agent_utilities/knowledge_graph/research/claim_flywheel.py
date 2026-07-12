#!/usr/bin/python
from __future__ import annotations

"""The epistemic mining flywheel — a governed lifecycle over mined ``Claim``s (X-3).

CONCEPT:AU-KG.evolution.mining-flywheel — mining outputs become PROPOSED CLAIMS
with full lineage → an agent or a validation step reviews or acts on them →
the outcome becomes an observation → retrieval ranking / routing / calibration
updates from that observation.

The engine already turns mining outputs into ``Claim`` nodes with lineage
(``candidate_insight.CandidateInsight.to_claim_node`` / ``to_evidence_bundle``,
persisted by ``loop_controller._run_insight_validation`` / ``_run_trace_mining``
with ``status: "proposal"``, ``is_verified=False``), and those two stages
ALREADY reuse :class:`~.promotion_governance.PromotionGovernanceValidator` and
:mod:`~agent_utilities.orchestration.action_policy` to decide whether a claim
may be promoted. What was missing is a single, explicit, QUERYABLE lifecycle
that distinguishes *proposal* from *validation* from *acceptance* from
*deprecation* from *retraction* — and a closed loop from an accepted claim's
real-world outcome back to the durable bandit/ranker.

This module is deliberately a thin OVERLAY, not a second governance stack:

* **Governance is reused, not reinvented.** A claim only reaches
  ``VALIDATED`` because :class:`~.promotion_governance.
  PromotionGovernanceValidator` said so, and only reaches ``ACCEPTED``
  because :func:`~agent_utilities.orchestration.action_policy.get_action_policy`
  independently allowed it (the SAME ``promote_mined_claim`` /
  ``route_policy_update`` kinds ``_run_insight_validation`` /
  ``_run_trace_mining`` already gate on) — this module never makes that call
  itself, it only RECORDS the outcome of calls the caller already made.
* **The bandit is reused, not reinvented.** Outcome feedback durably persists
  through :func:`~agent_utilities.graph.routing.enrichers.capability_designation.
  record_capability_outcome` (CONCEPT:AU-P1-3 — the SAME durable
  contextual-bandit spine ``OutcomeRouter``/``CapabilityIndex`` already use),
  never a parallel reward store.

State machine (five states, matching the task's own vocabulary)::

    proposed --> validated --> accepted --> deprecated --> retracted
       |            |             |                            ^
       +------------+-------------+----------------------------+
                (any pre-terminal state may be retracted directly)

``RETRACTED`` is terminal and STICKY: :meth:`ClaimFlywheel.propose` refuses to
re-open a retracted claim, so a rejected mined finding is never silently
re-proposed on the next mining pass over the SAME finding id (mining ids are
content-addressed — see ``candidate_insight._stable_id`` — so "the same
finding" really does mean "the same claim id" across cycles).

Every transition is persisted as an append-only ``ClaimLifecycleEvent`` node
(``claim_id``, ``from_state``, ``to_state``, ``reason``, ``actor``,
``governance_valid``, ``action_decision``, ``timestamp``) — the queryable
audit trail this workstream asked for — NEVER a silent mutation of the
``Claim`` node's own ``status``/``is_verified`` fields (those stay exactly
what ``_run_insight_validation``/``_run_trace_mining`` already set them to;
this module adds a parallel, richer lifecycle view rather than touching
fields other code/tests already depend on).

Within one :class:`ClaimFlywheel` instance (one mining pass) transitions are
also cached in-process — mirroring ``OutcomeRouter``/``CapabilityIndex``'s own
"in-process EMA + durable readback" split — so a sequence of
propose→validate→accept calls against even a minimal engine double (one that
does not round-trip ``query_cypher`` reads of nodes it just wrote) still
enforces the state machine correctly; only CROSS-cycle memory (a fresh
:class:`ClaimFlywheel` instance discovering a claim was retracted in an
earlier cycle) depends on the engine's ``query_cypher`` actually reflecting
prior writes, which any real engine does.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "ClaimLifecycleState",
    "IllegalTransition",
    "LifecycleTransition",
    "ClaimFlywheel",
]

#: Reward below which an ACCEPTED claim's observed outcome auto-deprecates it
#: (drift detection: the materialized/acted-on claim proved stale/wrong).
_DEFAULT_AUTO_DEPRECATE_BELOW = 0.2


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class ClaimLifecycleState(StrEnum):
    """The flywheel's five lifecycle states (CONCEPT:AU-KG.evolution.mining-flywheel, X-3)."""

    PROPOSED = "proposed"
    VALIDATED = "validated"
    ACCEPTED = "accepted"
    DEPRECATED = "deprecated"
    RETRACTED = "retracted"


# Legal transitions. RETRACTED is terminal — nothing leaves it (see
# ClaimFlywheel.propose()'s refusal to re-open a retracted claim).
_ALLOWED: dict[ClaimLifecycleState, frozenset[ClaimLifecycleState]] = {
    ClaimLifecycleState.PROPOSED: frozenset(
        {ClaimLifecycleState.VALIDATED, ClaimLifecycleState.RETRACTED}
    ),
    ClaimLifecycleState.VALIDATED: frozenset(
        {ClaimLifecycleState.ACCEPTED, ClaimLifecycleState.RETRACTED}
    ),
    ClaimLifecycleState.ACCEPTED: frozenset(
        {ClaimLifecycleState.DEPRECATED, ClaimLifecycleState.RETRACTED}
    ),
    ClaimLifecycleState.DEPRECATED: frozenset({ClaimLifecycleState.RETRACTED}),
    ClaimLifecycleState.RETRACTED: frozenset(),
}


class IllegalTransition(ValueError):
    """Raised when a requested lifecycle transition violates the state machine."""


@dataclass
class LifecycleTransition:
    """One recorded lifecycle event — the audit-visible unit this module persists."""

    claim_id: str
    from_state: str
    to_state: str
    reason: str
    actor: str = "loop_engine"
    governance_valid: bool | None = None
    action_decision: str | None = None
    timestamp: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "reason": self.reason,
            "actor": self.actor,
            "governance_valid": self.governance_valid,
            "action_decision": self.action_decision,
            "timestamp": self.timestamp,
        }


class ClaimFlywheel:
    """Governed proposed→validated→accepted→deprecated→retracted lifecycle over Claims.

    CONCEPT:AU-KG.evolution.mining-flywheel — Epistemic Mining Flywheel (X-3)

    Args:
        engine: KG engine (``add_node``/``query_cypher``) events persist to and
            cross-cycle state is read back from. Every write/read is
            best-effort — a missing/unreachable engine degrades gracefully
            (propose/validate/accept still work in-process for the duration
            of this instance; only cross-cycle retracted-memory is lost).
    """

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self._cache: dict[str, ClaimLifecycleState] = {}

    # ── state read ──────────────────────────────────────────────────────
    def current_state(self, claim_id: str) -> ClaimLifecycleState:
        """The claim's current lifecycle state (``PROPOSED`` if never transitioned)."""
        if claim_id in self._cache:
            return self._cache[claim_id]
        events = self.history(claim_id)
        if not events:
            return ClaimLifecycleState.PROPOSED
        try:
            return ClaimLifecycleState(str(events[-1]["to_state"]))
        except ValueError:
            return ClaimLifecycleState.PROPOSED

    def is_retracted(self, claim_id: str) -> bool:
        return self.current_state(claim_id) == ClaimLifecycleState.RETRACTED

    def history(self, claim_id: str) -> list[dict[str, Any]]:
        """The claim's full, chronological transition history (queryable audit trail)."""
        try:
            rows = (
                self.engine.query_cypher(
                    "MATCH (e:ClaimLifecycleEvent) WHERE e.claim_id = $id RETURN "
                    "e.from_state AS from_state, e.to_state AS to_state, "
                    "e.reason AS reason, e.actor AS actor, "
                    "e.governance_valid AS governance_valid, "
                    "e.action_decision AS action_decision, "
                    "e.timestamp AS timestamp ORDER BY e.timestamp",
                    {"id": claim_id},
                )
                or []
            )
        except Exception as e:  # noqa: BLE001 — history is best-effort introspection
            logger.debug("[X3] flywheel history query failed for %s: %s", claim_id, e)
            rows = []
        events = [
            dict(row) for row in rows if isinstance(row, dict) and row.get("to_state")
        ]
        events.sort(key=lambda e: str(e.get("timestamp") or ""))
        return events

    # ── transitions ─────────────────────────────────────────────────────
    def propose(
        self, claim_id: str, *, reason: str = "mined finding"
    ) -> LifecycleTransition | None:
        """Record the INITIAL proposal for a freshly-mined claim.

        Returns ``None`` (a no-op) when the claim already has lifecycle
        history — either because it was already RETRACTED (the durable
        rejection memory: a retracted claim is never re-proposed, even when
        re-mining produces the identical content-addressed finding id again)
        or because it is already somewhere further along its own lifecycle
        (idempotent re-mining of the same fact upserts the Claim node but
        does not mint a second proposal event).
        """
        events = self.history(claim_id)
        if events:
            try:
                last = ClaimLifecycleState(str(events[-1]["to_state"]))
            except ValueError:
                last = ClaimLifecycleState.PROPOSED
            if last == ClaimLifecycleState.RETRACTED:
                logger.info(
                    "[X3] flywheel refusing to re-propose retracted claim %s",
                    claim_id,
                )
            self._cache[claim_id] = last
            return None
        return self._append_event(claim_id, None, ClaimLifecycleState.PROPOSED, reason)

    def validate(
        self, claim_id: str, valid: bool, *, reason: str = ""
    ) -> LifecycleTransition | None:
        """Advance PROPOSED → VALIDATED when ``valid`` — REUSES the caller's own
        :class:`~.promotion_governance.PromotionGovernanceValidator` verdict;
        never re-derives governance validity itself.

        An invalid verdict does NOT retract the claim — a governance hold
        (e.g. a recorded regression-gate/ratchet hold) can resolve on a later
        cycle — it only records the held attempt (:meth:`record_hold`) so the
        hold is audit-visible rather than silent.
        """
        if valid:
            return self._transition(
                claim_id,
                ClaimLifecycleState.VALIDATED,
                reason=reason or "governance checks passed",
                governance_valid=True,
            )
        self.record_hold(claim_id, reason=reason or "governance checks failed")
        return None

    def accept(
        self,
        claim_id: str,
        *,
        reason: str = "action-gated promotion",
        action_decision: str | None = None,
    ) -> LifecycleTransition:
        """Advance VALIDATED → ACCEPTED — call ONLY after the caller's own
        ``action_policy.decide()`` independently allowed the promotion (this
        method never consults action_policy itself)."""
        return self._transition(
            claim_id,
            ClaimLifecycleState.ACCEPTED,
            reason=reason,
            action_decision=action_decision,
        )

    def reject(
        self,
        claim_id: str,
        *,
        reason: str = "rejected",
        action_decision: str | None = None,
    ) -> LifecycleTransition:
        """Retract a not-yet-accepted claim (PROPOSED/VALIDATED → RETRACTED) —
        e.g. a matched constitution forbid rule, or an explicit
        ``action_policy`` deny. Durable: :meth:`propose` never resurrects it."""
        return self._transition(
            claim_id,
            ClaimLifecycleState.RETRACTED,
            reason=reason,
            action_decision=action_decision,
        )

    def deprecate(
        self, claim_id: str, *, reason: str = "superseded or drifted"
    ) -> LifecycleTransition:
        """Advance ACCEPTED → DEPRECATED — a materialized/acted-on claim whose
        observed outcome (or a fresher contradicting claim) shows it is stale."""
        return self._transition(claim_id, ClaimLifecycleState.DEPRECATED, reason=reason)

    def retract(
        self, claim_id: str, *, reason: str = "retracted"
    ) -> LifecycleTransition:
        """Retract an ACCEPTED or DEPRECATED claim outright (skips the
        deprecation waypoint for a claim discovered to be flatly wrong)."""
        return self._transition(claim_id, ClaimLifecycleState.RETRACTED, reason=reason)

    def record_hold(self, claim_id: str, *, reason: str) -> LifecycleTransition:
        """Record a held (no state change) validation/action attempt — audit-visible,
        never silent, and never a state regression."""
        current = self.current_state(claim_id)
        return self._append_event(claim_id, current, current, reason)

    # ── outcome → observation → durable feedback (closes the loop) ─────
    def record_outcome(
        self,
        claim_id: str,
        *,
        reward: float,
        note: str = "",
        auto_deprecate_below: float = _DEFAULT_AUTO_DEPRECATE_BELOW,
        durable_key: str | None = None,
        durable_reward: float | None = None,
    ) -> dict[str, Any]:
        """Capture an accepted claim's real-world OUTCOME as an observation.

        Persists a queryable ``ClaimOutcome`` node (the observation) and feeds
        a reward back through the EXISTING durable contextual-bandit spine
        (:func:`~agent_utilities.graph.routing.enrichers.capability_designation.
        record_capability_outcome`, CONCEPT:AU-P1-3 — never a new reward
        store). The drift-detection tie-in: ``reward`` — how good ACCEPTING
        this claim itself turned out to be — auto-DEPRECATES the claim when it
        falls below ``auto_deprecate_below`` AND the claim is currently
        ACCEPTED (a bad outcome is itself the evidence a deprecation needs; no
        separate approval gate — deprecation only marks the claim stale, it
        does not undo anything already materialized).

        ``reward`` and the bandit's ``durable_reward`` are DELIBERATELY
        separate parameters: for a routing/process claim mined from a
        repeated FAILURE pattern, the claim itself may be a well-supported,
        confidently-accepted observation (``reward`` — high, no deprecation)
        even though what it teaches the bandit about the failing
        ``(task_class, choice)`` it routed away from is negative
        (``durable_reward=0.0``). Defaults to ``reward`` when omitted (the
        common case — e.g. an ontology-gap claim's own confidence doubles as
        the signal for both).

        ``durable_key`` lets a routing/ranking claim feed the SAME bandit key
        ``OutcomeRouter``/``CapabilityIndex`` already learn over (e.g.
        ``f"{namespace}:{task_class}:{choice}"``) instead of the claim id
        itself, so the durable observation lands on the thing that was
        actually acted on (a routing choice), not just the claim that
        proposed it.
        """
        reward = max(0.0, min(1.0, float(reward)))
        bandit_reward = (
            reward
            if durable_reward is None
            else max(0.0, min(1.0, float(durable_reward)))
        )
        outcome_id = f"claim_outcome:{claim_id}:{uuid.uuid4().hex[:10]}"
        try:
            self.engine.add_node(
                outcome_id,
                "ClaimOutcome",
                properties={
                    "claim_id": claim_id,
                    "reward": reward,
                    "durable_reward": bandit_reward,
                    "note": note,
                    "recorded_at": _now_iso(),
                },
            )
        except Exception as e:  # noqa: BLE001 — the observation node is best-effort
            logger.debug(
                "[X3] flywheel could not persist ClaimOutcome for %s: %s", claim_id, e
            )

        persisted_reward: float | None = None
        try:
            from agent_utilities.graph.routing.enrichers.capability_designation import (
                record_capability_outcome,
            )

            # X-6 / Seam 3 (CONCEPT:EG-KG.epistemic.truth-maintenance): the just-
            # written ``ClaimOutcome`` node above (``outcome_id``) is the REAL base
            # fact this reward was computed from — pass it through so the durable
            # reward on ``durable_key or claim_id`` registers as a live
            # TruthMaintenance materialization (see ``record_capability_outcome``'s
            # ``source_ids`` docstring). Never fabricated: the same id just persisted.
            persisted_reward = record_capability_outcome(
                self.engine,
                durable_key or claim_id,
                reward=bandit_reward,
                source_ids=[outcome_id],
            )
        except Exception as e:  # noqa: BLE001 — durability is an augmentation, never load-bearing
            logger.debug(
                "[X3] flywheel durable outcome feedback failed for %s: %s",
                claim_id,
                e,
            )

        deprecated: LifecycleTransition | None = None
        if (
            reward < auto_deprecate_below
            and self.current_state(claim_id) == ClaimLifecycleState.ACCEPTED
        ):
            try:
                deprecated = self.deprecate(
                    claim_id,
                    reason=(
                        f"outcome reward {reward:.3f} below drift threshold "
                        f"{auto_deprecate_below:.3f}"
                    ),
                )
            except IllegalTransition:
                deprecated = None

        return {
            "outcome_id": outcome_id,
            "reward": reward,
            "bandit_reward": bandit_reward,
            "persisted_reward": persisted_reward,
            "deprecated": deprecated is not None,
        }

    # ── internals ────────────────────────────────────────────────────────
    def _transition(
        self,
        claim_id: str,
        to_state: ClaimLifecycleState,
        *,
        reason: str,
        actor: str = "loop_engine",
        governance_valid: bool | None = None,
        action_decision: str | None = None,
    ) -> LifecycleTransition:
        current = self.current_state(claim_id)
        if to_state not in _ALLOWED.get(current, frozenset()):
            raise IllegalTransition(
                f"{claim_id}: {current.value} -> {to_state.value} is not a "
                "legal flywheel transition"
            )
        return self._append_event(
            claim_id,
            current,
            to_state,
            reason,
            actor=actor,
            governance_valid=governance_valid,
            action_decision=action_decision,
        )

    def _append_event(
        self,
        claim_id: str,
        from_state: ClaimLifecycleState | None,
        to_state: ClaimLifecycleState,
        reason: str,
        *,
        actor: str = "loop_engine",
        governance_valid: bool | None = None,
        action_decision: str | None = None,
    ) -> LifecycleTransition:
        record = LifecycleTransition(
            claim_id=claim_id,
            from_state=from_state.value if from_state is not None else "",
            to_state=to_state.value,
            reason=reason,
            actor=actor,
            governance_valid=governance_valid,
            action_decision=action_decision,
        )
        try:
            self.engine.add_node(
                f"claim_lifecycle:{claim_id}:{uuid.uuid4().hex[:12]}",
                "ClaimLifecycleEvent",
                properties=record.to_dict(),
            )
        except Exception as e:  # noqa: BLE001 — the audit event is best-effort
            logger.debug(
                "[X3] flywheel could not persist lifecycle event for %s: %s",
                claim_id,
                e,
            )
        self._cache[claim_id] = to_state
        return record
