#!/usr/bin/python
from __future__ import annotations

"""Unified artifact-promotion gate (CONCEPT:AU-AHE.harness.unified-promotion-gate).

``action_policy.py``'s reserved-kind pattern is already the right shape — six
independent artifact-mutation kinds (``promote_skill_version``,
``merge_promotion``, ``promote_mined_claim``, ``route_policy_update``,
``spec_promotion``, ``apply_placement_change``) all funnel through ONE
``ActionPolicy.decide()``, all default ``approval_required``, all audited as
``ActionDecision`` KG nodes. This module does not replace that substrate — it
generalizes the near-duplicate CALL SITES that build an ``ActionRequest`` and
interpret its ``ActionDecision`` (``skill_evolution.py``'s promotion tail and
``auto_merge.py``'s ``_consult_action_policy``) into one reusable pair:

- :func:`evaluate_promotion` — the candidate-vs-incumbent comparison gate,
  generalizing BOTH ``skill_gate.evaluate_promotion``'s strict ``cand > base``
  and ``program_optimization.should_promote``'s ``cand >= base + min_delta``
  into one function with a caller-chosen ``strict`` flag. Comparison-less
  vectors (a golden-loop spec/claim proposal, gated on quality + governance
  instead of a held-out score) pass ``incumbent_reward=None`` and this gate is
  skipped entirely — it is opt-in per vector, exactly like the design intends.
- :func:`promote` — the ONE entry point every optimizer's promotion boundary
  calls: (1) the comparison gate, when applicable, (2)
  ``action_policy.decide(kind=...)``, (3) a caller-legible verdict. It performs
  NO KG writes itself (persistence/edge-writing stays vector-specific — a
  skill's ``:SkillVersion`` upsert and a prompt's file write are shaped too
  differently to share one write path) — it is the DECISION boundary, mirroring
  how ``skill_gate``/``should_promote`` were always pure decision functions with
  the caller owning every side effect.

Never raises — mirrors ``run_reflact_cycle``'s / ``GovernedAutoMerger.consider``'s
best-effort discipline: a policy-consult failure degrades to a fail-closed
``deny`` verdict rather than crashing the caller's promotion cycle.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar

from agent_utilities.harness.reward_signal import RewardSignal
from agent_utilities.orchestration.action_policy import (
    ActionRequest,
    PolicyDisposition,
    PolicyReceipt,
    get_action_policy,
)

logger = logging.getLogger(__name__)

__all__ = ["PromotionCandidate", "PromotionOutcome", "evaluate_promotion", "promote"]

PROMOTION_OUTCOME_SCHEMA = "promotion-outcome.v1"


@dataclass(frozen=True)
class PromotionCandidate:
    """One artifact version proposed for promotion, ready to consult the gate.

    ``artifact_kind``/``artifact_id`` name WHAT is being promoted (e.g.
    ``"skill"``/``skill_id``, ``"prompt"``/``prompt component ref``) and drive the
    default reserved ``action_policy`` kind (``f"promote_{artifact_kind}_version"``)
    — override with ``policy_kind`` for a vector that already has its own reserved
    kind string (``auto_merge``'s ``"merge_promotion"``, which predates this
    generalization and must keep its existing ``DEFAULT_POLICY`` entry + tests).
    ``incumbent_reward=None`` skips :func:`evaluate_promotion` (the vector is
    gated on something other than a candidate-vs-incumbent score — quality +
    governance validity, for the spec/claim proposal shape).
    """

    artifact_kind: str
    artifact_id: str
    candidate_ref: str
    candidate_reward: RewardSignal
    incumbent_reward: RewardSignal | None = None
    incumbent_ref: str | None = None
    policy_kind: str = ""
    source: str = "loop_engine"
    reason: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    provenance_receipts: tuple[str, ...] = ()

    @property
    def policy_action_kind(self) -> str:
        return self.policy_kind or f"promote_{self.artifact_kind}_version"

    def normalized_provenance_receipts(self) -> tuple[str, ...]:
        """Validate the bounded, exact provenance set required for publication."""
        receipts = self.provenance_receipts
        if not receipts:
            raise ValueError("promotion requires at least one provenance receipt")
        if len(receipts) > 64 or len(receipts) != len(set(receipts)):
            raise ValueError("promotion provenance receipts must be unique and bounded")
        if any(not item or len(item) > 256 for item in receipts):
            raise ValueError(
                "promotion provenance receipt ids must be present and bounded"
            )
        return tuple(sorted(receipts))

    def intent_digest(self) -> str:
        """Stable identity of the exact candidate and provenance presented."""
        payload = {
            "artifact_id": self.artifact_id,
            "artifact_kind": self.artifact_kind,
            "candidate_ref": self.candidate_ref,
            "candidate_reward": self.candidate_reward.value,
            "candidate_reward_source": self.candidate_reward.source,
            "evidence": self.evidence,
            "incumbent_ref": self.incumbent_ref,
            "incumbent_reward": (
                self.incumbent_reward.value
                if self.incumbent_reward is not None
                else None
            ),
            "policy_kind": self.policy_action_kind,
            "provenance_receipts": self.normalized_provenance_receipts(),
            "reason": self.reason,
            "source": self.source,
        }
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PromotionOutcome:
    """Lossless effect-boundary result for one promotion intent."""

    eligible: bool
    disposition: PolicyDisposition
    intent_digest: str
    policy_request_digest: str
    reason: str
    provenance_receipts: tuple[str, ...]
    policy_receipt: PolicyReceipt | None = None
    approval_id: str | None = None
    schema: ClassVar[str] = PROMOTION_OUTCOME_SCHEMA

    @property
    def approved(self) -> bool:
        """Only an exact, receipt-backed approval can authorize publication."""
        receipt = self.policy_receipt
        return bool(
            self.eligible
            and self.disposition is PolicyDisposition.APPROVE
            and receipt is not None
            and receipt.disposition is PolicyDisposition.APPROVE
            and receipt.request_digest == self.policy_request_digest
            and self.provenance_receipts
        )


def evaluate_promotion(
    candidate: PromotionCandidate, *, min_delta: float = 0.0, strict: bool = True
) -> bool:
    """``True`` iff ``candidate`` clears the comparison gate.

    ``incumbent_reward is None`` -> ``True`` unconditionally (no comparison gate
    applies for this vector — e.g. a golden-loop spec proposal). Otherwise:

    - ``strict=True`` (default): ``candidate.value > incumbent.value + min_delta``
      — the ``skill_gate.evaluate_promotion`` rule (a tie never promotes).
    - ``strict=False``: ``candidate.value >= incumbent.value + min_delta`` — the
      ``program_optimization.should_promote`` rule.
    """
    if candidate.incumbent_reward is None:
        return True
    threshold = candidate.incumbent_reward.value + min_delta
    if strict:
        return candidate.candidate_reward.value > threshold
    return candidate.candidate_reward.value >= threshold


def _promotion_identity(candidate: PromotionCandidate) -> tuple[tuple[str, ...], str]:
    receipts = candidate.normalized_provenance_receipts()
    return receipts, candidate.intent_digest()


def _promotion_request(
    candidate: PromotionCandidate,
    provenance_receipts: tuple[str, ...],
    intent_digest: str,
) -> ActionRequest:
    params = {
        **candidate.evidence,
        "candidate_ref": candidate.candidate_ref,
        "candidate_reward": candidate.candidate_reward.value,
        "candidate_reward_source": candidate.candidate_reward.source,
        "promotion_intent_digest": intent_digest,
        "provenance_receipts": list(provenance_receipts),
    }
    if candidate.incumbent_reward is not None:
        params["incumbent_reward"] = candidate.incumbent_reward.value
    return ActionRequest(
        kind=candidate.policy_action_kind,
        target=candidate.artifact_id,
        params=params,
        source=candidate.source,
        reason=candidate.reason
        or f"{candidate.artifact_kind} candidate {candidate.candidate_ref} promotion",
    )


def _policy_unavailable(
    request: ActionRequest,
    intent_digest: str,
    provenance_receipts: tuple[str, ...],
    error: Exception,
) -> PromotionOutcome:
    logger.warning(
        "artifact_promotion: action_policy consult failed for %s: %s",
        request.summary(),
        error,
    )
    return PromotionOutcome(
        eligible=True,
        disposition=PolicyDisposition.UNAVAILABLE,
        intent_digest=intent_digest,
        policy_request_digest=request.digest(),
        reason=f"action policy unavailable (fail closed): {error}",
        provenance_receipts=provenance_receipts,
    )


def _receipt_matches_approval(receipt: Any, request_digest: str) -> bool:
    return bool(
        isinstance(receipt, PolicyReceipt)
        and receipt.request_digest == request_digest
        and receipt.disposition is PolicyDisposition.APPROVE
    )


def _promotion_outcome(
    request: ActionRequest,
    intent_digest: str,
    provenance_receipts: tuple[str, ...],
    decision: Any,
) -> PromotionOutcome:
    disposition = getattr(decision, "disposition", PolicyDisposition.UNAVAILABLE)
    receipt = getattr(decision, "receipt", None)
    request_digest = request.digest()
    if disposition is PolicyDisposition.APPROVE and not _receipt_matches_approval(
        receipt, request_digest
    ):
        disposition = PolicyDisposition.UNAVAILABLE
        receipt = None
    return PromotionOutcome(
        eligible=True,
        disposition=disposition,
        intent_digest=intent_digest,
        policy_request_digest=request_digest,
        reason=decision.reason,
        provenance_receipts=provenance_receipts,
        policy_receipt=receipt,
        approval_id=decision.approval_id,
    )


def promote(
    engine: Any,
    candidate: PromotionCandidate,
    *,
    min_delta: float = 0.0,
    strict: bool = True,
    policy: Any = None,
) -> PromotionOutcome:
    """The ONE promotion entry point every optimizer's promotion boundary calls.

    1. :func:`evaluate_promotion` (the comparison gate, when applicable). Not
       eligible -> return immediately WITHOUT consulting ``action_policy`` (a
       benchmark loss has nothing to decide).
    2. ``action_policy.decide(kind=candidate.policy_action_kind, ...)`` —
       ``policy`` overrides the resolved gate (auto_merge's own injectable
       ``action_policy=...``); ``None`` resolves ``get_action_policy(engine)``,
       the same default every other reserved-kind call site uses.
    3. A :class:`PromotionOutcome` the caller applies: ``approved`` gates
       whatever vector-specific write the caller performs next (flip a
       ``:SkillVersion`` to ``active`` + write ``SUPERSEDES``, flip a golden-loop
       proposal's lifecycle, write a hardened prompt to source, ...).

    Performs NO KG writes itself and never raises — a policy-consult failure
    degrades to a fail-closed ``deny`` verdict, mirroring
    ``GovernedAutoMerger._consult_action_policy``'s existing fail-closed handling.
    """
    eligible = evaluate_promotion(candidate, min_delta=min_delta, strict=strict)
    if not eligible:
        return PromotionOutcome(
            eligible=False,
            disposition=PolicyDisposition.DENY,
            intent_digest="",
            policy_request_digest="",
            reason="candidate did not beat incumbent",
            provenance_receipts=(),
        )

    try:
        provenance_receipts, intent_digest = _promotion_identity(candidate)
    except (TypeError, ValueError) as exc:
        return PromotionOutcome(
            eligible=True,
            disposition=PolicyDisposition.UNAVAILABLE,
            intent_digest="",
            policy_request_digest="",
            reason=f"promotion provenance unavailable: {exc}",
            provenance_receipts=(),
        )

    active_policy = policy or get_action_policy(engine)
    request = _promotion_request(candidate, provenance_receipts, intent_digest)
    try:
        decision = active_policy.decide(request)
    except Exception as e:  # noqa: BLE001 — gate failure => fail closed, never crash
        return _policy_unavailable(request, intent_digest, provenance_receipts, e)
    return _promotion_outcome(request, intent_digest, provenance_receipts, decision)
