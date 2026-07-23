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

import logging
from dataclasses import dataclass, field
from typing import Any

from agent_utilities.harness.reward_signal import RewardSignal

logger = logging.getLogger(__name__)

__all__ = ["PromotionCandidate", "PromotionVerdict", "evaluate_promotion", "promote"]


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

    @property
    def policy_action_kind(self) -> str:
        return self.policy_kind or f"promote_{self.artifact_kind}_version"


@dataclass(frozen=True)
class PromotionVerdict:
    """The gate's answer: whether the candidate cleared BOTH checks in play.

    ``eligible`` is the comparison-gate result (or ``True`` when the vector has
    no comparison gate, i.e. ``incumbent_reward is None``). ``decision`` is the
    raw ``action_policy`` verdict string (``"allow"`` | ``"allow_notify"`` |
    ``"queue_approval"`` | ``"deny"``), empty when the comparison gate itself
    rejected the candidate (``action_policy`` was never consulted — mirrors
    ``run_reflact_cycle``'s "a benchmark loss never reaches action_policy").
    ``approved`` is the single boolean a caller should act on: eligible AND the
    policy allowed it.
    """

    eligible: bool
    decision: str
    approved: bool
    reason: str = ""
    approval_id: str | None = None


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


def promote(
    engine: Any,
    candidate: PromotionCandidate,
    *,
    min_delta: float = 0.0,
    strict: bool = True,
    policy: Any = None,
) -> PromotionVerdict:
    """The ONE promotion entry point every optimizer's promotion boundary calls.

    1. :func:`evaluate_promotion` (the comparison gate, when applicable). Not
       eligible -> return immediately WITHOUT consulting ``action_policy`` (a
       benchmark loss has nothing to decide).
    2. ``action_policy.decide(kind=candidate.policy_action_kind, ...)`` —
       ``policy`` overrides the resolved gate (auto_merge's own injectable
       ``action_policy=...``); ``None`` resolves ``get_action_policy(engine)``,
       the same default every other reserved-kind call site uses.
    3. A :class:`PromotionVerdict` the caller applies: ``approved`` gates
       whatever vector-specific write the caller performs next (flip a
       ``:SkillVersion`` to ``active`` + write ``SUPERSEDES``, flip a golden-loop
       proposal's lifecycle, write a hardened prompt to source, ...).

    Performs NO KG writes itself and never raises — a policy-consult failure
    degrades to a fail-closed ``deny`` verdict, mirroring
    ``GovernedAutoMerger._consult_action_policy``'s existing fail-closed handling.
    """
    eligible = evaluate_promotion(candidate, min_delta=min_delta, strict=strict)
    if not eligible:
        return PromotionVerdict(
            eligible=False,
            decision="",
            approved=False,
            reason="candidate did not beat incumbent",
        )

    from agent_utilities.orchestration.action_policy import (
        ActionRequest,
        get_action_policy,
    )

    active_policy = policy or get_action_policy(engine)
    request = ActionRequest(
        kind=candidate.policy_action_kind,
        target=candidate.artifact_id,
        params={
            "candidate_ref": candidate.candidate_ref,
            "candidate_reward": candidate.candidate_reward.value,
            "candidate_reward_source": candidate.candidate_reward.source,
            **(
                {"incumbent_reward": candidate.incumbent_reward.value}
                if candidate.incumbent_reward is not None
                else {}
            ),
            **candidate.evidence,
        },
        source=candidate.source,
        reason=candidate.reason
        or f"{candidate.artifact_kind} candidate {candidate.candidate_ref} promotion",
    )
    try:
        decision = active_policy.decide(request)
    except Exception as e:  # noqa: BLE001 — gate failure => fail closed, never crash
        logger.warning(
            "artifact_promotion: action_policy consult failed for %s: %s",
            request.summary(),
            e,
        )
        return PromotionVerdict(
            eligible=True,
            decision="deny",
            approved=False,
            reason=f"action policy unavailable (fail closed): {e}",
        )

    approved = decision.decision in ("allow", "allow_notify")
    return PromotionVerdict(
        eligible=True,
        decision=decision.decision,
        approved=approved,
        reason=decision.reason,
        approval_id=decision.approval_id,
    )
