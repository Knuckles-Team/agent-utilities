"""Generative-model deferral policy (AU-HARNESS-R005).

AU defers requesting a microGPT-style generative model, in-engine autograd, a
generative UQL-producing model, or a KLPO loss implementation until a
separate evidence-backed need demonstrates held-out benefit, cost and safety.
In the meantime, AU reuses the existing typed Decide ladder (legal-option
derivation, calibrated probabilities, cost-sensitive optimal choice,
provenance, abstention, natural-language templating, constrained UQL
generation) and a bounded resident scorer is never treated as authorization
for a generative model.

This module is the single gate any call site must pass through before
proceeding with generative-model-class work (a microGPT-style generator,
in-engine autograd, a generative UQL-producing model, or a KLPO loss
implementation). It holds no opinion on *how* such work should be approved —
only that an explicit, evidence-backed approval record must exist first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class GenerativeModelWorkKind(StrEnum):
    """The classes of work this deferral covers (CONCEPT:AU-HARNESS-R005)."""

    MICROGPT_GENERATOR = "microgpt_generator"
    IN_ENGINE_AUTOGRAD = "in_engine_autograd"
    GENERATIVE_UQL_MODEL = "generative_uql_model"
    KLPO_LOSS = "klpo_loss"


@dataclass(frozen=True)
class EvidenceRecord:
    """An approved, evidence-backed need record for one work kind.

    A bounded resident scorer's existence is deliberately *not* a field here:
    scorer promotion (AU-HARNESS-R006) is a separate, already-governed gate
    and must never be read as authorization for generative-model work.
    """

    held_out_benefit: bool
    cost_assessed: bool
    safety_assessed: bool
    approved_by: str

    def is_sufficient(self) -> bool:
        return bool(
            self.held_out_benefit
            and self.cost_assessed
            and self.safety_assessed
            and self.approved_by.strip()
        )


class GenerativeModelWorkRefused(RuntimeError):
    """Raised when generative-model-class work is attempted without an
    approved, evidence-backed record (AU-HARNESS-R005)."""


@dataclass(frozen=True)
class GenerativeModelPolicyDecision:
    """The outcome of evaluating a generative-model-work request."""

    kind: GenerativeModelWorkKind
    permitted: bool
    reason: str
    evidence: EvidenceRecord | None = field(default=None)


def evaluate_generative_model_request(
    kind: GenerativeModelWorkKind,
    evidence: EvidenceRecord | None,
) -> GenerativeModelPolicyDecision:
    """Evaluate whether generative-model-class work of ``kind`` may proceed.

    Returns a decision; never raises. Use :func:`require_generative_model_approval`
    at the actual call site to enforce (raise on refusal).
    """
    if evidence is None:
        return GenerativeModelPolicyDecision(
            kind=kind,
            permitted=False,
            reason=(
                f"{kind.value} is deferred pending evidence-backed need "
                "(AU-HARNESS-R005): no evidence record supplied."
            ),
            evidence=None,
        )
    if not evidence.is_sufficient():
        return GenerativeModelPolicyDecision(
            kind=kind,
            permitted=False,
            reason=(
                f"{kind.value} is deferred pending evidence-backed need "
                "(AU-HARNESS-R005): evidence record is incomplete "
                "(requires held-out benefit, cost and safety assessment, "
                "and a named approver)."
            ),
            evidence=evidence,
        )
    return GenerativeModelPolicyDecision(
        kind=kind,
        permitted=True,
        reason=f"{kind.value} approved by {evidence.approved_by}.",
        evidence=evidence,
    )


def require_generative_model_approval(
    kind: GenerativeModelWorkKind,
    evidence: EvidenceRecord | None,
) -> GenerativeModelPolicyDecision:
    """Enforce the deferral: raise :class:`GenerativeModelWorkRefused` unless
    an approved, evidence-backed record for ``kind`` is supplied."""
    decision = evaluate_generative_model_request(kind, evidence)
    if not decision.permitted:
        raise GenerativeModelWorkRefused(decision.reason)
    return decision
