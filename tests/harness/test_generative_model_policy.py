"""Tests for the generative-model deferral gate (AU-HARNESS-R005)."""

from __future__ import annotations

import pytest

from agent_utilities.harness.generative_model_policy import (
    EvidenceRecord,
    GenerativeModelWorkKind,
    GenerativeModelWorkRefused,
    evaluate_generative_model_request,
    require_generative_model_approval,
)


@pytest.mark.parametrize(
    "kind",
    list(GenerativeModelWorkKind),
)
@pytest.mark.spec("AU-HARNESS-R005")
def test_no_evidence_is_refused(kind: GenerativeModelWorkKind) -> None:
    decision = evaluate_generative_model_request(kind, None)
    assert decision.permitted is False
    assert "AU-HARNESS-R005" in decision.reason

    with pytest.raises(GenerativeModelWorkRefused):
        require_generative_model_approval(kind, None)


def test_incomplete_evidence_is_refused() -> None:
    incomplete = EvidenceRecord(
        held_out_benefit=True,
        cost_assessed=False,
        safety_assessed=True,
        approved_by="reviewer",
    )
    decision = evaluate_generative_model_request(
        GenerativeModelWorkKind.MICROGPT_GENERATOR, incomplete
    )
    assert decision.permitted is False

    with pytest.raises(GenerativeModelWorkRefused):
        require_generative_model_approval(GenerativeModelWorkKind.KLPO_LOSS, incomplete)


def test_unapproved_evidence_without_approver_is_refused() -> None:
    unapproved = EvidenceRecord(
        held_out_benefit=True,
        cost_assessed=True,
        safety_assessed=True,
        approved_by="",
    )
    assert unapproved.is_sufficient() is False
    with pytest.raises(GenerativeModelWorkRefused):
        require_generative_model_approval(
            GenerativeModelWorkKind.GENERATIVE_UQL_MODEL, unapproved
        )


def test_complete_evidence_is_permitted() -> None:
    approved = EvidenceRecord(
        held_out_benefit=True,
        cost_assessed=True,
        safety_assessed=True,
        approved_by="eng-review-2026-10-09",
    )
    decision = require_generative_model_approval(
        GenerativeModelWorkKind.IN_ENGINE_AUTOGRAD, approved
    )
    assert decision.permitted is True
    assert decision.evidence is approved
