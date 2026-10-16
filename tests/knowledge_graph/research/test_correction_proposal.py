#!/usr/bin/python
from __future__ import annotations

"""Tests for AU-HARNESS-R004.1 — typed correction-proposal model."""

import pytest

from agent_utilities.knowledge_graph.research.correction_proposal import (
    STATUS_AUTHORIZED,
    STATUS_MATERIALIZED,
    STATUS_PROPOSED,
    STATUS_REFUSED,
    CorrectionProposal,
    CorrectionProposalRefused,
    MaterializationReceipt,
    submit_for_authorization,
)


def _proposal(**overrides: object) -> CorrectionProposal:
    fields: dict[str, object] = {
        "gap_ref": "gap:123",
        "specification": "Fix the widget alignment bug.",
        "change_ref": "changeset:abc",
        "evidence": ("trace:1",),
    }
    fields.update(overrides)
    return CorrectionProposal(**fields)  # type: ignore[arg-type]


@pytest.mark.spec("AU-HARNESS-R004.1")
def test_valid_proposal_constructs() -> None:
    proposal = _proposal()
    assert proposal.status == STATUS_PROPOSED
    assert proposal.eligible_to_resolve is False


@pytest.mark.parametrize(
    "overrides",
    [
        {"gap_ref": ""},
        {"specification": ""},
        {"change_ref": ""},
        {"evidence": ()},
    ],
)
@pytest.mark.spec("AU-HARNESS-R004.1")
def test_missing_required_field_is_refused(overrides: dict[str, object]) -> None:
    with pytest.raises(CorrectionProposalRefused):
        _proposal(**overrides)


@pytest.mark.spec("AU-HARNESS-R004.1")
def test_unknown_status_is_refused() -> None:
    with pytest.raises(CorrectionProposalRefused):
        _proposal(status="merged")


def test_materialized_without_receipt_is_refused() -> None:
    with pytest.raises(CorrectionProposalRefused):
        _proposal(status=STATUS_MATERIALIZED)


@pytest.mark.parametrize(
    "field_name",
    ["repo_path", "worktree_path", "commit_sha", "branch", "patch", "diff"],
)
def test_direct_git_metadata_field_is_refused(field_name: str) -> None:
    with pytest.raises(CorrectionProposalRefused):
        _proposal(metadata={field_name: "/some/local/path"})


def test_materialization_receipt_requires_all_fields() -> None:
    with pytest.raises(CorrectionProposalRefused):
        MaterializationReceipt(receipt_id="", commit_ref="sha1", materialized_at="now")
    with pytest.raises(CorrectionProposalRefused):
        MaterializationReceipt(receipt_id="r1", commit_ref="", materialized_at="now")
    with pytest.raises(CorrectionProposalRefused):
        MaterializationReceipt(receipt_id="r1", commit_ref="sha1", materialized_at="")


def test_materialized_proposal_with_receipt_is_eligible_to_resolve() -> None:
    receipt = MaterializationReceipt(
        receipt_id="receipt:1",
        commit_ref="commit:deadbeef",
        materialized_at="2026-10-09T00:00:00Z",
    )
    proposal = _proposal(status=STATUS_MATERIALIZED, receipt=receipt)
    assert proposal.eligible_to_resolve is True


def test_authorized_proposal_not_yet_eligible_to_resolve() -> None:
    proposal = _proposal(status=STATUS_AUTHORIZED)
    assert proposal.eligible_to_resolve is False


def test_submit_for_authorization_accepts_proposed() -> None:
    proposal = _proposal()
    assert submit_for_authorization(proposal) is proposal


def test_submit_for_authorization_refuses_non_proposed_status() -> None:
    proposal = _proposal(status=STATUS_AUTHORIZED)
    with pytest.raises(CorrectionProposalRefused):
        submit_for_authorization(proposal)


def test_refused_status_is_valid_terminal_state() -> None:
    proposal = _proposal(status=STATUS_REFUSED)
    assert proposal.eligible_to_resolve is False
