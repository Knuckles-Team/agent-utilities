"""AU-SEMANTIC-R019.1 typed model and refusal tests."""

from __future__ import annotations

import pytest

from agent_utilities.governance.eg_relational_authority_handoff import (
    RelationalAuthorityDelegationClaim,
    RelationalAuthorityDelegationError,
)


def test_claim_is_typed() -> None:
    claim = RelationalAuthorityDelegationClaim(domain="usage_store")
    assert claim.domain == "usage_store"
    assert claim.delegated_to == "epistemic-graph"


def test_undelegated_claim_refuses() -> None:
    claim = RelationalAuthorityDelegationClaim(
        domain="usage_store", delegated_to="agent-utilities"
    )
    with pytest.raises(RelationalAuthorityDelegationError):
        claim.require_delegated()


def test_delegated_claim_passes() -> None:
    RelationalAuthorityDelegationClaim(domain="usage_store").require_delegated()
