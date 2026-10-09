"""AU-SEMANTIC-R009.1: typed contract + fail-closed refusal tests."""

from __future__ import annotations

import pytest

from agent_utilities.api.skill_derivation_client import (
    RunnableSkillDerivationClient,
    RunnableSkillDerivationRequest,
    RunnableSkillDerivationResponse,
    RunnableSkillDerivationUnavailableError,
    resolve_runnable_skill_derivation_client,
)


def test_request_and_response_are_typed_and_frozen() -> None:
    request = RunnableSkillDerivationRequest(
        source_envelope_id="env-1", tenant_id="tenant-1"
    )
    response = RunnableSkillDerivationResponse(
        skill_id="skill-1", derivation_receipt_id="receipt-1"
    )

    assert request.source_envelope_id == "env-1"
    assert response.skill_id == "skill-1"
    with pytest.raises(AttributeError):
        request.source_envelope_id = "env-2"  # type: ignore[misc]


def test_client_protocol_requires_runnable_skill_derivation_method() -> None:
    class _NotAClient:
        pass

    class _Client:
        def runnable_skill_derivation(
            self, request: RunnableSkillDerivationRequest
        ) -> RunnableSkillDerivationResponse:
            raise NotImplementedError

    assert not isinstance(_NotAClient(), RunnableSkillDerivationClient)
    assert isinstance(_Client(), RunnableSkillDerivationClient)


def test_resolve_fails_closed_while_eg_op_is_absent() -> None:
    """EG has not shipped runnable_skill_derivation yet (EG-REPO-INGEST-R002).

    The resolver must raise, never return a stub client or a default
    response, for as long as that is true.
    """
    with pytest.raises(RunnableSkillDerivationUnavailableError):
        resolve_runnable_skill_derivation_client()


def test_unavailable_error_is_a_runtime_error_not_a_default_value() -> None:
    assert issubclass(RunnableSkillDerivationUnavailableError, RuntimeError)
