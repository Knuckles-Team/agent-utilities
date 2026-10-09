"""AU-SEMANTIC-R015.1: typed contract + fail-closed refusal tests."""

from __future__ import annotations

import pytest

from agent_utilities.api.workflow_derivation_client import (
    WorkflowDerivationClient,
    WorkflowDerivationRequest,
    WorkflowDerivationResponse,
    WorkflowDerivationUnavailableError,
    resolve_workflow_derivation_client,
)


def test_request_and_response_are_typed_and_frozen() -> None:
    request = WorkflowDerivationRequest(
        source_envelope_id="env-1", tenant_id="tenant-1", manifest_preset="default"
    )
    response = WorkflowDerivationResponse(
        workflow_id="workflow-1", derivation_receipt_id="receipt-1"
    )

    assert request.manifest_preset == "default"
    assert response.workflow_id == "workflow-1"
    with pytest.raises(AttributeError):
        request.tenant_id = "tenant-2"  # type: ignore[misc]


def test_client_protocol_requires_workflow_derivation_method() -> None:
    class _NotAClient:
        pass

    class _Client:
        def workflow_derivation(
            self, request: WorkflowDerivationRequest
        ) -> WorkflowDerivationResponse:
            raise NotImplementedError

    assert not isinstance(_NotAClient(), WorkflowDerivationClient)
    assert isinstance(_Client(), WorkflowDerivationClient)


def test_resolve_fails_closed_while_eg_op_is_absent() -> None:
    """EG has not shipped workflow_derivation yet (EG-REPO-INGEST-R002).

    The resolver must raise, never return a stub client or a default
    response, for as long as that is true.
    """
    with pytest.raises(WorkflowDerivationUnavailableError):
        resolve_workflow_derivation_client()


def test_unavailable_error_is_a_runtime_error_not_a_default_value() -> None:
    assert issubclass(WorkflowDerivationUnavailableError, RuntimeError)
