"""AU-SEMANTIC-R008.1: typed contract + fail-closed refusal tests."""

from __future__ import annotations

import pytest

from agent_utilities.api.topology_decision_policy_client import (
    TopologyDecisionPolicyClient,
    TopologyDecisionPolicyUnavailableError,
    TopologyDecisionRequest,
    TopologyDecisionResponse,
    resolve_topology_decision_policy_client,
)


@pytest.mark.spec("AU-SEMANTIC-R008.1")
def test_request_and_response_are_typed_and_frozen() -> None:
    request = TopologyDecisionRequest(
        task_digest="digest-1", candidate_topologies=("cot", "tot")
    )
    response = TopologyDecisionResponse(
        selected_topology="cot", confidence=0.9, abstained=False
    )

    assert request.task_digest == "digest-1"
    assert response.selected_topology == "cot"
    with pytest.raises(AttributeError):
        request.task_digest = "digest-2"  # type: ignore[misc]


@pytest.mark.spec("AU-SEMANTIC-R008.1")
def test_response_supports_explicit_abstention() -> None:
    response = TopologyDecisionResponse(
        selected_topology=None, confidence=0.0, abstained=True
    )
    assert response.abstained is True
    assert response.selected_topology is None


@pytest.mark.spec("AU-SEMANTIC-R008.1")
def test_client_protocol_requires_select_topology_method() -> None:
    class _NotAClient:
        pass

    class _Client:
        def select_topology(
            self, request: TopologyDecisionRequest
        ) -> TopologyDecisionResponse:
            raise NotImplementedError

    assert not isinstance(_NotAClient(), TopologyDecisionPolicyClient)
    assert isinstance(_Client(), TopologyDecisionPolicyClient)


@pytest.mark.spec("AU-SEMANTIC-R008.1")
def test_resolve_fails_closed_while_no_served_policy_exists() -> None:
    """No served topology decision policy exists yet.

    The resolver must raise, never fall back to reading the local EMA
    outcome store, for as long as that is true.
    """
    with pytest.raises(TopologyDecisionPolicyUnavailableError):
        resolve_topology_decision_policy_client()


@pytest.mark.spec("AU-SEMANTIC-R008.1")
def test_unavailable_error_is_a_runtime_error_not_a_fallback() -> None:
    assert issubclass(TopologyDecisionPolicyUnavailableError, RuntimeError)
