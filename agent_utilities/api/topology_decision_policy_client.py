"""Typed client contract for a served, calibrated topology decision policy.

AU-SEMANTIC-R008.1: this module exists before any served topology
decision-policy op exists. It declares the typed request/response shape
(including an explicit abstention response) AU's reasoning-topology
selection (``agent_utilities/graph/reasoning/topology.py``) will call once a
served, calibrated decision policy is reachable, and it fails closed --
raising, never falling back to the local exponential-moving-average outcome
store -- for as long as no such policy is installed.

AU-SEMANTIC-R008.2 wires ``topology.py``'s selection/outcome-recording path
to the real call and deletes the local EMA outcome store; the typed
request/response/Protocol shapes here are expected to survive that swap
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = [
    "TopologyDecisionPolicyClient",
    "TopologyDecisionRequest",
    "TopologyDecisionResponse",
    "TopologyDecisionPolicyUnavailableError",
    "resolve_topology_decision_policy_client",
]


@dataclass(frozen=True, slots=True)
class TopologyDecisionRequest:
    """Typed request for a served, calibrated topology decision policy."""

    task_digest: str
    candidate_topologies: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TopologyDecisionResponse:
    """Typed response from a served topology decision policy.

    ``abstained`` is explicit: a calibrated policy that lacks confidence
    must say so rather than guessing, so callers can fail closed instead of
    silently selecting from an uncalibrated default.
    """

    selected_topology: str | None
    confidence: float
    abstained: bool


@runtime_checkable
class TopologyDecisionPolicyClient(Protocol):
    """Contract AU calls through once a served decision policy exists."""

    def select_topology(
        self, request: TopologyDecisionRequest
    ) -> TopologyDecisionResponse: ...


class TopologyDecisionPolicyUnavailableError(RuntimeError):
    """Raised fail-closed while no served topology decision policy exists.

    This is never converted into a fallback to the local
    exponential-moving-average outcome store. A caller that catches this
    error must treat topology selection as unavailable, not silently read
    the local store.
    """


def resolve_topology_decision_policy_client() -> TopologyDecisionPolicyClient:
    """Resolve the served, calibrated topology decision policy client.

    Fails closed with :class:`TopologyDecisionPolicyUnavailableError` until
    a served decision-policy op exists and is installed (AU-SEMANTIC-R008.2
    wires the real call here and deletes the local EMA outcome store).
    """
    try:
        from epistemic_graph.generated.reasoning_policy import (  # type: ignore[import-not-found]
            ReasoningPolicyClient,
        )
    except ImportError as exc:
        raise TopologyDecisionPolicyUnavailableError(
            "no served topology decision-policy client is installed; "
            "topology selection is unavailable rather than falling back to "
            "the local EMA outcome store (AU-SEMANTIC-R008.1 fail-closed "
            "refusal)"
        ) from exc

    if not hasattr(ReasoningPolicyClient, "select_topology"):
        raise TopologyDecisionPolicyUnavailableError(
            "the installed generated client does not yet expose "
            "select_topology; refusing rather than reading the local EMA "
            "outcome store (AU-SEMANTIC-R008.1 fail-closed refusal)"
        )

    return ReasoningPolicyClient()
