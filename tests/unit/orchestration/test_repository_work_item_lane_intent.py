"""RMDD-28: typed immutable lane-intent fields on Repository WorkItems.

Proves immutability by attempted mutation (not by convention), and proves
the model_validator wiring that ties ``lane_intent``/``lane_cleanup_intent``
to exactly the ``lane.lifecycle``/``lane.cleanup`` operations.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_utilities.orchestration.repository_work_item import (
    RepositoryOperation,
    RepositoryWorkItemKind,
    RepositoryWorkItemRequest,
)
from agent_utilities.protocols.epistemic_operations._generated import (
    DevelopmentLaneCleanupIntent,
    DevelopmentLaneIntent,
)

_INTENT = DevelopmentLaneIntent(
    schema_version="1",
    tenant_ref="tenant:golden",
    request_id="request:golden",
    lane_id="lane:golden",
    repository_id="repo:golden",
    base_ref="refs/heads/main",
    base_sha="0123456789abcdef0123456789abcdef01234567",
    branch="rmdd-28/golden",
    host_target_kind="inventory_alias",
    host_target_alias="host:golden",
    host_ref="host-ref:golden",
    resource_reservation_id="reservation:golden",
    workspace_ref="workspace:golden",
    worktree_locator="lanes/golden",
    owner_id="agent:golden",
    session_id="session:golden",
    fairness_group="fairness:golden",
    quota_policy_name="default",
    quota_policy_version="1",
    predicted_disk_bytes=4096,
    ttl_ms=60000,
    input_fingerprint=(
        "v1:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    ),
)


def _base_kwargs(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "request_id": "request:one",
        "idempotency_key": "idem:one",
        "operation": RepositoryOperation.LANE_LIFECYCLE,
        "repository_id": "repo:one",
        "base_ref": "refs/heads/main",
        "base_sha": "0123456789abcdef0123456789abcdef01234567",
        "owner_id": "agent:one",
        "session_id": "session:one",
        "tenant_id": "tenant:one",
        "lane_intent": _INTENT,
    }
    values.update(overrides)
    return values


def test_lane_lifecycle_request_requires_and_accepts_a_typed_lane_intent() -> None:
    request = RepositoryWorkItemRequest(**_base_kwargs())
    assert request.lane_intent == _INTENT
    assert request.lane_cleanup_intent is None
    # ``use_enum_values=True`` stores the plain string value, not the enum
    # member -- compare by value, matching the model's own convention.
    assert request.operation == RepositoryOperation.LANE_LIFECYCLE.value


def test_lane_lifecycle_kind_is_distinct_from_lane_allocate() -> None:
    assert RepositoryWorkItemKind.LANE_LIFECYCLE == "repository.lane.lifecycle"
    assert RepositoryWorkItemKind.LANE_ALLOCATE == "repository.lane.allocate"
    assert RepositoryWorkItemKind.LANE_LIFECYCLE != RepositoryWorkItemKind.LANE_ALLOCATE


def test_lane_lifecycle_request_without_lane_intent_is_refused() -> None:
    with pytest.raises(ValidationError, match="lane.lifecycle requires"):
        RepositoryWorkItemRequest(**_base_kwargs(lane_intent=None))


def test_non_lifecycle_request_with_lane_intent_is_refused() -> None:
    with pytest.raises(ValidationError, match="only valid on a lane.lifecycle"):
        RepositoryWorkItemRequest(
            **_base_kwargs(operation=RepositoryOperation.REPOSITORY)
        )


def test_lane_intent_and_lane_cleanup_intent_are_mutually_exclusive() -> None:
    cleanup = DevelopmentLaneCleanupIntent(
        schema_version="1",
        hold_id="v1:" + "a" * 64,
        lane_id="lane:golden",
        expected_hold_revision=7,
    )
    with pytest.raises(ValidationError, match="mutually exclusive"):
        RepositoryWorkItemRequest(
            **_base_kwargs(
                operation=RepositoryOperation.LANE_LIFECYCLE,
                lane_cleanup_intent=cleanup,
            )
        )


def test_lane_intent_is_never_smuggled_through_consent_or_preferred_target() -> None:
    """``consent``/``preferred_target`` remain ``extra="forbid"`` typed models.

    A caller cannot slip a ``lane_event``/lane-intent-shaped payload through
    either field; both reject unknown keys outright.
    """

    with pytest.raises(ValidationError):
        RepositoryWorkItemRequest(**_base_kwargs(consent={"lane_event": "smuggled"}))
    with pytest.raises(ValidationError):
        RepositoryWorkItemRequest(
            **_base_kwargs(preferred_target={"lane_event": "smuggled"})
        )


def test_lane_intent_field_is_genuinely_immutable_not_by_convention() -> None:
    """Attempted post-construction mutation is refused by pydantic frozen=True.

    This is the immutability proof required by the RMDD-28 lane brief: a
    test that attempts mutation and is refused, not an assertion resting on
    convention alone.
    """

    request = RepositoryWorkItemRequest(**_base_kwargs())
    with pytest.raises(ValidationError) as excinfo:
        request.lane_intent = None  # type: ignore[misc]
    assert "frozen" in str(excinfo.value).lower()

    other_intent = _INTENT.model_copy(update={"lane_id": "lane:different"})
    with pytest.raises(ValidationError) as excinfo_second:
        request.lane_intent = other_intent  # type: ignore[misc]
    assert "frozen" in str(excinfo_second.value).lower()

    # The generated DevelopmentLaneIntent DTO itself is also frozen+strict
    # (ProtocolModel base), so even the nested typed value cannot be mutated
    # in place.
    with pytest.raises(ValidationError):
        request.lane_intent.lane_id = "lane:mutated"  # type: ignore[misc]
