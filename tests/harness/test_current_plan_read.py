"""Public validation contract for run scaffold snapshots."""

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from agent_utilities.api import (
    CurrentPlanRead,
    CurrentPlanRequest,
    CurrentPlanSnapshot,
    PlanItem,
)


def snapshot(**changes: object) -> CurrentPlanSnapshot:
    values = {
        "run_id": "run-one",
        "revision": 0,
        "observed_at": datetime(2026, 1, 1, tzinfo=UTC),
        "expires_at": datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=30),
        "items": (),
    }
    values.update(changes)
    return CurrentPlanSnapshot.model_validate(values)


@pytest.mark.spec("AU-HARNESS-R014.6")
def test_snapshot_contract_and_empty_plan() -> None:
    available = CurrentPlanRead(availability="available", snapshot=snapshot())
    unavailable = CurrentPlanRead(availability="unavailable")
    assert available.snapshot is not None
    assert available.snapshot.items == ()
    assert unavailable.snapshot is None
    assert available.model_dump(mode="json") != unavailable.model_dump(mode="json")
    assert CurrentPlanRead.model_validate_json(available.model_dump_json()) == available
    assert (
        CurrentPlanRead.model_validate_json(unavailable.model_dump_json())
        == unavailable
    )
    assert CurrentPlanRequest(run_id="run-one").model_dump() == {"run_id": "run-one"}


@pytest.mark.spec("AU-HARNESS-R014.6")
def test_snapshot_preserves_order_and_is_deeply_immutable() -> None:
    items = tuple(
        PlanItem(text=status, status=status)
        for status in ("done", "in_progress", "pending")
    )
    result = snapshot(items=items)
    assert result.items == items
    assert [item.status for item in result.items] == ["done", "in_progress", "pending"]
    with pytest.raises(ValidationError):
        result.revision = 2
    with pytest.raises(ValidationError):
        result.items[0].text = "changed"
    with pytest.raises(TypeError):
        result.items[0] = items[1]
    assert CurrentPlanSnapshot.model_validate_json(result.model_dump_json()) == result


@pytest.mark.spec("AU-HARNESS-R014.6")
@pytest.mark.parametrize(
    "changes",
    [
        {"revision": -1},
        {"revision": True},
        {"revision": "1"},
        {"run_id": ""},
        {"run_id": 42},
        {"extra": "rejected"},
        {"observed_at": datetime(2026, 1, 1)},
        {"expires_at": datetime(2026, 1, 1)},
        {"observed_at": datetime(2026, 1, 1, tzinfo=timezone(timedelta(hours=1)))},
        {"expires_at": datetime(2026, 1, 1, tzinfo=timezone(timedelta(hours=1)))},
        {"expires_at": datetime(2026, 1, 1, tzinfo=UTC)},
        {"expires_at": datetime(2025, 1, 1, tzinfo=UTC)},
        {"items": ({"text": "item", "status": "unknown"},)},
        {"items": ({"text": "", "status": "pending"},)},
        {"items": ({"text": "item", "status": "done", "extra": True},)},
    ],
)
def test_malformed_snapshot_fails_closed(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        snapshot(**changes)


@pytest.mark.spec("AU-HARNESS-R014.6")
def test_read_rejects_inconsistent_availability_and_caller_claims() -> None:
    for values in (
        {"availability": "available"},
        {"availability": "unavailable", "snapshot": snapshot()},
        {"availability": "unknown"},
        {"availability": "unavailable", "items": ()},
    ):
        with pytest.raises(ValidationError):
            CurrentPlanRead.model_validate(values)
    for claim in ("tenant_id", "principal_id", "session", "scope"):
        with pytest.raises(ValidationError):
            CurrentPlanRequest.model_validate(
                {"run_id": "run-one", claim: "caller-input"}
            )
