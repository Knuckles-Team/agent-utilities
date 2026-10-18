"""Public scaffold state behavior; no run loop or snapshot availability is implied."""

import pytest
from pydantic import ValidationError

from agent_utilities.harness import PlanItem, PlanState


@pytest.mark.spec("AU-BOUNDARY-R043.4.1")
def test_replaces_whole_ordered_list_without_mutable_aliases():
    state = PlanState()
    assert state.items == ()
    supplied = [
        PlanItem(text="first", status="done"),
        PlanItem(text="second", status="in_progress"),
        PlanItem(text="third", status="pending"),
    ]
    before = state.replace(supplied)
    supplied.clear()
    assert [item.text for item in state.items] == ["first", "second", "third"]
    assert [item.status for item in state.items] == ["done", "in_progress", "pending"]
    after = state.replace([PlanItem(text="replacement", status="pending")])
    assert state.items == after
    assert [item.text for item in before] == ["first", "second", "third"]
    assert [item.text for item in after] == ["replacement"]
    assert state.replace([]) == ()


@pytest.mark.spec("AU-BOUNDARY-R043.4.1")
def test_states_are_isolated_and_items_are_frozen():
    first, second = PlanState(), PlanState()
    first.replace([PlanItem(text="private", status="pending")])
    assert second.items == ()
    with pytest.raises(ValidationError, match="frozen"):
        first.items[0].text = "changed"
    assert first.items[0].text == "private"


@pytest.mark.spec("AU-BOUNDARY-R043.4.1")
@pytest.mark.parametrize(
    "value",
    [
        {"text": "", "status": "pending"},
        {"text": 1, "status": "pending"},
        {"text": "item", "status": "unknown"},
        {"text": "item"},
        {"text": "item", "status": "done", "model": "chosen"},
    ],
)
def test_items_reject_malformed_fields(value):
    with pytest.raises(ValidationError):
        PlanItem.model_validate(value)


@pytest.mark.spec("AU-BOUNDARY-R043.4.1")
def test_invalid_later_item_leaves_entire_old_state_intact():
    state = PlanState()
    before = state.replace([PlanItem(text="original", status="pending")])
    invalid = PlanItem.model_construct(text="bad", status="unknown")
    with pytest.raises(ValidationError):
        state.replace([PlanItem(text="new", status="done"), invalid])
    assert state.items is before
