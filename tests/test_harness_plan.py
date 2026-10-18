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
    expected = tuple(supplied)
    before = state.replace(supplied)
    supplied.clear()
    assert state.items == expected
    after = state.replace([PlanItem(text="replacement", status="pending")])
    assert state.items == after
    assert before == expected
    assert after == (PlanItem(text="replacement", status="pending"),)
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


@pytest.mark.spec("AU-BOUNDARY-R043.5.1")
def test_revisions_count_accepted_replacements_only():
    state = PlanState()
    assert state.read_versioned() == (0, ())
    item = PlanItem(text="step", status="pending")
    state.replace([item])
    old = state.read_versioned()
    assert old == (1, (item,))
    assert state.read_versioned() == old
    state.replace([item])
    assert state.read_versioned() == (2, (item,))
    invalid = PlanItem.model_construct(text="invalid", status="unknown")
    with pytest.raises(ValidationError):
        state.replace([invalid])
    assert state.read_versioned() == (2, (item,))
    state.replace([])
    assert state.read_versioned() == (3, ())
    assert old == (1, (item,))
    assert PlanState().read_versioned() == (0, ())


@pytest.mark.spec("AU-BOUNDARY-R043.5.1")
def test_competing_updates_have_unique_revisions_and_consistent_views():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    state = PlanState()
    barrier = Barrier(3, timeout=5)
    observed = []

    def writer(label):
        item = PlanItem(text=label, status="pending")
        for _ in range(10):
            barrier.wait()
            state.replace([item, item])
            barrier.wait()

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(writer, label) for label in ("first", "second")]
        for index in range(10):
            barrier.wait()
            revision, items = state.read_versioned()
            assert 2 * index <= revision <= 2 * index + 2
            assert items == () or (len(items) == 2 and items[0] == items[1])
            observed.append((revision, items))
            barrier.wait()
            assert state.read_versioned()[0] == 2 * (index + 1)
        for future in futures:
            future.result()
    assert state.read_versioned()[0] == 20
    assert all(
        left[0] <= right[0] for left, right in zip(observed, observed[1:], strict=False)
    )


@pytest.mark.spec("AU-BOUNDARY-R043.5.1")
def test_interleaved_read_never_pairs_revision_with_another_update():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    state = PlanState()
    barrier = Barrier(2, timeout=5)

    def writer():
        for revision in range(1, 11):
            barrier.wait()
            state.replace([PlanItem(text=str(revision), status="pending")])
            barrier.wait()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(writer)
        for _ in range(10):
            barrier.wait()
            revision, items = state.read_versioned()
            assert items == (
                ()
                if revision == 0
                else (PlanItem(text=str(revision), status="pending"),)
            )
            barrier.wait()
        future.result()
