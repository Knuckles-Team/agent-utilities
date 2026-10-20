"""The propose–probe–compose–persist–reflect harness evolution cycle."""

from __future__ import annotations

import dataclasses

import pytest

from agent_utilities.harness.harness_cycle import (
    ABSTAIN_REASONS,
    CycleOutcome,
    CycleRefusal,
    HarnessCandidate,
    propose,
)


def _candidate() -> HarnessCandidate:
    return HarnessCandidate(
        candidate_id="c-1",
        task_ref="task-1",
        component_kind="system_prompt",
        component_version_ref="eg:component/system_prompt/3",
        seed="seed-1",
        model_ref="model-1",
        budget_ref="budget-1",
    )


@pytest.mark.spec("AU-HARNESS-CYCLE-R001.1")
def test_model_round_trips() -> None:
    candidate = _candidate()
    assert HarnessCandidate(**dataclasses.asdict(candidate)) == candidate

    outcome = CycleOutcome(
        task_ref="task-1",
        k=2,
        adopted_version_ref=None,
        abstain_reason="budget_below_contrast",
        contrast_ref=None,
        lesson_ref=None,
    )
    assert CycleOutcome(**dataclasses.asdict(outcome)) == outcome

    adopted = CycleOutcome(
        task_ref="task-1",
        k=2,
        adopted_version_ref="eg:component/system_prompt/3",
        abstain_reason=None,
        contrast_ref="eg:contrast/1",
        lesson_ref="eg:lesson/1",
    )
    assert CycleOutcome(**dataclasses.asdict(adopted)) == adopted

    for code in ABSTAIN_REASONS:
        abstained = CycleOutcome(
            task_ref="task-1",
            k=1,
            adopted_version_ref=None,
            abstain_reason=code,
            contrast_ref=None,
            lesson_ref=None,
        )
        assert CycleOutcome(**dataclasses.asdict(abstained)) == abstained

    with pytest.raises(ValueError, match="open refusal code"):
        CycleOutcome(
            task_ref="task-1",
            k=2,
            adopted_version_ref=None,
            abstain_reason="because",
            contrast_ref=None,
            lesson_ref=None,
        )
    with pytest.raises(ValueError, match="adopts nothing"):
        CycleOutcome(
            task_ref="task-1",
            k=2,
            adopted_version_ref="eg:component/system_prompt/3",
            abstain_reason="compose_abstention",
            contrast_ref=None,
            lesson_ref=None,
        )


def test_cycle_refusal_rejects_open_code() -> None:
    for code in ABSTAIN_REASONS:
        assert CycleRefusal(code).code == code
    with pytest.raises(ValueError, match="open refusal code"):
        CycleRefusal("because")


@pytest.mark.spec("AU-HARNESS-CYCLE-R001.2")
@pytest.mark.parametrize(
    ("returned", "k", "expected_refs"),
    [
        (
            [
                {"component_kind": "system_prompt", "component_version_ref": "v-a"},
                {"component_kind": "guardrails", "component_version_ref": "v-b"},
            ],
            2,
            ("v-a", "v-b"),
        ),
        (
            [
                {"component_kind": "system_prompt", "component_version_ref": "v-a"},
                {"component_kind": "guardrails", "component_version_ref": "v-b"},
                {"component_kind": "memory", "component_version_ref": "v-c"},
            ],
            2,
            ("v-a", "v-b"),
        ),
    ],
)
def test_propose_queries_lesson_library(returned, k, expected_refs) -> None:
    queried: list[str] = []

    def library(task_class: str) -> list[dict[str, str]]:
        queried.append(task_class)
        return [dict(entry) for entry in returned]

    candidates = propose(
        task_ref="task-1",
        task_class="classification",
        library=library,
        seed="seed-1",
        model_ref="model-1",
        budget_ref="budget-1",
        k=k,
    )

    assert queried == ["classification"]
    assert tuple(c.component_version_ref for c in candidates) == expected_refs
    for candidate in candidates:
        assert candidate.task_ref == "task-1"
        assert candidate.seed == "seed-1"
        assert candidate.model_ref == "model-1"
        assert candidate.budget_ref == "budget-1"
        with pytest.raises(dataclasses.FrozenInstanceError):
            candidate.seed = "other"  # type: ignore[misc]

    # The cycle stores nothing: a rerun re-queries the library and stores no
    # answer of its own (no write, no cache).
    again = propose(
        task_ref="task-1",
        task_class="classification",
        library=library,
        seed="seed-1",
        model_ref="model-1",
        budget_ref="budget-1",
        k=k,
    )
    assert again == candidates
    assert queried == ["classification", "classification"]

    with pytest.raises(CycleRefusal, match="no_candidates"):
        propose(
            task_ref="task-1",
            task_class="classification",
            library=lambda _class: [],
            seed="seed-1",
            model_ref="model-1",
            budget_ref="budget-1",
            k=k,
        )
