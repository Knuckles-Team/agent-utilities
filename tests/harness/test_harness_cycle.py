"""The propose–probe–compose–persist–reflect harness evolution cycle."""

from __future__ import annotations

import dataclasses

import pytest

from agent_utilities.harness.harness_cycle import (
    ABSTAIN_REASONS,
    CycleOutcome,
    HarnessCandidate,
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
    from agent_utilities.harness.harness_cycle import CycleRefusal

    for code in ABSTAIN_REASONS:
        assert CycleRefusal(code).code == code
    with pytest.raises(ValueError, match="open refusal code"):
        CycleRefusal("because")
