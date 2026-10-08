"""AU semantic tiers mirror the EG S1-S6 queue classes (spec: baseline-ingestion)."""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.core import semantic_tiers as tiers


def test_stage_classes_match_the_eg_stage_table() -> None:
    assert tiers.STAGE_QUEUE_CLASS == {
        "S1": "fast",
        "S2": "medium",
        "S3": "medium",
        "S4": "slow_heavy",
        "S5": "slow_heavy",
        "S6": "slow_heavy",
    }
    assert tiers.QUEUE_CLASS_ORDER == ("fast", "medium", "slow_heavy")


@pytest.mark.parametrize(
    ("task_type", "stage", "queue_class", "priority"),
    [
        ("skill_workflows", "S1", "fast", 1),
        ("scheduled_job", "S1", "fast", 1),
        ("codebase", "S2", "medium", 2),
        ("document", "S2", "medium", 2),
        ("enrichment_backfill", "S4", "slow_heavy", 3),
        ("unknown_type", "S2", "medium", 2),
        (None, "S2", "medium", 2),
    ],
)
def test_task_type_routing(
    task_type: str | None, stage: str, queue_class: str, priority: int
) -> None:
    assert tiers.entry_stage_for_task_type(task_type) == stage
    assert tiers.queue_class_for_task_type(task_type) == queue_class
    assert tiers.priority_for_queue_class(queue_class) == priority


def test_unknown_class_is_background_and_drains_last() -> None:
    assert tiers.priority_for_queue_class("bogus") == 3
    assert tiers.queue_class_rank("bogus") == 3
    ranks = [tiers.queue_class_rank(name) for name in tiers.QUEUE_CLASS_ORDER]
    assert ranks == [0, 1, 2]
