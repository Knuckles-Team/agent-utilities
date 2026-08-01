"""Tests for :mod:`agent_utilities.patterns.prioritization`.

CONCEPT:AU-KG.memory.checkpoint-worthiness-scoring (D-KCI-2) — the honest
context-*overlap* predictor. The dependency graph (``blocking_ids``/``blocked_by_ids``)
tells you two tasks are *related*; it does not tell you they would hit the same warm
context. These tests prove ``predicted_reuse_from_context_overlap`` genuinely
discriminates the two rather than treating dependency-adjacency as reuse.
"""

from __future__ import annotations

from agent_utilities.patterns.prioritization import (
    PrioritizationEngine,
    PrioritizedTask,
)


def _engine_with(*tasks: PrioritizedTask) -> PrioritizationEngine:
    engine = PrioritizationEngine()
    for task in tasks:
        engine.add_task(task)
    return engine


def test_no_recorded_context_keys_abstains_rather_than_reporting_zero():
    """No fingerprint anywhere -> (None, None), never (0, 0). A caller wiring this
    straight into CheckpointObservation must see the scorer still abstain."""
    engine = _engine_with(
        PrioritizedTask(id="a", description="root"),
        PrioritizedTask(id="b", description="dependent", blocked_by_ids=["a"]),
    )
    assert engine.predicted_reuse_from_context_overlap("a") == (None, None)


def test_dependency_adjacency_alone_is_not_counted_as_reuse():
    """B depends on A (A blocks B), but B's own recorded context does not overlap A's
    -- must NOT be counted. This is the exact gap D-KCI-2 named: adjacency != overlap."""
    engine = _engine_with(
        PrioritizedTask(
            id="a",
            description="root",
            blocking_ids=["b"],
            context_keys=frozenset({"kg:node:1"}),
        ),
        PrioritizedTask(
            id="b",
            description="dependent, disjoint context",
            blocked_by_ids=["a"],
            context_keys=frozenset({"kg:node:999"}),
        ),
    )
    siblings, queued = engine.predicted_reuse_from_context_overlap("a")
    assert (siblings, queued) == (0, 0)


def test_genuine_overlap_is_counted_and_non_overlapping_siblings_are_excluded():
    """Of two dependency-adjacent tasks, only the one with a real fingerprint overlap
    counts -- proving the signal discriminates rather than counting all dependents."""
    engine = _engine_with(
        PrioritizedTask(
            id="a",
            description="root",
            blocking_ids=["b", "c"],
            context_keys=frozenset({"kg:node:1", "kg:node:2"}),
        ),
        PrioritizedTask(
            id="b",
            description="overlapping sibling",
            blocked_by_ids=["a"],
            context_keys=frozenset({"kg:node:2", "kg:node:5"}),
        ),
        PrioritizedTask(
            id="c",
            description="non-overlapping sibling (dependency-adjacent only)",
            blocked_by_ids=["a"],
            context_keys=frozenset({"kg:node:77"}),
        ),
        PrioritizedTask(
            id="d",
            description="not dependency-adjacent at all",
            context_keys=frozenset({"kg:node:1"}),
        ),
    )
    siblings, queued = engine.predicted_reuse_from_context_overlap("a")
    # Only "b" is both dependency-adjacent AND genuinely context-overlapping.
    assert siblings == 1
    # "d" is not dependency-adjacent (not in blocking_ids/blocked_by_ids) but is
    # still pending and genuinely overlaps -- counted as a queued reuse candidate.
    assert queued == 1


def test_explicit_context_keys_override_the_tasks_own_recorded_fingerprint():
    engine = _engine_with(
        PrioritizedTask(id="a", description="root", blocking_ids=["b"]),
        PrioritizedTask(
            id="b",
            description="sibling",
            blocked_by_ids=["a"],
            context_keys=frozenset({"kg:node:42"}),
        ),
    )
    # "a" itself has no recorded context_keys, but a caller may pass the CURRENT
    # prospective context explicitly instead of relying on what's on file.
    siblings, _queued = engine.predicted_reuse_from_context_overlap(
        "a", context_keys=frozenset({"kg:node:42"})
    )
    assert siblings == 1


def test_unknown_task_id_raises():
    engine = _engine_with(PrioritizedTask(id="a", description="root"))
    try:
        engine.predicted_reuse_from_context_overlap("nonexistent")
    except KeyError as exc:
        assert "nonexistent" in str(exc)
    else:
        raise AssertionError("expected KeyError for an unknown task id")
