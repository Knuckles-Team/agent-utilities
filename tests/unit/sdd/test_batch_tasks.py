"""Tests for dependency-aware SDD parallel task batching.

CONCEPT:ORCH-1.3 — Spec-Driven Development Pipeline

Covers ``SDDManager.get_parallel_opportunities`` (the real topological batcher)
and the ``get_sdd_parallel_batches`` tool that delegates to it. Batches are
"waves": every task in a wave has its dependencies satisfied by an earlier wave
(or already completed), and no two tasks in a wave touch the same files.
"""

import tempfile
from unittest.mock import MagicMock, patch

import pytest
from pydantic_ai import RunContext

from agent_utilities.models import AgentDeps, Task, Tasks
from agent_utilities.sdd import SDDManager
from agent_utilities.tools.sdd_tools import get_sdd_parallel_batches


def _wave_of(batches: list[list[str]], task_id: str) -> int:
    """Return the index of the wave containing ``task_id`` (-1 if absent)."""
    for i, batch in enumerate(batches):
        if task_id in batch:
            return i
    return -1


def _no_dependency_violations(batches: list[list[str]], tasks: Tasks) -> bool:
    """Assert no task shares a wave with (or precedes) one of its dependencies."""
    deps = {t.id: t.depends_on for t in tasks.tasks}
    for task_id, dependencies in deps.items():
        w = _wave_of(batches, task_id)
        if w < 0:
            continue
        for dep in dependencies:
            dw = _wave_of(batches, dep)
            if dw < 0:
                # Dependency completed/unknown -> already satisfied.
                continue
            if dw >= w:
                return False
    return True


@pytest.fixture
def manager():
    return SDDManager(tempfile.mkdtemp())


def test_independent_tasks_share_first_batch(manager):
    """A and B are independent; C depends on A."""
    tasks = Tasks(
        feature_id="feat",
        tasks=[
            Task(id="A", file_paths=["a.py"]),
            Task(id="B", file_paths=["b.py"]),
            Task(id="C", depends_on=["A"], file_paths=["c.py"]),
        ],
    )
    batches = manager.get_parallel_opportunities(tasks)

    # (a) A and B share the first batch.
    assert _wave_of(batches, "A") == 0
    assert _wave_of(batches, "B") == 0

    # (b) C is in a later batch than A.
    assert _wave_of(batches, "C") > _wave_of(batches, "A")

    # (c) No batch contains a task whose dependency is in the same/later batch.
    assert _no_dependency_violations(batches, tasks)


def test_no_task_dropped(manager):
    """Every pending task must appear in exactly one wave."""
    tasks = Tasks(
        feature_id="feat",
        tasks=[
            Task(id="A"),
            Task(id="B"),
            Task(id="C", depends_on=["A"]),
            Task(id="D", depends_on=["C"]),
        ],
    )
    batches = manager.get_parallel_opportunities(tasks)
    flat = [tid for batch in batches for tid in batch]
    assert sorted(flat) == ["A", "B", "C", "D"]
    # Chain ordering A < C < D.
    assert _wave_of(batches, "A") < _wave_of(batches, "C") < _wave_of(batches, "D")
    assert _no_dependency_violations(batches, tasks)


def test_completed_dependency_is_satisfied(manager):
    """A completed dependency does not hold back its dependent."""
    tasks = Tasks(
        feature_id="feat",
        tasks=[
            Task(id="A", status="completed"),
            Task(id="C", depends_on=["A"]),
        ],
    )
    batches = manager.get_parallel_opportunities(tasks)
    # A is completed -> excluded; C is immediately schedulable.
    assert batches == [["C"]]


def test_file_collision_splits_into_separate_waves(manager):
    """Two independent tasks touching the same file cannot share a wave."""
    tasks = Tasks(
        feature_id="feat",
        tasks=[
            Task(id="A", file_paths=["shared.py"]),
            Task(id="B", file_paths=["shared.py"]),
        ],
    )
    batches = manager.get_parallel_opportunities(tasks)
    assert _wave_of(batches, "A") != _wave_of(batches, "B")
    flat = [tid for batch in batches for tid in batch]
    assert sorted(flat) == ["A", "B"]


def test_dependency_cycle_terminates(manager):
    """A cyclic dependency must not loop forever; every task still scheduled."""
    tasks = Tasks(
        feature_id="feat",
        tasks=[
            Task(id="X", depends_on=["Y"]),
            Task(id="Y", depends_on=["X"]),
        ],
    )
    batches = manager.get_parallel_opportunities(tasks)
    flat = [tid for batch in batches for tid in batch]
    assert sorted(flat) == ["X", "Y"]


@pytest.mark.asyncio
async def test_get_sdd_parallel_batches_tool_delegates():
    """The MCP tool returns the manager's real dependency-aware batches."""
    tasks = Tasks(
        feature_id="feat",
        tasks=[
            Task(id="A", file_paths=["a.py"]),
            Task(id="B", file_paths=["b.py"]),
            Task(id="C", depends_on=["A"], file_paths=["c.py"]),
        ],
    )

    deps = MagicMock(spec=AgentDeps)
    deps.workspace_path = tempfile.mkdtemp()
    ctx = MagicMock(spec=RunContext)
    ctx.deps = deps

    with patch("agent_utilities.tools.sdd_tools.SDDManager") as MockManager:
        instance = MockManager.return_value
        instance.load.return_value = tasks
        # Delegate to the real batching algorithm bound to the mock instance.
        instance.get_parallel_opportunities = lambda t: (
            SDDManager.get_parallel_opportunities(instance, t)
        )

        batches = await get_sdd_parallel_batches(ctx, "feat")

    assert _wave_of(batches, "A") == 0
    assert _wave_of(batches, "B") == 0
    assert _wave_of(batches, "C") > _wave_of(batches, "A")


@pytest.mark.asyncio
async def test_get_sdd_parallel_batches_empty():
    """No tasks -> empty batch list."""
    deps = MagicMock(spec=AgentDeps)
    deps.workspace_path = tempfile.mkdtemp()
    ctx = MagicMock(spec=RunContext)
    ctx.deps = deps

    with patch("agent_utilities.tools.sdd_tools.SDDManager") as MockManager:
        MockManager.return_value.load.return_value = Tasks(feature_id="feat", tasks=[])
        batches = await get_sdd_parallel_batches(ctx, "feat")

    assert batches == []
