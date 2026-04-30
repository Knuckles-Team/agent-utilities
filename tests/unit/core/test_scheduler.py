"""Coverage push for agent_utilities.scheduler.

Targets the mostly-pure-function paths: task CRUD on the Knowledge Graph
(mocked), logging, reload, and a single-iteration exercise of the
``background_processor`` happy + error branches.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_utilities.core import scheduler
from agent_utilities.models.scheduling import (
    CronLogModel,
    CronRegistryModel,
    CronTaskModel,
    PeriodicTask,
)


@pytest.fixture(autouse=True)
def reset_scheduler_state() -> Any:
    """Clear the module-level tasks list before each test."""
    scheduler.tasks.clear()
    yield
    scheduler.tasks.clear()


def _fake_engine_with_rows(rows: list[dict[str, Any]] | None = None) -> MagicMock:
    """Return a mock engine whose backend.execute returns the given rows."""
    engine = MagicMock()
    engine.backend = MagicMock()
    engine.backend.execute.return_value = rows or []
    return engine


# ---------------------------------------------------------------------------
# get_cron_tasks
# ---------------------------------------------------------------------------


def test_get_cron_tasks_no_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """No active engine -> empty registry."""
    fake_kg = MagicMock()
    fake_kg.IntelligenceGraphEngine.get_active.return_value = None
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent_utilities.knowledge_graph.engine",
        fake_kg,
    )
    result = scheduler.get_cron_tasks()
    assert isinstance(result, CronRegistryModel)
    assert result.tasks == []


def test_get_cron_tasks_engine_no_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Engine exists but backend is None -> empty registry."""
    engine = MagicMock(backend=None)
    fake_kg = MagicMock()
    fake_kg.IntelligenceGraphEngine.get_active.return_value = engine
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent_utilities.knowledge_graph.engine",
        fake_kg,
    )
    result = scheduler.get_cron_tasks()
    assert isinstance(result, CronRegistryModel)
    assert result.tasks == []


def test_get_cron_tasks_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy path: rows are converted to CronTaskModel instances."""
    rows = [
        {
            "j.id": "t1",
            "j.name": "Task One",
            "j.schedule": "10",
            "j.command": "do stuff",
            "j.last_run": "12:00",
            "j.next_approx": "12:10",
        },
        {
            "j.id": "t2",
            "j.name": "Task Two",
            "j.schedule": "5",
            "j.command": "do other stuff",
            "j.last_run": "12:05",
            "j.next_approx": "12:10",
        },
    ]
    engine = _fake_engine_with_rows(rows)
    fake_kg = MagicMock()
    fake_kg.IntelligenceGraphEngine.get_active.return_value = engine
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent_utilities.knowledge_graph.engine",
        fake_kg,
    )
    result = scheduler.get_cron_tasks()
    assert len(result.tasks) == 2
    assert result.tasks[0].id == "t1"
    assert result.tasks[0].interval_minutes == 10
    assert result.tasks[1].interval_minutes == 5


def test_get_cron_tasks_execute_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """execute raising returns an empty registry."""
    engine = MagicMock()
    engine.backend = MagicMock()
    engine.backend.execute.side_effect = RuntimeError("db down")
    fake_kg = MagicMock()
    fake_kg.IntelligenceGraphEngine.get_active.return_value = engine
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent_utilities.knowledge_graph.engine",
        fake_kg,
    )
    result = scheduler.get_cron_tasks()
    assert isinstance(result, CronRegistryModel)
    assert result.tasks == []


# ---------------------------------------------------------------------------
# get_cron_logs
# ---------------------------------------------------------------------------


def test_get_cron_logs_no_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """No active engine -> empty log model."""
    fake_kg = MagicMock()
    fake_kg.IntelligenceGraphEngine.get_active.return_value = None
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent_utilities.knowledge_graph.engine",
        fake_kg,
    )
    result = scheduler.get_cron_logs()
    assert isinstance(result, CronLogModel)
    assert result.entries == []


def test_get_cron_logs_engine_no_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Engine with None backend -> empty log model."""
    engine = MagicMock(backend=None)
    fake_kg = MagicMock()
    fake_kg.IntelligenceGraphEngine.get_active.return_value = engine
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent_utilities.knowledge_graph.engine",
        fake_kg,
    )
    result = scheduler.get_cron_logs()
    assert result.entries == []


def test_get_cron_logs_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rows converted to CronLogEntryModel."""
    rows = [
        {
            "l.timestamp": "2024-01-01 12:00:00",
            "l.task_id": "t1",
            "l.task_name": "Task",
            "l.status": "success",
            "l.output": "ok",
            "l.chat_id": "chat-1",
        },
        {
            "l.timestamp": "2024-01-01 13:00:00",
            "l.task_id": "t2",
            "l.task_name": "Task2",
            "l.status": "error",
            "l.output": "oops",
            "l.chat_id": None,
        },
    ]
    engine = _fake_engine_with_rows(rows)  # type: ignore[arg-type]
    fake_kg = MagicMock()
    fake_kg.IntelligenceGraphEngine.get_active.return_value = engine
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent_utilities.knowledge_graph.engine",
        fake_kg,
    )
    result = scheduler.get_cron_logs()
    assert len(result.entries) == 2
    assert result.entries[0].task_id == "t1"
    assert result.entries[1].status == "error"


def test_get_cron_logs_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exception in execute is caught."""
    engine = MagicMock()
    engine.backend = MagicMock()
    engine.backend.execute.side_effect = ValueError("bad query")
    fake_kg = MagicMock()
    fake_kg.IntelligenceGraphEngine.get_active.return_value = engine
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent_utilities.knowledge_graph.engine",
        fake_kg,
    )
    result = scheduler.get_cron_logs()
    assert result.entries == []


# ---------------------------------------------------------------------------
# update_cron_task_in_cron_md
# ---------------------------------------------------------------------------


def test_update_cron_task_no_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """No active engine returns silently."""
    fake_kg = MagicMock()
    fake_kg.IntelligenceGraphEngine.get_active.return_value = None
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent_utilities.knowledge_graph.engine",
        fake_kg,
    )
    # Should not raise
    scheduler.update_cron_task_in_cron_md("t1", "Task", 5, "do stuff")
    assert True, 'No-engine cron update should not raise'


def test_update_cron_task_no_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Engine with no backend returns silently."""
    engine = MagicMock(backend=None)
    fake_kg = MagicMock()
    fake_kg.IntelligenceGraphEngine.get_active.return_value = engine
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent_utilities.knowledge_graph.engine",
        fake_kg,
    )
    scheduler.update_cron_task_in_cron_md("t1", "Task", 5, "do stuff")
    assert True, 'No-backend cron update should not raise'


def test_update_cron_task_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy path executes MERGE query with correct params."""
    engine = _fake_engine_with_rows()
    fake_kg = MagicMock()
    fake_kg.IntelligenceGraphEngine.get_active.return_value = engine
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent_utilities.knowledge_graph.engine",
        fake_kg,
    )
    scheduler.update_cron_task_in_cron_md("t1", "Task", 5, "do stuff")
    engine.backend.execute.assert_called_once()
    call_args = engine.backend.execute.call_args
    params = call_args[0][1]
    assert params["id"] == "t1"
    assert params["name"] == "Task"
    assert params["schedule"] == "5"
    assert params["command"] == "do stuff"
    assert "last_run" in params


# ---------------------------------------------------------------------------
# schedule_task
# ---------------------------------------------------------------------------


def test_schedule_task_invalid_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    """Interval < 1 returns validation error string."""
    # Mock knowledge graph so no persistence happens
    fake_kg = MagicMock()
    fake_kg.IntelligenceGraphEngine.get_active.return_value = None
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent_utilities.knowledge_graph.engine",
        fake_kg,
    )
    msg = scheduler.schedule_task("t1", "Task", 0, "do stuff")
    assert "≥ 1 minute" in msg
    assert scheduler.tasks == []


def test_schedule_task_new_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scheduling a new task appends to the in-memory list."""
    fake_kg = MagicMock()
    fake_kg.IntelligenceGraphEngine.get_active.return_value = None
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent_utilities.knowledge_graph.engine",
        fake_kg,
    )
    msg = scheduler.schedule_task("t1", "Task", 10, "do stuff")
    assert "Scheduled 'Task'" in msg
    assert "ID: t1" in msg
    assert len(scheduler.tasks) == 1
    assert scheduler.tasks[0].id == "t1"
    assert scheduler.tasks[0].interval_minutes == 10


def test_schedule_task_existing_task_updates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scheduling an existing ID updates in place, does not duplicate."""
    fake_kg = MagicMock()
    fake_kg.IntelligenceGraphEngine.get_active.return_value = None
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent_utilities.knowledge_graph.engine",
        fake_kg,
    )
    scheduler.tasks.append(
        PeriodicTask(
            id="t1",
            name="Old",
            interval_minutes=5,
            prompt="old",
            last_run=datetime.now() - timedelta(minutes=30),
        )
    )
    scheduler.schedule_task("t1", "New", 15, "new prompt")
    assert len(scheduler.tasks) == 1
    assert scheduler.tasks[0].name == "New"
    assert scheduler.tasks[0].interval_minutes == 15
    assert scheduler.tasks[0].prompt == "new prompt"


# ---------------------------------------------------------------------------
# delete_scheduled_task
# ---------------------------------------------------------------------------


def test_delete_scheduled_task_no_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """No engine, still removes from in-memory list."""
    fake_kg = MagicMock()
    fake_kg.IntelligenceGraphEngine.get_active.return_value = None
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent_utilities.knowledge_graph.engine",
        fake_kg,
    )
    scheduler.tasks.append(
        PeriodicTask(
            id="t1",
            name="Task",
            interval_minutes=5,
            prompt="do",
        )
    )
    msg = scheduler.delete_scheduled_task("t1")
    assert "Deleted" in msg
    assert scheduler.tasks == []


def test_delete_scheduled_task_with_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Engine present calls DETACH DELETE query."""
    engine = _fake_engine_with_rows()
    fake_kg = MagicMock()
    fake_kg.IntelligenceGraphEngine.get_active.return_value = engine
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent_utilities.knowledge_graph.engine",
        fake_kg,
    )
    scheduler.tasks.append(
        PeriodicTask(id="t1", name="T", interval_minutes=5, prompt="x")
    )
    msg = scheduler.delete_scheduled_task("t1")
    engine.backend.execute.assert_called_once()
    call_args = engine.backend.execute.call_args[0]
    assert "DETACH DELETE" in call_args[0]
    assert call_args[1] == {"id": "t1"}
    assert scheduler.tasks == []
    assert "Deleted" in msg


def test_delete_scheduled_task_nonexistent_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deleting a nonexistent task does not crash and still reports 'Deleted'."""
    fake_kg = MagicMock()
    fake_kg.IntelligenceGraphEngine.get_active.return_value = None
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent_utilities.knowledge_graph.engine",
        fake_kg,
    )
    msg = scheduler.delete_scheduled_task("does_not_exist")
    assert "Deleted" in msg


# ---------------------------------------------------------------------------
# list_scheduled_tasks
# ---------------------------------------------------------------------------


def test_list_scheduled_tasks_delegates_to_get_cron_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """list_scheduled_tasks returns whatever get_cron_tasks returns."""
    sentinel = CronRegistryModel(
        tasks=[
            CronTaskModel(id="x", name="X", interval_minutes=1, prompt="p"),
        ]
    )
    monkeypatch.setattr(scheduler, "get_cron_tasks", lambda: sentinel)
    result = scheduler.list_scheduled_tasks()
    assert result is sentinel


# ---------------------------------------------------------------------------
# append_cron_log
# ---------------------------------------------------------------------------


def test_append_cron_log_no_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """No engine returns silently."""
    fake_kg = MagicMock()
    fake_kg.IntelligenceGraphEngine.get_active.return_value = None
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent_utilities.knowledge_graph.engine",
        fake_kg,
    )
    scheduler.append_cron_log("t1", "Task", "output")
    assert True, 'No-engine cron log append should not raise'


def test_append_cron_log_no_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Engine with no backend returns silently."""
    engine = MagicMock(backend=None)
    fake_kg = MagicMock()
    fake_kg.IntelligenceGraphEngine.get_active.return_value = engine
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent_utilities.knowledge_graph.engine",
        fake_kg,
    )
    scheduler.append_cron_log("t1", "Task", "output")
    assert True, 'No-backend cron log append should not raise'


def test_append_cron_log_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy path executes CREATE log query."""
    engine = _fake_engine_with_rows()
    fake_kg = MagicMock()
    fake_kg.IntelligenceGraphEngine.get_active.return_value = engine
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent_utilities.knowledge_graph.engine",
        fake_kg,
    )
    scheduler.append_cron_log("t1", "Task", "output", status="success", chat_id="c1")
    engine.backend.execute.assert_called_once()
    call_args = engine.backend.execute.call_args[0]
    assert "CREATE (l:Log" in call_args[0]
    params = call_args[1]
    assert params["task_id"] == "t1"
    assert params["task_name"] == "Task"
    assert params["status"] == "success"
    assert params["chat_id"] == "c1"
    assert params["id"].startswith("log:t1:")


def test_append_cron_log_error_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit error status propagates to params."""
    engine = _fake_engine_with_rows()
    fake_kg = MagicMock()
    fake_kg.IntelligenceGraphEngine.get_active.return_value = engine
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent_utilities.knowledge_graph.engine",
        fake_kg,
    )
    scheduler.append_cron_log("t1", "Task", "bad", status="error")
    params = engine.backend.execute.call_args[0][1]
    assert params["status"] == "error"


# ---------------------------------------------------------------------------
# cleanup_cron_log
# ---------------------------------------------------------------------------


def test_cleanup_cron_log_no_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """No engine returns silently."""
    fake_kg = MagicMock()
    fake_kg.IntelligenceGraphEngine.get_active.return_value = None
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent_utilities.knowledge_graph.engine",
        fake_kg,
    )
    scheduler.cleanup_cron_log()
    assert True, 'No-engine cron log cleanup should not raise'


def test_cleanup_cron_log_no_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Engine with no backend returns silently."""
    engine = MagicMock(backend=None)
    fake_kg = MagicMock()
    fake_kg.IntelligenceGraphEngine.get_active.return_value = engine
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent_utilities.knowledge_graph.engine",
        fake_kg,
    )
    scheduler.cleanup_cron_log()
    assert True, 'No-backend cron log cleanup should not raise'


def test_cleanup_cron_log_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy path: deletes old logs via SKIP query."""
    engine = _fake_engine_with_rows()
    fake_kg = MagicMock()
    fake_kg.IntelligenceGraphEngine.get_active.return_value = engine
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent_utilities.knowledge_graph.engine",
        fake_kg,
    )
    scheduler.cleanup_cron_log(max_entries=50)
    engine.backend.execute.assert_called_once()
    call_args = engine.backend.execute.call_args[0]
    assert "DETACH DELETE" in call_args[0]
    assert call_args[1] == {"skip": 50}


def test_cleanup_cron_log_default_max_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default max_entries uses config value."""
    from agent_utilities.core.config import DEFAULT_MAX_CRON_LOG_ENTRIES

    engine = _fake_engine_with_rows()
    fake_kg = MagicMock()
    fake_kg.IntelligenceGraphEngine.get_active.return_value = engine
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent_utilities.knowledge_graph.engine",
        fake_kg,
    )
    scheduler.cleanup_cron_log()
    params = engine.backend.execute.call_args[0][1]
    assert params["skip"] == DEFAULT_MAX_CRON_LOG_ENTRIES


# ---------------------------------------------------------------------------
# reload_cron_tasks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reload_cron_tasks_fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fresh reload populates in-memory tasks from registry."""
    registry = CronRegistryModel(
        tasks=[
            CronTaskModel(
                id="t1",
                name="Task",
                interval_minutes=5,
                prompt="do",
            ),
        ]
    )
    monkeypatch.setattr(scheduler, "get_cron_tasks", lambda: registry)
    await scheduler.reload_cron_tasks()
    assert len(scheduler.tasks) == 1
    assert scheduler.tasks[0].id == "t1"


@pytest.mark.asyncio
async def test_reload_cron_tasks_preserves_existing_last_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reload preserves last_run and active of existing tasks with unchanged interval."""
    old_last_run = datetime(2020, 1, 1, 12, 0, 0)
    scheduler.tasks.append(
        PeriodicTask(
            id="t1",
            name="Task",
            interval_minutes=5,
            prompt="do",
            last_run=old_last_run,
            active=False,
        )
    )
    registry = CronRegistryModel(
        tasks=[
            CronTaskModel(
                id="t1",
                name="Task",
                interval_minutes=5,
                prompt="do",
            ),
        ]
    )
    monkeypatch.setattr(scheduler, "get_cron_tasks", lambda: registry)
    await scheduler.reload_cron_tasks()
    assert len(scheduler.tasks) == 1
    assert scheduler.tasks[0].last_run == old_last_run
    assert scheduler.tasks[0].active is False


@pytest.mark.asyncio
async def test_reload_cron_tasks_resets_on_interval_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reload with different interval does NOT preserve last_run/active."""
    old_last_run = datetime(2020, 1, 1, 12, 0, 0)
    scheduler.tasks.append(
        PeriodicTask(
            id="t1",
            name="Task",
            interval_minutes=5,
            prompt="do",
            last_run=old_last_run,
            active=False,
        )
    )
    registry = CronRegistryModel(
        tasks=[
            CronTaskModel(
                id="t1",
                name="Task",
                interval_minutes=10,  # different
                prompt="do",
            ),
        ]
    )
    monkeypatch.setattr(scheduler, "get_cron_tasks", lambda: registry)
    await scheduler.reload_cron_tasks()
    # interval differs, so we expect a fresh PeriodicTask (active=True, last_run set now)
    assert len(scheduler.tasks) == 1
    assert scheduler.tasks[0].interval_minutes == 10
    assert scheduler.tasks[0].active is True
    assert scheduler.tasks[0].last_run != old_last_run


# ---------------------------------------------------------------------------
# background_processor (single-iteration exercise)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_background_processor_runs_due_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """background_processor picks up a due task, calls agent.run, and logs."""
    # A short-circuiting asyncio.sleep
    async def instant_sleep(n: float) -> None:  # noqa: ARG001
        # Only sleep once, then raise to break the loop on the second call
        if not hasattr(instant_sleep, "_calls"):
            instant_sleep._calls = 0  # type: ignore[attr-defined]
        instant_sleep._calls += 1  # type: ignore[attr-defined]
        if instant_sleep._calls >= 2:  # type: ignore[attr-defined]
            raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "sleep", instant_sleep)
    monkeypatch.setattr(scheduler, "reload_cron_tasks", AsyncMock(return_value=None))
    monkeypatch.setattr(scheduler, "resolve_prompt", lambda s: f"[resolved] {s}")
    monkeypatch.setattr(scheduler, "append_cron_log", MagicMock())
    monkeypatch.setattr(scheduler, "save_chat_to_disk", MagicMock())

    scheduler.tasks.append(
        PeriodicTask(
            id="t1",
            name="DueTask",
            interval_minutes=1,
            prompt="say hi",
            last_run=datetime.now() - timedelta(minutes=10),
            active=True,
        )
    )

    class FakeResult:
        output = "Agent response text"

    fake_agent = MagicMock()
    fake_agent.run = AsyncMock(return_value=FakeResult())

    with pytest.raises(asyncio.CancelledError):
        await scheduler.background_processor(fake_agent)

    fake_agent.run.assert_called_once_with("[resolved] say hi")
    scheduler.append_cron_log.assert_called_once()  # type: ignore[attr-defined]
    call_kwargs = scheduler.append_cron_log.call_args.kwargs  # type: ignore[attr-defined]
    assert call_kwargs["task_id"] == "t1"
    assert call_kwargs["task_name"] == "DueTask"


@pytest.mark.asyncio
async def test_background_processor_handles_internal_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """background_processor runs __internal:cleanup_cron_log without calling agent."""

    async def instant_sleep(n: float) -> None:  # noqa: ARG001
        if not hasattr(instant_sleep, "_calls"):
            instant_sleep._calls = 0  # type: ignore[attr-defined]
        instant_sleep._calls += 1  # type: ignore[attr-defined]
        if instant_sleep._calls >= 2:  # type: ignore[attr-defined]
            raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "sleep", instant_sleep)
    monkeypatch.setattr(scheduler, "reload_cron_tasks", AsyncMock(return_value=None))
    cleanup_mock = MagicMock()
    monkeypatch.setattr(scheduler, "cleanup_cron_log", cleanup_mock)

    scheduler.tasks.append(
        PeriodicTask(
            id="internal-cleanup",
            name="Log cleanup",
            interval_minutes=1,
            prompt="__internal:cleanup_cron_log",
            last_run=datetime.now() - timedelta(minutes=10),
            active=True,
        )
    )

    fake_agent = MagicMock()
    fake_agent.run = AsyncMock()
    with pytest.raises(asyncio.CancelledError):
        await scheduler.background_processor(fake_agent)

    # agent.run should NOT have been called for __internal: tasks
    fake_agent.run.assert_not_called()
    cleanup_mock.assert_called_once()


@pytest.mark.asyncio
async def test_background_processor_skips_inactive_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inactive tasks are not executed even if they are overdue."""

    async def instant_sleep(n: float) -> None:  # noqa: ARG001
        if not hasattr(instant_sleep, "_calls"):
            instant_sleep._calls = 0  # type: ignore[attr-defined]
        instant_sleep._calls += 1  # type: ignore[attr-defined]
        if instant_sleep._calls >= 2:  # type: ignore[attr-defined]
            raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "sleep", instant_sleep)
    monkeypatch.setattr(scheduler, "reload_cron_tasks", AsyncMock(return_value=None))

    scheduler.tasks.append(
        PeriodicTask(
            id="inactive",
            name="Inactive",
            interval_minutes=1,
            prompt="noop",
            last_run=datetime.now() - timedelta(minutes=10),
            active=False,  # inactive
        )
    )

    fake_agent = MagicMock()
    fake_agent.run = AsyncMock()
    with pytest.raises(asyncio.CancelledError):
        await scheduler.background_processor(fake_agent)
    fake_agent.run.assert_not_called()


@pytest.mark.asyncio
async def test_background_processor_handles_agent_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exception from agent.run is caught and logged as error."""

    async def instant_sleep(n: float) -> None:  # noqa: ARG001
        if not hasattr(instant_sleep, "_calls"):
            instant_sleep._calls = 0  # type: ignore[attr-defined]
        instant_sleep._calls += 1  # type: ignore[attr-defined]
        if instant_sleep._calls >= 2:  # type: ignore[attr-defined]
            raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "sleep", instant_sleep)
    monkeypatch.setattr(scheduler, "reload_cron_tasks", AsyncMock(return_value=None))
    monkeypatch.setattr(scheduler, "resolve_prompt", lambda s: s)
    append_mock = MagicMock()
    monkeypatch.setattr(scheduler, "append_cron_log", append_mock)

    scheduler.tasks.append(
        PeriodicTask(
            id="t1",
            name="BuggyTask",
            interval_minutes=1,
            prompt="say hi",
            last_run=datetime.now() - timedelta(minutes=10),
            active=True,
        )
    )

    fake_agent = MagicMock()
    fake_agent.run = AsyncMock(side_effect=RuntimeError("agent exploded"))

    with pytest.raises(asyncio.CancelledError):
        await scheduler.background_processor(fake_agent)

    append_mock.assert_called_once()
    kwargs = append_mock.call_args.kwargs
    assert "❌ ERROR" in kwargs["output"]
    assert kwargs["task_id"] == "t1"


@pytest.mark.asyncio
async def test_background_processor_reload_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exception from reload_cron_tasks is caught, loop continues."""

    async def instant_sleep(n: float) -> None:  # noqa: ARG001
        if not hasattr(instant_sleep, "_calls"):
            instant_sleep._calls = 0  # type: ignore[attr-defined]
        instant_sleep._calls += 1  # type: ignore[attr-defined]
        if instant_sleep._calls >= 2:  # type: ignore[attr-defined]
            raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "sleep", instant_sleep)

    async def bad_reload() -> None:
        raise RuntimeError("reload failed")

    monkeypatch.setattr(scheduler, "reload_cron_tasks", bad_reload)

    fake_agent = MagicMock()
    fake_agent.run = AsyncMock()
    with pytest.raises(asyncio.CancelledError):
        await scheduler.background_processor(fake_agent)


@pytest.mark.asyncio
async def test_background_processor_save_chat_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exception from save_chat_to_disk is caught, chat_id becomes None."""

    async def instant_sleep(n: float) -> None:  # noqa: ARG001
        if not hasattr(instant_sleep, "_calls"):
            instant_sleep._calls = 0  # type: ignore[attr-defined]
        instant_sleep._calls += 1  # type: ignore[attr-defined]
        if instant_sleep._calls >= 2:  # type: ignore[attr-defined]
            raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "sleep", instant_sleep)
    monkeypatch.setattr(scheduler, "reload_cron_tasks", AsyncMock(return_value=None))
    monkeypatch.setattr(scheduler, "resolve_prompt", lambda s: s)
    append_mock = MagicMock()
    monkeypatch.setattr(scheduler, "append_cron_log", append_mock)

    def boom(*_a: Any, **_kw: Any) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(scheduler, "save_chat_to_disk", boom)

    scheduler.tasks.append(
        PeriodicTask(
            id="t1",
            name="Task",
            interval_minutes=1,
            prompt="say hi",
            last_run=datetime.now() - timedelta(minutes=10),
            active=True,
        )
    )

    class FakeResult:
        output = "agent reply"

    fake_agent = MagicMock()
    fake_agent.run = AsyncMock(return_value=FakeResult())

    with pytest.raises(asyncio.CancelledError):
        await scheduler.background_processor(fake_agent)

    kwargs = append_mock.call_args.kwargs
    assert kwargs["chat_id"] is None
