"""Goal loop durability through the sole engine-native WorkItem.

These live-path tests drive ``run_goal_loop`` and prove that a terminal native
WorkItem prevents a redelivered goal from re-running its completed iterations.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time

import pytest

from agent_utilities.core import sessions as _sessions


@pytest.fixture
def loop_env(tmp_path, monkeypatch):
    # These are LIVE-PATH tests: run_goal_loop spins a real IntelligenceGraphEngine
    # whose iteration validation only fires when the engine is reachable. With no
    # isolated test engine (a bare pre-commit run), run_loop fails internally and
    # the side-effect marker is never written, so skip rather than report a
    # spurious failure. CI / canonical autostart the engine and run these for real.
    import tests.conftest as _ct

    if not getattr(_ct, "_TEST_ENGINE_AVAILABLE", False):
        pytest.skip(
            "epistemic-graph engine not reachable; goal-loop live path needs it"
        )

    db = tmp_path / "sessions.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(_sessions._SQLITE_DDL)
    conn.execute(
        "INSERT INTO sessions (id, status, created_at, updated_at, turn_count) "
        "VALUES (?, 'running', ?, ?, 0)",
        ("sess-1", time.time(), time.time()),
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(_sessions, "_get_db_path", lambda: db)
    monkeypatch.setattr(_sessions, "_rehydrated", False)
    monkeypatch.setattr(_sessions, "active_goals", {})
    monkeypatch.setattr(_sessions, "background_goal_runs", {})
    # Pin sqlite state regardless of an ambient STATE_DB_URI (a dev checkout's
    # .env may externalize session metadata to Postgres, which would bypass the
    # sqlite monkeypatching above).
    monkeypatch.delenv("STATE_DB_URI", raising=False)
    monkeypatch.setattr(
        "agent_utilities.core.state_store.postgres_state_enabled", lambda: False
    )
    # The loop sleeps 2s between failing iterations; fast-forward it so the
    # exactly-once assertions don't pay real wall-clock. (Subprocess execution
    # does not depend on asyncio.sleep.) Only collapse the loop's SHORT retry
    # sleep — a fresh ``IntelligenceGraphEngine`` starts a background subscriber
    # that idles on ``await asyncio.sleep(3600)`` in a loop, and no-oping that
    # would busy-spin the bridge thread and hang the test; let long idle sleeps
    # sleep for real (they run on a daemon thread and never block the test).
    _real_sleep = asyncio.sleep

    async def _fast_sleep(delay=0, *args, **kwargs):
        if delay and delay >= 60:
            return await _real_sleep(delay, *args, **kwargs)
        return None

    monkeypatch.setattr(asyncio, "sleep", _fast_sleep)
    return tmp_path


async def test_terminal_work_item_replay_does_not_rerun_validation(loop_env):
    marker = loop_env / "marker.txt"
    # Always-failing command (exit 1) so the loop runs all iterations; each run
    # appends one byte to the marker — the observable side effect.
    cmd = f"printf x >> {marker}; exit 1"

    await _sessions.run_goal_loop(
        session_id="sess-1",
        goal_id="g-replay",
        objective="obj",
        validation_cmd=cmd,
        max_iterations=2,
        constraints=[],
    )
    assert marker.read_text() == "xx"  # two iterations, one effect each

    # Replay the SAME goal (e.g. an at-least-once redelivery). Every iteration's
    # effect is already COMPLETED, so the marker must not grow.
    await _sessions.run_goal_loop(
        session_id="sess-1",
        goal_id="g-replay",
        objective="obj",
        validation_cmd=cmd,
        max_iterations=2,
        constraints=[],
    )
    assert marker.read_text() == "xx"


async def test_goal_loop_distinct_goals_isolated(loop_env):
    marker = loop_env / "marker.txt"
    cmd = f"printf y >> {marker}; exit 1"

    await _sessions.run_goal_loop(
        session_id="sess-1",
        goal_id="g-a",
        objective="obj",
        validation_cmd=cmd,
        max_iterations=1,
        constraints=[],
    )
    # A different goal id is a different idempotency namespace — it runs.
    await _sessions.run_goal_loop(
        session_id="sess-1",
        goal_id="g-b",
        objective="obj",
        validation_cmd=cmd,
        max_iterations=1,
        constraints=[],
    )
    assert marker.read_text() == "yy"
