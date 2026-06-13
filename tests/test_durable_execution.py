"""
Tests for Durable Execution (CONCEPT:ECO-4.0 / ORCH-1.36).

Covers checkpoint/resume, crash-resume across manager instances, exactly-once
effects via idempotency keys, and at-least-once retry semantics.
"""

import pytest

from agent_utilities.orchestration.durable_execution import DurableExecutionManager
from agent_utilities.orchestration.resilience import ResiliencePolicy

# Fast, deterministic retry policy for tests (no real backoff sleeps).
_FAST = ResiliencePolicy(max_attempts=3, backoff_base_s=0.0, jitter=False)


@pytest.fixture
def db(tmp_path):
    return tmp_path / "durable.db"


def test_durable_execution_flow(db):
    manager = DurableExecutionManager(session_id="s1", db_path=db)

    node_id = manager.save_checkpoint("trade_step_1", {"asset": "BTC", "qty": 1.5})
    assert node_id == "trade_step_1"

    resumed = manager.resume_session()
    assert resumed is not None
    assert resumed["node_id"] == "trade_step_1"
    assert "BTC" in resumed["state"]

    manager.mark_completed("trade_step_1")
    assert manager.resume_session() is None


def test_crash_resume_across_instances(db):
    # Process A writes a pending checkpoint then "crashes".
    DurableExecutionManager("s2", db_path=db).save_checkpoint("leg_a", {"side": "buy"})
    # Process B (fresh instance, same durable file) recovers it.
    recovered = DurableExecutionManager("s2", db_path=db).resume_session()
    assert recovered is not None
    assert recovered["node_id"] == "leg_a"


def test_idempotency_exactly_once(db):
    manager = DurableExecutionManager("s3", db_path=db)
    calls = {"n": 0}

    def critical():
        calls["n"] += 1
        return {"order_id": "abc"}

    first = manager.run_durable_action("place_order", critical, idempotency_key="ORD-1")
    # A retry / replay with the same key must NOT re-run the side effect.
    second = manager.run_durable_action("place_order", critical, idempotency_key="ORD-1")

    assert calls["n"] == 1
    assert first == {"order_id": "abc"}
    assert second == {"order_id": "abc"}


def test_at_least_once_retries_then_succeeds(db):
    manager = DurableExecutionManager("s4", db_path=db)
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise ConnectionError("transient")
        return "ok"

    result = manager.run_durable_action(
        "flaky_step", flaky, idempotency_key="F-1", policy=_FAST
    )
    assert result == "ok"
    assert attempts["n"] == 2  # failed once, retried, succeeded


def test_idempotency_survives_restart(db):
    # Complete an action, then a brand-new manager (post-restart) must skip it.
    DurableExecutionManager("s5", db_path=db).run_durable_action(
        "step", lambda: "done", idempotency_key="K-1"
    )
    ran_again = {"n": 0}

    def again():
        ran_again["n"] += 1
        return "second"

    out = DurableExecutionManager("s5", db_path=db).run_durable_action(
        "step", again, idempotency_key="K-1"
    )
    assert ran_again["n"] == 0
    assert out == "done"


async def test_arun_durable_action_exactly_once(db):
    # Async twin used by the live goal loop / dispatch worker: a redelivery with
    # the same key must not re-run the awaitable effect.
    manager = DurableExecutionManager("s6", db_path=db)
    calls = {"n": 0}

    async def effect():
        calls["n"] += 1
        return {"id": "xyz"}

    first = await manager.arun_durable_action("turn", effect, idempotency_key="T-1")
    second = await manager.arun_durable_action("turn", effect, idempotency_key="T-1")

    assert calls["n"] == 1
    assert first == {"id": "xyz"} == second


async def test_arun_durable_action_survives_restart(db):
    await DurableExecutionManager("s7", db_path=db).arun_durable_action(
        "turn", lambda: "done", idempotency_key="T-2"
    )
    ran = {"n": 0}

    async def again():
        ran["n"] += 1
        return "second"

    out = await DurableExecutionManager("s7", db_path=db).arun_durable_action(
        "turn", again, idempotency_key="T-2"
    )
    assert ran["n"] == 0
    assert out == "done"
