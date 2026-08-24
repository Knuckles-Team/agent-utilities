"""LANE-6 (KG-2.134) — ``_run_background_task`` must not block the shared
asyncio event loop.

Root cause: the graph-os pod runs the KG daemon in-process with the webui's
uvicorn server. ``_run_background_task`` (``engine_tasks.py``) is declared
``async def`` but used to call several genuinely blocking operations
directly and un-offloaded: a synchronous engine read (``query_cypher``), two
synchronous engine writes (``add_node``, ``submit_task``), a blocking file
read (``Path.read_text``), and a blocking remote-embedder call
(``get_text_embedding``). Each is now hopped off the loop via
``asyncio.to_thread`` — chosen specifically because (unlike a bare
``ThreadPoolExecutor.submit()``, the exact defect that just bit
``loop_controller.py``) it propagates ``contextvars``, which the ambient
``GraphSession`` and the QoS ``PriorityClass`` both depend on.

These tests drive the ``diff`` task-type branch of ``_run_background_task``
(read -> embed -> ``add_node``) because it is the smallest branch that
exercises a real offloaded engine write end-to-end, with the fewest
supporting fakes.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent_utilities.core.resource_priority import (
    PriorityClass,
    current_priority,
    priority_scope,
)
from agent_utilities.knowledge_graph.core.engine_tasks import TaskManagerMixin
from agent_utilities.knowledge_graph.core.session import (
    GraphSession,
    current_session,
    use_session,
)
from agent_utilities.security.brain_context import ActorContext, ActorType


class _RecordingEngine(TaskManagerMixin):
    """Minimal ``TaskManagerMixin`` stand-in for the ``diff`` branch of
    ``_run_background_task`` — exactly what that branch touches on ``self``:
    ``add_node``, ``_update_task_status`` (success path), ``_fail_or_retry_task``
    (failure path), and the unconditional ``finally: self._checkpoint_db()``.

    Deliberately does NOT call ``TaskManagerMixin.__init__`` (it stands up a
    real durable queue backend, worker locks, etc. — none of which the
    ``diff`` branch needs), matching the existing fake-engine pattern in
    ``test_engine_tasks_skill_workflows_atomic_pairing.py``.
    """

    def __init__(self, add_node_impl) -> None:
        self.backend = None
        self.updates: list[tuple[str, str, dict]] = []
        self.failures: list[tuple[str, str, dict | None]] = []
        self._add_node_impl = add_node_impl
        self.add_node_calls = 0

    def add_node(self, node_id, node_type, properties=None):  # noqa: ANN001
        self.add_node_calls += 1
        return self._add_node_impl(node_id, node_type, properties)

    def _update_task_status(self, job_id, status, payload):  # noqa: ANN001
        self.updates.append((job_id, status, payload))

    def _fail_or_retry_task(self, job_id, error, details=None):  # noqa: ANN001
        self.failures.append((job_id, error, details))

    def _checkpoint_db(self) -> None:
        pass


def _write_diff_file(tmp_path: Path) -> Path:
    p = tmp_path / "sample.diff"
    p.write_text("--- a\n+++ b\n+hello\n", encoding="utf-8")
    return p


def _patched_embedder():
    """The ``diff`` branch's ``create_embedding_model()`` call, stubbed to a
    fast, deterministic embedder — the embed call itself is not what these
    tests are exercising."""
    return patch(
        "agent_utilities.core.embedding_utilities.create_embedding_model",
        return_value=SimpleNamespace(get_text_embedding=lambda text: [0.1, 0.2, 0.3]),
    )


@pytest.mark.asyncio
async def test_run_background_task_offloads_blocking_engine_write(tmp_path):
    """Load-bearing test: a synchronous engine write inside
    ``_run_background_task`` must not block the event loop. A concurrent,
    trivial coroutine (a simple heartbeat) has to keep making progress while
    the write is in flight — with the offload it should tick roughly every
    ``tick_interval``; without it (a direct, un-offloaded sync call) the
    single-threaded loop cannot run the heartbeat AT ALL until the blocking
    call returns.
    """
    block_seconds = 0.4
    tick_interval = 0.02

    def _blocking_add_node(node_id, node_type, properties):  # noqa: ANN001
        time.sleep(block_seconds)

    engine = _RecordingEngine(_blocking_add_node)
    diff_file = _write_diff_file(tmp_path)

    ticks = 0
    stop = False

    async def _heartbeat():
        nonlocal ticks
        while not stop:
            ticks += 1
            await asyncio.sleep(tick_interval)

    with _patched_embedder():
        hb_task = asyncio.create_task(_heartbeat())
        # Give the heartbeat one scheduling turn before the blocking call
        # starts, so a low tick count can't be blamed on startup ordering.
        await asyncio.sleep(0)
        start = time.monotonic()
        await engine._run_background_task(
            job_id="job:diff-1",
            target=diff_file,
            is_codebase=False,
            task_type="diff",
        )
        elapsed = time.monotonic() - start
        stop = True
        await hb_task

    assert engine.add_node_calls == 1
    assert elapsed >= block_seconds

    # A genuinely blocked loop produces ~0 additional heartbeat ticks during
    # the whole blocking window (it already got exactly one tick from the
    # `await asyncio.sleep(0)` above, then nothing until the sync call
    # returns). A freed loop should tick close to block_seconds/tick_interval
    # times; require at least a quarter of that as a flake-tolerant bound.
    expected_ticks = block_seconds / tick_interval
    assert ticks >= expected_ticks * 0.25, (
        f"heartbeat only ticked {ticks} times during a {block_seconds}s "
        "blocking engine write -- the event loop looks blocked"
    )

    assert engine.updates
    job_id, status, _payload = engine.updates[-1]
    assert (job_id, status) == ("job:diff-1", "completed")


@pytest.mark.asyncio
async def test_run_background_task_propagates_contextvars_to_offloaded_call(tmp_path):
    """Regression guard for the exact defect that just hit
    ``loop_controller.py``: a bare ``ThreadPoolExecutor.submit()`` silently
    drops ``contextvars``, losing the ambient ``GraphSession`` and
    ``PriorityClass`` inside the offloaded call. ``asyncio.to_thread``
    (used here, not a bare executor) must carry both through.
    """
    session = GraphSession(
        actor=ActorContext(
            actor_id="agent-lane6",
            actor_type=ActorType.AUTOMATED_SERVICE,
            tenant_id="tenant-lane6",
            authenticated=True,
        ),
        tenant="tenant-lane6",
        scopes=frozenset({"kg:write"}),
        graph="tenant-lane6",
    )

    observed: dict[str, object] = {}

    def _observing_add_node(node_id, node_type, properties):  # noqa: ANN001
        # Runs inside the to_thread executor thread -- proves the ambient
        # session/priority contextvars crossed the thread hop.
        observed["session"] = current_session()
        observed["priority"] = current_priority()

    engine = _RecordingEngine(_observing_add_node)
    diff_file = _write_diff_file(tmp_path)

    with (
        _patched_embedder(),
        use_session(session),
        priority_scope(PriorityClass.BACKGROUND_INGESTION),
    ):
        await engine._run_background_task(
            job_id="job:diff-2",
            target=diff_file,
            is_codebase=False,
            task_type="diff",
        )

    assert engine.add_node_calls == 1
    assert observed["session"] is session
    assert observed["priority"] is PriorityClass.BACKGROUND_INGESTION


@pytest.mark.asyncio
async def test_run_background_task_offloaded_exception_propagates_with_cause(tmp_path):
    """``asyncio.to_thread`` re-raises the offloaded call's exception; the
    existing failure handling (capture + full traceback + route to
    ``_fail_or_retry_task``) must see the SAME exception, not a swallowed or
    generic one -- this repo gates against the swallowed-error anti-pattern.
    """

    class _BoomError(RuntimeError):
        pass

    def _raising_add_node(node_id, node_type, properties):  # noqa: ANN001
        raise _BoomError("simulated engine write failure")

    engine = _RecordingEngine(_raising_add_node)
    diff_file = _write_diff_file(tmp_path)

    with _patched_embedder():
        # _run_background_task's own except-block catches everything and
        # routes it to _fail_or_retry_task; it does not re-raise to the
        # caller (an app-level failure, not a crash). Assert that handler
        # actually observed the real exception, with type/message and a full
        # traceback intact, not an opaque/blank failure.
        await engine._run_background_task(
            job_id="job:diff-3",
            target=diff_file,
            is_codebase=False,
            task_type="diff",
        )

    assert engine.add_node_calls == 1
    assert engine.updates == []  # no success status was ever recorded
    assert len(engine.failures) == 1
    job_id, error, details = engine.failures[0]
    assert job_id == "job:diff-3"
    assert "simulated engine write failure" in error
    assert details is not None
    assert "_BoomError" in details["traceback"]
    assert "simulated engine write failure" in details["traceback"]
