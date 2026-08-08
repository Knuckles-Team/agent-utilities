"""D-CDX-22: a grounding timeout must not leave the abandoned compile worker
able to block process/loop teardown, or silently overlap the NEXT measurement.

Before this fix, ``_compiled_evidence_and_bundle_bounded`` bounded a stuck
compile with ``asyncio.wait_for(asyncio.to_thread(...))``. Cancelling that
``wait_for`` only abandons the AWAITING coroutine -- ``asyncio.to_thread``
submits to the process-wide DEFAULT ``ThreadPoolExecutor``, whose worker
threads are (deliberately, so CPython's global ``_python_exit`` atexit hook
and ``asyncio.run()``'s own ``loop.shutdown_default_executor()`` can join them
cleanly on normal shutdown) NON-daemon. So an abandoned post-timeout compile
kept a non-daemon thread alive indefinitely, and process/loop teardown then
blocked joining it -- the profiled timeout run attributed ~700s to exactly
that teardown path (``_do_shutdown``, 17 thread joins, six thread-pool
shutdowns), not to any stage's own work. The two SUCCESSFUL functional
profiles also showed ~80s of the same teardown tail, proving this was not
timeout-exclusive (tracked separately as D-CDX-98).

The fix (``_run_isolated`` / ``_compile_isolated_bounded`` in
``contextual_model.py``) runs the compile on a dedicated DAEMON thread
instead, and tracks it in ``_inflight_compiles`` so a caller can bound-wait
and OBSERVE a straggler (``drain_inflight_compiles``/``inflight_compile_count``)
rather than silently racing the next measurement against it -- which is
exactly how one incomplete, abandoned ~90s timeout sample was once mistaken
for a real 90s->147s regression against a completed sample (never compare a
timed-out sample against a completed one).

Every test below FAILS against the pre-fix ``asyncio.to_thread`` version:
that version starts no thread named ``ctx-compile-isolated``, its worker
threads are never daemonic, and it has no ``_inflight_compiles`` registry at
all (``inflight_compile_count``/``drain_inflight_compiles`` did not exist).
"""

from __future__ import annotations

import asyncio
import threading
import time
from contextvars import ContextVar

import pytest

from agent_utilities.core import contextual_model
from agent_utilities.core.contextual_model import (
    GroundingUnavailableError,
    _compiled_evidence_and_bundle_bounded,
    drain_inflight_compiles,
    inflight_compile_count,
)


class _Msg:
    pass


@pytest.fixture
def messages() -> list[object]:
    return [_Msg()]


@pytest.fixture(autouse=True)
def reset_state():
    contextual_model._ctx_compile_degradation_streak = 0
    contextual_model._ctx_compile_breaker_reopen_at = 0.0
    contextual_model._grounding_policy.set("required")
    contextual_model._grounding_outcome.set(None)
    with contextual_model._inflight_compiles_lock:
        contextual_model._inflight_compiles.clear()
    yield
    with contextual_model._inflight_compiles_lock:
        contextual_model._inflight_compiles.clear()
    contextual_model._ctx_compile_degradation_streak = 0
    contextual_model._ctx_compile_breaker_reopen_at = 0.0


async def test_timed_out_compile_runs_on_a_daemon_thread(monkeypatch, messages):
    """The abandoned worker must be a DAEMON thread so it can never block
    process/loop teardown -- the exact ~700s _do_shutdown tail this defect
    describes. A non-daemon thread (the pre-fix asyncio.to_thread default
    executor worker) would make this assertion fail."""
    monkeypatch.setattr(contextual_model, "_CONTEXT_COMPILE_TIMEOUT_S", 0.2)
    release = threading.Event()
    observed: dict[str, threading.Thread] = {}

    def _slow_blocking(_messages, _model_name):
        # Find ourselves in the live thread list while still "stuck".
        observed["thread"] = threading.current_thread()
        release.wait(timeout=5.0)
        return _messages, None

    monkeypatch.setattr(
        contextual_model, "_compiled_evidence_and_bundle", _slow_blocking
    )

    with pytest.raises(GroundingUnavailableError, match="timeout"):
        await _compiled_evidence_and_bundle_bounded(messages, "m")

    # Give the worker a moment to reach the observation point.
    for _ in range(50):
        if "thread" in observed:
            break
        await asyncio.sleep(0.02)

    assert "thread" in observed, "compile worker never started"
    worker = observed["thread"]
    assert worker.daemon is True, (
        "compile worker must be a daemon thread -- a non-daemon worker is what "
        "made process/loop teardown block on it for hundreds of seconds"
    )
    assert worker.name == "ctx-compile-isolated"

    release.set()
    worker.join(timeout=5.0)
    assert not worker.is_alive()


async def test_inflight_compile_count_tracks_and_clears_the_straggler(
    monkeypatch, messages
):
    """``inflight_compile_count()`` must reflect a still-running post-timeout
    worker, then drop back to 0 once it actually finishes -- the observability
    this defect's acceptance criteria required ("prove no compile worker ...
    remains"). The pre-fix code had no such registry at all."""
    monkeypatch.setattr(contextual_model, "_CONTEXT_COMPILE_TIMEOUT_S", 0.2)
    release = threading.Event()

    def _slow_blocking(_messages, _model_name):
        release.wait(timeout=5.0)
        return _messages, None

    monkeypatch.setattr(
        contextual_model, "_compiled_evidence_and_bundle", _slow_blocking
    )

    assert inflight_compile_count() == 0
    with pytest.raises(GroundingUnavailableError, match="timeout"):
        await _compiled_evidence_and_bundle_bounded(messages, "m")

    assert inflight_compile_count() == 1, (
        "the abandoned worker must still be tracked as in-flight immediately "
        "after the timeout is reported"
    )

    release.set()
    for _ in range(100):
        if inflight_compile_count() == 0:
            break
        await asyncio.sleep(0.02)
    assert inflight_compile_count() == 0, (
        "the registry must clear once the straggler actually completes"
    )


async def test_drain_inflight_compiles_never_blocks_past_its_own_timeout(
    monkeypatch, messages
):
    """A caller sequencing around a straggler must get a BOUNDED answer, never
    an indefinite hang -- there is no way to force an uncooperative blocking
    RPC thread to return early."""
    monkeypatch.setattr(contextual_model, "_CONTEXT_COMPILE_TIMEOUT_S", 0.1)
    release = threading.Event()

    def _slow_blocking(_messages, _model_name):
        release.wait(timeout=5.0)
        return _messages, None

    monkeypatch.setattr(
        contextual_model, "_compiled_evidence_and_bundle", _slow_blocking
    )

    with pytest.raises(GroundingUnavailableError, match="timeout"):
        await _compiled_evidence_and_bundle_bounded(messages, "m")

    start = time.monotonic()
    result = await drain_inflight_compiles(timeout=0.3)
    elapsed = time.monotonic() - start

    assert elapsed < 2.0, "drain must respect its own bounded timeout"
    assert result["remaining"] == 1
    release.set()


async def test_run_isolated_propagates_contextvars_like_to_thread(monkeypatch):
    """The isolated daemon-thread path must preserve to_thread's contextvar
    propagation (the verified GraphSession/resource-priority class), or the
    D-CDX-22 fix would silently regress grounding correctness while fixing
    teardown."""
    marker: ContextVar[str] = ContextVar("d_cdx_22_probe", default="unset")
    token = marker.set("ambient-value")
    try:
        seen = {}

        def _read_marker():
            seen["value"] = marker.get()

        future = contextual_model._run_isolated(_read_marker)
        future.result(timeout=5.0)
    finally:
        marker.reset(token)

    assert seen["value"] == "ambient-value"
