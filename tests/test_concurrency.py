"""run_blocking offloads sync work off the event loop."""

from __future__ import annotations

import threading

import anyio
import pytest

from agent_utilities.mcp.concurrency import run_blocking
from agent_utilities.mcp_utilities import run_blocking as run_blocking_reexport


def test_run_blocking_runs_in_worker_thread_and_returns():
    async def main():
        loop_thread = threading.current_thread().name

        def work(a, b, *, c):
            return (threading.current_thread().name, a + b + c)

        worker_thread, total = await run_blocking(work, 1, 2, c=3)
        assert total == 6
        assert worker_thread != loop_thread  # ran off the event loop thread

    anyio.run(main)


def test_run_blocking_propagates_exceptions():
    async def main():
        def boom():
            raise ValueError("nope")

        with pytest.raises(ValueError, match="nope"):
            await run_blocking(boom)

    anyio.run(main)


def test_reexport_is_same_callable():
    assert run_blocking_reexport is run_blocking
