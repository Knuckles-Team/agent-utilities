"""CONCEPT:AU-ORCH.sandbox.tiered-rlm-sandbox — LocalSandbox stdout-capture regression (B-21).

``LocalSandbox.execute()`` used to swap the process-global ``sys.stdout`` around its
``await``ed exec (``try/finally``-scoped, but not concurrency-safe): another coroutine
running during that ``await``, or a concurrent ``execute()`` call, could cross-contaminate
captured output. The fix replaces the per-call swap-and-restore with a single, stable
``sys.stdout`` replacement that routes each write by the calling asyncio Task's own
:class:`contextvars.ContextVar` capture buffer (see ``local_backend.py``'s module
docstring for why a subprocess boundary — the wasm/forkserver precedent — isn't available
to this specific backend).
"""

from __future__ import annotations

import asyncio
import sys

import pytest

from agent_utilities.rlm.sandboxes.base import SandboxEnv
from agent_utilities.rlm.sandboxes.local_backend import (
    LocalSandbox,
    _ContextRoutedStdout,
)


@pytest.mark.asyncio
async def test_execute_captures_print_output_and_syncs_vars():
    sandbox = LocalSandbox()
    env = SandboxEnv(vars={}, tool_sources={}, helpers={})

    result = await sandbox.execute("print('hello')\nglobal x\nx = 1", env)

    assert result.stdout == "hello\n"
    assert result.error is None
    assert result.updated_vars["x"] == 1


@pytest.mark.asyncio
async def test_execute_captures_traceback_on_in_sandbox_error():
    sandbox = LocalSandbox()
    env = SandboxEnv(vars={}, tool_sources={}, helpers={})

    result = await sandbox.execute("print('before')\nraise ValueError('boom')", env)

    assert result.error == "boom"
    assert "before" in result.stdout
    assert "ValueError" in result.stdout


def _staggered_wait_for(starts: list[asyncio.Event], gates: list[asyncio.Event]):
    """A helper whose N-th call sets ``starts[N]`` then blocks on ``gates[N]`` --
    lets a test drive a snippet through multiple precise interleaving checkpoints."""
    calls = 0

    async def wait_for() -> None:
        nonlocal calls
        starts[calls].set()
        await gates[calls].wait()
        calls += 1

    return wait_for


@pytest.mark.asyncio
async def test_two_concurrent_executions_do_not_cross_contaminate_stdout():
    """The core B-21 regression test: two concurrent ``execute()`` calls, each with
    TWO internal await checkpoints, driven through an interleaving proven (against a
    verbatim copy of the pre-fix implementation, in the accompanying investigation)
    to leak output between them AND straight through to the real terminal:

    old code, this exact interleaving -> a's capture = "1\\n" (missing "2","3" --
    they landed in b's buffer instead), b's capture = "1\\n2\\n2\\n3\\n" (b's own
    output PLUS a's leaked "2"), and a bare "3" written directly to the real
    process stdout. The fix must reduce all of that to each side capturing exactly
    its own three lines, with nothing reaching the real stream.
    """
    sandbox = LocalSandbox()

    started_a = [asyncio.Event(), asyncio.Event()]
    gate_a = [asyncio.Event(), asyncio.Event()]
    started_b = [asyncio.Event(), asyncio.Event()]
    gate_b = [asyncio.Event(), asyncio.Event()]

    env_a = SandboxEnv(
        vars={},
        tool_sources={},
        helpers={"wait_for": _staggered_wait_for(started_a, gate_a)},
    )
    env_b = SandboxEnv(
        vars={},
        tool_sources={},
        helpers={"wait_for": _staggered_wait_for(started_b, gate_b)},
    )

    code = "print('1')\nawait wait_for()\nprint('2')\nawait wait_for()\nprint('3')"

    task_a = asyncio.create_task(sandbox.execute(code, env_a))
    await started_a[0].wait()
    task_b = asyncio.create_task(sandbox.execute(code, env_b))
    await started_b[0].wait()

    # Interleave A and B's second stage in the opposite order from how they
    # started -- exactly the pattern that made the old process-global swap leak.
    gate_a[0].set()
    await started_a[1].wait()
    gate_b[0].set()
    await started_b[1].wait()
    gate_a[1].set()
    await asyncio.sleep(0.01)
    gate_b[1].set()

    result_a, result_b = await asyncio.gather(task_a, task_b)

    assert result_a.stdout == "1\n2\n3\n"
    assert result_b.stdout == "1\n2\n3\n"
    assert result_a.error is None
    assert result_b.error is None


@pytest.mark.asyncio
async def test_concurrent_execution_does_not_leak_into_an_unrelated_coroutine():
    """A coroutine that is NOT a sandbox execution at all must never see sandboxed
    output, even while a LocalSandbox execution is actively capturing on the same
    event loop."""
    sandbox = LocalSandbox()

    started = asyncio.Event()
    release = asyncio.Event()

    async def wait_for() -> None:
        started.set()
        await release.wait()

    env = SandboxEnv(vars={}, tool_sources={}, helpers={"wait_for": wait_for})
    code = "print('sandboxed-output')\nawait wait_for()\nprint('sandboxed-output-2')"

    task = asyncio.create_task(sandbox.execute(code, env))
    await started.wait()

    # A plain coroutine on the SAME loop, running WHILE the sandbox's capture is
    # active, writing directly through the real (unwrapped-by-any-capture) stdout
    # path -- must reach the real stream, not the sandbox's buffer.
    assert isinstance(sys.stdout, _ContextRoutedStdout)
    marker = "unrelated-coroutine-output\n"
    sys.stdout.write(marker)

    release.set()
    result = await task

    assert marker not in result.stdout
    assert result.stdout == "sandboxed-output\nsandboxed-output-2\n"
