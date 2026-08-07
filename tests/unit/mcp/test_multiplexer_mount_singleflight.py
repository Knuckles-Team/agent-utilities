"""Regression tests for D-CDX-44: concurrent first loads of the SAME
not-yet-mounted MCP child must not start duplicate child runtimes or race
catalog/tool/forwarder registration.

Before the fix, ``MCPMultiplexer.mount_child`` had no per-server
singleflight/lock around the "is this server already mounted?" check: two
concurrent callers for the same unmounted ``server_name`` could both observe
``server_name not in self.children``, each call ``_start_child`` (a real
provider process/connection), and race ``_register_child_result``. These
tests deterministically interleave two concurrent ``mount_child`` calls
(via an artificial await inside a faked ``_start_child``) to prove exactly
one child start, one registration, no leaked singleflight state, and that a
failed leader does not poison a later retry.
"""

from __future__ import annotations

import asyncio

import pytest
from unittest.mock import AsyncMock

from tests.unit.mcp.test_multiplexer_dynamic_gateway import (
    CNT,
    CNT_PREFIXED,
    CNT_TOOL,
    _fake_tool,
    _mux_with_children,
)


@pytest.mark.asyncio
async def test_concurrent_first_loads_start_exactly_one_child(tmp_path) -> None:
    """Two callers racing ``mount_child(CNT)`` while it is still unmounted
    must produce exactly ONE ``_start_child`` invocation, ONE registration,
    and both callers must observe the SAME successfully mounted result."""
    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "containers")]})

    start_calls = 0
    entered = asyncio.Event()
    release = asyncio.Event()

    async def slow_start_child(server_name, cfg):
        nonlocal start_calls
        start_calls += 1
        entered.set()
        # Yield control back to the event loop so a second concurrent
        # caller has the chance to observe "not yet mounted" BEFORE this
        # (the leader's) attempt finishes — the exact race window D-CDX-44
        # describes.
        await release.wait()
        tools = [_fake_tool(CNT_TOOL, "containers")]
        session = AsyncMock()
        return server_name, session, tools, cfg

    mux._start_child = AsyncMock(side_effect=slow_start_child)  # type: ignore[method-assign]

    task_a = asyncio.create_task(mux.mount_child(CNT))
    await entered.wait()  # the leader is now inside _start_child, blocked

    task_b = asyncio.create_task(mux.mount_child(CNT))
    # Give task_b a real chance to run up to (and join) the singleflight
    # before the leader is released.
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    release.set()
    result_a, result_b = await asyncio.gather(task_a, task_b)

    assert start_calls == 1, "duplicate child runtime was started"
    assert mux._start_child.await_count == 1
    assert [t.name for t in result_a] == [CNT_PREFIXED]
    assert [t.name for t in result_b] == [CNT_PREFIXED]
    assert CNT in mux.children
    # No leaked singleflight state once both callers have resolved.
    assert mux._mount_inflight == {}


@pytest.mark.asyncio
async def test_many_concurrent_first_loads_still_start_exactly_one_child(
    tmp_path,
) -> None:
    """A wider fan-out (10 concurrent callers) than the minimal 2-caller
    case, to catch an ownership bug that only a real thundering herd would
    expose (e.g. a check that is safe for exactly 2 tasks but not N)."""
    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "containers")]})

    start_calls = 0
    entered = asyncio.Event()
    release = asyncio.Event()

    async def slow_start_child(server_name, cfg):
        nonlocal start_calls
        start_calls += 1
        entered.set()
        await release.wait()
        tools = [_fake_tool(CNT_TOOL, "containers")]
        session = AsyncMock()
        return server_name, session, tools, cfg

    mux._start_child = AsyncMock(side_effect=slow_start_child)  # type: ignore[method-assign]

    tasks = [asyncio.create_task(mux.mount_child(CNT)) for _ in range(10)]
    await entered.wait()
    for _ in range(5):
        await asyncio.sleep(0)
    release.set()

    results = await asyncio.gather(*tasks)

    assert start_calls == 1
    assert mux._start_child.await_count == 1
    assert all([t.name for t in r] == [CNT_PREFIXED] for r in results)
    assert mux._mount_inflight == {}


@pytest.mark.asyncio
async def test_different_servers_mount_fully_in_parallel(tmp_path) -> None:
    """The singleflight is per-server, never global: two DIFFERENT unmounted
    servers must both start concurrently, not serialize behind each other."""
    other = "other-mcp"
    mux = _mux_with_children(
        tmp_path, {CNT: [(CNT_TOOL, "containers")], other: [("op", "desc")]}
    )

    concurrently_active = 0
    max_concurrent = 0
    release = asyncio.Event()

    async def slow_start_child(server_name, cfg):
        nonlocal concurrently_active, max_concurrent
        concurrently_active += 1
        max_concurrent = max(max_concurrent, concurrently_active)
        await release.wait()
        concurrently_active -= 1
        tools = [_fake_tool("op", "desc")]
        session = AsyncMock()
        return server_name, session, tools, cfg

    mux._start_child = AsyncMock(side_effect=slow_start_child)  # type: ignore[method-assign]

    task_a = asyncio.create_task(mux.mount_child(CNT))
    task_b = asyncio.create_task(mux.mount_child(other))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    release.set()
    await asyncio.gather(task_a, task_b)

    assert max_concurrent == 2, "different servers must mount in parallel"
    assert mux._start_child.await_count == 2


@pytest.mark.asyncio
async def test_retry_after_failed_leader_starts_a_fresh_attempt(tmp_path) -> None:
    """A leader whose ``_start_child`` fails (returns ``None``, the existing
    ``_start_child`` failure contract) must not poison the singleflight —
    a later caller for the same server gets a FRESH attempt, not a cached
    failure."""
    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "containers")]})

    attempt = 0

    async def flaky_start_child(server_name, cfg):
        nonlocal attempt
        attempt += 1
        if attempt == 1:
            return None  # `_start_child`'s documented failure contract
        tools = [_fake_tool(CNT_TOOL, "containers")]
        session = AsyncMock()
        return server_name, session, tools, cfg

    mux._start_child = AsyncMock(side_effect=flaky_start_child)  # type: ignore[method-assign]

    first = await mux.mount_child(CNT)
    assert first == []
    assert CNT not in mux.children
    assert mux._mount_inflight == {}, "a failed leader must release ownership"

    second = await mux.mount_child(CNT)
    assert [t.name for t in second] == [CNT_PREFIXED]
    assert CNT in mux.children
    assert mux._start_child.await_count == 2


@pytest.mark.asyncio
async def test_retry_after_cancelled_leader_starts_a_fresh_attempt(tmp_path) -> None:
    """A leader cancelled mid-mount (e.g. its own caller's request timed out)
    must release ownership so a later caller is not stuck awaiting a future
    that will never resolve to a live mount."""
    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "containers")]})

    entered = asyncio.Event()
    hang = asyncio.Event()

    async def hanging_start_child(server_name, cfg):
        entered.set()
        await hang.wait()  # never released — simulates a stuck handshake
        raise AssertionError("unreachable")

    mux._start_child = AsyncMock(side_effect=hanging_start_child)  # type: ignore[method-assign]

    leader_task = asyncio.create_task(mux.mount_child(CNT))
    await entered.wait()
    leader_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await leader_task

    assert mux._mount_inflight == {}, "a cancelled leader must release ownership"

    # A fresh, non-hanging attempt now succeeds.
    async def fast_start_child(server_name, cfg):
        tools = [_fake_tool(CNT_TOOL, "containers")]
        session = AsyncMock()
        return server_name, session, tools, cfg

    mux._start_child = AsyncMock(side_effect=fast_start_child)  # type: ignore[method-assign]
    result = await mux.mount_child(CNT)
    assert [t.name for t in result] == [CNT_PREFIXED]
    assert CNT in mux.children


@pytest.mark.asyncio
async def test_follower_observes_leader_exception_without_retrying_itself(
    tmp_path,
) -> None:
    """If the leader's attempt raises an unexpected exception (not the
    normal ``_start_child`` -> ``None`` failure contract, but a genuine bug),
    every follower observes that SAME exception rather than silently
    swallowing it into a different, possibly-misleading empty result."""
    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "containers")]})

    entered = asyncio.Event()
    release = asyncio.Event()

    async def exploding_start_child(server_name, cfg):
        entered.set()
        await release.wait()
        raise RuntimeError("synthetic child-start failure")

    mux._start_child = AsyncMock(side_effect=exploding_start_child)  # type: ignore[method-assign]

    task_a = asyncio.create_task(mux.mount_child(CNT))
    await entered.wait()
    task_b = asyncio.create_task(mux.mount_child(CNT))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    release.set()

    results = await asyncio.gather(task_a, task_b, return_exceptions=True)
    assert all(isinstance(r, RuntimeError) for r in results)
    assert mux._mount_inflight == {}
    assert mux._start_child.await_count == 1

    # And a later retry is still possible (ownership was released).
    async def fast_start_child(server_name, cfg):
        tools = [_fake_tool(CNT_TOOL, "containers")]
        session = AsyncMock()
        return server_name, session, tools, cfg

    mux._start_child = AsyncMock(side_effect=fast_start_child)  # type: ignore[method-assign]
    result = await mux.mount_child(CNT)
    assert [t.name for t in result] == [CNT_PREFIXED]
