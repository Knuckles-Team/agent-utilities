"""Fleet-scale hardening of the MCP multiplexer (CONCEPT:ECO-4.34).

Per-child concurrency limits with bounded queueing, HTTP session pools,
cancellation-safe dispatch, restart-on-crash, and circuit breakers — all
exercised with in-process fake child sessions (no subprocesses, no network).
"""

from __future__ import annotations

import asyncio
from typing import Any

import mcp.types
import pytest

from agent_utilities.mcp.child_resilience import (
    ChildRuntime,
    MCPChildBusyError,
    MCPChildCallTimeoutError,
    MCPChildCircuitOpenError,
    MCPChildUnavailableError,
)
from agent_utilities.mcp.multiplexer import MCPMultiplexer


class GatedSession:
    """Fake child session whose calls block until ``release`` is set."""

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.started = 0
        self.completed = 0
        self.active = 0
        self.max_active = 0

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.started += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await self.release.wait()
        finally:
            self.active -= 1
        self.completed += 1
        return mcp.types.CallToolResult(
            content=[mcp.types.TextContent(type="text", text=f"ok:{name}")]
        )


class EchoSession:
    """Fake child session that answers immediately."""

    def __init__(self, tag: str = "echo") -> None:
        self.tag = tag
        self.calls: list[str] = []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append(name)
        return mcp.types.CallToolResult(
            content=[mcp.types.TextContent(type="text", text=f"{self.tag}:{name}")]
        )


# ---------------------------------------------------------------------------
# Work item 1 — per-server concurrency limits + bounded queue
# ---------------------------------------------------------------------------


async def test_concurrency_limit_enforced_and_excess_call_gets_busy_error():
    session = GatedSession()
    runtime = ChildRuntime(
        "limited", {"max_concurrency": 2, "queue_timeout": 0.05}
    )
    runtime.adopt_sessions([session])

    first = asyncio.create_task(runtime.call_tool("t", {}))
    second = asyncio.create_task(runtime.call_tool("t", {}))
    await asyncio.sleep(0.01)
    assert session.active == 2

    # Third call queues, times out, and fails typed — it never reaches the child.
    with pytest.raises(MCPChildBusyError) as exc:
        await runtime.call_tool("t", {})
    assert "limited" in str(exc.value)
    assert session.started == 2

    session.release.set()
    results = await asyncio.gather(first, second)
    assert all(not r.isError for r in results)
    assert session.max_active == 2
    assert runtime.in_flight == 0


async def test_queued_call_proceeds_when_slot_frees_within_timeout():
    session = GatedSession()
    runtime = ChildRuntime("queued", {"max_concurrency": 1, "queue_timeout": 5.0})
    runtime.adopt_sessions([session])

    first = asyncio.create_task(runtime.call_tool("t", {}))
    await asyncio.sleep(0.01)
    second = asyncio.create_task(runtime.call_tool("t", {}))
    await asyncio.sleep(0.01)
    assert runtime.queued == 1

    session.release.set()
    results = await asyncio.gather(first, second)
    assert [r.isError for r in results] == [False, False]
    assert session.max_active == 1  # never overlapped
    assert runtime.queued == 0


async def test_per_server_max_concurrency_override_beats_global_default():
    runtime = ChildRuntime("custom", {"max_concurrency": 3})
    assert runtime.max_concurrency == 3

    from agent_utilities.core.config import config

    runtime_default = ChildRuntime("default", {})
    assert runtime_default.max_concurrency == config.mcp_child_max_concurrency


async def test_zero_max_concurrency_disables_the_limit():
    session = GatedSession()
    runtime = ChildRuntime("unlimited", {"max_concurrency": 0})
    runtime.adopt_sessions([session])

    tasks = [asyncio.create_task(runtime.call_tool("t", {})) for _ in range(20)]
    await asyncio.sleep(0.01)
    assert session.active == 20
    session.release.set()
    await asyncio.gather(*tasks)


async def test_multiplexer_surfaces_busy_error_as_typed_tool_result(tmp_path):
    mux = MCPMultiplexer(tmp_path / "c.json")
    session = GatedSession()
    runtime = ChildRuntime("child", {"max_concurrency": 1, "queue_timeout": 0.05})
    runtime.adopt_sessions([session])
    mux.children["child"] = runtime
    mux.tool_to_server["ch__tool"] = ("child", "tool")

    blocker = asyncio.create_task(mux.call_proxied_tool("ch__tool", {}))
    await asyncio.sleep(0.01)

    result = await mux.call_proxied_tool("ch__tool", {})
    assert result.isError
    assert "MCPChildBusyError" in result.content[0].text
    assert "child" in result.content[0].text

    session.release.set()
    ok = await blocker
    assert not ok.isError


# ---------------------------------------------------------------------------
# Work item 2 — HTTP session pools + cancellation-safe dispatch
# ---------------------------------------------------------------------------


async def test_session_pool_round_robins_parallel_calls_across_connections():
    pool = [GatedSession(), GatedSession()]
    runtime = ChildRuntime("pooled", {"max_concurrency": 4})
    runtime.adopt_sessions(pool)

    tasks = [asyncio.create_task(runtime.call_tool("t", {})) for _ in range(4)]
    await asyncio.sleep(0.01)
    # Round-robin: 4 in-flight calls split 2/2 across the two connections.
    assert [s.active for s in pool] == [2, 2]

    for s in pool:
        s.release.set()
    results = await asyncio.gather(*tasks)
    assert all(not r.isError for r in results)


async def test_multiplexer_opens_pool_size_connections_for_http_child(
    tmp_path, monkeypatch
):
    import contextlib
    from unittest.mock import MagicMock

    from agent_utilities.mcp import multiplexer as mod

    connects: list[str] = []

    @contextlib.asynccontextmanager
    async def fake_http(url, headers=None):
        connects.append(url)
        yield ("r", "w", "sid")

    class FakeSessionCM:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            inner = MagicMock()

            async def initialize():
                return None

            async def list_tools():
                result = MagicMock()
                result.tools = []
                return result

            inner.initialize = initialize
            inner.list_tools = list_tools
            return inner

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(mod, "streamablehttp_client", fake_http)
    monkeypatch.setattr(mod, "ClientSession", FakeSessionCM)

    mux = MCPMultiplexer(tmp_path / "c.json")
    res = await mux._start_child(
        "pooled-http", {"url": "http://pooled.arpa/mcp", "pool_size": 3}
    )
    assert res is not None
    assert len(connects) == 3
    assert isinstance(res[1], ChildRuntime)
    assert res[1].status()["sessions"] == 3
    await res[1].aclose()


async def test_stdio_child_ignores_pool_size_and_keeps_one_pipe(
    tmp_path, monkeypatch
):
    import contextlib
    from unittest.mock import MagicMock

    from agent_utilities.mcp import multiplexer as mod

    connects: list[Any] = []

    @contextlib.asynccontextmanager
    async def fake_stdio(params):
        connects.append(params)
        yield ("r", "w")

    class FakeSessionCM:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            inner = MagicMock()

            async def initialize():
                return None

            async def list_tools():
                result = MagicMock()
                result.tools = []
                return result

            inner.initialize = initialize
            inner.list_tools = list_tools
            return inner

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(mod, "stdio_client", fake_stdio)
    monkeypatch.setattr(mod, "ClientSession", FakeSessionCM)

    mux = MCPMultiplexer(tmp_path / "c.json")
    res = await mux._start_child(
        "stdio-child", {"command": "child", "args": [], "pool_size": 3}
    )
    assert res is not None
    assert len(connects) == 1
    assert isinstance(res[1], ChildRuntime)
    assert res[1].status()["sessions"] == 1
    await res[1].aclose()


async def test_call_timeout_detaches_cleanly_and_keeps_session_usable():
    session = GatedSession()
    runtime = ChildRuntime(
        "slowpoke", {"max_concurrency": 2, "call_timeout": 0.05}
    )
    runtime.adopt_sessions([session])

    with pytest.raises(MCPChildCallTimeoutError) as exc:
        await runtime.call_tool("slow", {})
    assert "slowpoke" in str(exc.value)

    # The abandoned call still holds its slot until the child finishes.
    assert runtime.in_flight == 1
    assert session.active == 1

    # The shared session is NOT corrupted: a second call works fine.
    session.release.set()
    result = await runtime.call_tool("slow", {})
    assert not result.isError
    await asyncio.sleep(0)  # let the detached task's done-callback run
    assert runtime.in_flight == 0
    assert session.completed == 2


async def test_caller_cancellation_does_not_cancel_the_child_side_call():
    session = GatedSession()
    runtime = ChildRuntime("cancelled", {"max_concurrency": 2})
    runtime.adopt_sessions([session])

    caller = asyncio.create_task(runtime.call_tool("t", {}))
    await asyncio.sleep(0.01)
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller

    # Child-side call keeps running (shielded) and finishes normally.
    assert session.active == 1
    session.release.set()
    await asyncio.sleep(0.01)
    assert session.completed == 1
    assert runtime.in_flight == 0


async def test_detached_timeouts_apply_backpressure_until_child_recovers():
    session = GatedSession()
    runtime = ChildRuntime(
        "wedged",
        {"max_concurrency": 1, "call_timeout": 0.05, "queue_timeout": 0.05},
    )
    runtime.adopt_sessions([session])

    with pytest.raises(MCPChildCallTimeoutError):
        await runtime.call_tool("t", {})
    # Slot is still held by the wedged call -> next caller gets BUSY, fast.
    with pytest.raises(MCPChildBusyError):
        await runtime.call_tool("t", {})

    session.release.set()
    await asyncio.sleep(0.01)
    result = await runtime.call_tool("t", {})
    assert not result.isError


# ---------------------------------------------------------------------------
# Work item 3 — restart-on-crash supervisor + health surface
# ---------------------------------------------------------------------------


class DeadPipeSession:
    """Fake session over a dead transport: every call hits a closed pipe."""

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        raise ConnectionResetError("child process exited")


class GenerationConnector:
    """Scripted connect factory: one plan entry per connection generation.

    An exception entry makes that generation's connect fail; a session entry
    becomes the generation's (single-session) pool."""

    def __init__(self, plan: list[Any]) -> None:
        self.plan = plan
        self.connects = 0

    async def __call__(self, stack: Any) -> tuple[list[Any], list[Any]]:
        item = self.plan[min(self.connects, len(self.plan) - 1)]
        self.connects += 1
        if isinstance(item, BaseException):
            raise item
        if item == "hang":
            await asyncio.Event().wait()
        return [item], ["tool_a"]


async def _wait_for(predicate, timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        assert asyncio.get_running_loop().time() < deadline, "condition not reached"
        await asyncio.sleep(0.005)


async def test_crash_triggers_restart_and_child_recovers():
    healthy = EchoSession("gen2")
    connector = GenerationConnector([DeadPipeSession(), healthy])
    runtime = ChildRuntime(
        "phoenix",
        {"max_concurrency": 2, "queue_timeout": 0.5},
        connect=connector,
        restart_backoff_base=0.01,
        restart_backoff_cap=0.02,
    )
    tools = await runtime.start()
    assert tools == ["tool_a"]
    assert runtime.state == "up"

    # The dead pipe surfaces to the caller AND trips the restart cycle.
    with pytest.raises(ConnectionResetError):
        await runtime.call_tool("t", {})

    await _wait_for(lambda: runtime.state == "up" and connector.connects == 2)
    assert runtime.restart_count == 1

    result = await runtime.call_tool("t", {})
    assert result.content[0].text == "gen2:t"
    assert runtime.status()["state"] == "up"
    assert runtime.status()["restart_count"] == 1
    await runtime.aclose()


async def test_calls_during_restart_fail_fast_with_typed_error():
    connector = GenerationConnector([DeadPipeSession(), "hang"])
    runtime = ChildRuntime(
        "limbo",
        {"max_concurrency": 2, "queue_timeout": 0.05},
        connect=connector,
        restart_backoff_base=0.01,
        restart_backoff_cap=0.02,
    )
    await runtime.start()

    with pytest.raises(ConnectionResetError):
        await runtime.call_tool("t", {})
    await _wait_for(lambda: runtime.state == "restarting")

    # Reconnect hangs forever -> a call waits briefly, then fails typed.
    with pytest.raises(MCPChildUnavailableError) as exc:
        await runtime.call_tool("t", {})
    assert "limbo" in str(exc.value)
    assert exc.value.state == "restarting"
    await runtime.aclose()


async def test_restart_budget_exhaustion_parks_child_as_failed():
    connector = GenerationConnector(
        [EchoSession(), ConnectionRefusedError("child gone for good")]
    )
    runtime = ChildRuntime(
        "doomed",
        {"max_concurrency": 2, "queue_timeout": 0.05, "max_restarts": 2},
        connect=connector,
        restart_backoff_base=0.01,
        restart_backoff_cap=0.02,
    )
    await runtime.start()
    runtime.request_restart(reason="test crash")

    await _wait_for(lambda: runtime.state == "failed")
    # 1 boot connect + 2 allowed restart attempts, then parked.
    assert connector.connects == 3

    with pytest.raises(MCPChildUnavailableError) as exc:
        await runtime.call_tool("t", {})
    assert exc.value.state == "failed"
    assert "doomed" in str(exc.value)
    await runtime.aclose()


async def test_zero_max_restarts_disables_auto_restart():
    connector = GenerationConnector([DeadPipeSession(), EchoSession()])
    runtime = ChildRuntime(
        "frozen",
        {"max_concurrency": 2, "max_restarts": 0, "queue_timeout": 0.05},
        connect=connector,
        restart_backoff_base=0.01,
    )
    await runtime.start()
    with pytest.raises(ConnectionResetError):
        await runtime.call_tool("t", {})

    await _wait_for(lambda: runtime.state == "failed")
    assert connector.connects == 1  # never reconnected
    await runtime.aclose()


async def test_boot_connect_failure_raises_and_does_not_retry():
    connector = GenerationConnector([ConnectionRefusedError("nobody home")])
    runtime = ChildRuntime("stillborn", {}, connect=connector)
    with pytest.raises(ConnectionRefusedError):
        await runtime.start()
    assert connector.connects == 1
    assert runtime.state == "closed"  # aclose() ran in start()'s error path


async def test_multiplexer_status_snapshot_reports_every_child(tmp_path):
    mux = MCPMultiplexer(tmp_path / "c.json")
    up = ChildRuntime("alive", {"max_concurrency": 4})
    up.adopt_sessions([EchoSession()])
    mux.children["alive"] = up

    snapshot = mux.status_snapshot()
    assert snapshot["total_children"] == 1
    child = snapshot["children"]["alive"]
    assert child["state"] == "up"
    assert child["restart_count"] == 0
    assert child["max_concurrency"] == 4


# ---------------------------------------------------------------------------
# Work item 4 — per-child circuit breaker + metrics
# ---------------------------------------------------------------------------


class FlakySession:
    """Scripted session: fails the first N calls at the transport level,
    then answers normally."""

    def __init__(self, failures: int) -> None:
        self.remaining = failures
        self.calls = 0

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.calls += 1
        if self.remaining > 0:
            self.remaining -= 1
            raise ConnectionResetError("transport hiccup")
        return mcp.types.CallToolResult(
            content=[mcp.types.TextContent(type="text", text=f"ok:{name}")]
        )


async def test_breaker_opens_after_consecutive_transport_failures_then_recovers():
    session = FlakySession(failures=2)
    runtime = ChildRuntime(
        "breaky", {"breaker_threshold": 2, "breaker_cooldown": 0.05}
    )
    runtime.adopt_sessions([session])

    for _ in range(2):
        with pytest.raises(ConnectionResetError):
            await runtime.call_tool("t", {})
    assert runtime.breaker.state == "open"
    assert runtime.status()["breaker"] == "open"

    # Open circuit: fail fast, the child is never touched.
    with pytest.raises(MCPChildCircuitOpenError) as exc:
        await runtime.call_tool("t", {})
    assert "breaky" in str(exc.value)
    assert session.calls == 2

    # After the cooldown the half-open probe goes through and closes it.
    await asyncio.sleep(0.06)
    result = await runtime.call_tool("t", {})
    assert not result.isError
    assert runtime.breaker.state == "closed"
    assert session.calls == 3


async def test_failed_half_open_probe_reopens_the_circuit():
    session = FlakySession(failures=10)
    runtime = ChildRuntime(
        "relapse", {"breaker_threshold": 1, "breaker_cooldown": 0.05}
    )
    runtime.adopt_sessions([session])

    with pytest.raises(ConnectionResetError):
        await runtime.call_tool("t", {})
    assert runtime.breaker.state == "open"

    await asyncio.sleep(0.06)
    with pytest.raises(ConnectionResetError):  # the probe itself fails
        await runtime.call_tool("t", {})
    assert runtime.breaker.state == "open"
    with pytest.raises(MCPChildCircuitOpenError):
        await runtime.call_tool("t", {})
    assert session.calls == 2


async def test_zero_breaker_threshold_disables_short_circuiting():
    session = FlakySession(failures=10)
    runtime = ChildRuntime("unbroken", {"breaker_threshold": 0})
    runtime.adopt_sessions([session])

    for _ in range(4):
        with pytest.raises(ConnectionResetError):
            await runtime.call_tool("t", {})
    assert runtime.breaker.state == "closed"
    assert session.calls == 4


async def test_application_errors_do_not_trip_the_breaker():
    class AppErrorSession:
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
            raise ValueError("bad arguments for this tool")

    runtime = ChildRuntime("chatty", {"breaker_threshold": 2})
    runtime.adopt_sessions([AppErrorSession()])

    for _ in range(5):
        with pytest.raises(ValueError):
            await runtime.call_tool("t", {})
    # The child answered (with errors) — it is alive, circuit stays closed.
    assert runtime.breaker.state == "closed"


async def test_busy_rejections_do_not_count_as_breaker_failures():
    session = GatedSession()
    runtime = ChildRuntime(
        "swamped",
        {"max_concurrency": 1, "queue_timeout": 0.02, "breaker_threshold": 2},
    )
    runtime.adopt_sessions([session])

    blocker = asyncio.create_task(runtime.call_tool("t", {}))
    await asyncio.sleep(0.01)
    for _ in range(3):
        with pytest.raises(MCPChildBusyError):
            await runtime.call_tool("t", {})
    assert runtime.breaker.state == "closed"

    session.release.set()
    assert not (await blocker).isError


async def test_multiplexer_surfaces_circuit_open_as_typed_tool_result(tmp_path):
    mux = MCPMultiplexer(tmp_path / "c.json")
    runtime = ChildRuntime(
        "fused", {"breaker_threshold": 1, "breaker_cooldown": 60.0}
    )
    runtime.adopt_sessions([FlakySession(failures=1)])
    mux.children["fused"] = runtime
    mux.tool_to_server["fu__tool"] = ("fused", "tool")

    first = await mux.call_proxied_tool("fu__tool", {})
    assert first.isError  # the transport failure itself

    second = await mux.call_proxied_tool("fu__tool", {})
    assert second.isError
    assert "MCPChildCircuitOpenError" in second.content[0].text
    assert "fused" in second.content[0].text


async def test_metrics_are_noop_safe_without_prometheus():
    # The multiplexer runs standalone; every metric must degrade to a no-op
    # when prometheus_client (the optional extra) is absent — and be a real
    # series when it is installed. Either way these calls must not raise.
    from agent_utilities.observability import gateway_metrics as gm

    gm.MCP_CHILD_CALLS.labels(server="x", outcome="ok").inc()
    gm.MCP_CHILD_BREAKER_STATE.labels(server="x").set(2.0)
    gm.MCP_CHILD_RESTARTS.labels(server="x").inc()
    gm.MCP_CHILD_QUEUE_DEPTH.labels(server="x").set(3)

    if gm.PROMETHEUS_AVAILABLE:
        value = gm.MCP_CHILD_RESTARTS.labels(server="x")._value.get()
        assert value >= 1.0


async def test_runtime_status_reports_limits_and_load():
    session = GatedSession()
    runtime = ChildRuntime("statusy", {"max_concurrency": 2})
    runtime.adopt_sessions([session])

    task = asyncio.create_task(runtime.call_tool("t", {}))
    await asyncio.sleep(0.01)
    status = runtime.status()
    assert status["server"] == "statusy"
    assert status["max_concurrency"] == 2
    assert status["in_flight"] == 1
    assert status["sessions"] == 1

    session.release.set()
    await task
    assert runtime.status()["in_flight"] == 0
