"""Fleet-scale hardening of the MCP multiplexer (CONCEPT:AU-ECO.mcp.profile-differences-from-client).

Per-child concurrency limits with bounded queueing, HTTP session pools,
cancellation-safe dispatch, restart-on-crash, and circuit breakers — all
exercised with in-process fake child sessions (no subprocesses, no network).
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

import mcp.types
import pytest

from agent_utilities.mcp.child_resilience import (
    ChildRuntime,
    MCPChildBusyError,
    MCPChildCallTimeoutError,
    MCPChildCircuitOpenError,
    MCPChildUnavailableError,
    MCPError,
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
    runtime = ChildRuntime("limited", {"max_concurrency": 2, "queue_timeout": 0.05})
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
    assert all(not r.is_error for r in results)
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
    assert [r.is_error for r in results] == [False, False]
    assert session.max_active == 1  # never overlapped
    assert runtime.queued == 0


async def test_per_server_max_concurrency_override_beats_global_default():
    runtime = ChildRuntime("custom", {"max_concurrency": 3})
    assert runtime.max_concurrency == 3

    from agent_utilities.core.config import config

    runtime_default = ChildRuntime("default", {})
    assert runtime_default.max_concurrency == config.mcp_child_max_concurrency


async def test_zero_max_concurrency_is_rejected():
    with pytest.raises(ValueError, match="max_concurrency"):
        ChildRuntime("unlimited", {"max_concurrency": 0})


async def test_multiplexer_surfaces_busy_error_as_typed_tool_result(tmp_path):
    mux = MCPMultiplexer(tmp_path / "c.json")
    # Seed the catalog directly (bypassing the on-disk config read) so
    # call_proxied_tool's "is this child still declared" gate — a deliberate
    # truthfulness check, not something a test may route around — sees
    # "child" as a live, configured server.
    mux._catalog = {"child": {}}
    session = GatedSession()
    runtime = ChildRuntime("child", {"max_concurrency": 1, "queue_timeout": 0.05})
    runtime.adopt_sessions([session])
    mux.children["child"] = runtime
    mux.tool_to_server["ch__tool"] = ("child", "tool")

    blocker = asyncio.create_task(mux.call_proxied_tool("ch__tool", {}))
    await asyncio.sleep(0.01)

    result = await mux.call_proxied_tool("ch__tool", {})
    assert result.is_error
    # The caller-facing text is deliberately just the class name (so callers
    # can branch on it); the full "which server" message goes to the
    # server-side log only (see call_proxied_tool's MCPChildError handler).
    assert result.content[0].text == "MCPChildBusyError"

    session.release.set()
    ok = await blocker
    assert not ok.is_error


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
    assert all(not r.is_error for r in results)


async def test_multiplexer_opens_pool_size_connections_for_http_child(
    tmp_path, monkeypatch
):
    import contextlib
    from unittest.mock import MagicMock

    from agent_utilities.mcp import multiplexer as mod

    connects: list[str] = []

    @contextlib.asynccontextmanager
    async def fake_http(url, *, http_client=None):
        # MCP SDK v2 signature: (url, *, http_client, terminate_on_close), and
        # it yields (read, write) — no third get_session_id element.
        connects.append(url)
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

    monkeypatch.setattr(mod, "streamable_http_client", fake_http)
    monkeypatch.setattr(mod, "ClientSession", FakeSessionCM)

    mux = MCPMultiplexer(tmp_path / "c.json")
    # A non-loopback remote child must be HTTPS (fail-closed transport gate
    # in _open_one_session) — this test is about pool sizing, not that gate,
    # so use a scheme the gate actually admits.
    res = await mux._start_child(
        "pooled-http", {"url": "https://pooled.example/mcp", "pool_size": 3}
    )
    assert res is not None
    assert len(connects) == 3
    assert isinstance(res[1], ChildRuntime)
    assert res[1].status()["sessions"] == 3
    await res[1].aclose()


async def test_stdio_child_ignores_pool_size_and_keeps_one_pipe(tmp_path, monkeypatch):
    import contextlib
    from unittest.mock import MagicMock

    from agent_utilities.mcp import multiplexer as mod

    connects: list[Any] = []

    @contextlib.asynccontextmanager
    async def fake_stdio(params, *, errlog):
        connects.append(params)
        assert errlog is not None
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
    runtime = ChildRuntime("slowpoke", {"max_concurrency": 2, "call_timeout": 0.05})
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
    assert not result.is_error
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
    assert not result.is_error


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

    # ChildRuntime.call_tool retries ONCE on a transient child death,
    # synchronously waiting for the restart to complete — so the dead pipe
    # trips the restart cycle AND is invisible to this caller: the single
    # call transparently returns gen2's answer instead of raising.
    result = await runtime.call_tool("t", {})
    assert result.content[0].text == "gen2:t"

    await _wait_for(lambda: runtime.state == "up" and connector.connects == 2)
    assert runtime.restart_count == 1
    assert runtime.status()["state"] == "up"
    assert runtime.status()["restart_count"] == 1
    await runtime.aclose()


async def test_generation_callback_gates_calls_until_recovered_catalog_is_published():
    """A recovered transport must wait for its owner to publish metadata.

    The callback models the multiplexer replacing its forwarding schemas.  A
    second concurrent caller must not slip through the fresh child generation
    before that update finishes.
    """
    healthy = EchoSession("gen2")
    connector = GenerationConnector([DeadPipeSession(), healthy])
    callback_started = asyncio.Event()
    release_callback = asyncio.Event()
    observed: list[list[Any]] = []

    async def on_generation(tools: list[Any]) -> None:
        observed.append(tools)
        callback_started.set()
        await release_callback.wait()

    runtime = ChildRuntime(
        "catalog-gated",
        {"max_concurrency": 2, "queue_timeout": 0.5},
        connect=connector,
        restart_backoff_base=0.005,
        restart_backoff_cap=0.005,
        on_generation=on_generation,
    )
    await runtime.start()

    first = asyncio.create_task(runtime.call_tool("first", {}))
    await callback_started.wait()
    second = asyncio.create_task(runtime.call_tool("second", {}))
    await asyncio.sleep(0.01)
    assert healthy.calls == []

    release_callback.set()
    results = await asyncio.gather(first, second)
    assert sorted(result.content[0].text for result in results) == [
        "gen2:first",
        "gen2:second",
    ]
    assert observed == [["tool_a"]]
    await runtime.aclose()


async def test_generation_callback_failure_does_not_strand_recovered_child(caplog):
    """Observer failures are isolated from the child transport itself."""
    healthy = EchoSession("gen2")
    connector = GenerationConnector([DeadPipeSession(), healthy])

    async def on_generation(_tools: list[Any]) -> None:
        raise RuntimeError("synthetic observer failure")

    runtime = ChildRuntime(
        "observer-isolated",
        {"max_concurrency": 2, "queue_timeout": 0.5},
        connect=connector,
        restart_backoff_base=0.005,
        restart_backoff_cap=0.005,
        on_generation=on_generation,
    )
    await runtime.start()

    result = await runtime.call_tool("recover", {})
    assert result.content[0].text == "gen2:recover"
    assert runtime.state == "up"
    assert "generation observer failed" in caplog.text
    await runtime.aclose()


def _session_terminated_error() -> BaseException:
    """The MCP protocol error a redeployed backend raises, on EITHER SDK line.

    SDK v1's ``McpError.__init__`` takes a pre-built ``ErrorData``; SDK v2
    renamed the class to ``MCPError`` and changed the signature to
    ``(code, message, data=None)``, building its own ``.error``. Both populate
    ``exc.error.code``/``.message``, which is all ``is_session_dead`` reads —
    so branch on the actual signature rather than pinning one SDK line.
    """
    if "code" in inspect.signature(MCPError.__init__).parameters:
        return MCPError(code=32600, message="Session terminated")
    return MCPError(mcp.types.ErrorData(code=32600, message="Session terminated"))


class SessionTerminatedSession:
    """Fake session whose call hits a server-terminated streamable-http session
    (what a redeployed backend does): MCP protocol error with code=32600."""

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        raise _session_terminated_error()


async def test_terminated_session_auto_reconnects_and_retries(monkeypatch):
    # A backend redeploy drops the session; the next call must transparently
    # reconnect and retry on the fresh generation — no manual reconnect, no
    # error surfaced to the caller (ECO-4.36 hardening).
    healthy = EchoSession("gen2")
    connector = GenerationConnector([SessionTerminatedSession(), healthy])
    runtime = ChildRuntime(
        "redeployed",
        {"max_concurrency": 2, "queue_timeout": 0.5},
        connect=connector,
        restart_backoff_base=0.01,
        restart_backoff_cap=0.02,
    )
    await runtime.start()
    assert runtime.state == "up"

    result = await runtime.call_tool("t", {})
    assert result.content[0].text == "gen2:t"  # served by the reconnected gen
    assert runtime.restart_count == 1
    assert connector.connects == 2
    await runtime.aclose()


async def test_terminated_session_surfaces_if_reconnect_also_dead():
    # If the reconnect's session is ALSO terminated, the single retry is
    # exhausted and the typed error surfaces (no infinite retry loop).
    connector = GenerationConnector(
        [SessionTerminatedSession(), SessionTerminatedSession()]
    )
    runtime = ChildRuntime(
        "still-dead",
        {"max_concurrency": 2, "queue_timeout": 0.5},
        connect=connector,
        restart_backoff_base=0.01,
        restart_backoff_cap=0.02,
    )
    await runtime.start()
    with pytest.raises(MCPError):
        await runtime.call_tool("t", {})
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

    # ChildRuntime.call_tool retries ONCE on a transient child death, waiting
    # (bounded) for the restart to finish first. Reconnect hangs forever
    # here, so that wait always elapses -- even this FIRST call already
    # fails typed instead of raising the raw transport error.
    with pytest.raises(MCPChildUnavailableError) as exc:
        await runtime.call_tool("t", {})
    assert "limbo" in str(exc.value)
    assert exc.value.state == "restarting"
    await _wait_for(lambda: runtime.state == "restarting")

    # A second call while still restarting fails the same typed way.
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
    # With no restart budget the transient death parks the child FAILED
    # immediately (no reconnect attempt); ChildRuntime.call_tool's one retry
    # then hits that failed state and surfaces the typed unavailable error,
    # not the raw transport exception.
    with pytest.raises(MCPChildUnavailableError) as exc:
        await runtime.call_tool("t", {})
    assert exc.value.state == "failed"

    assert runtime.state == "failed"
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
    runtime = ChildRuntime("breaky", {"breaker_threshold": 2, "breaker_cooldown": 0.05})
    runtime.adopt_sessions([session])

    # ChildRuntime.call_tool retries once on a transient child death, so a
    # SINGLE top-level call already makes 2 session-level attempts here
    # (FlakySession(failures=2)) -- both fail, tripping breaker_threshold=2
    # within that one call. The second top-level call then finds the
    # circuit already open and fails typed instead of hitting the child.
    with pytest.raises(ConnectionResetError):
        await runtime.call_tool("t", {})
    with pytest.raises(MCPChildCircuitOpenError):
        await runtime.call_tool("t", {})
    assert runtime.breaker.state == "open"
    assert runtime.status()["breaker"] == "open"

    # Open circuit: fail fast, the child is never touched.
    with pytest.raises(MCPChildCircuitOpenError) as exc:
        await runtime.call_tool("t", {})
    # MCPChildCircuitOpenError's message is just "circuit_open" (str(exc)
    # deliberately doesn't repeat the server name); the server it belongs to
    # lives on the typed `.server` attribute instead.
    assert exc.value.server == "breaky"
    assert session.calls == 2

    # After the cooldown the half-open probe goes through and closes it.
    await asyncio.sleep(0.06)
    result = await runtime.call_tool("t", {})
    assert not result.is_error
    assert runtime.breaker.state == "closed"
    assert session.calls == 3


async def test_failed_half_open_probe_reopens_the_circuit():
    session = FlakySession(failures=10)
    runtime = ChildRuntime(
        "relapse", {"breaker_threshold": 1, "breaker_cooldown": 0.05}
    )
    runtime.adopt_sessions([session])

    # breaker_threshold=1 opens the circuit on the very first recorded
    # failure, which happens during ChildRuntime.call_tool's own retry
    # attempt inside this first top-level call -- so it already surfaces
    # the typed circuit-open result, not the raw transport failure.
    with pytest.raises(MCPChildCircuitOpenError):
        await runtime.call_tool("t", {})
    assert runtime.breaker.state == "open"

    await asyncio.sleep(0.06)
    # Same story for the half-open probe: it fails, and the retry inside
    # that same call reopens the breaker before returning, so this call
    # ALSO surfaces the typed circuit-open result (not the raw error).
    with pytest.raises(MCPChildCircuitOpenError):  # the probe itself fails
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
    # With short-circuiting disabled the breaker never blocks the retry, so
    # each top-level call makes 2 session-level attempts (ChildRuntime's one
    # transient-death retry) -- 4 calls x 2 attempts.
    assert session.calls == 8


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
    assert not (await blocker).is_error


async def test_multiplexer_surfaces_circuit_open_as_typed_tool_result(tmp_path):
    mux = MCPMultiplexer(tmp_path / "c.json")
    # Seed the catalog directly (bypassing the on-disk config read) so
    # call_proxied_tool's "is this child still declared" gate — a deliberate
    # truthfulness check, not something a test may route around — sees
    # "fused" as a live, configured server.
    mux._catalog = {"fused": {}}
    runtime = ChildRuntime("fused", {"breaker_threshold": 1, "breaker_cooldown": 60.0})
    runtime.adopt_sessions([FlakySession(failures=1)])
    mux.children["fused"] = runtime
    mux.tool_to_server["fu__tool"] = ("fused", "tool")

    # ChildRuntime.call_tool retries once on a transient child death; with
    # breaker_threshold=1 the single failure already opens the circuit
    # before that retry runs, so even this FIRST call surfaces the typed
    # circuit-open result, not the raw transport failure. The caller-facing
    # text is deliberately just the class name (server detail stays in the
    # server-side log — see call_proxied_tool's MCPChildError handler).
    first = await mux.call_proxied_tool("fu__tool", {})
    assert first.is_error
    assert first.content[0].text == "MCPChildCircuitOpenError"

    second = await mux.call_proxied_tool("fu__tool", {})
    assert second.is_error
    assert second.content[0].text == "MCPChildCircuitOpenError"


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


# ---------------------------------------------------------------------------
# Session recycle before token expiry (CONCEPT:AU-OS.identity.so-jwt-protected-children)
# ---------------------------------------------------------------------------


async def test_session_recycled_before_token_ttl_not_counted_as_restart():
    """A service-authenticated child whose session outlives its bearer reconnects
    BEFORE the next call (so the call never lands on a dead-token session), and
    the planned recycle is NOT charged to the crash-restart budget."""
    connector = GenerationConnector([EchoSession("gen1"), EchoSession("gen2")])
    runtime = ChildRuntime(
        "expiring",
        {"max_concurrency": 2, "queue_timeout": 0.5},
        connect=connector,
        session_max_age=0.05,
        restart_backoff_base=0.01,
        restart_backoff_cap=0.02,
    )
    await runtime.start()
    assert connector.connects == 1 and runtime.state == "up"

    await asyncio.sleep(0.08)  # outlive the (tiny) token window
    result = await runtime.call_tool("foo", {})
    assert "gen2:foo" in result.content[0].text  # served by the FRESH generation
    assert connector.connects == 2  # recycled exactly once, lazily, on the call
    assert runtime.restart_count == 0  # planned recycle, not a crash

    # A prompt follow-up reuses the fresh generation (still inside its window).
    result2 = await runtime.call_tool("bar", {})
    assert "gen2:bar" in result2.content[0].text
    assert connector.connects == 2

    await runtime.aclose()


async def test_no_recycle_without_session_max_age():
    """Children with no token to manage (session_max_age=None) never recycle."""
    connector = GenerationConnector([EchoSession("only"), EchoSession("unused")])
    runtime = ChildRuntime("static", {}, connect=connector)
    await runtime.start()
    await asyncio.sleep(0.05)
    await runtime.call_tool("foo", {})
    assert connector.connects == 1  # never recycled

    await runtime.aclose()


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
