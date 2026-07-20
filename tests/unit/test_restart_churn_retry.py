"""A child that dies mid-call self-heals: the call retries on the fresh child.

Regression for the restart-churn we kept hitting — the FIRST call after a
graph-os child (re)start crashed with an empty ``Error executing tool:`` (the
mid-call transport crash wasn't classified as session-death, so the existing
retry-once never fired), then a manual retry worked.
"""

from __future__ import annotations

import asyncio
from typing import Any

import anyio
import pytest

from agent_utilities.mcp.child_resilience import (
    ChildRuntime,
    is_transient_child_death,
)


def test_is_transient_child_death_classifies_transport_and_groups() -> None:
    assert is_transient_child_death(anyio.ClosedResourceError())
    assert is_transient_child_death(anyio.BrokenResourceError())
    assert is_transient_child_death(OSError("pipe closed"))
    assert is_transient_child_death(EOFError())
    # anyio wraps the real transport error in a group whose own str() is opaque.
    assert is_transient_child_death(
        BaseExceptionGroup("unhandled errors", [anyio.ClosedResourceError()])
    )
    # A live child answering with an application error is NOT a transient death.
    assert not is_transient_child_death(ValueError("tool said no"))
    assert not is_transient_child_death(BaseExceptionGroup("g", [ValueError("x")]))


def _runtime() -> ChildRuntime:
    rt = ChildRuntime.__new__(ChildRuntime)  # skip the connecting __init__
    rt._ready = asyncio.Event()
    rt._ready.set()  # respawned generation is already ready in the test
    rt.connect_timeout = 1.0
    # call_tool consults the lazy session-recycle guard first; with no
    # service-auth session window this is a no-op (None ⇒ early return).
    rt.session_max_age = None
    return rt


@pytest.mark.asyncio
async def test_call_tool_retries_once_on_midcall_crash() -> None:
    state = {"calls": 0, "restarts": 0}
    rt = _runtime()
    rt.request_restart = lambda reason="": state.__setitem__(
        "restarts", state["restarts"] + 1
    )  # type: ignore[method-assign]

    async def fake_call_once(_name: str, _args: dict[str, Any]) -> str:
        state["calls"] += 1
        if state["calls"] == 1:
            # child process crashed mid-call (anyio group wrapping a closed pipe)
            raise BaseExceptionGroup("crash", [anyio.ClosedResourceError()])
        return "ok"

    rt._call_once = fake_call_once  # type: ignore[method-assign]

    out = await rt.call_tool("some_tool", {})
    assert out == "ok"  # self-healed on the fresh generation
    assert state["calls"] == 2  # original + one retry
    assert state["restarts"] == 1  # reconnect was requested


@pytest.mark.asyncio
async def test_call_tool_does_not_retry_application_errors() -> None:
    state = {"calls": 0, "restarts": 0}
    rt = _runtime()
    rt.request_restart = lambda reason="": state.__setitem__(
        "restarts", state["restarts"] + 1
    )  # type: ignore[method-assign]

    async def boom(_name: str, _args: dict[str, Any]) -> str:
        state["calls"] += 1
        raise ValueError("application error")

    rt._call_once = boom  # type: ignore[method-assign]

    with pytest.raises(ValueError):
        await rt.call_tool("some_tool", {})
    assert state["calls"] == 1  # no retry for a live child's error
    assert state["restarts"] == 0
