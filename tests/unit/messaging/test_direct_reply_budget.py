"""Direct-turn reply budget and follow-up delivery (au-integration-reliability spec, direct reply budget requirement)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agent_utilities.messaging import router
from agent_utilities.orchestration.execution_profile import ExecutionProfile


def _direct_shape() -> ExecutionProfile:
    return ExecutionProfile(
        name="chat", router_timeout=12.0, verifier_timeout=12.0, direct_complete=True
    )


@pytest.mark.spec("AU-INTEGRATION-R019")
def test_direct_budget_defaults_to_sixty_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MESSAGING_DIRECT_REPLY_BUDGET_S", raising=False)
    shape = _direct_shape()
    assert shape.reply_budget_s == 60.0
    assert shape.is_interactive


@pytest.mark.spec("AU-INTEGRATION-R019")
def test_direct_budget_respects_the_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MESSAGING_DIRECT_REPLY_BUDGET_S", "120")
    shape = _direct_shape()
    assert shape.reply_budget_s == 120.0
    # A direct turn stays inline even when its budget passes the interactive ceiling.
    assert shape.is_interactive


@pytest.mark.spec("AU-INTEGRATION-R019")
def test_invalid_direct_budget_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MESSAGING_DIRECT_REPLY_BUDGET_S", "not-a-number")
    assert _direct_shape().reply_budget_s == 60.0


class _SlowOrch:
    calls = 0

    def __init__(self, _engine: Any) -> None: ...

    async def execute_agent(self, **kwargs: Any) -> str:
        type(self).calls += 1
        await asyncio.sleep(0.3)
        return "the late answer"


@pytest.mark.asyncio
async def test_overrun_schedules_one_follow_up_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_utilities.orchestration import manager as mgr

    _SlowOrch.calls = 0
    monkeypatch.setattr(mgr, "Orchestrator", _SlowOrch)
    monkeypatch.setenv("AGENT_UTILITIES_TESTING", "true")
    delivered: list[str] = []

    async def _late(text: str) -> None:
        delivered.append(text)

    reply = await router._graph_agent_reply(
        object(),
        "hello",
        session="messaging:telegram:1",
        budget=0.05,
        shape=_direct_shape(),
        on_late_reply=_late,
    )
    assert reply == router._LATE_REPLY_MESSAGE
    assert "try again" not in reply.lower()
    assert delivered == []
    await asyncio.sleep(0.6)
    assert len(delivered) == 1
    assert "the late answer" in delivered[0]
    assert _SlowOrch.calls == 1


@pytest.mark.asyncio
async def test_overrun_without_follow_up_path_keeps_graceful_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_utilities.orchestration import manager as mgr

    monkeypatch.setattr(mgr, "Orchestrator", _SlowOrch)
    monkeypatch.setenv("AGENT_UTILITIES_TESTING", "true")
    reply = await router._graph_agent_reply(
        object(),
        "hello",
        session="messaging:telegram:1",
        budget=0.05,
        shape=_direct_shape(),
    )
    assert reply.startswith(router._SLOW_BACKEND_MESSAGE)
