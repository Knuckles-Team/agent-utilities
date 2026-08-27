"""Characterization tests for ``run_agent`` (CX-AU-02, CCN 156 -> decomposed).

These tests pin ``run_agent``'s OBSERVED behaviour against the unmodified function,
before any decomposition, per the two-commit discipline: this file must be added and
green in commit 1 (characterize), with zero behaviour change, before commit 2
(refactor) touches ``agent_runner.py``. They deliberately mock only the KG/network
boundary functions that already existed on ``agent_runner`` before the refactor
(``_resolve_agent_from_kg``, ``_build_execution_config``, ``_execute_graph``,
``_execute_single_server``, ``_record_execution_trace``, ``_prime_recent_mementos``,
``_prime_code_context``) so the same mocks are valid both before and after the
decomposition -- none of those names were renamed, only their internal call sites
moved into new private helpers.

Each assertion here was verified, by hand, to FAIL when the corresponding behaviour
is broken (the known-bad discipline): flipping an expected value, or removing a
patch, turns the test red. See the lane report (CX-AU-02) for the specific mutations
tried.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from agent_utilities.orchestration import agent_runner


def _patch_common(monkeypatch, *, agent_meta: dict[str, Any] | None = None) -> None:
    """Stub the KG/network boundary so only run_agent's own control flow is under
    test. Mirrors the pattern in tests/unit/test_orchestrate_mermaid_surfacing.py."""
    monkeypatch.setattr(
        agent_runner,
        "_resolve_agent_from_kg",
        lambda e, n: agent_meta if agent_meta is not None else {"type": "stub"},
    )
    monkeypatch.setattr(
        agent_runner,
        "_build_execution_config",
        lambda e, n, m, **kw: {
            "agent_model": "synthetic-model",
            "selected_model_class": "standard",
        },
    )
    # Returning True (not None/falsy) matters: _persist_run_outcome treats a falsy
    # trace-write result as a genuine write FAILURE (BUG-015 atomic-outcome
    # contract) and downgrades a would-be "ok" run_summary to "degraded".
    monkeypatch.setattr(agent_runner, "_record_execution_trace", lambda *a, **k: True)
    monkeypatch.setattr(
        agent_runner,
        "_prime_recent_mementos",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        agent_runner,
        "_prime_code_context",
        AsyncMock(return_value=None),
    )


@pytest.mark.asyncio
async def test_run_agent_full_graph_success_returns_bare_string(monkeypatch):
    """An unresolved agent (no server/template match) routes through the full
    multi-agent graph and, by default (return_mermaid=False), returns a bare
    string -- not a JSON envelope."""
    _patch_common(monkeypatch)

    async def _fake_execute_graph(**kwargs):
        return {"results": {"output": "the answer"}}

    monkeypatch.setattr(agent_runner, "_execute_graph", _fake_execute_graph)

    out = await agent_runner.run_agent(
        agent_name="unresolved-agent",
        task="q",
        engine=object(),
    )
    assert out == "the answer"
    with pytest.raises(json.JSONDecodeError):
        json.loads(out)


@pytest.mark.asyncio
async def test_run_agent_return_mermaid_wraps_output_and_run_id(monkeypatch):
    """return_mermaid=True always wraps the result in a JSON envelope carrying
    ``output`` and ``run_id`` (ORCH-1.97), even with no diagram produced."""
    _patch_common(monkeypatch)

    async def _fake_execute_graph(**kwargs):
        return {"results": {"output": "the answer"}, "mermaid": None}

    monkeypatch.setattr(agent_runner, "_execute_graph", _fake_execute_graph)

    out = await agent_runner.run_agent(
        agent_name="unresolved-agent",
        task="q",
        engine=object(),
        return_mermaid=True,
    )
    payload = json.loads(out)
    assert payload["output"] == "the answer"
    assert "run_id" in payload
    assert payload["mermaid"] is None


@pytest.mark.asyncio
async def test_run_agent_single_server_route_binds_tool_grounding(monkeypatch):
    """A resolved single MCP-server agent (type=='server' + bound mcp_toolsets)
    dispatches through ``_execute_single_server`` with ``bound_tool_grounding=True``
    -- the deterministic direct tool loop, not the planning graph."""
    monkeypatch.setattr(
        agent_runner, "_resolve_agent_from_kg", lambda e, n: {"type": "server"}
    )
    monkeypatch.setattr(
        agent_runner,
        "_build_execution_config",
        lambda e, n, m, **kw: {
            "agent_model": "synthetic-model",
            "selected_model_class": "standard",
            "mcp_toolsets": ["fake-server"],
        },
    )
    monkeypatch.setattr(agent_runner, "_record_execution_trace", lambda *a, **k: None)
    monkeypatch.setattr(
        agent_runner, "_prime_recent_mementos", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        agent_runner, "_prime_code_context", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        agent_runner,
        "_select_relevant_tool_names",
        AsyncMock(return_value=None),
    )

    captured_kwargs: dict[str, Any] = {}

    async def _fake_execute_single_server(**kwargs):
        captured_kwargs.update(kwargs)
        return {
            "results": {"output": "server said hi"},
            "tool_calls": [{"tool_name": "fake_tool"}],
        }

    monkeypatch.setattr(
        agent_runner, "_execute_single_server", _fake_execute_single_server
    )

    out = await agent_runner.run_agent(
        agent_name="fake-server",
        task="q",
        engine=object(),
    )
    assert out == "server said hi"
    assert captured_kwargs.get("bound_tool_grounding") is True


@pytest.mark.asyncio
async def test_run_agent_dispatch_exception_returns_failed_message(monkeypatch):
    """An exception raised during dispatch (e.g. the full graph) is flattened and
    reported as ``"Agent execution failed: <message>"`` — never propagated to the
    caller as a raw exception."""
    _patch_common(monkeypatch)

    async def _raise_execute_graph(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(agent_runner, "_execute_graph", _raise_execute_graph)

    out = await agent_runner.run_agent(
        agent_name="unresolved-agent",
        task="q",
        engine=object(),
    )
    # _flatten_exception_group renders as "<ExceptionType>: <message>", not the
    # bare message -- observed behaviour, pinned as-is.
    assert out == "Agent execution failed: RuntimeError: boom"


@pytest.mark.asyncio
async def test_run_agent_cancelled_error_propagates(monkeypatch):
    """A cooperative cancellation (asyncio.CancelledError) during dispatch MUST
    propagate to the caller — it is never flattened into a failure string, so an
    outer wall-clock timeout still surfaces as a real cancellation."""
    _patch_common(monkeypatch)

    async def _raise_cancelled(**kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(agent_runner, "_execute_graph", _raise_cancelled)

    with pytest.raises(asyncio.CancelledError):
        await agent_runner.run_agent(
            agent_name="unresolved-agent",
            task="q",
            engine=object(),
        )


@pytest.mark.asyncio
async def test_run_agent_required_tools_without_toolcall_is_degraded(monkeypatch):
    """``required_tools`` demands real ToolCall provenance: a result with no
    ``tool_calls`` is replaced with a truthful degraded envelope, never reported as
    a plain success."""
    _patch_common(monkeypatch)

    async def _fake_execute_graph(**kwargs):
        return {"results": {"output": "looks fine"}}

    monkeypatch.setattr(agent_runner, "_execute_graph", _fake_execute_graph)

    out = await agent_runner.run_agent(
        agent_name="unresolved-agent",
        task="q",
        engine=object(),
        # validate_tool_contract requires required_tools' members to be a subset
        # of an explicit allowed_tools catalog.
        allowed_tools=["some_tool"],
        required_tools=["some_tool"],
    )
    assert "tool-required execution finished without recorded ToolCall" in out


@pytest.mark.asyncio
async def test_run_agent_include_run_summary_reports_ok_outcome(monkeypatch):
    """``include_run_summary=True`` attaches a ``run_summary`` with
    ``outcome=="ok"`` for a plain successful full-graph run."""
    _patch_common(monkeypatch)

    async def _fake_execute_graph(**kwargs):
        return {"results": {"output": "the answer"}}

    monkeypatch.setattr(agent_runner, "_execute_graph", _fake_execute_graph)

    out = await agent_runner.run_agent(
        agent_name="unresolved-agent",
        task="q",
        engine=object(),
        include_run_summary=True,
    )
    payload = json.loads(out)
    assert payload["run_summary"]["outcome"] == "ok"
    assert payload["run_summary"]["route"]["agents"] == ["multi-agent-graph"]


@pytest.mark.asyncio
async def test_run_agent_skill_name_mismatch_raises_before_dispatch(monkeypatch):
    """``skill_name`` must match ``agent_name`` when both are given — validated
    up front, before any KG/network call."""
    with pytest.raises(ValueError, match="skill_name must match"):
        await agent_runner.run_agent(
            agent_name="alpha",
            task="q",
            engine=object(),
            skill_name="beta",
        )


@pytest.mark.asyncio
async def test_run_agent_tool_server_without_skill_name_raises(monkeypatch):
    """``tool_server`` requires an explicit ``skill_name`` — validated up front."""
    with pytest.raises(ValueError, match="tool_server requires skill_name"):
        await agent_runner.run_agent(
            agent_name="alpha",
            task="q",
            engine=object(),
            tool_server="some-server",
        )
