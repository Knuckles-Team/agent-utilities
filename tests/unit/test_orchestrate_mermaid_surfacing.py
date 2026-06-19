"""Tests for orchestration flow-diagram surfacing (CONCEPT:ORCH-1.37).

Covers the trickiest part of the feature — ``run_agent``'s backward-compatible
return shaping (AC4/AC5): a bare string by default, a JSON ``{"output","mermaid"}``
wrapper only when ``return_mermaid=True`` AND a diagram is present. The additive
``mermaid`` keys on the swarm/compile/execute_workflow MCP handlers are dict
additions exercised end-to-end by ``test_workflow_e2e.py``.

@pytest.mark.concept("ORCH-1.37")
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.orchestration import agent_runner


@pytest.fixture
def _patched_run_agent(monkeypatch):
    """Stub out KG resolution + graph execution so only return-shaping is under test."""
    fake_response = {
        "results": {"output": "the answer"},
        "mermaid": "```mermaid\ngraph TD\n  A-->B\n```",
    }

    monkeypatch.setattr(agent_runner, "_resolve_agent_from_kg", lambda e, n: {"type": "stub"})
    monkeypatch.setattr(agent_runner, "_build_execution_config", lambda e, n, m, **kw: {})
    monkeypatch.setattr(agent_runner, "_record_execution_trace", lambda *a, **k: None)

    async def _fake_execute_graph(**kwargs):
        return dict(fake_response)

    monkeypatch.setattr(agent_runner, "_execute_graph", _fake_execute_graph)
    return fake_response


@pytest.mark.asyncio
@pytest.mark.concept("ORCH-1.37")
async def test_run_agent_default_returns_bare_string(_patched_run_agent):
    """AC5: default (return_mermaid=False) preserves the bare-string contract."""
    out = await agent_runner.run_agent(
        agent_name="unregistered-stub-agent",
        task="q",
        engine=object(),
    )
    assert out == "the answer"
    # must not be a JSON wrapper
    with pytest.raises(json.JSONDecodeError):
        json.loads(out)


@pytest.mark.asyncio
@pytest.mark.concept("ORCH-1.37")
async def test_run_agent_return_mermaid_wraps_when_present(_patched_run_agent):
    """AC4: return_mermaid=True yields a JSON wrapper carrying output + mermaid."""
    out = await agent_runner.run_agent(
        agent_name="unregistered-stub-agent",
        task="q",
        engine=object(),
        return_mermaid=True,
    )
    payload = json.loads(out)
    assert payload["output"] == "the answer"
    assert "mermaid" in payload["mermaid"]
    assert payload["mermaid"].startswith("```mermaid")


@pytest.mark.asyncio
@pytest.mark.concept("ORCH-1.37")
async def test_run_agent_no_mermaid_stays_bare_string(monkeypatch):
    """AC4 edge: return_mermaid=True but no diagram -> still a bare string."""
    monkeypatch.setattr(agent_runner, "_resolve_agent_from_kg", lambda e, n: {"type": "stub"})
    monkeypatch.setattr(agent_runner, "_build_execution_config", lambda e, n, m, **kw: {})
    monkeypatch.setattr(agent_runner, "_record_execution_trace", lambda *a, **k: None)

    async def _fake_execute_graph(**kwargs):
        return {"results": {"output": "no-diagram"}, "mermaid": None}

    monkeypatch.setattr(agent_runner, "_execute_graph", _fake_execute_graph)

    out = await agent_runner.run_agent(
        agent_name="unregistered-stub-agent",
        task="q",
        engine=object(),
        return_mermaid=True,
    )
    assert out == "no-diagram"
