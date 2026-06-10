"""Invoker→spawned-agent curated-context handoff (CONCEPT:ORCH-1.38, MVP Phase 1).

Covers: the budgeted ``### INVOKER CONTEXT`` section helper, the ``GraphState.invoker_context``
field, and that ``run_agent(context=...)`` threads the curated context into the execution
config that seeds ``GraphState`` (and thus the spawn assemblers).

@pytest.mark.concept("ORCH-1.38")
"""

from __future__ import annotations

import pytest

from agent_utilities.graph.executor import invoker_context_section
from agent_utilities.graph.state import GraphState


@pytest.mark.concept("ORCH-1.38")
def test_section_empty_when_no_context():
    state = GraphState(query="q")
    assert state.invoker_context == ""
    assert invoker_context_section(state) == ""


@pytest.mark.concept("ORCH-1.38")
def test_section_rendered_when_present():
    state = GraphState(query="q", invoker_context="The user prefers metric units.")
    section = invoker_context_section(state)
    assert "### INVOKER CONTEXT" in section
    assert "metric units" in section


@pytest.mark.concept("ORCH-1.38")
def test_section_budgeted_to_window():
    # 200k chars of context must be trimmed to fit a 32K-token window fraction (~19.6K chars).
    big = "x" * 200_000
    state = GraphState(query="q", invoker_context=big)
    section = invoker_context_section(state, window_tokens=32768)
    assert "truncated to fit model window" in section
    # well under the full blob, comfortably within budget + header overhead
    assert len(section) < 25_000


@pytest.mark.asyncio
@pytest.mark.concept("ORCH-1.38")
async def test_run_agent_threads_context_into_config(monkeypatch):
    """run_agent(context=...) must place the curated context on the execution config
    that seeds GraphState (proves the entrypoint→state thread)."""
    from agent_utilities.orchestration import agent_runner

    captured = {}

    monkeypatch.setattr(agent_runner, "_resolve_agent_from_kg", lambda e, n: {"type": "stub"})
    monkeypatch.setattr(
        agent_runner, "_build_execution_config", lambda e, n, m: {"tag_prompts": {}}
    )
    monkeypatch.setattr(agent_runner, "_record_execution_trace", lambda *a, **k: None)

    async def _fake_execute_graph(*, config, **kwargs):
        captured["invoker_context"] = config.get("invoker_context")
        return {"results": {"output": "ok"}}

    monkeypatch.setattr(agent_runner, "_execute_graph", _fake_execute_graph)

    await agent_runner.run_agent(
        agent_name="unregistered-stub",
        task="do it",
        engine=object(),
        context="INVOKER SAYS: use the staging cluster only.",
    )
    assert captured["invoker_context"] == "INVOKER SAYS: use the staging cluster only."
