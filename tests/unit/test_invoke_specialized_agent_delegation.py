"""invoke_specialized_agent is a thin wrapper onto the orchestration core (CONCEPT:AU-ECO.toolkit.unified-delegation-surface).

One delegation surface: ``invoke_specialized_agent`` and ``graph_orchestrate(action=
"execute_agent")`` both converge on ``Orchestrator.execute_agent`` — there is no separate
discovery / A2A / sub-agent-build path in the tool. These tests pin that contract.
"""

from __future__ import annotations

from typing import Any

import pytest


class _Deps:
    knowledge_engine: Any = None


class _Ctx:
    def __init__(self, engine: Any) -> None:
        self.deps = _Deps()
        self.deps.knowledge_engine = engine


@pytest.mark.asyncio
async def test_invoke_specialized_agent_routes_through_orchestrator(
    monkeypatch,
) -> None:
    from agent_utilities.orchestration import manager
    from agent_utilities.tools.agent_tools import invoke_specialized_agent

    seen: dict[str, Any] = {}

    class _FakeOrch:
        def __init__(self, engine: Any) -> None:
            seen["engine"] = engine

        async def execute_agent(self, *, agent_name: str, task: str) -> str:
            seen["agent_name"] = agent_name
            seen["task"] = task
            return "delegated result"

    monkeypatch.setattr(manager, "Orchestrator", _FakeOrch)

    engine = object()
    out = await invoke_specialized_agent(
        _Ctx(engine), agent_name="github", prompt="fetch issues"
    )
    assert out == "delegated result"
    # The SAME core graph_orchestrate(execute_agent) uses was invoked with our args.
    assert seen == {
        "engine": engine,
        "agent_name": "github",
        "task": "fetch issues",
    }


@pytest.mark.asyncio
async def test_invoke_specialized_agent_no_active_engine(monkeypatch) -> None:
    from agent_utilities.knowledge_graph.core import engine as engine_mod
    from agent_utilities.tools.agent_tools import invoke_specialized_agent

    monkeypatch.setattr(
        engine_mod.IntelligenceGraphEngine, "get_active", staticmethod(lambda: None)
    )
    out = await invoke_specialized_agent(
        _Ctx(None), agent_name="x", prompt="do a thing"
    )
    assert "not active" in out.lower()
