"""Synthesis → execution bridge (CONCEPT:KG-2.10), with injected runner/facade."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from agent_utilities.knowledge_graph.enrichment.execute import (
    execute_agent_spec,
    execute_team_spec,
    make_capability_search,
    persist_as_runnable,
)
from agent_utilities.knowledge_graph.enrichment.orchestration import AgentSpec, TeamSpec


@dataclass
class FakeDesignation:
    id: str
    score: float
    capabilities: set


class FakeFacade:
    def designate(self, vec, k=8, required_caps=None):
        return [
            FakeDesignation("tool:graph_query", 0.91, {"query"}),
            FakeDesignation("skill:kg-ingest", 0.84, {"ingest"}),
        ]


from tests.kg_recording_backend import RecordingGraphBackend as FakeBackend


def _embed(texts):
    return [[1.0, 0.0] for _ in texts]


def test_make_capability_search_uses_designate():
    search = make_capability_search(FakeFacade(), _embed)
    res = search("ingest a codebase", 5)
    assert res[0] == {
        "id": "tool:graph_query",
        "type": "Tool",
        "name": "graph_query",
        "score": 0.91,
        "capabilities": ["query"],
    }
    assert res[1]["type"] == "Skill" and res[1]["name"] == "kg-ingest"


def test_persist_as_runnable_writes_callable_resource():
    spec = AgentSpec(name="KG Bot", description="ingests", tools=["graph_query"])
    backend = FakeBackend()
    n, e = persist_as_runnable(backend, spec)
    assert n == 1 and e == 1
    node = backend.nodes["resource:agent:kg-bot"]
    assert node["type"] == "CallableResource"
    assert node["resource_type"] == "AGENT_SKILL"
    assert ("resource:agent:kg-bot", "tool:graph_query", "USES_TOOL") in backend.edges


@pytest.mark.asyncio
async def test_execute_agent_spec_calls_runner():
    calls = {}

    async def fake_runner(name, task, max_steps=30):
        calls["name"] = name
        calls["task"] = task
        return f"ran {name}"

    spec = AgentSpec(name="Retriever", goal="answer")
    out = await execute_agent_spec(spec, "what calls foo?", runner=fake_runner)
    assert out == "ran Retriever"
    assert calls["name"] == "Retriever" and "foo" in calls["task"]


@pytest.mark.asyncio
async def test_execute_team_spec_runs_all_members():
    async def fake_runner(name, task, max_steps=30):
        return f"{name}-done"

    team = TeamSpec(name="Squad", lead="Lead", members=["Lead", "Worker"])
    members = [
        AgentSpec(name="Lead", goal="coordinate"),
        AgentSpec(name="Worker", goal="do the work"),
    ]
    results = await execute_team_spec(team, members, "ship it", runner=fake_runner)
    assert results == {"Lead": "Lead-done", "Worker": "Worker-done"}
