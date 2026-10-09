"""graph.assemble() before LLM composition; the mapping is a claim."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from typing import Any

import pytest

from agent_utilities.decide.consumers.assembly import Assembler, install_assembler
from agent_utilities.knowledge_graph.enrichment.synthesize import synthesize_agent
from tests.unit.decide.fakes import FakeGraphs

GOAL = "find the owner of the billing service"
AGENT = {
    "agent_id": "agent:researcher",
    "tools": [{"component_id": "tool.search"}],
    "skills": [{"component_id": "skill.cite"}],
    "system_prompt": {"component_id": "prompt.research"},
    "model_identity": "model:qwen",
}


def _llm(prompt: str) -> str:
    if "task IRIs" in prompt:
        return json.dumps(["eg:task/research", "eg:task/invented"])
    return json.dumps({"name": "LLM agent", "tools": [], "skills": []})


@pytest.fixture
def installed() -> Iterator[list[FakeGraphs]]:
    box: list[FakeGraphs] = []
    yield box
    install_assembler(None)


def _install(box: list[FakeGraphs], result: dict[str, Any], **kw: Any) -> FakeGraphs:
    graphs = FakeGraphs(result)
    box.append(graphs)
    install_assembler(Assembler(graphs, "tenant-t", **kw), asyncio.run)
    return graphs


def test_eg_assembles_the_agent_and_the_goal_is_only_a_claim(installed) -> None:
    solved = {"record": {"outcome": {"outcome": "solved"}}, "agents": [AGENT]}

    async def context(record: Any) -> dict[str, Any]:
        return {"record": record, "context": "minted-by-graph-os"}

    graphs = _install(installed, solved, commit_context=context)
    spec = synthesize_agent(GOAL, lambda goal, limit: [], _llm)
    assert (spec.name, spec.tools, spec.model) == (
        "agent:researcher",
        ["tool.search"],
        "model:qwen",
    )
    request = graphs.requests[0]
    mapping = request["requirements"]["task_mappings"][0]
    assert mapping["task_iris"] == ["eg:task/research"], "unknown IRIs are dropped"
    assert GOAL not in json.dumps(request), "free text never reaches EG"
    assert graphs.commits and graphs.commits[0]["context"] == "minted-by-graph-os"


def test_an_abstention_leaves_the_llm_composition(installed) -> None:
    abstained = {
        "record": {
            "outcome": {
                "outcome": "abstained",
                "reasons": [{"reason": "uncovered_capability", "iri": "eg:x"}],
            }
        }
    }
    graphs = _install(installed, abstained)
    spec = synthesize_agent(GOAL, lambda goal, limit: [], _llm)
    assert spec.name == "LLM agent"
    assert graphs.requests and graphs.commits == []


@pytest.mark.spec("AU-CONTROL-R024", "AU-CONTROL-R025", "AU-CONTROL-R026")
def test_without_an_assembler_nothing_changes() -> None:
    calls: list[str] = []

    def llm(prompt: str) -> str:
        calls.append(prompt)
        return json.dumps({"name": "LLM agent"})

    assert synthesize_agent(GOAL, lambda goal, limit: [], llm).name == "LLM agent"
    assert len(calls) == 1, "no mapping call when EG is not wired"


GRAPH = {"graph_id": "graph:billing", "shape": {"nodes": [], "edges": []}}
SOLVED_GRAPH = {
    "record": {"outcome": {"outcome": "solved"}},
    "agents": [AGENT],
    "graph": GRAPH,
}


async def _commit_context(record: Any) -> dict[str, Any]:
    return {"record": record, "context": "minted-by-graph-os"}


async def _publish_context(graph: Any) -> dict[str, Any]:
    return {"principal": "graph-os", "graph_id": graph["graph_id"]}


@pytest.mark.spec("AU-CONTROL-R024", "AU-CONTROL-R025", "AU-CONTROL-R026")
def test_a_solved_committed_graph_is_published_and_saved_to_the_library(
    installed,
) -> None:
    """Commit, publish and library save are all bound."""
    from agent_utilities.orchestration.agent_library import AgentLibrary
    from tests.unit.orchestration.agent_library_fakes import FakeLibraryEngine

    library = AgentLibrary(FakeLibraryEngine())
    graphs = _install(
        installed,
        SOLVED_GRAPH,
        commit_context=_commit_context,
        publish_context=_publish_context,
        library=library,
    )
    synthesize_agent(GOAL, lambda goal, limit: [], _llm)

    assert graphs.commits, "the decision is committed before the publish"
    draft, context, evidence = graphs.published[0]
    assert draft["graph_id"] == "graph:billing"
    assert context == {"principal": "graph-os", "graph_id": "graph:billing"}
    assert evidence["component_id"] == "decision:abc"
    saved = library.list(kind="agent_graph")
    assert [r.name for r in saved] == ["graph:billing"]
    reuse = library.get(saved[0].agent_id)
    assert reuse is not None and reuse.graph is not None
    assert reuse.graph["decision_record_id"] == "decision:abc"
    assert reuse.graph["published"] == {"graph_id": "graph:billing"}


@pytest.mark.spec("AU-CONTROL-R024", "AU-CONTROL-R025", "AU-CONTROL-R026")
def test_an_uncommitted_graph_is_saved_but_never_published(installed) -> None:
    from agent_utilities.orchestration.agent_library import AgentLibrary
    from tests.unit.orchestration.agent_library_fakes import FakeLibraryEngine

    library = AgentLibrary(FakeLibraryEngine())
    graphs = _install(
        installed, SOLVED_GRAPH, publish_context=_publish_context, library=library
    )
    synthesize_agent(GOAL, lambda goal, limit: [], _llm)

    assert graphs.published == [], "no committed decision, no publish"
    saved = library.list(kind="agent_graph")
    assert saved and saved[0].graph is not None
    assert saved[0].graph["decision_record_id"] is None


def test_a_publish_failure_keeps_the_assembled_agent(installed) -> None:
    async def refuse(graph: Any) -> Any:
        raise PermissionError("no mutation context")

    graphs = _install(
        installed, SOLVED_GRAPH, commit_context=_commit_context, publish_context=refuse
    )
    spec = synthesize_agent(GOAL, lambda goal, limit: [], _llm)
    assert spec.name == "agent:researcher"
    assert graphs.published == []


def test_the_library_assembler_binds_session_clients_and_the_library() -> None:
    from types import SimpleNamespace

    from agent_utilities.decide.consumers.assembly import install_library_assembler
    from agent_utilities.orchestration.agent_library import AgentLibrary

    session = SimpleNamespace(tenant="tenant-t", graph="tenant-t")
    try:
        assembler = install_library_assembler(
            object(),
            session,
            engine=SimpleNamespace(query_cypher=lambda *a: []),
            run=asyncio.run,
            commit_context=_commit_context,
            publish_context=_publish_context,
        )
        assert assembler.tenant == "tenant-t"
        assert isinstance(assembler.library, AgentLibrary)
        assert assembler.commit_context is _commit_context
        assert assembler.publish_context is _publish_context
    finally:
        install_assembler(None)
