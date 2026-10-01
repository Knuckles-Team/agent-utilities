"""AU-CONTEXT-R001 through the LIVE callers.

* ``HybridRetriever.plan_and_retrieve`` (reached by ``engine.search_hybrid``
  for ``hyde``/``deep`` searches) keys the proven-path lookup by the
  retriever's task class and its graph's committed composed schema;
* a delegated run (``run_agent`` executes inside ``run_scoped``) attests each
  retrieval it planned once its answer is in (``record_answer_usage``): the
  units the answer names are cited, and a template-planned run carries the
  typed template it executed -- anchored on its cited units' class, under the
  skill ``run_agent`` pinned -- so a judged success becomes a proven path;
* ``synthesize_agent`` -> ``assemble_goal`` pins the skills EG proved for the
  mapped task classes into the ``AgentAssemble`` request.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Iterator
from typing import Any

import pytest

from agent_utilities.decide.consumers.assembly import Assembler, install_assembler
from agent_utilities.decide.learning import runs
from agent_utilities.decide.learning.run_scope import (
    current_skill_ref,
    run_scoped,
)
from agent_utilities.knowledge_graph.enrichment.synthesize import synthesize_agent
from agent_utilities.knowledge_graph.retrieval.hybrid_retriever import (
    HybridRetriever,
)
from agent_utilities.orchestration import agent_runner
from tests.unit.decide.fakes import FakeGraphs, FakeTransport, acted

SCHEMA = "sha256:composed"
QUERY = "which incident took billing down?"


class _Schemas:
    composed_digest = SCHEMA


class _Graph:
    graph_name = "kg"

    def graph_schema_list(self) -> _Schemas:
        return _Schemas()


class _Report:
    gate_passed = True


class _Retriever:
    """The real ``plan_and_retrieve``/``record_answer_usage`` over a canned
    candidate pool (no backend, no model)."""

    plan_and_retrieve = HybridRetriever.plan_and_retrieve
    record_answer_usage = HybridRetriever.record_answer_usage
    usage_telemetry = HybridRetriever.usage_telemetry
    last_quality_report = _Report()
    embed_model = None

    def __init__(self) -> None:
        self.engine = type("Engine", (), {"graph": _Graph()})()

    def retrieve_hybrid(self, query: str, **kw: Any) -> list[dict[str, Any]]:
        return [
            {"id": "inc-1", "type": "Incident", "_score": 0.9, "name": "outage"},
            {"id": "doc-1", "type": "Document", "_score": 0.8, "name": "runbook"},
        ]


@pytest.fixture(autouse=True)
def _fresh_schema_cache() -> Iterator[None]:
    runs._SCHEMA_DIGESTS.clear()
    yield
    runs._SCHEMA_DIGESTS.clear()


def test_the_live_retriever_keys_proven_paths_by_task_and_schema(
    eg: FakeTransport,
) -> None:
    eg.answer = acted("standard")
    _Retriever().plan_and_retrieve(QUERY, mode="deep")
    (paths_sql,) = [q for q in eg.queries if "decision_proven_paths" in q]
    assert f"task_class = '{runs.DEFAULT_TASK_CLASS}'" in paths_sql
    assert f"composed_digest = '{SCHEMA}'" in paths_sql


def test_a_run_answer_attests_its_retrievals_with_their_typed_template(
    eg: FakeTransport,
) -> None:
    eg.answer = acted("standard")
    retriever = _Retriever()
    skills: list[str | None] = []

    def tool_call() -> None:
        # A delegated run's tools reach the retriever off its event loop.
        skills.append(current_skill_ref())
        retriever.plan_and_retrieve(QUERY, mode="deep")

    @run_scoped
    async def delegated(*, skill_name: str | None = None) -> str:
        await asyncio.to_thread(tool_call)
        return "Billing went down in incident inc-1 (see the runbook)."

    asyncio.run(delegated(skill_name="Triage Incidents"))
    assert skills == ["skill://triage-incidents"]
    assert current_skill_ref() is None, "the run scope ends with the run"
    outcome = eg.ops[-1]["write"]["outcome"]
    assert [r["evidence_id"] for r in outcome["returned"]] == ["inc-1", "doc-1"]
    assert outcome["cited"] == ["inc-1"], "the answer names inc-1 only"
    assert outcome["path"] == {
        "task_class": runs.DEFAULT_TASK_CLASS,
        "composed_digest": SCHEMA,
        "policy_version": runs.PLAN_POLICY_VERSION,
        "anchor_class": "Incident",
        "edges": [],
        "rank": "text",
        "slots": ["query_text"],
        "skill_ref": "skill://triage-incidents",
    }


def test_run_agent_is_the_scoped_delegated_run() -> None:
    async def probe() -> None:
        return None

    run_agent = agent_runner.run_agent
    assert inspect.unwrap(run_agent) is not run_agent
    assert run_agent.__code__ is run_scoped(probe).__code__


class _Components:
    def __init__(self) -> None:
        self.asked: list[Any] = []

    async def current(self, op: Any) -> Any:
        self.asked.append(op)
        entry = type("Entry", (), {})()
        entry.component_id, entry.kind = op["component_id"], "skill"
        entry.definition_digest = "sha256:skill-rev"
        return entry


def _proven_skills(query: str) -> Any:
    assert "FROM decision_proven_paths" in query
    rows = [
        ["skill://triage-incidents", 5, 1],
        ["skill://lost-more", 1, 3],
    ]
    return {"columns": ["skill_ref", "successes", "failures"], "rows": rows}


def test_assembly_pins_the_skills_eg_proved_for_the_task(eg: FakeTransport) -> None:
    eg.sql_answer = _proven_skills
    graphs, components = FakeGraphs({"record": {}}), _Components()
    install_assembler(Assembler(graphs, "tenant-t", components=components), asyncio.run)
    try:
        synthesize_agent(
            "triage the billing outage",
            lambda goal, limit: [],
            lambda prompt: (
                json.dumps(["eg:task/research"])
                if "task IRIs" in prompt
                else json.dumps({"name": "LLM agent", "tools": [], "skills": []})
            ),
        )
    finally:
        install_assembler(None)
    pins = graphs.requests[0]["requirements"]["pins"]
    assert pins == [
        {
            "component_id": "skill://triage-incidents",
            "kind": "skill",
            "definition_digest": "sha256:skill-rev",
        }
    ], "only a skill that won more judged runs than it lost is pinned"
    assert "task_class = 'eg:task/research'" in eg.queries[0]
