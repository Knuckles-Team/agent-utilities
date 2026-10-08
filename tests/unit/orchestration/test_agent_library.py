"""Agent Library store, role agents and assembled graphs."""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.ingestion.skill_workflow_ingest import (
    runnable_skill_digest,
)
from agent_utilities.orchestration.agent_library import (
    AGENT_LIBRARY_PROVIDER_REF,
    AgentLibrary,
    AgentRecord,
    assembled_graph_record,
    role_agent_from_blueprint,
    seed_role_agents,
)
from tests.unit.orchestration.agent_library_fakes import FakeLibraryEngine

ANALYST = AgentRecord(
    name="billing-analyst",
    system_prompt="You analyse billing incidents.",
    description="Billing analyst",
    role="Billing Analyst",
    tools=("tool:search",),
    skills=("skill://cite",),
    model_profile="qwen-27b",
    context_policy={"budget_tokens": 8000},
)

SPECIALIST = {
    "task": "wger_fitness_specialist",
    "source": "wger-agent",
    "description": "wger fitness specialist",
    "identity": {"role": "wger Fitness Specialist"},
    "instructions": {"core_directive": "You are a fitness specialist."},
}


@pytest.fixture
def engine() -> FakeLibraryEngine:
    return FakeLibraryEngine()


def test_a_saved_agent_is_listed_and_read_back_with_every_field(engine) -> None:
    library = AgentLibrary(engine)
    saved = library.save(ANALYST)

    assert saved.agent_id == "resource:skill:billing-analyst"
    assert [r.agent_id for r in library.list()] == [saved.agent_id]
    got = library.get(saved.agent_id)
    assert got is not None
    assert (got.role, got.model_profile, got.skills) == (
        "Billing Analyst",
        "qwen-27b",
        ("skill://cite",),
    )
    assert got.context_policy == {"budget_tokens": 8000}
    assert [ref["id"] for ref in got.tool_refs] == ["tool:search"]


def test_a_saved_agent_keeps_the_runnable_skill_contract(engine) -> None:
    saved = AgentLibrary(engine).save(ANALYST)
    node = engine.nodes[saved.agent_id]
    assert node["resource_type"] == "AGENT_SKILL"
    assert node["provider_ref"] == AGENT_LIBRARY_PROVIDER_REF
    assert node["instruction_digest"] == runnable_skill_digest(ANALYST.system_prompt)
    assert ("skill:billing-analyst", saved.agent_id, "BINDS_RUNNABLE") in engine.edges


def test_a_bound_server_expands_to_its_ingested_tools(engine) -> None:
    engine.add_tool("tool:a", "a", "demo-mcp")
    engine.add_tool("tool:b", "b", "other-mcp")
    record = AgentRecord(name="server-agent", system_prompt="x", mcp_server="demo-mcp")
    assert AgentLibrary(engine).save(record).tools == ("tool:a",)


def test_archived_and_foreign_entries_are_not_listed(engine) -> None:
    library = AgentLibrary(engine)
    saved = library.save(ANALYST)
    engine.nodes[saved.agent_id]["status"] = "ARCHIVED"
    engine.add_node(
        "resource:skill:fleet",
        "CallableResource",
        {"name": "fleet", "resource_type": "AGENT_SKILL", "provider_ref": "x"},
    )
    assert library.list() == []


@pytest.mark.parametrize(
    "record",
    [
        AgentRecord(name="", system_prompt="x"),
        AgentRecord(name="no-prompt", system_prompt=""),
        AgentRecord(name="bad-kind", system_prompt="x", kind="other"),
        AgentRecord(name="huge", system_prompt="x" * 40_000),
    ],
)
def test_an_invalid_record_is_refused_before_any_write(engine, record) -> None:
    with pytest.raises(ValueError):
        AgentLibrary(engine).save(record)
    assert engine.nodes == {}


def test_role_agents_come_from_packaged_prompts_with_a_role(engine) -> None:
    engine.add_prompt("prompt:wger", SPECIALIST)
    engine.add_prompt("prompt:main", {"task": "main-agent", "content": "orchestrate"})
    engine.add_tool("tool:routines", "routines", "wger-agent")

    saved = seed_role_agents(AgentLibrary(engine))

    assert [r.name for r in saved] == ["role-wger-fitness-specialist"]
    role = AgentLibrary(engine).list(kind="role")[0]
    assert (role.role, role.mcp_server) == ("wger Fitness Specialist", "wger-agent")
    assert role.source == "prompt:wger-agent/wger_fitness_specialist"
    assert saved[0].tools == ("tool:routines",)


def test_a_blueprint_without_a_role_is_not_a_role_agent() -> None:
    assert role_agent_from_blueprint({"instructions": {"core_directive": "x"}}) is None
    assert role_agent_from_blueprint({"identity": {"role": "r"}}) is None


def test_an_assembled_graph_is_saved_for_reuse_and_never_runnable(engine) -> None:
    result = {
        "graph": {"graph_id": "graph:billing", "shape": {"nodes": []}},
        "agents": [
            {
                "agent_id": "agent:researcher",
                "tools": [{"component_id": "tool.search"}],
                "skills": [{"component_id": "skill.cite"}],
                "system_prompt": {"component_id": "prompt.research"},
                "model_identity": "model:qwen",
            }
        ],
    }
    record = assembled_graph_record(result, committed={"record_id": "decision:1"})
    assert record is not None
    library = AgentLibrary(engine)
    saved = library.save(record)

    listed = library.list(kind="agent_graph")
    assert [r.agent_id for r in listed] == [saved.agent_id]
    got = library.get(saved.agent_id)
    assert got is not None and got.runnable is False
    assert got.graph is not None
    assert got.graph["decision_record_id"] == "decision:1"
    assert got.graph["graph"]["graph_id"] == "graph:billing"


def test_an_unsolved_result_makes_no_graph_record() -> None:
    assert assembled_graph_record({}) is None


def test_an_a2a_agent_is_listed_with_its_endpoint(engine) -> None:
    engine.add_node(
        "agent:outside",
        "CallableResource",
        {
            "name": "outside",
            "resource_type": "A2A_AGENT",
            "endpoint": "https://agent.example.com",
            "agent_card": {"name": "outside"},
        },
    )
    library = AgentLibrary(engine)
    (listed,) = library.list()
    assert listed.view()["kind"] == "a2a"
    assert listed.view()["endpoint"] == "https://agent.example.com"
    got = library.get("agent:outside")
    assert got is not None and got.agent_card == {"name": "outside"}
