"""Orchestration substrate: specs → graph batches (CONCEPT:AU-KG.enrichment.a2a-capability-extraction S4b)."""

from __future__ import annotations

from agent_utilities.knowledge_graph.enrichment.orchestration import (
    AgentSpec,
    PromptSpec,
    TeamSpec,
    WorkflowSpec,
    agent_to_batch,
    prompt_to_batch,
    team_to_batch,
    workflow_to_batch,
)
from agent_utilities.knowledge_graph.enrichment.registry import write_batch
from tests.kg_recording_backend import RecordingGraphBackend as FakeBackend


def test_agent_spec_to_graph():
    a = AgentSpec(
        name="Retriever Bot",
        goal="answer from KG",
        prompt_id="prompt:retriever",
        tools=["graph_query", "embed"],
        skills=["kg-ingest"],
    )
    b = agent_to_batch(a)
    backend = FakeBackend()
    write_batch(backend, b)
    assert backend.nodes["agent:retriever-bot"]["type"] == "Agent"
    rels = {(s.split(":")[0], t, r) for s, t, r in backend.edges}
    assert ("agent", "tool:graph-query", "USES_TOOL") in rels
    assert ("agent", "skill:kg-ingest", "HAS_SKILL") in rels
    assert ("agent", "prompt:retriever", "HAS_PROMPT") in rels
    assert any(r == "SOLVES" for _, _, r in backend.edges)


def test_team_hierarchy_defaults_to_lead():
    t = TeamSpec(
        name="KG Squad",
        goal="build the graph",
        lead="Lead",
        members=["Lead", "Retriever", "Linker"],
    )
    b = team_to_batch(t)
    backend = FakeBackend()
    write_batch(backend, b)
    assert backend.nodes["team:kg-squad"]["type"] == "Team"
    reports = {(s, t_) for s, t_, r in backend.edges if r == "REPORTS_TO"}
    # non-lead members report to lead
    assert ("agent:retriever", "agent:lead") in reports
    assert ("agent:linker", "agent:lead") in reports
    members = {(s, t_) for s, t_, r in backend.edges if r == "MEMBER_OF_TEAM"}
    assert ("agent:lead", "team:kg-squad") in members


def test_prompt_lineage_and_workflow():
    p = PromptSpec(
        name="Retriever v2",
        content="You retrieve...",
        evolved_from="prompt:retriever-v1",
        rationale="add grounding",
    )
    pb = prompt_to_batch(p)
    assert any(e.rel_type == "EVOLVED_FROM" for e in pb.edges)
    assert pb.nodes[0].type == "Prompt"

    w = WorkflowSpec(
        name="Ingest Flow",
        steps=["scan", "parse", "enrich"],
        orchestrates=["agent:retriever", "skill:enrich"],
    )
    wb = workflow_to_batch(w)
    assert wb.nodes[0].type == "Workflow"
    assert {e.rel_type for e in wb.edges} == {"ORCHESTRATES"}
