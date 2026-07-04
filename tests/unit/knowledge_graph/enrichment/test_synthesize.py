"""KG-driven orchestration synthesis (CONCEPT:KG-2.10 S4c)."""

from __future__ import annotations

import json

from agent_utilities.knowledge_graph.enrichment.orchestration import (
    AgentSpec,
    PromptSpec,
    TeamSpec,
)
from agent_utilities.knowledge_graph.enrichment.synthesize import (
    evolve_prompts,
    persist_synthesis,
    synthesize_agent,
    synthesize_team,
)
from tests.kg_recording_backend import RecordingGraphBackend as FakeBackend


def fake_capability_search(query, k):
    """A few Tool/Skill/Prompt candidates from the KG."""
    return [
        {"id": "tool:graph-query", "type": "Tool", "name": "graph_query"},
        {"id": "tool:embed", "type": "Tool", "name": "embed"},
        {
            "id": "skill:kg-ingest",
            "type": "Skill",
            "name": "kg-ingest",
        },
        {"id": "prompt:retriever", "type": "Prompt", "name": "retriever"},
    ]


# ── synthesize_agent ──────────────────────────────────────────────────────────
def test_synthesize_agent_grounds_in_candidates():
    def fake_llm(prompt):
        # candidate tools/skills must be listed for grounding to be meaningful
        assert "graph_query" in prompt and "kg-ingest" in prompt
        return json.dumps(
            {
                "name": "Retriever Bot",
                "system_prompt": "You retrieve answers from the KG.",
                "tools": ["graph_query", "embed"],
                "skills": ["kg-ingest"],
                "description": "Answers questions from the knowledge graph.",
            }
        )

    spec = synthesize_agent(
        "answer questions from the KG", fake_capability_search, fake_llm
    )
    assert isinstance(spec, AgentSpec)
    assert spec.name == "Retriever Bot"
    assert spec.goal == "answer questions from the KG"
    assert spec.tools == ["graph_query", "embed"]
    assert spec.skills == ["kg-ingest"]


def test_synthesize_agent_lenient_parse_and_defaults():
    def fake_llm(prompt):
        return (
            "Sure! Here is the agent:\n"
            '{"name": "Linker", "tools": ["graph_query"]} -- done'
        )

    spec = synthesize_agent("link concepts", fake_capability_search, fake_llm)
    assert spec.name == "Linker"
    assert spec.tools == ["graph_query"]
    assert spec.skills == []  # missing key tolerated


# ── synthesize_team ───────────────────────────────────────────────────────────
def test_synthesize_team_builds_lead_and_members():
    def fake_llm(prompt):
        if "Decompose this goal" in prompt:
            return json.dumps(
                {
                    "lead": "Coordinator",
                    "members": [
                        {"name": "Coordinator", "subgoal": "coordinate the build"},
                        {"name": "Retriever", "subgoal": "fetch from KG"},
                        {"name": "Linker", "subgoal": "link concepts"},
                    ],
                }
            )
        # per-member agent synthesis
        return json.dumps(
            {
                "name": "ignored",
                "system_prompt": "do the subgoal",
                "tools": ["graph_query"],
                "skills": [],
                "description": "member",
            }
        )

    team, members = synthesize_team("build the graph", fake_capability_search, fake_llm)
    assert isinstance(team, TeamSpec)
    assert team.lead == "Coordinator"
    assert team.members == ["Coordinator", "Retriever", "Linker"]
    assert {m.name for m in members} == {"Coordinator", "Retriever", "Linker"}
    assert all(isinstance(m, AgentSpec) for m in members)
    # member sub-goals carried onto the specs
    by_name = {m.name: m for m in members}
    assert by_name["Retriever"].goal == "fetch from KG"


def test_synthesize_team_respects_max_members():
    def fake_llm(prompt):
        if "Decompose this goal" in prompt:
            return json.dumps(
                {
                    "lead": "L",
                    "members": [{"name": f"M{i}", "subgoal": "s"} for i in range(10)],
                }
            )
        return json.dumps({"name": "x", "tools": [], "skills": []})

    team, members = synthesize_team(
        "big goal", fake_capability_search, fake_llm, max_members=3
    )
    assert len(team.members) == 3
    assert len(members) == 3


# ── evolve_prompts ────────────────────────────────────────────────────────────
def test_evolve_prompts_records_lineage():
    def fake_llm(prompt):
        return json.dumps(
            {
                "content": "You retrieve with grounding.",
                "rationale": "adds grounding to reduce hallucination",
            }
        )

    specs = evolve_prompts(
        [
            {
                "name": "Retriever v2",
                "problem": "hallucinates",
                "prior_prompt_id": "prompt:retriever-v1",
            },
            {"name": "Fresh Prompt", "problem": "no prompt yet"},
        ],
        fake_llm,
    )
    assert len(specs) == 2
    assert all(isinstance(s, PromptSpec) for s in specs)
    evolved = specs[0]
    assert evolved.name == "Retriever v2"
    assert evolved.evolved_from == "prompt:retriever-v1"
    assert evolved.content == "You retrieve with grounding."
    assert specs[1].evolved_from is None


# ── persist_synthesis ─────────────────────────────────────────────────────────
def test_persist_synthesis_writes_nodes_and_edges():
    agent = AgentSpec(
        name="Retriever Bot",
        goal="answer from KG",
        tools=["graph_query"],
        skills=["kg-ingest"],
    )
    team = TeamSpec(
        name="KG Squad",
        goal="build the graph",
        lead="Lead",
        members=["Lead", "Retriever"],
    )
    prompt = PromptSpec(
        name="Retriever v2", content="grounded", evolved_from="prompt:retriever-v1"
    )

    backend = FakeBackend()
    nodes, edges = persist_synthesis(backend, agent, team, prompt)

    assert nodes > 0 and edges > 0
    # nodes landed
    assert backend.nodes["agent:retriever-bot"]["type"] == "Agent"
    assert backend.nodes["team:kg-squad"]["type"] == "Team"
    assert backend.nodes["prompt:retriever-v2"]["type"] == "Prompt"
    # edges landed
    rels = {(s, t, r) for s, t, r in backend.edges}
    assert ("agent:retriever-bot", "tool:graph-query", "USES_TOOL") in rels
    assert ("agent:retriever", "agent:lead", "REPORTS_TO") in rels
    assert ("prompt:retriever-v2", "prompt:retriever-v1", "EVOLVED_FROM") in rels
    # returned totals match what the backend actually stored
    assert nodes == len(backend.nodes)
    assert edges == len(backend.edges)
