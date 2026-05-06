"""Tests for CONCEPT:ORCH-1.4 — Swarm Preset Template Engine."""

from __future__ import annotations

import pytest

from agent_utilities.graph.swarm_preset import (
    SwarmAgentDef,
    SwarmPresetDef,
    SwarmPresetEngine,
    SwarmTaskDef,
)
from agent_utilities.models.knowledge_graph import (
    RegistryEdgeType,
    RegistryNodeType,
    SwarmPresetNode,
    SwarmRunNode,
    SwarmTaskRecordNode,
)


SAMPLE_YAML = """
name: research_team
description: Multi-agent research workflow
agents:
  - id: researcher
    role: Research Analyst
  - id: writer
    role: Technical Writer
  - id: reviewer
    role: Peer Reviewer
tasks:
  - id: research
    agent_id: researcher
    prompt: "Research {topic} thoroughly."
  - id: draft
    agent_id: writer
    prompt: "Write a report on {topic} based on research."
    depends_on: [research]
  - id: review
    agent_id: reviewer
    prompt: "Review the draft for accuracy."
    depends_on: [draft]
variables:
  topic: AI safety
"""


class TestSwarmPresetDef:
    def test_creation(self):
        preset = SwarmPresetDef(
            name="test",
            agents=[SwarmAgentDef(id="a1", role="Analyst")],
            tasks=[SwarmTaskDef(id="t1", agent_id="a1", prompt="Do work")],
            variables={"topic": "test"},
        )
        assert preset.name == "test"
        assert len(preset.agents) == 1
        assert len(preset.tasks) == 1

    def test_serialization(self):
        preset = SwarmPresetDef(
            name="ser_test",
            agents=[SwarmAgentDef(id="a1", role="Analyst")],
            tasks=[SwarmTaskDef(id="t1", agent_id="a1")],
        )
        data = preset.model_dump()
        restored = SwarmPresetDef.model_validate(data)
        assert restored.name == "ser_test"


class TestSwarmPresetEngine:
    def setup_method(self):
        self.engine = SwarmPresetEngine()

    def test_load_yaml(self):
        preset = self.engine.load_from_yaml(SAMPLE_YAML)
        assert preset.name == "research_team"
        assert len(preset.agents) == 3
        assert len(preset.tasks) == 3
        assert preset.variables["topic"] == "AI safety"

    def test_load_invalid_yaml(self):
        with pytest.raises(ValueError, match="Invalid YAML"):
            self.engine.load_from_yaml("{{invalid yaml")

    def test_load_non_mapping(self):
        with pytest.raises(ValueError, match="must be a mapping"):
            self.engine.load_from_yaml("- just a list")

    def test_validate_dag_valid(self):
        preset = self.engine.load_from_yaml(SAMPLE_YAML)
        errors = self.engine.validate_dag(preset)
        assert errors == []

    def test_validate_dag_missing_agent(self):
        preset = SwarmPresetDef(
            name="bad",
            agents=[SwarmAgentDef(id="a1", role="A")],
            tasks=[SwarmTaskDef(id="t1", agent_id="nonexistent")],
        )
        errors = self.engine.validate_dag(preset)
        assert any("unknown agent" in e for e in errors)

    def test_validate_dag_missing_dep(self):
        preset = SwarmPresetDef(
            name="bad",
            agents=[SwarmAgentDef(id="a1", role="A")],
            tasks=[SwarmTaskDef(id="t1", agent_id="a1", depends_on=["ghost"])],
        )
        errors = self.engine.validate_dag(preset)
        assert any("unknown task" in e for e in errors)

    def test_validate_dag_cycle(self):
        preset = SwarmPresetDef(
            name="cyclic",
            agents=[SwarmAgentDef(id="a1", role="A")],
            tasks=[
                SwarmTaskDef(id="t1", agent_id="a1", depends_on=["t2"]),
                SwarmTaskDef(id="t2", agent_id="a1", depends_on=["t1"]),
            ],
        )
        errors = self.engine.validate_dag(preset)
        assert any("cycle" in e for e in errors)

    def test_resolve_dag(self):
        preset = self.engine.load_from_yaml(SAMPLE_YAML)
        layers = self.engine.resolve_dag(preset)
        assert len(layers) == 3
        assert layers[0] == ["research"]
        assert layers[1] == ["draft"]
        assert layers[2] == ["review"]

    def test_resolve_dag_parallel(self):
        preset = SwarmPresetDef(
            name="parallel",
            agents=[
                SwarmAgentDef(id="a1", role="A"),
                SwarmAgentDef(id="a2", role="B"),
            ],
            tasks=[
                SwarmTaskDef(id="t1", agent_id="a1"),
                SwarmTaskDef(id="t2", agent_id="a2"),
                SwarmTaskDef(id="t3", agent_id="a1", depends_on=["t1", "t2"]),
            ],
        )
        layers = self.engine.resolve_dag(preset)
        assert len(layers) == 2
        assert set(layers[0]) == {"t1", "t2"}  # parallel
        assert layers[1] == ["t3"]  # depends on both

    def test_resolve_dag_invalid(self):
        preset = SwarmPresetDef(
            name="bad",
            agents=[SwarmAgentDef(id="a1", role="A")],
            tasks=[
                SwarmTaskDef(id="t1", agent_id="a1", depends_on=["t2"]),
                SwarmTaskDef(id="t2", agent_id="a1", depends_on=["t1"]),
            ],
        )
        with pytest.raises(ValueError, match="DAG validation failed"):
            self.engine.resolve_dag(preset)

    def test_substitute_variables(self):
        preset = self.engine.load_from_yaml(SAMPLE_YAML)
        substituted = self.engine.substitute_variables(preset)
        assert "{topic}" not in substituted.tasks[0].prompt
        assert "AI safety" in substituted.tasks[0].prompt

    def test_substitute_user_override(self):
        preset = self.engine.load_from_yaml(SAMPLE_YAML)
        substituted = self.engine.substitute_variables(
            preset, user_vars={"topic": "Climate change"}
        )
        assert "Climate change" in substituted.tasks[0].prompt
        assert "AI safety" not in substituted.tasks[0].prompt

    def test_get_parallel_tasks(self):
        preset = self.engine.load_from_yaml(SAMPLE_YAML)
        roots = self.engine.get_parallel_tasks(preset)
        assert roots == ["research"]

    def test_to_yaml(self):
        preset = self.engine.load_from_yaml(SAMPLE_YAML)
        yaml_str = self.engine.to_yaml(preset)
        assert "research_team" in yaml_str
        assert "researcher" in yaml_str


class TestSwarmPresetKGNodes:
    def test_preset_node(self):
        node = SwarmPresetNode(
            id="sp:001",
            name="Investment Committee",
            preset_name="investment_committee",
            agent_specs=[{"id": "analyst", "role": "Analyst"}],
            task_graph=[{"id": "analyze", "agent_id": "analyst"}],
            variables={"universe": "SP500"},
            success_count=10,
            total_runs=12,
        )
        assert node.type == RegistryNodeType.SWARM_PRESET
        assert node.preset_name == "investment_committee"

    def test_run_node(self):
        node = SwarmRunNode(
            id="sr:001",
            name="Run #1",
            preset_id="sp:001",
            status="completed",
            total_input_tokens=5000,
            total_output_tokens=3000,
            task_count=3,
        )
        assert node.type == RegistryNodeType.SWARM_RUN
        assert node.status == "completed"

    def test_task_record_node(self):
        node = SwarmTaskRecordNode(
            id="str:001",
            name="Analysis Task",
            task_id="analyze",
            agent_id="analyst",
            status="completed",
            depends_on=["gather"],
        )
        assert node.type == RegistryNodeType.SWARM_TASK_RECORD

    def test_edge_types(self):
        assert RegistryEdgeType.PRESET_OF == "preset_of"
        assert RegistryEdgeType.RAN_PRESET == "ran_preset"
        assert RegistryEdgeType.TASK_DEPENDS_ON == "task_depends_on"
