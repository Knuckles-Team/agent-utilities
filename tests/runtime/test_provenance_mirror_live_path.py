"""CONCEPT:AU-KG.enrichment.atomic-triple-extraction — Workspace-Action Provenance: actions/observations mirror into the KG and
file edits ground to Code symbols.

Exercises the real :class:`ProvenanceMirror` write path against a recording engine that honours
the ``IntelligenceGraphEngine`` API contract (``add_node``/``add_edge``/``backend.execute``), so
the test asserts the exact graph shape the golden loop (AHE-3.23) later queries.
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.observability.trace_ontology import trace_id
from agent_utilities.runtime import DevWorkspace, LocalWorkspace
from agent_utilities.runtime.events import (
    CmdRunAction,
    ErrorObservation,
    FileEditAction,
    FileWriteAction,
)
from agent_utilities.runtime.provenance import ProvenanceMirror


class _RecordingBackend:
    def __init__(self, code_rows):
        self._code_rows = code_rows
        self.queries = []

    def execute(self, query, params=None):
        self.queries.append((query, params))
        # Only the Code-symbol lookup returns rows.
        if "MATCH (c:Code)" in query:
            return self._code_rows
        return []

    def add_edge(self, src, tgt, rel, **props):
        pass


class _RecordingEngine:
    def __init__(self, code_rows):
        self.backend = _RecordingBackend(code_rows)
        self.nodes = []
        self.edges = []

    def add_node(self, node_id, node_type, properties=None):
        self.nodes.append((node_id, node_type, properties or {}))

    def add_edge(self, source, target, rel_type="", **properties):
        self.edges.append((source, target, rel_type))


@pytest.fixture()
def recording_engine(monkeypatch):
    engine = _RecordingEngine(code_rows=[{"id": "Code:pkg/m.py::foo"}])
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    monkeypatch.setattr(
        IntelligenceGraphEngine, "get_active", classmethod(lambda cls: engine)
    )
    return engine


async def test_actions_and_observations_become_nodes_and_edges(recording_engine):
    ws = DevWorkspace(LocalWorkspace(), run_id="prov1")
    async with ws:
        await ws.act(CmdRunAction(command="true"))

    node_types = {t for _, t, _ in recording_engine.nodes}
    assert "WorkspaceAction" in node_types
    assert "WorkspaceObservation" in node_types

    rels = {r for _, _, r in recording_engine.edges}
    assert "HAS_ACTION" in rels  # (:RunTrace)-[:HAS_ACTION]->(:WorkspaceAction)
    assert "PRODUCED" in rels  # (:WorkspaceAction)-[:PRODUCED]->(:WorkspaceObservation)


async def test_next_replay_edge_links_consecutive_actions(recording_engine):
    ws = DevWorkspace(LocalWorkspace(), run_id="prov2")
    async with ws:
        await ws.act(CmdRunAction(command="true"))
        await ws.act(CmdRunAction(command="true"))

    next_edges = [(s, t) for s, t, r in recording_engine.edges if r == "NEXT"]
    run_ref = trace_id("prov2").removeprefix("trace:")
    assert next_edges == [
        (f"wsaction:{run_ref}:1", f"wsaction:{run_ref}:2")
    ]


async def test_file_edit_grounds_to_code_symbol(recording_engine):
    ws = DevWorkspace(LocalWorkspace(), run_id="prov3")
    async with ws:
        await ws.act(FileWriteAction(path="pkg/m.py", content="foo = 1\n"))
        await ws.act(FileEditAction(path="pkg/m.py", old="foo = 1", new="foo = 2"))

    mutated = [(s, t) for s, t, r in recording_engine.edges if r == "MUTATED"]
    run_ref = trace_id("prov3").removeprefix("trace:")
    assert (f"wsaction:{run_ref}:2", "Code:pkg/m.py::foo") in mutated


async def test_mirror_is_silent_when_no_engine(monkeypatch):
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    monkeypatch.setattr(
        IntelligenceGraphEngine, "get_active", classmethod(lambda cls: None)
    )
    ws = DevWorkspace(LocalWorkspace(), run_id="prov4")
    async with ws:
        obs = await ws.act(CmdRunAction(command="true"))  # must not raise
    assert obs.exit_code == 0


def test_durable_provenance_is_metadata_only(recording_engine, tmp_path):
    """Commands, paths, reports, errors, and identities never enter node properties."""

    command = "run --token top-secret --file /private/location/input.txt"
    actor = "person@example.test"
    error = "failed for person@example.test at /private/location/input.txt"
    action = CmdRunAction(
        run_id="privacy-run",
        step=7,
        ts=1.0,
        actor=actor,
        command=command,
        cwd="/private/location",
    )
    observation = ErrorObservation(
        run_id="privacy-run",
        step=7,
        ts=2.0,
        actor=actor,
        message=error,
        action_kind="cmd_run",
    )

    mirror = ProvenanceMirror()
    mirror.mirror_action(action)
    mirror.mirror_observation(action, observation, tmp_path)

    serialized = json.dumps(recording_engine.nodes, sort_keys=True)
    for raw in (command, actor, error, "top-secret", "/private/location"):
        assert raw not in serialized
    for _node_id, _node_type, properties in recording_engine.nodes:
        assert "summary" not in properties
        assert "payload_ref" in properties
        assert properties["payload_character_count"] > 0
