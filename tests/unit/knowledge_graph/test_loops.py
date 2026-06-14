"""Unit tests for the Loop model — the long-running-objective unit (CONCEPT:KG-2.78)."""

from __future__ import annotations

from agent_utilities.knowledge_graph.research.loops import (
    TERMINAL_STATUS,
    active_loops,
    submit_loop,
)


class _Engine:
    def __init__(self, concept_rows=None, addressed=None):
        self.nodes: dict[str, dict] = {}
        self._concept_rows = concept_rows or []
        self._addressed = addressed or []

    def add_node(self, node_id, node_type, properties=None):
        self.nodes[node_id] = {"type": node_type, **(properties or {})}

    def query_cypher(self, q: str, params: dict | None = None):
        if "ADDRESSED_BY" in q:
            return [{"id": i} for i in self._addressed]
        return self._concept_rows


def test_submit_loop_creates_kinded_node():
    eng = _Engine()
    loop = submit_loop(
        eng, "Make the build pass", kind="develop", validation_cmd="pytest -q"
    )
    assert loop["kind"] == "develop"
    nid = loop["id"]
    node = eng.nodes[nid]
    assert node["loop_kind"] == "develop"
    assert node["validation_cmd"] == "pytest -q"
    assert node["objective"] == "Make the build pass"
    assert node["status"] == "pending"


def test_submit_research_loop_default_kind():
    eng = _Engine()
    loop = submit_loop(eng, "self-improving agent harnesses")
    assert loop["kind"] == "research"
    assert eng.nodes[loop["id"]]["loop_kind"] == "research"


def test_active_loops_dispatch_by_kind_and_status():
    rows = [
        {
            "id": "loop:research:a",
            "name": "A",
            "loop_kind": "research",
            "status": "pending",
        },
        {
            "id": "loop:research:done",
            "name": "D",
            "loop_kind": "research",
            "status": "pending",
        },
        {
            "id": "loop:develop:b",
            "name": "B",
            "loop_kind": "develop",
            "status": "running",
            "validation_cmd": "pytest",
        },
        {
            "id": "loop:develop:fin",
            "name": "C",
            "loop_kind": "develop",
            "status": "completed",
        },
        {
            "id": "loop:skill:s",
            "name": "S",
            "loop_kind": "skill",
            "status": "pending",
            "skill_ref": "deploy-stack",
        },
        {
            "id": "concept:legacy",
            "name": "legacy topic",
            "loop_kind": None,
            "status": None,
        },
    ]
    eng = _Engine(concept_rows=rows, addressed=["loop:research:done"])
    out = {lp["id"]: lp for lp in active_loops(eng, limit=10)}
    assert "loop:research:a" in out  # research, unaddressed
    assert "loop:research:done" not in out  # research, ADDRESSED_BY → resolved
    assert "loop:develop:b" in out  # develop, non-terminal
    assert "loop:develop:fin" not in out  # develop, completed → terminal
    assert "loop:skill:s" in out and out["loop:skill:s"]["kind"] == "skill"
    assert "concept:legacy" in out  # bare Concept → treated as research loop
    assert out["loop:develop:b"]["validation_cmd"] == "pytest"


def test_terminal_status_set():
    assert "completed" in TERMINAL_STATUS and "running" not in TERMINAL_STATUS
