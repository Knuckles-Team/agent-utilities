"""Loop engine — develop/skill execution stages + unified dispatch (CONCEPT:KG-2.78 L3)."""

from __future__ import annotations

from agent_utilities.knowledge_graph.research.loop_controller import LoopController
from agent_utilities.knowledge_graph.research.loops import (
    mark_loop_status,
    submit_loop,
)


class _Engine:
    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self._rows: list[dict] = []

    def add_node(self, nid, ntype, properties=None):
        # upsert/merge by id (mirrors backend upsert semantics)
        cur = self.nodes.get(nid, {})
        cur.update({"type": ntype, **(properties or {})})
        self.nodes[nid] = cur

    def query_cypher(self, q, params=None):
        return self._rows


def _controller(eng, **kw):
    return LoopController(eng, **kw)


# ── status lifecycle ────────────────────────────────────────────────────────


def test_mark_loop_status_upserts_lifecycle():
    eng = _Engine()
    submit_loop(eng, "do a thing", kind="develop", loop_id="loop:develop:x")
    assert mark_loop_status(eng, "loop:develop:x", "completed", iteration=2)
    node = eng.nodes["loop:develop:x"]
    assert node["status"] == "completed" and node["iteration"] == 2
    # objective preserved through the status upsert (merge, not replace)
    assert node["objective"] == "do a thing"


# ── develop stage ───────────────────────────────────────────────────────────


def test_develop_completes_on_validation_success():
    eng = _Engine()
    c = _controller(eng, develop_runner=lambda cmd, cwd: (True, "exit=0"))
    out = c._run_execute_loops(
        [{"id": "loop:develop:x", "kind": "develop", "validation_cmd": "pytest -q"}]
    )
    assert out["develop"] == 1 and out["completed"] == 1
    assert eng.nodes["loop:develop:x"]["status"] == "completed"


def test_develop_stays_active_on_validation_failure():
    eng = _Engine()
    c = _controller(eng, develop_runner=lambda cmd, cwd: (False, "exit=1"))
    out = c._run_execute_loops(
        [{"id": "loop:develop:x", "kind": "develop", "validation_cmd": "pytest -q"}]
    )
    assert out["completed"] == 0
    assert eng.nodes["loop:develop:x"]["status"] == "pending"


# ── skill stage ─────────────────────────────────────────────────────────────


def test_skill_completes_on_runner_success():
    eng = _Engine()
    c = _controller(eng, skill_runner=lambda ref, obj: (True, "ran"))
    out = c._run_execute_loops(
        [
            {
                "id": "loop:skill:y",
                "kind": "skill",
                "skill_ref": "deploy",
                "objective": "go",
            }
        ]
    )
    assert out["skill"] == 1 and out["completed"] == 1
    assert eng.nodes["loop:skill:y"]["status"] == "completed"


def test_skill_without_ref_fails():
    eng = _Engine()
    _controller(eng)._run_execute_loops([{"id": "loop:skill:z", "kind": "skill"}])
    assert eng.nodes["loop:skill:z"]["status"] == "failed"


# ── one hot path: a single cycle advances research + develop + skill ─────────


def test_run_one_cycle_advances_non_research_loops():
    eng = _Engine()
    # active_loops returns a develop + skill loop (research path is a no-op w/o sources)
    eng._rows = []  # addressed query empty
    c = _controller(
        eng,
        develop_runner=lambda cmd, cwd: (True, "ok"),
        skill_runner=lambda ref, obj: (True, "ok"),
    )
    topics = [
        {"id": "loop:develop:a", "kind": "develop", "validation_cmd": "true"},
        {"id": "loop:skill:b", "kind": "skill", "skill_ref": "s"},
    ]
    rep = c.run_one_cycle(
        topics=topics,
        assimilate=False,
        reason=False,
        synthesize=False,
        breadth=False,
    )
    assert rep["executed"]["completed"] == 2
    assert {r["status"] for r in rep["executed"]["results"]} == {"completed"}
