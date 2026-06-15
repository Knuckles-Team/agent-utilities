"""Durable, resumable LoopController.run_loop — cross-cutting checkpointing (KG-2.78/OS-5.16)."""

from __future__ import annotations

import asyncio

from agent_utilities.knowledge_graph.research.loop_controller import LoopController
from agent_utilities.orchestration.durable_execution import DurableExecutionManager


class _Engine:
    def __init__(self):
        self.nodes: dict[str, dict] = {}

    def add_node(self, nid, ntype, properties=None):
        cur = self.nodes.get(nid, {})
        cur.update({"type": ntype, **(properties or {})})
        self.nodes[nid] = cur

    def query_cypher(self, q, params=None):
        return []


def _durable(tmp_path, sid):
    return DurableExecutionManager(session_id=sid, db_path=str(tmp_path / "ckpt.db"))


def test_run_loop_completes_develop_and_checkpoints(tmp_path):
    eng = _Engine()
    calls = {"n": 0}

    def dev(cmd, cwd):
        calls["n"] += 1
        return (calls["n"] >= 3, f"try {calls['n']}")

    c = LoopController(eng, develop_runner=dev)
    loop = {"id": "loop:develop:g", "kind": "develop", "validation_cmd": "x"}
    res = asyncio.run(
        c.run_loop(
            loop, max_iterations=10, sleep_s=0, durable=_durable(tmp_path, "loop:develop:g")
        )
    )
    assert res["status"] == "completed" and res["iterations"] == 3
    assert calls["n"] == 3
    assert eng.nodes["loop:develop:g"]["status"] == "completed"


def test_run_loop_resumes_from_checkpoint(tmp_path):
    # a crash left a PENDING checkpoint at iteration 3 → resume re-runs from 3, not 1
    dm = _durable(tmp_path, "loop:develop:r")
    dm.save_checkpoint("loop:develop:r:iter:3", {"iteration": 3}, status="PENDING")
    eng = _Engine()
    seen: list[str] = []

    def dev(cmd, cwd):
        seen.append(cmd)
        return (True, "ok")  # completes on the first iteration it actually runs

    c = LoopController(eng, develop_runner=dev)
    res = asyncio.run(
        c.run_loop(
            {"id": "loop:develop:r", "kind": "develop", "validation_cmd": "v"},
            max_iterations=10,
            sleep_s=0,
            durable=dm,
        )
    )
    # resumed at iteration 3 (2 already applied) → first executed iter is 3
    assert res["iterations"] == 3
    assert len(seen) == 1


def test_run_loop_idempotent_replay(tmp_path):
    # re-running with the same durable store does NOT re-execute completed iterations
    dm = _durable(tmp_path, "loop:develop:i")
    eng = _Engine()
    calls = {"n": 0}

    def dev(cmd, cwd):
        calls["n"] += 1
        return (calls["n"] >= 2, "x")

    c = LoopController(eng, develop_runner=dev)
    loop = {"id": "loop:develop:i", "kind": "develop", "validation_cmd": "v"}
    asyncio.run(c.run_loop(loop, max_iterations=5, sleep_s=0, durable=dm))
    first = calls["n"]
    # replay with a fresh status so the while-loop runs again
    asyncio.run(
        c.run_loop(
            {**loop, "status": "running"}, max_iterations=5, sleep_s=0, durable=dm
        )
    )
    # completed iterations returned cached results → runner not called again
    assert calls["n"] == first


def test_run_loop_corrigible_interruption(tmp_path):
    eng = _Engine()
    c = LoopController(eng, develop_runner=lambda cmd, cwd: (False, "x"))
    res = asyncio.run(
        c.run_loop(
            {"id": "loop:develop:p", "kind": "develop", "validation_cmd": "v"},
            max_iterations=5,
            desired_state=lambda: "kill",
            sleep_s=0,
            durable=_durable(tmp_path, "loop:develop:p"),
        )
    )
    assert res["interrupted"] is True
    assert res["status"] == "cancelled"  # kill → cancelled
    assert eng.nodes["loop:develop:p"]["status"] == "cancelled"


def test_run_loop_drives_skill_kind(tmp_path):
    eng = _Engine()
    c = LoopController(eng, skill_runner=lambda ref, obj: (True, "ran"))
    res = asyncio.run(
        c.run_loop(
            {"id": "loop:skill:s", "kind": "skill", "skill_ref": "deploy"},
            max_iterations=5,
            sleep_s=0,
            durable=_durable(tmp_path, "loop:skill:s"),
        )
    )
    assert res["status"] == "completed" and res["iterations"] == 1
