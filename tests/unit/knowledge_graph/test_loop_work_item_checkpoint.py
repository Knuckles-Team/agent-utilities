"""Resumable Loop execution through the sole native WorkItem authority."""

from __future__ import annotations

import asyncio

from agent_utilities.knowledge_graph.research.loop_controller import LoopController
from agent_utilities.knowledge_graph.research.loops import claim_loop, submit_loop
from agent_utilities.orchestration import work_item as wi
from tests.unit.knowledge_graph.test_loops import LoopEngine, _authority


def _submit(engine: LoopEngine, *, loop_id: str, kind: str = "develop") -> dict:
    with _authority():
        return submit_loop(
            engine,
            "bounded objective",
            loop_id=loop_id,
            kind=kind,
            validation_cmd="validate" if kind == "develop" else "",
            skill_ref="runtime-governance" if kind == "skill" else "",
        )


def test_run_loop_checkpoints_and_commits_on_one_work_item():
    engine = LoopEngine()
    loop = _submit(engine, loop_id="loop:develop:checkpoint")
    calls = 0

    def develop(_cmd, _cwd):
        nonlocal calls
        calls += 1
        return calls >= 3, f"try {calls}"

    with _authority():
        result = asyncio.run(
            LoopController(engine, develop_runner=develop).run_loop(
                loop, max_iterations=10
            )
        )
    item = wi.get_work_item(engine, wi.loop_work_item_id(loop["id"]))
    assert result == {"id": loop["id"], "status": "completed", "iterations": 3}
    assert calls == 3
    assert item is not None
    assert item["checkpoint_id"] == "checkpoint:iteration:3"
    assert item["status"] == "succeeded"


def test_expired_lease_resumes_after_last_fenced_checkpoint():
    engine = LoopEngine()
    loop = _submit(engine, loop_id="loop:develop:resume")
    item_id = wi.loop_work_item_id(loop["id"])
    with _authority():
        assert claim_loop(engine, loop["id"])
        claim = wi.current_work_item_claim(engine, item_id)
        assert claim is not None
        assert wi.checkpoint_work_item(engine, item_id, claim, "checkpoint:iteration:2")
    engine.nodes[item_id]["lease_expires_at"] = 0.0
    calls = 0

    def develop(_cmd, _cwd):
        nonlocal calls
        calls += 1
        return True, "done"

    with _authority():
        result = asyncio.run(
            LoopController(engine, develop_runner=develop).run_loop(
                loop, max_iterations=10
            )
        )
    assert result["iterations"] == 3
    assert calls == 1
    assert (
        wi.get_work_item(engine, item_id)["checkpoint_id"] == "checkpoint:iteration:3"
    )


def test_terminal_replay_is_a_noop():
    engine = LoopEngine()
    loop = _submit(engine, loop_id="loop:develop:terminal")
    calls = 0

    def develop(_cmd, _cwd):
        nonlocal calls
        calls += 1
        return True, "done"

    controller = LoopController(engine, develop_runner=develop)
    with _authority():
        first = asyncio.run(controller.run_loop(loop, max_iterations=3))
        second = asyncio.run(controller.run_loop(loop, max_iterations=3))
    assert first["status"] == "completed"
    assert second["skipped"] is True
    assert second["iterations"] == 1
    assert calls == 1


def test_corrigibility_commits_cancelled_without_running_a_step():
    engine = LoopEngine()
    loop = _submit(engine, loop_id="loop:develop:cancel")
    calls = 0

    def develop(_cmd, _cwd):
        nonlocal calls
        calls += 1
        return False, "pending"

    with _authority():
        result = asyncio.run(
            LoopController(engine, develop_runner=develop).run_loop(
                loop,
                max_iterations=5,
                desired_state=lambda: "kill",
            )
        )
    item = wi.get_work_item(engine, wi.loop_work_item_id(loop["id"]))
    assert result["interrupted"] is True and result["status"] == "cancelled"
    assert calls == 0
    assert item is not None and item["status"] == "cancelled"


def test_skill_loop_uses_the_same_checkpoint_path():
    engine = LoopEngine()
    loop = _submit(engine, loop_id="loop:skill:checkpoint", kind="skill")
    with _authority():
        result = asyncio.run(
            LoopController(
                engine, skill_runner=lambda _ref, _objective: (True, "ran")
            ).run_loop(loop, max_iterations=5)
        )
    item = wi.get_work_item(engine, wi.loop_work_item_id(loop["id"]))
    assert result["status"] == "completed" and result["iterations"] == 1
    assert item is not None and item["checkpoint_id"] == "checkpoint:iteration:1"
