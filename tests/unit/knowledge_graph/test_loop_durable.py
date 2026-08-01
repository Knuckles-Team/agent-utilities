"""Resumable, corrigible ``LoopController.run_loop`` — cross-cutting checkpointing (KG-2.78/OS-5.16).

The ``durable=DurableExecutionManager(...)`` sidecar this file originally drove
is retired: durability now lives entirely on the Loop's native engine
``WorkItem`` (one fenced lease + checkpoint authority — CONCEPT:
AU-AHE.harness.loop-exit-conditions, see ``LoopController.run_loop``'s own
docstring: "Durability is unchanged: **resume** from the WorkItem's fenced
``checkpoint_id``; **one authority** ... no sidecar"). ``run_loop`` no longer
accepts a ``durable=`` keyword at all, and the toy ``_Engine`` stub this file
used (bare ``add_node``/``query_cypher``) cannot satisfy the native
``claim_loop``/``checkpoint_work_item`` contract those functions now require.

Rewritten against the same ``LoopEngine`` double + ``_authority`` session
harness the sibling suites already use for exactly this — ``test_loops.py``,
``test_loop_exits.py``, and ``test_loop_work_item_checkpoint.py`` — which
directly demonstrate the resume-after-crash, idempotent-terminal-replay,
human-interrupt, and skill-kind cases this file covers under the current
WorkItem-only model. Test names are kept for node-id stability.
"""

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
            skill_ref="deploy" if kind == "skill" else "",
        )


def _run(controller: LoopController, loop: dict, **kw) -> dict:
    with _authority():
        return asyncio.run(controller.run_loop(loop, **kw))


def test_run_loop_completes_develop_and_checkpoints():
    engine = LoopEngine()
    loop = _submit(engine, loop_id="loop:develop:g")
    calls = {"n": 0}

    def dev(cmd, cwd):
        calls["n"] += 1
        return (calls["n"] >= 3, f"try {calls['n']}")

    c = LoopController(engine, develop_runner=dev)
    res = _run(c, loop, max_iterations=10, sleep_s=0)
    assert res["status"] == "completed" and res["iterations"] == 3
    assert calls["n"] == 3
    item = wi.get_work_item(engine, wi.loop_work_item_id(loop["id"]))
    assert item is not None and item["status"] == "succeeded"


def test_run_loop_resumes_from_checkpoint():
    # A crash left the native WorkItem's lease expired with a fenced checkpoint
    # at iteration 2 -> resume continues from iteration 3, not from scratch
    # (mirrors test_loop_work_item_checkpoint.py's
    # test_expired_lease_resumes_after_last_fenced_checkpoint).
    engine = LoopEngine()
    loop = _submit(engine, loop_id="loop:develop:r")
    item_id = wi.loop_work_item_id(loop["id"])
    with _authority():
        assert claim_loop(engine, loop["id"])
        claim = wi.current_work_item_claim(engine, item_id)
        assert claim is not None
        assert wi.checkpoint_work_item(
            engine, item_id, claim, "checkpoint:iteration:2"
        )
    engine.nodes[item_id]["lease_expires_at"] = 0.0

    seen: list[str] = []

    def dev(cmd, cwd):
        seen.append(cmd)
        return (True, "ok")  # completes on the first iteration it actually runs

    c = LoopController(engine, develop_runner=dev)
    res = _run(c, loop, max_iterations=10, sleep_s=0)
    # resumed at iteration 3 (2 already fenced) → first executed iter is 3
    assert res["iterations"] == 3
    assert len(seen) == 1


def test_run_loop_idempotent_replay():
    # Re-running against the SAME already-completed WorkItem does not
    # re-execute completed iterations — a terminal replay is a no-op (mirrors
    # test_loop_work_item_checkpoint.py's test_terminal_replay_is_a_noop).
    engine = LoopEngine()
    loop = _submit(engine, loop_id="loop:develop:i")
    calls = {"n": 0}

    def dev(cmd, cwd):
        calls["n"] += 1
        return (calls["n"] >= 2, "x")

    c = LoopController(engine, develop_runner=dev)
    first = _run(c, loop, max_iterations=5, sleep_s=0)
    first_calls = calls["n"]
    second = _run(c, loop, max_iterations=5, sleep_s=0)
    assert first["status"] == "completed"
    assert second.get("skipped") is True
    # completed iterations returned cached results → runner not called again
    assert calls["n"] == first_calls


def test_run_loop_corrigible_interruption():
    engine = LoopEngine()
    loop = _submit(engine, loop_id="loop:develop:p")
    c = LoopController(engine, develop_runner=lambda cmd, cwd: (False, "x"))
    res = _run(c, loop, max_iterations=5, desired_state=lambda: "kill", sleep_s=0)
    assert res["interrupted"] is True
    assert res["status"] == "cancelled"  # kill → cancelled
    item = wi.get_work_item(engine, wi.loop_work_item_id(loop["id"]))
    assert item is not None and item["status"] == "cancelled"


def test_run_loop_drives_skill_kind():
    engine = LoopEngine()
    loop = _submit(engine, loop_id="loop:skill:s", kind="skill")
    c = LoopController(engine, skill_runner=lambda ref, obj: (True, "ran"))
    res = _run(c, loop, max_iterations=5, sleep_s=0)
    assert res["status"] == "completed" and res["iterations"] == 1
