"""Live-path execution tests for native WorkItem-backed Loops."""

from __future__ import annotations

from agent_utilities.knowledge_graph.research.loop_controller import (
    LoopController,
    _default_develop_runner,
)
from agent_utilities.knowledge_graph.research.loops import claim_loop, submit_loop
from agent_utilities.orchestration import work_item as wi
from tests.unit.knowledge_graph.test_loops import LoopEngine, _authority


def _submit(engine: LoopEngine, objective: str, **kwargs):
    with _authority():
        return submit_loop(engine, objective, **kwargs)


def test_develop_completion_commits_the_native_work_item():
    engine = LoopEngine()
    loop = _submit(
        engine,
        "validate",
        kind="develop",
        validation_cmd="pytest -q",
        loop_id="loop:develop:complete",
    )
    controller = LoopController(engine, develop_runner=lambda _cmd, _cwd: (True, "ok"))
    with _authority():
        report = controller._run_execute_loops([loop])

    item = wi.get_work_item(engine, wi.loop_work_item_id(loop["id"]))
    assert report["completed"] == 1
    assert item is not None and item["status"] == "succeeded"
    assert "status" not in engine.nodes[loop["id"]]


def test_develop_pending_retains_the_native_lease_for_the_driver():
    engine = LoopEngine()
    loop = _submit(
        engine,
        "retry",
        kind="develop",
        validation_cmd="pytest -q",
        loop_id="loop:develop:pending",
    )
    controller = LoopController(engine, develop_runner=lambda _cmd, _cwd: (False, "no"))
    with _authority():
        report = controller._run_execute_loops([loop])

    item = wi.get_work_item(engine, wi.loop_work_item_id(loop["id"]))
    assert report["completed"] == 0
    assert item is not None and item["status"] == "leased"


def test_skill_completion_uses_the_same_work_item_path():
    engine = LoopEngine()
    loop = _submit(
        engine,
        "deploy",
        kind="skill",
        skill_ref="runtime-governance",
        loop_id="loop:skill:complete",
    )
    controller = LoopController(engine, skill_runner=lambda _ref, _obj: (True, "ran"))
    with _authority():
        report = controller._run_execute_loops([loop])
    item = wi.get_work_item(engine, wi.loop_work_item_id(loop["id"]))
    assert report["skill"] == 1 and report["completed"] == 1
    assert item is not None and item["status"] == "succeeded"


def test_second_driver_cannot_execute_an_owned_loop():
    engine = LoopEngine()
    loop = _submit(
        engine,
        "single owner",
        kind="develop",
        validation_cmd="true",
        loop_id="loop:develop:owned",
    )
    with _authority():
        assert claim_loop(engine, loop["id"])
    calls = 0

    def runner(_cmd, _cwd):
        nonlocal calls
        calls += 1
        return True, "ok"

    with _authority():
        report = LoopController(engine, develop_runner=runner)._run_execute_loops(
            [loop]
        )
    assert report["skipped"] == 1
    assert calls == 0


def test_default_develop_runner_requires_explicit_host_permission(monkeypatch):
    from agent_utilities.core.config import config

    monkeypatch.setattr(config, "kg_loop_allow_host_validation", False)
    ok, output = _default_develop_runner("pytest -q", ".")
    assert ok is False
    assert "disabled" in output


def test_host_validation_rejects_shell_wrappers(monkeypatch):
    from agent_utilities.core.config import config

    monkeypatch.setattr(config, "kg_loop_allow_host_validation", True)
    monkeypatch.setattr(config, "kg_loop_host_validation_executables", "pytest,sh,bash")
    ok, output = _default_develop_runner("sh -c 'echo unsafe'", ".")
    assert ok is False
    assert "not operator-allowlisted" in output


def test_develop_runner_output_is_privacy_sanitized():
    controller = LoopController(
        LoopEngine(),
        develop_runner=lambda _cmd, _cwd: (
            False,
            "contact local.person@example.com with token sk-example-secret-value-123456",
        ),
    )
    output = controller._advance_develop(
        {"kind": "develop", "validation_cmd": "pytest -q"}
    )["output"]
    assert "local.person@example.com" not in output
    assert "sk-example-secret-value-123456" not in output
