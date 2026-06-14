"""CONCEPT:AHE-3.22 — SWE-bench harness end-to-end on a local fixture (no LLM, no Docker)."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from agent_utilities.harness.swebench_corpus import SweBenchInstance, load_instances
from agent_utilities.harness.swebench_harness import (
    aggregate_report,
    evaluate_instance,
    is_resolved,
    run_suite,
)
from agent_utilities.runtime import DevWorkspace, LocalWorkspace
from agent_utilities.runtime.events import FileEditAction


def _new_file_test_patch() -> str:
    """Generate a valid git new-file patch for test_calc.py via a throwaway repo."""
    d = Path(tempfile.mkdtemp(prefix="au-patch-"))
    subprocess.run(["git", "init", "-q"], cwd=d, check=True)
    (d / "test_calc.py").write_text(
        "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )
    subprocess.run(["git", "add", "test_calc.py"], cwd=d, check=True)
    return subprocess.run(
        ["git", "diff", "--cached"], cwd=d, capture_output=True, text=True, check=True
    ).stdout


def _fixture_instance(resolved_ok: bool) -> SweBenchInstance:
    # setup seeds a buggy repo: add() subtracts. The gold test (test_patch) asserts add(2,3)==5.
    return SweBenchInstance(
        instance_id=f"calc-{'fix' if resolved_ok else 'nofix'}",
        repo="fixtures/calc",
        base_commit="HEAD",
        problem_statement="add() returns a-b but should return a+b",
        fail_to_pass=["test_calc.py::test_add"],
        pass_to_pass=[],
        test_patch=_new_file_test_patch(),
        setup_commands=[
            "git init -q .",
            "printf 'def add(a, b):\\n    return a - b\\n' > calc.py",
            "git add -A && git -c user.email=t@t.io -c user.name=t commit -q -m init",
        ],
    )


def test_is_resolved_logic():
    assert is_resolved(1, 1, 0, 0, 0, 0) is True
    assert is_resolved(0, 1, 0, 0, 1, 0) is False  # a fail_to_pass still fails
    assert is_resolved(1, 1, 2, 3, 0, 1) is False  # a pass_to_pass regressed


async def _good_solver(instance, workspace):
    await workspace.act(
        FileEditAction(path="calc.py", old="return a - b", new="return a + b")
    )
    return "patch"


async def _noop_solver(instance, workspace):
    return ""  # leaves the bug in place


async def test_resolved_instance_scores_resolved():
    inst = _fixture_instance(resolved_ok=True)
    ws = DevWorkspace(LocalWorkspace(), run_id="swe-ok")
    await ws.start()
    try:
        result = await evaluate_instance(inst, workspace=ws, solve=_good_solver)
        assert result.resolved is True
        assert result.fail_to_pass_passed == 1
    finally:
        await ws.stop()


async def test_unresolved_instance_scores_unresolved():
    inst = _fixture_instance(resolved_ok=False)
    ws = DevWorkspace(LocalWorkspace(), run_id="swe-no")
    await ws.start()
    try:
        result = await evaluate_instance(inst, workspace=ws, solve=_noop_solver)
        assert result.resolved is False
    finally:
        await ws.stop()


async def test_run_suite_aggregate_report():
    instances = [_fixture_instance(True), _fixture_instance(False)]
    n = iter([_good_solver, _noop_solver])

    async def mixed_solver(inst, ws):
        return await next(n)(inst, ws)

    report = await run_suite(
        instances,
        workspace_factory=lambda inst: DevWorkspace(
            LocalWorkspace(), run_id=inst.instance_id
        ),
        solve=mixed_solver,
    )
    rep = report.report
    assert rep["total"] == 2
    assert rep["resolved"] == 1
    assert rep["resolved_rate"] == 0.5
    assert "fixtures/calc" in rep["by_repo"]


def test_load_instances_parses_swebench_fields():
    rows = [
        {
            "instance_id": "x__1",
            "repo": "x/y",
            "base_commit": "abc",
            "problem_statement": "p",
            "FAIL_TO_PASS": '["t::a", "t::b"]',
            "PASS_TO_PASS": '["t::c"]',
        }
    ]
    insts = load_instances(rows)
    assert insts[0].fail_to_pass == ["t::a", "t::b"]
    assert insts[0].pass_to_pass == ["t::c"]


def test_aggregate_report_empty():
    rep = aggregate_report([])
    assert rep["total"] == 0 and rep["resolved_rate"] == 0.0
