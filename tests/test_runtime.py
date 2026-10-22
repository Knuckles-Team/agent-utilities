"""Developer workspace: action/observation protocol, stateful shell, tiers, policy gate."""

from __future__ import annotations

import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from agent_utilities.runtime import DevWorkspace, LocalWorkspace, create_workspace
from agent_utilities.runtime.bridge import ActionDispatcher
from agent_utilities.runtime.browser_tier import NullBrowserDriver
from agent_utilities.runtime.computer_use_tier import NullComputerUseDriver
from agent_utilities.runtime.events import (
    ACTION_ADAPTER,
    OBSERVATION_ADAPTER,
    WORKSPACE_MUTATING_ACTIONS,
    BrowseAction,
    BrowserObservation,
    CmdOutputObservation,
    CmdRunAction,
    ComputerUseAction,
    ErrorObservation,
    FileEditAction,
    FileReadAction,
    FileWriteAction,
    ScreenObservation,
    TestRunAction,
    mutating_action_name,
)

_ACTIONS = [
    (CmdRunAction(command="pytest"), "workspace.cmd"),
    (FileReadAction(path="a.py", start=1, end=10), None),
    (FileWriteAction(path="a.py", content="x=1"), "workspace.write"),
    (FileEditAction(path="a.py", old="x=1", new="x=2"), "workspace.edit"),
    (TestRunAction(selector="tests/test_a.py::test_b"), "workspace.cmd"),
    (BrowseAction(url="http://x"), "workspace.browse"),
]


@pytest.mark.parametrize(("action", "gate_name"), _ACTIONS)
def test_action_roundtrip_and_gate_name(action, gate_name):
    back = ACTION_ADAPTER.validate_python(action.model_dump())
    assert type(back) is type(action) and back == action
    assert mutating_action_name(action) == gate_name
    with pytest.raises(Exception):  # noqa: B017 - frozen model rejects mutation
        action.kind = "other"


def test_observation_roundtrip_and_mutating_set():
    back = OBSERVATION_ADAPTER.validate_python(
        CmdOutputObservation(exit_code=0, stdout="hi", cwd="/x").model_dump()
    )
    assert isinstance(back, CmdOutputObservation) and back.exit_code == 0
    assert WORKSPACE_MUTATING_ACTIONS == {
        "workspace.cmd",
        "workspace.write",
        "workspace.edit",
        "workspace.browse",
        "workspace.computer_use",
    }


async def test_shell_state_persists_and_steps_stamp(workspace):
    workspace.state.env["AU_TEST_VAR"] = "42"
    first = await workspace.act(CmdRunAction(command="mkdir -p pkg && cd pkg"))
    second = await workspace.act(CmdRunAction(command='pwd; echo "val=$AU_TEST_VAR"'))
    assert first.exit_code == 0
    assert second.cwd.endswith("/pkg") and "val=42" in second.stdout
    assert (first.step, second.step) == (1, 2) and first.run_id == "testrun"


async def test_file_write_read_edit(workspace):
    await workspace.act(FileWriteAction(path="m.py", content="a = 1\nb = 2\nx\nx\n"))
    edit = await workspace.act(FileEditAction(path="m.py", old="a = 1", new="a = 100"))
    assert edit.applied and edit.replacements == 1
    assert "a = 100" in (await workspace.act(FileReadAction(path="m.py"))).content
    dup = await workspace.act(FileEditAction(path="m.py", old="x", new="y"))
    assert isinstance(dup, ErrorObservation) and "not unique" in dup.message


async def test_write_rejects_symlink_escape(tmp_path):
    root, outside = tmp_path / "ws", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "redirect").symlink_to(outside, target_is_directory=True)
    async with DevWorkspace(LocalWorkspace(root=root), run_id="s") as ws:
        result = await ws.act(FileWriteAction(path="redirect/x.txt", content="no"))
    assert isinstance(result, ErrorObservation)
    assert not (outside / "x.txt").exists()


async def test_test_run_parses_pytest_summary(workspace):
    await workspace.act(
        FileWriteAction(
            path="test_sample.py",
            content="def test_pass():\n    assert True\n\ndef test_fail():\n    assert False\n",
        )
    )
    result = await workspace.act(TestRunAction(selector="test_sample.py"))
    assert (result.passed, result.failed) == (1, 1) and result.exit_code != 0


async def test_policy_gate_denies_only_gated_actions(tmp_path):
    def deny_shell(name: str) -> tuple[bool, str]:
        return (name != "workspace.cmd", "shell disabled")

    async with DevWorkspace(
        LocalWorkspace(root=tmp_path), run_id="p", policy_gate=deny_shell
    ) as ws:
        denied = await ws.act(CmdRunAction(command="echo nope"))
        ok = await ws.act(FileWriteAction(path="ok.txt", content="hi"))
    assert isinstance(denied, ErrorObservation) and "denied by policy" in denied.message
    assert ok.kind == "file_write_ok"


async def test_optional_tiers_refuse_without_driver_and_route_with_one(workspace):
    browse = await workspace.act(BrowseAction(url="http://example.com"))
    screen = await NullComputerUseDriver().run(ComputerUseAction(op="capture"))
    assert isinstance(browse, BrowserObservation) and "not provisioned" in browse.error
    assert isinstance(screen, ScreenObservation) and "not provisioned" in screen.error
    assert (await NullBrowserDriver().browse(BrowseAction(url="u"))).error

    class _Driver:
        async def browse(self, action):
            return BrowserObservation(url=action.url, status=200, text="hello")

        async def run(self, action):
            return ScreenObservation(image_b64="abc")

    ws = DevWorkspace(
        LocalWorkspace(root=workspace.backend.root), run_id="d", browser=_Driver()
    )
    async with ws:
        assert (await ws.act(BrowseAction(url="http://e"))).text == "hello"
    obs = await ActionDispatcher().dispatch(
        ComputerUseAction(op="capture"),
        backend=None,
        state=None,
        computer_use=_Driver(),
    )
    assert obs.image_b64 == "abc"


def test_create_workspace_falls_back_to_local(monkeypatch):
    from agent_utilities.runtime import docker_workspace

    monkeypatch.setattr(
        docker_workspace.DockerWorkspace, "is_available", lambda self: False
    )
    assert create_workspace(prefer_docker=True).backend.name == "local"


def test_docker_workspace_host_directory_is_owner_only():
    from agent_utilities.runtime.docker_workspace import DockerWorkspace

    ws = DockerWorkspace(run_id="perm")
    try:
        assert stat.S_IMODE(ws.root.stat().st_mode) == 0o700
    finally:
        shutil.rmtree(ws.root)


@pytest.mark.parametrize(
    ("script", "timeout", "expected"),
    [
        ("echo out; echo err >&2", 5.0, (0, "out\n", "err\n")),
        ("exit 3", 5.0, (3, "", "")),
        ("sleep 5", 0.1, (124, "", "command timed out after 0.1s")),
    ],
)
async def test_run_shell_process_returns_exit_code_and_output(
    script, timeout, expected
):
    from agent_utilities.runtime.workspace import run_shell_process

    assert await run_shell_process(["bash", "-c", script], timeout) == expected


RUNTIME_RUN_EXPORTS = (
    "AgentRunExecutor",
    "DependencyPlan",
    "RunBudget",
    "RunFailed",
    "RunPlan",
    "RunPlanFailure",
    "RunTask",
    "StopRule",
    "default_run_plan",
)


@pytest.mark.spec("AU-BOUNDARY-R043.7")
@pytest.mark.parametrize("name", RUNTIME_RUN_EXPORTS)
def test_runtime_exports_the_same_public_run_objects(name):
    from agent_utilities import graph, runtime

    assert name in runtime.__all__
    assert name in graph.__all__
    assert getattr(runtime, name) is getattr(graph, name)


@pytest.mark.spec("AU-BOUNDARY-R043.7")
def test_runtime_run_exports_are_lazy_and_cycle_safe_in_a_fresh_process():
    script = """
import os
import sys
from agent_utilities import runtime

assert "agent_utilities.graph" not in sys.modules
resolved = {name: getattr(runtime, name) for name in sys.argv[1:]}
from agent_utilities import graph

for name, exported in resolved.items():
    assert name in runtime.__all__
    assert exported is getattr(graph, name), name
# This probe checks imports, not third-party interpreter-shutdown hooks.
print("runtime aliases resolved", flush=True)
os._exit(0)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, *RUNTIME_RUN_EXPORTS],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "runtime aliases resolved"
