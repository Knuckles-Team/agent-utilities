"""CONCEPT:AU-OS.scaling.bridge-developer-workspace-mutating — Developer-Workspace lifecycle: stateful shell, file ops, tests, policy gate.

Runs against the local backend (always available — the zero-infra floor), so it needs no Docker
daemon and is not marked ``live``.
"""

from __future__ import annotations

from agent_utilities.runtime import DevWorkspace, LocalWorkspace, create_workspace
from agent_utilities.runtime.events import (
    CmdRunAction,
    ErrorObservation,
    FileEditAction,
    FileReadAction,
    FileWriteAction,
    TestRunAction,
)


def _local_ws(**kw) -> DevWorkspace:
    return DevWorkspace(LocalWorkspace(), run_id="testrun", **kw)


async def test_stateful_cwd_persists_across_commands():
    async with _local_ws() as ws:
        out = await ws.act(CmdRunAction(command="mkdir -p pkg && cd pkg"))
        assert out.exit_code == 0
        # the cd must persist to the next, independent command
        out2 = await ws.act(CmdRunAction(command="pwd"))
        assert out2.stdout.strip().endswith("/pkg")
        assert out2.cwd.endswith("/pkg")


async def test_env_applied_to_commands():
    async with _local_ws() as ws:
        ws.state.env["AU_TEST_VAR"] = "42"
        out = await ws.act(CmdRunAction(command='echo "val=$AU_TEST_VAR"'))
        assert "val=42" in out.stdout


async def test_file_write_read_edit_cycle():
    async with _local_ws() as ws:
        await ws.act(FileWriteAction(path="m.py", content="a = 1\nb = 2\n"))
        read = await ws.act(FileReadAction(path="m.py"))
        assert read.content == "a = 1\nb = 2\n"
        edit = await ws.act(FileEditAction(path="m.py", old="a = 1", new="a = 100"))
        assert edit.applied and edit.replacements == 1
        assert "a = 100" in (await ws.act(FileReadAction(path="m.py"))).content


async def test_edit_nonunique_without_replace_all_is_error():
    async with _local_ws() as ws:
        await ws.act(FileWriteAction(path="d.py", content="x\nx\n"))
        obs = await ws.act(FileEditAction(path="d.py", old="x", new="y"))
        assert isinstance(obs, ErrorObservation)
        assert "not unique" in obs.message


async def test_test_run_parses_pytest_summary():
    async with _local_ws() as ws:
        await ws.act(
            FileWriteAction(
                path="test_sample.py",
                content="def test_pass():\n    assert 1 == 1\n\ndef test_fail():\n    assert 1 == 2\n",
            )
        )
        result = await ws.act(TestRunAction(selector="test_sample.py"))
        assert result.passed == 1
        assert result.failed == 1
        assert result.exit_code != 0


async def test_policy_gate_denies_mutating_action():
    def deny_writes(name: str) -> tuple[bool, str]:
        return (name != "workspace.cmd", "shell disabled in this run")

    async with _local_ws(policy_gate=deny_writes) as ws:
        denied = await ws.act(CmdRunAction(command="echo nope"))
        assert isinstance(denied, ErrorObservation)
        assert "denied by policy" in denied.message
        # a write is still allowed by this gate
        ok = await ws.act(FileWriteAction(path="ok.txt", content="hi"))
        assert ok.kind == "file_write_ok"


async def test_steps_increment_and_stamp():
    async with _local_ws() as ws:
        o1 = await ws.act(CmdRunAction(command="true"))
        o2 = await ws.act(CmdRunAction(command="true"))
        assert (o1.step, o2.step) == (1, 2)
        assert o1.run_id == "testrun"


def test_create_workspace_falls_back_to_local(monkeypatch):
    # force docker unavailable -> local floor
    from agent_utilities.runtime import docker_workspace

    monkeypatch.setattr(
        docker_workspace.DockerWorkspace, "is_available", lambda self: False
    )
    ws = create_workspace(prefer_docker=True)
    assert ws.backend.name == "local"
