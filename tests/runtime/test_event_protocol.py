"""CONCEPT:ORCH-1.46 — Action/Observation event protocol round-trip + policy mapping."""

from __future__ import annotations

from agent_utilities.runtime.events import (
    ACTION_ADAPTER,
    OBSERVATION_ADAPTER,
    WORKSPACE_MUTATING_ACTIONS,
    CmdOutputObservation,
    CmdRunAction,
    FileEditAction,
    FileReadAction,
    FileWriteAction,
    TestRunAction,
    mutating_action_name,
)


def test_actions_are_frozen():
    a = CmdRunAction(command="ls")
    try:
        a.command = "rm -rf /"  # type: ignore[misc]
    except Exception:  # noqa: BLE001
        return
    raise AssertionError("action should be immutable (frozen)")


def test_action_union_roundtrip_preserves_subtype():
    for action in (
        CmdRunAction(command="pytest"),
        FileReadAction(path="a.py", start=1, end=10),
        FileWriteAction(path="a.py", content="x=1"),
        FileEditAction(path="a.py", old="x=1", new="x=2"),
        TestRunAction(selector="tests/test_a.py::test_b"),
    ):
        wire = action.model_dump()
        back = ACTION_ADAPTER.validate_python(wire)
        assert type(back) is type(action)
        assert back == action


def test_observation_union_roundtrip():
    obs = CmdOutputObservation(exit_code=0, stdout="hi", cwd="/x")
    back = OBSERVATION_ADAPTER.validate_python(obs.model_dump())
    assert isinstance(back, CmdOutputObservation)
    assert back.exit_code == 0


def test_mutating_action_mapping():
    assert mutating_action_name(CmdRunAction(command="x")) == "workspace.cmd"
    assert (
        mutating_action_name(FileWriteAction(path="a", content="b"))
        == "workspace.write"
    )
    assert (
        mutating_action_name(FileEditAction(path="a", old="b", new="c"))
        == "workspace.edit"
    )
    assert mutating_action_name(TestRunAction()) == "workspace.cmd"
    # read-only actions bypass the gate
    assert mutating_action_name(FileReadAction(path="a")) is None
    assert WORKSPACE_MUTATING_ACTIONS == {
        "workspace.cmd",
        "workspace.write",
        "workspace.edit",
        "workspace.browse",
    }
