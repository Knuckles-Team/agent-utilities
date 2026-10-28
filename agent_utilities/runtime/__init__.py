"""Developer workspaces and the lazy host-facing run contract facade.

Pillars: OS-5.33 (runtime/backends), ORCH-1.46 (action/observation protocol + dispatcher),
the provenance sink port. See ``.specify/design/os-5.33-developer-workspace-runtime``.

Typical use::

    from agent_utilities.runtime import create_workspace
    from agent_utilities.runtime.events import CmdRunAction, FileEditAction, TestRunAction

    ws = create_workspace(run_id="abc123")
    async with ws:
        await ws.act(CmdRunAction(command="git clone <repo> ."))
        await ws.act(FileEditAction(path="src/x.py", old="foo", new="bar"))
        obs = await ws.act(TestRunAction(selector="tests/test_x.py"))
        print(obs.passed, obs.failed)
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_utilities.graph import (
        AgentRunExecutor,
        DependencyPlan,
        LiveSharedBudget,
        RunBudget,
        RunFailed,
        RunPlan,
        RunPlanFailure,
        RunRequest,
        RunTask,
        StopRule,
        default_run_plan,
    )

    from .computer_use_tier import ComputerUseDriver

from .docker_workspace import DockerWorkspace
from .local_workspace import LocalWorkspace
from .provenance import ProvenanceMirror, ProvenanceSink
from .workspace import DevWorkspace, PolicyGate, WorkspaceBackend

_RUN_EXPORTS = (
    "AgentRunExecutor",
    "DependencyPlan",
    "LiveSharedBudget",
    "RunBudget",
    "RunFailed",
    "RunPlan",
    "RunPlanFailure",
    "RunRequest",
    "RunTask",
    "StopRule",
    "default_run_plan",
)

__all__ = [
    "AgentRunExecutor",
    "DependencyPlan",
    "LiveSharedBudget",
    "RunBudget",
    "RunFailed",
    "RunPlan",
    "RunPlanFailure",
    "RunRequest",
    "RunTask",
    "StopRule",
    "default_run_plan",
    "DevWorkspace",
    "DockerWorkspace",
    "LocalWorkspace",
    "ProvenanceMirror",
    "ProvenanceSink",
    "PolicyGate",
    "WorkspaceBackend",
    "create_workspace",
]


def __getattr__(name: str) -> Any:
    """Expose the same run contracts without eagerly loading the model runtime."""
    if name in _RUN_EXPORTS:
        from agent_utilities import graph

        return getattr(graph, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def create_workspace(
    run_id: str | None = None,
    *,
    prefer_docker: bool = True,
    actor: str | None = None,
    policy_gate: PolicyGate | None = None,
    image: str = "python:3.14-slim",
    root: str | None = None,
    computer_use: ComputerUseDriver | None = None,
) -> DevWorkspace:
    """Build a :class:`DevWorkspace`, picking the isolation tier when available.

    Selection mirrors the RLM router's "fastest acceptable tier" idea: prefer Docker isolation
    when a daemon is reachable, else fall back to the always-available local backend (the
    zero-infra floor) so the runtime works out-of-the-box.
    """
    rid = run_id or uuid.uuid4().hex
    backend: WorkspaceBackend
    if prefer_docker and root is None:
        candidate = DockerWorkspace(run_id=rid, image=image)
        backend = candidate if candidate.is_available() else LocalWorkspace()
    else:
        backend = LocalWorkspace(root=root)
    return DevWorkspace(
        backend,
        run_id=rid,
        actor=actor,
        policy_gate=policy_gate,
        computer_use=computer_use,
    )
