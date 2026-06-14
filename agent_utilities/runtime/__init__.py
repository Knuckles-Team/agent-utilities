"""Developer-Workspace Runtime — the persistent, sandboxed substrate for the SWE agent.

Pillars: OS-5.33 (runtime/backends), ORCH-1.46 (action/observation protocol + dispatcher),
KG-2.64 (provenance mirroring). See ``.specify/design/os-5.33-developer-workspace-runtime``.

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

from .docker_workspace import DockerWorkspace
from .local_workspace import LocalWorkspace
from .policy import action_policy_gate
from .provenance import ProvenanceMirror
from .workspace import DevWorkspace, PolicyGate, WorkspaceBackend

__all__ = [
    "DevWorkspace",
    "DockerWorkspace",
    "LocalWorkspace",
    "ProvenanceMirror",
    "PolicyGate",
    "WorkspaceBackend",
    "action_policy_gate",
    "create_workspace",
]


def create_workspace(
    run_id: str | None = None,
    *,
    prefer_docker: bool = True,
    actor: str | None = None,
    policy_gate: PolicyGate | None = None,
    image: str = "python:3.11-slim",
    root: str | None = None,
) -> DevWorkspace:
    """Build a :class:`DevWorkspace`, picking the isolation tier when available.

    Selection mirrors the RLM router's "fastest acceptable tier" idea: prefer Docker isolation
    when a daemon is reachable, else fall back to the always-available local backend (the
    zero-infra floor) so the runtime works out-of-the-box.
    """
    rid = run_id or uuid.uuid4().hex[:12]
    backend: WorkspaceBackend
    if prefer_docker and root is None:
        candidate = DockerWorkspace(run_id=rid, image=image)
        backend = candidate if candidate.is_available() else LocalWorkspace()
    else:
        backend = LocalWorkspace(root=root)
    return DevWorkspace(backend, run_id=rid, actor=actor, policy_gate=policy_gate)
