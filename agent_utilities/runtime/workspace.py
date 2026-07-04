"""CONCEPT:AU-OS.scaling.bridge-developer-workspace-mutating — Developer-Workspace Runtime: a long-lived, stateful workspace the SWE
agent (ORCH-1.47) drives via typed actions.

``DevWorkspace`` is the facade. It owns:

* a :class:`WorkspaceBackend` (the execution substrate — :class:`~.local_workspace.LocalWorkspace`
  for the zero-infra floor, :class:`~.docker_workspace.DockerWorkspace` for isolation),
* the shell :class:`~.bridge.WorkspaceState` (cwd/env carried across actions),
* the fail-closed policy gate (OS-5.24) for mutating actions, and
* the provenance mirror (KG-2.64) that records every action/observation into the KG.

Contract: ``await ws.start()`` once, then ``await ws.act(action)`` per step (each returns the
typed observation and is stamped with ``run_id``/``step``/``ts``/``actor``), then
``await ws.stop()``. ``act`` never raises for in-workspace failures — it returns an
:class:`~.events.ErrorObservation` the agent can read and retry.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .bridge import ActionDispatcher, WorkspaceState

if TYPE_CHECKING:
    from .browser_tier import BrowserDriver
    from .computer_use_tier import ComputerUseDriver
from .events import (  # noqa: F401  (re-exported for callers)
    Action,
    ErrorObservation,
    Observation,
    mutating_action_name,
)
from .provenance import ProvenanceMirror

# A policy gate: given the ActionPolicy action name (e.g. "workspace.cmd"), return
# (allowed, reason). Pluggable so the runtime stays decoupled from a concrete ActionPolicy.
PolicyGate = Callable[[str], "tuple[bool, str]"]


@runtime_checkable
class WorkspaceBackend(Protocol):
    """The execution substrate behind a :class:`DevWorkspace`.

    ``workdir`` is the path commands execute relative to *in the execution context* (the host
    path for local, ``/workspace`` for docker). ``root`` is the host path that maps to
    ``workdir`` — file operations happen host-side on ``root`` (a bind-mount for docker).
    """

    name: str
    workdir: str
    root: Path

    def is_available(self) -> bool: ...

    async def start(self) -> None: ...

    async def exec_shell(
        self, script: str, env: dict[str, str], timeout: float
    ) -> tuple[int, str, str]: ...

    async def stop(self) -> None: ...

    def exposed_url(self, port: int) -> str: ...


class DevWorkspace:
    """Stateful developer workspace — drive it with typed actions (OS-5.33)."""

    def __init__(
        self,
        backend: WorkspaceBackend,
        *,
        run_id: str,
        actor: str | None = None,
        policy_gate: PolicyGate | None = None,
        mirror: ProvenanceMirror | None = None,
        browser: BrowserDriver | None = None,
        computer_use: ComputerUseDriver | None = None,
    ) -> None:
        self.backend = backend
        self.run_id = run_id
        self.actor = actor
        self._policy_gate = policy_gate
        self._mirror = mirror if mirror is not None else ProvenanceMirror()
        self._browser = browser  # optional ECO-4.44 browser driver
        self._computer_use = computer_use  # optional ECO-4.93 computer-use driver
        self._dispatcher = ActionDispatcher()
        self._state: WorkspaceState | None = None
        self._step = 0
        self._started = False

    @property
    def state(self) -> WorkspaceState:
        if self._state is None:
            raise RuntimeError("workspace not started; call await start() first")
        return self._state

    async def start(self) -> None:
        if self._started:
            return
        await self.backend.start()
        self._state = WorkspaceState(cwd=self.backend.workdir, env={})
        self._started = True

    async def act(self, action: Action) -> Observation:
        """Execute one action, returning its observation. Stamps correlation + mirrors to KG."""
        if not self._started:
            await self.start()
        self._step += 1
        action = action.model_copy(
            update={
                "run_id": self.run_id,
                "step": self._step,
                "ts": time.time(),
                "actor": self.actor,
            }
        )

        # Fail-closed policy gate for mutating actions (OS-5.24).
        policy_name = mutating_action_name(action)
        if policy_name and self._policy_gate is not None:
            allowed, reason = self._policy_gate(policy_name)
            if not allowed:
                obs = self._stamp(
                    ErrorObservation(
                        message=f"action denied by policy: {reason}",
                        action_kind=getattr(action, "kind", ""),
                    )
                )
                self._mirror.mirror_action(action)
                self._mirror.mirror_observation(action, obs, self.backend.root)
                return obs

        self._mirror.mirror_action(action)
        observation = await self._dispatcher.dispatch(
            action,
            self.backend,
            self.state,
            browser=self._browser,
            computer_use=self._computer_use,
        )
        observation = self._stamp(observation)
        self._mirror.mirror_observation(action, observation, self.backend.root)
        return observation

    def _stamp(self, observation: Observation) -> Observation:
        return observation.model_copy(
            update={
                "run_id": self.run_id,
                "step": self._step,
                "ts": time.time(),
                "actor": self.actor,
            }
        )

    async def stop(self) -> None:
        if not self._started:
            return
        await self.backend.stop()
        self._started = False

    async def __aenter__(self) -> DevWorkspace:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()
