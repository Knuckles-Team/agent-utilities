"""L4 selection and one-call execution over :class:`HarnessPort`.

:class:`HarnessRegistry` holds the adapters a process offers (the five ruled
harnesses by default) and is the L4 client other layers use to see their
descriptors. :func:`run_to_completion` is the single negotiate -> start ->
drain -> result sequence every caller uses, so no caller re-implements the
lifecycle. :func:`host_workspace_lease` gives a ``caller-managed-host`` run a
private per-run workspace without claiming any sandbox containment.
"""

from __future__ import annotations

import shutil
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_utilities.layers.contracts import (
    HarnessDescriptor,
    HarnessError,
    HarnessRefused,
    RunEvent,
    RunResult,
    RunSpec,
    SandboxBoundaryError,
)
from agent_utilities.layers.negotiation import HarnessPolicy
from agent_utilities.layers.ports import HarnessPort, SandboxLease

#: Backend label of a workspace-only lease for ``caller-managed-host`` runs.
HOST_WORKSPACE_BACKEND = "caller-managed-host"


class HarnessRegistry:
    """The harness adapters available in this process, by descriptor name."""

    def __init__(self, harnesses: Mapping[str, HarnessPort]) -> None:
        for name, harness in harnesses.items():
            if harness.describe().name != name:
                raise ValueError(
                    f"harness registered as {name!r} describes itself "
                    f"as {harness.describe().name!r}"
                )
        self._harnesses = dict(harnesses)

    def get(self, name: str) -> HarnessPort:
        harness = self._harnesses.get(name)
        if harness is None:
            raise HarnessRefused(name, ("the harness is not registered",))
        return harness

    def descriptors(self) -> tuple[HarnessDescriptor, ...]:
        return tuple(
            self._harnesses[name].describe() for name in sorted(self._harnesses)
        )


def default_registry(
    runner: Any,
    *,
    devin_org_id: str | None = None,
    launcher: Any = None,
    credentials: Any = None,
) -> HarnessRegistry:
    """All five ruled adapters (RF-ADR-010 §6.4) over one AU runtime."""
    from agent_utilities.layers.adapters.claude_code import ClaudeCodeHarness
    from agent_utilities.layers.adapters.codex import CodexHarness
    from agent_utilities.layers.adapters.devin import DevinHarness
    from agent_utilities.layers.adapters.grok import GrokHarness
    from agent_utilities.layers.adapters.pydantic_ai import PydanticAiHarness

    cli = {"launcher": launcher, "credentials": credentials}
    harnesses: list[HarnessPort] = [
        PydanticAiHarness(runner),
        ClaudeCodeHarness(**cli),
        CodexHarness(**cli),
        GrokHarness(**cli),
        DevinHarness(org_id=devin_org_id, credentials=credentials),
    ]
    return HarnessRegistry({h.describe().name: h for h in harnesses})


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """A finished run: its result, full normalized trace and typed failure."""

    result: RunResult
    trace: tuple[RunEvent, ...]
    failure: HarnessError | None


async def run_to_completion(
    harness: HarnessPort,
    spec: RunSpec,
    *,
    policy: HarnessPolicy,
    lease: SandboxLease | None,
) -> RunOutcome:
    """Negotiate, start, drain every event and return the terminal outcome."""
    negotiated = harness.negotiate(spec, policy)
    handle = await harness.start(negotiated, lease)
    async for _event in harness.events(handle):
        continue
    result = await harness.result(handle)
    return RunOutcome(
        result=result, trace=harness.trace(handle), failure=harness.failure(handle)
    )


def host_workspace_lease(root: str, run_id: str) -> SandboxLease:
    """A private workspace for a ``caller-managed-host`` run (no containment)."""
    base = Path(root).resolve()
    lease_id = f"host-{uuid.uuid4().hex}"
    workspace = base / lease_id
    workspace.mkdir(parents=True, mode=0o700)
    return SandboxLease(
        lease_id=lease_id,
        backend=HOST_WORKSPACE_BACKEND,
        workspace=str(workspace),
        reason={"mode": "caller-managed-host", "run_id": run_id},
    )


def release_workspace(root: str, lease: SandboxLease) -> None:
    """Remove a lease's workspace, refusing any path outside ``root``."""
    base = Path(root).resolve()
    workspace = Path(lease.workspace).resolve()
    if workspace.parent != base:
        raise SandboxBoundaryError("lease workspace is outside the workspace root")
    if workspace.exists():
        shutil.rmtree(workspace)


__all__ = [
    "HOST_WORKSPACE_BACKEND",
    "HarnessRegistry",
    "RunOutcome",
    "default_registry",
    "host_workspace_lease",
    "release_workspace",
    "run_to_completion",
]
