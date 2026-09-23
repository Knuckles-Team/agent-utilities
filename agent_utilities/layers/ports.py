"""The L4 ports: :class:`HarnessPort` and :class:`SandboxPort` (RF-ADR-010 §6-§7).

One abstraction for every runtime. The in-process AU runtime and the external
harnesses (Claude Code, Codex, Devin, Grok) are adapters behind
:class:`HarnessPort`; local isolation backends are adapters behind
:class:`SandboxPort`. No caller reaches a harness or sandbox any other way.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from agent_utilities.layers.contracts import (
    HarnessDescriptor,
    HarnessError,
    NegotiatedRunSpec,
    RunEvent,
    RunResult,
    RunSpec,
)
from agent_utilities.layers.negotiation import HarnessPolicy

NetworkNeed = Literal["none", "egress"]


class SandboxRequirements(BaseModel):
    """What a job needs from a local sandbox; the sole selection input (§7)."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    network: NetworkNeed = "none"
    third_party_libs: bool = False
    classes: bool = False
    host_callbacks: bool = False
    pin: str | None = Field(default=None, max_length=64)
    deny: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class SandboxLease:
    """One leased backend plus the per-run workspace it confines work to.

    ``reason`` records why this backend was chosen (requirements matched,
    candidates excluded and why) so the choice is visible in L5.
    """

    lease_id: str
    backend: str
    workspace: str
    reason: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RunHandle:
    """Opaque handle of one started run."""

    run_id: str
    harness: str
    spec_digest: str
    workspace: str | None


@runtime_checkable
class HarnessPort(Protocol):
    """L4 harness adapter contract (§6.2)."""

    def describe(self) -> HarnessDescriptor: ...

    def negotiate(self, spec: RunSpec, policy: HarnessPolicy) -> NegotiatedRunSpec: ...

    async def start(
        self, negotiated: NegotiatedRunSpec, lease: SandboxLease | None
    ) -> RunHandle: ...

    def events(self, handle: RunHandle) -> AsyncIterator[RunEvent]: ...

    async def cancel(self, handle: RunHandle) -> None: ...

    async def result(self, handle: RunHandle) -> RunResult: ...

    def trace(self, handle: RunHandle) -> tuple[RunEvent, ...]: ...

    def failure(self, handle: RunHandle) -> HarnessError | None: ...


@runtime_checkable
class SandboxPort(Protocol):
    """L4 local-isolation contract, promoting the RLM sandbox router (§7)."""

    def lease(self, requirements: SandboxRequirements) -> SandboxLease: ...

    async def execute(self, lease: SandboxLease, code: str) -> dict[str, object]: ...

    def release(self, lease: SandboxLease) -> None: ...

    def health(self) -> dict[str, bool]: ...


__all__ = [
    "HarnessPort",
    "NetworkNeed",
    "RunHandle",
    "SandboxLease",
    "SandboxPort",
    "SandboxRequirements",
]
