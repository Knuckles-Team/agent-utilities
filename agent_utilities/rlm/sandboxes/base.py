"""CONCEPT:AU-ORCH.sandbox.tiered-rlm-sandbox — Sandbox contract, capabilities, and the data passed across it.

Every RLM execution backend implements :class:`Sandbox`. The router (``router.py``) reasons
purely over :class:`SandboxCapabilities`, so adding a backend is: implement ``execute`` +
``is_available`` and declare what it can do — no router changes needed.

Two failure modes are kept distinct on purpose:

* :class:`SandboxRejected` — *this* backend cannot run *this* code (a capability gap or a
  parse/compile rejection, e.g. monty seeing a ``class``). The router catches it and escalates
  to the next backend in the chain. Recoverable by definition.
* :class:`~agent_utilities.rlm.telemetry.SandboxFatalError` — the backend's *infrastructure*
  died (container daemon gone, mount lost). The run fast-fails; the router does NOT escalate,
  matching the pre-existing ORCH-1.29 semantics.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# The canonical RLM host-helper names injected into the REPL namespace
# (see RLMEnvironment.__init__ in repl.py). A backend that advertises
# ``host_callbacks`` must make every one of these callable from inside the
# sandbox; the router treats any call to one of these as "needs host_callbacks".
HELPER_NAMES: frozenset[str] = frozenset(
    {
        "rlm_query",
        "run_parallel_sub_calls",
        "magma_view",
        "graph_query",
        "owl_query",
        "kg_bulk_export",
        "ephemeral_graph_query",
        "sub_agent_call",
        "FINAL_VAR",
    }
)


@dataclass(frozen=True)
class SandboxCapabilities:
    """What a backend can run — the sole input to routing decisions.

    ``preference_rank`` orders candidates the router considers equally capable: lower wins.
    It encodes "fastest *acceptable* tier first", which is why ``local`` (fast but
    unsandboxed) is ranked LAST despite its speed — it is the always-available floor, not a
    preferred destination.
    """

    host_callbacks: bool
    """Can serve the RLM host helpers (``rlm_query`` etc.) from inside the sandbox."""
    third_party_libs: bool
    """Can ``import`` non-stdlib packages (numpy, pandas, ...)."""
    classes: bool
    """Supports ``class`` / ``@dataclass`` definitions."""
    full_stdlib: bool
    """Full CPython stdlib (vs monty's ~8-module subset)."""
    network: bool
    """Code can reach the network (false = isolated egress)."""
    isolated: bool
    """Provides a real isolation boundary (false only for ``local`` exec)."""
    preference_rank: int
    """Lower = preferred when multiple backends satisfy the requirements."""
    workspace: bool = False
    """Advertises the persistent developer-workspace contract (OS-5.33), NOT the snippet
    contract. The snippet router never selects a workspace backend; this flag exists so the two
    runtimes share one capability vocabulary without the router conflating them."""
    warm_fork: bool = False
    """Implements the warm-fork lifecycle (CONCEPT:AU-ORCH.sandbox.shared-host-helper-bridge): a warmed parent is paid for
    once, then children are spawned from copy-on-write state instead of cold-booting each.
    Backends advertising this also implement :class:`ForkableSandbox`. It is a *property of how
    the backend spawns*, orthogonal to the routing filters above — the router does not gate on
    it; ``execute`` still works (warm-or-reuse a parent, fork one child, run). It exists so the
    capability layer and the warm-parent registry can reason about which rungs amortise startup
    across a fan-out cohort (CONCEPT:AU-ORCH.sandbox.warmforkfanoutcapability :WarmForkFanoutCapability)."""


@dataclass
class SandboxEnv:
    """The execution context a backend needs to run one snippet.

    ``vars`` is the persistent REPL namespace (read in, synced back out). ``helpers`` are the
    host callbacks a ``host_callbacks`` backend wires up (monty as ``external_functions``,
    Docker via the UDS bridge). ``local_globals`` are objects only the in-process ``local``
    backend can inject (live module/class refs like ``json``, ``asyncio``,
    ``GraphComputeEngine``) — isolated backends ignore them.
    """

    vars: dict[str, Any]
    tool_sources: dict[str, str] = field(default_factory=dict)
    helpers: dict[str, Callable[..., Any]] = field(default_factory=dict)
    local_globals: dict[str, Any] = field(default_factory=dict)


@dataclass
class SandboxResult:
    """Outcome of one execution: the synced-back namespace plus captured stdout.

    ``error`` is a human-readable message for an *in-sandbox* error (the model can read it and
    retry) — distinct from :class:`SandboxRejected` (wrong backend) and
    :class:`~agent_utilities.rlm.telemetry.SandboxFatalError` (dead infra).
    """

    updated_vars: dict[str, Any]
    stdout: str
    error: str | None = None


class SandboxRejected(Exception):
    """Raised when a backend cannot run a snippet — the router escalates to the next tier.

    Carries the backend name and a terse reason for telemetry/debugging. NOT a
    ``SandboxFatalError``: rejection is expected control flow (e.g. monty meets a ``class``),
    not infrastructure death.
    """

    def __init__(self, backend: str, reason: str):
        self.backend = backend
        self.reason = reason
        super().__init__(f"[{backend}] rejected: {reason}")


class Sandbox(abc.ABC):
    """A single RLM code-execution backend.

    Subclasses set :attr:`name` and :attr:`capabilities`, and implement :meth:`is_available`
    (probe imports/daemons; cache it) and :meth:`execute`. ``execute`` MUST:

    * run ``code`` against ``env`` and return a :class:`SandboxResult` with the synced-back
      ``vars`` and captured stdout;
    * raise :class:`SandboxRejected` if the snippet is outside this backend's capabilities
      (so the router escalates);
    * raise :class:`~agent_utilities.rlm.telemetry.SandboxFatalError` only on irreversible
      infrastructure failure (so the run fast-fails).
    """

    name: str
    capabilities: SandboxCapabilities

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Whether this backend can run at all in the current environment (cache the result)."""

    @abc.abstractmethod
    async def execute(self, code: str, env: SandboxEnv) -> SandboxResult:
        """Execute ``code`` against ``env``; see class docstring for the error contract."""

    def __repr__(self) -> str:
        return f"<Sandbox {self.name!r} rank={self.capabilities.preference_rank}>"


@dataclass
class WarmSpec:
    """The content-addressable description of a warm parent (CONCEPT:AU-ORCH.sandbox.shared-host-helper-bridge).

    Two parents with the same ``key`` are interchangeable, so the registry can reuse one
    instead of paying warm-up again. ``preload`` names the heavy imports the parent loads
    before snapshotting (the cost we amortise across the fork cohort); ``base_key`` lets a
    spec declare it derives from another (the diff-snapshot-chain edge, CONCEPT:AU-ORCH.sandbox.warmforkfanoutcapability —
    "is there a warm parent that is a superset of what I need?").
    """

    backend: str
    preload: tuple[str, ...] = ()
    base_key: str | None = None
    extra: tuple[tuple[str, str], ...] = ()

    @property
    def key(self) -> str:
        """Stable content hash of the spec — the registry/pool key."""
        import hashlib

        payload = repr(
            (self.backend, tuple(self.preload), self.base_key, tuple(self.extra))
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _noop_close() -> None:
    """Default teardown for a :class:`ParentHandle` with no backend resource to free."""
    return None


@dataclass
class ParentHandle:
    """A live warmed parent a backend forks children from. ``ref`` is backend-private (a
    forkserver context, a warmed container id, a microVM snapshot tag). Borrow/idle accounting
    is the warm-parent registry's job; ``close`` tears the parent down (sync, idempotent)."""

    backend: str
    spec: WarmSpec
    ref: Any
    close: Callable[[], None] = _noop_close


class ForkableSandbox(Sandbox):
    """A :class:`Sandbox` that spawns via warm-fork (CONCEPT:AU-ORCH.sandbox.shared-host-helper-bridge) instead of cold-boot.

    A rung implements just two atoms — :meth:`warm` (pay start-up once for a :class:`WarmSpec`)
    and :meth:`run_forked` (fork ONE copy-on-write child off a warm parent, run the snippet,
    return its :class:`SandboxResult`) — plus :meth:`warm_spec` describing its parent. It then
    gets a working :meth:`Sandbox.execute` *for free* from this base: execute warms-or-reuses a
    parent through the host :class:`~agent_utilities.runtime.warm_registry.WarmParentRegistry`
    (CONCEPT:AU-OS.host.so-they-are-idle) and forks one child. Fan-out is simply many concurrent ``execute`` /
    ``run_forked`` calls — each forks its own child off the *one* warm parent, which is the
    copy-on-write amortisation (imports/deps/weights resident once, shared across the cohort).

    Mid-execution ``branch`` (snapshot a *running* child into a new parent) is a microVM-only
    capability (it needs a memory snapshot os.fork can't take) and is therefore NOT on this
    base — the ``firecracker`` rung adds it as its own method.

    Subclasses set ``capabilities.warm_fork = True``. :meth:`warm` / :meth:`run_forked` MUST
    raise :class:`~agent_utilities.rlm.telemetry.SandboxFatalError` on irreversible
    infrastructure failure (dead daemon, lost snapshot) so callers fast-fail.
    """

    @abc.abstractmethod
    def warm_spec(self) -> WarmSpec:
        """The :class:`WarmSpec` describing this rung's warm parent (its registry/pool key)."""

    @abc.abstractmethod
    async def warm(self, spec: WarmSpec) -> ParentHandle:
        """Boot + warm a parent for ``spec`` (pay the expensive start-up once)."""

    @abc.abstractmethod
    async def run_forked(
        self, parent: ParentHandle, code: str, env: SandboxEnv
    ) -> SandboxResult:
        """Fork one CoW child off ``parent``, run ``code`` against ``env``, return the result."""

    async def execute(self, code: str, env: SandboxEnv) -> SandboxResult:
        """Warm-or-reuse a parent (host registry, OS-5.58), then fork one child to run ``code``.

        Concrete for every forkable rung — the warm-fork win is structural here: the registry
        hands back an already-warmed parent across calls, so only the first ``execute`` pays
        start-up and every subsequent one is a cheap fork.
        """
        from agent_utilities.runtime.warm_registry import WarmParentRegistry

        registry = WarmParentRegistry.get()
        spec = self.warm_spec()
        parent = registry.acquire(spec.key)
        if parent is None:
            parent = await self.warm(spec)
            registry.register(spec.key, parent, close=parent.close, kind=self.name)
        return await self.run_forked(parent, code, env)
