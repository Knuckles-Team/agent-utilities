"""CONCEPT:ORCH-1.38 — Sandbox contract, capabilities, and the data passed across it.

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
