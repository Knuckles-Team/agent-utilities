"""CONCEPT:AU-ORCH.sandbox.tiered-rlm-sandbox — Tiered RLM code sandbox with a uniform Sandbox contract and capability routing.

The RLM REPL runs LLM-generated Python glue code through confined backends
(monty / wasm / docker / firecracker), each advertising its
:class:`SandboxCapabilities`, plus a
deterministic :class:`SandboxRouter` that picks the cheapest backend able to run a given
snippet and escalates when one rejects it.

The keystone is that a backend can serve the *host-side* RLM helpers (``rlm_query`` etc.)
while still isolating the code — monty does this natively via pause/resume external
functions, Docker via a UDS bridge.
"""

from .base import (
    HELPER_NAMES,
    Sandbox,
    SandboxCapabilities,
    SandboxEnv,
    SandboxRejected,
    SandboxResult,
)

__all__ = [
    "HELPER_NAMES",
    "Sandbox",
    "SandboxCapabilities",
    "SandboxEnv",
    "SandboxRejected",
    "SandboxResult",
]
