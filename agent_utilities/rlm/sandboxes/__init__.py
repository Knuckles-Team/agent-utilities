"""CONCEPT:ORCH-1.38 — Tiered RLM code-execution sandboxes.

The RLM REPL runs LLM-generated Python glue code. Historically that was a hardcoded
``if use_wasm / elif use_container / else local`` chain in :mod:`agent_utilities.rlm.repl`,
where WASM was a stub, Docker could not serve the RLM host helpers, and the only working
path (``local``) was unsandboxed ``exec()``.

This package replaces that with a uniform :class:`Sandbox` contract and four real backends
(local / monty / wasm / docker), each advertising its :class:`SandboxCapabilities`, plus a
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
from .local_backend import LocalSandbox

__all__ = [
    "HELPER_NAMES",
    "Sandbox",
    "SandboxCapabilities",
    "SandboxEnv",
    "SandboxRejected",
    "SandboxResult",
    "LocalSandbox",
]
