"""CONCEPT:ORCH-1.38 — monty sandbox: the fast, isolated, host-callback-capable default tier.

monty (Pydantic's pure-Rust Python-subset interpreter) is the keystone of the tiering: it is
the only *isolating* backend that can still serve the RLM host helpers, because async external
functions suspend the VM and let the host fulfil ``await rlm_query(...)``. It starts in ~0.06ms
(vs Docker's 100-500ms), needs no daemon or root, and enforces memory/time/recursion limits.

Semantics deliberately match :class:`~.local_backend.LocalSandbox`: a fresh ``Monty`` per call,
so a snippet's plain locals do NOT persist across turns (only host vars updated via the
``FINAL_VAR`` helper and the seeded inputs do) — exactly like the legacy ``_execute_local``
``async def`` wrapping. Fresh-per-call is also *faster* than a reused ``MontyRepl`` for the
small snippets RLM emits (the REPL offloads each resume to a thread).

Rejection safety: monty rejects unsupported features (``class``, ``@dataclass``, unsupported
syntax) at *construction* time — before any host helper fires — so escalating to Docker has no
side effects. A runtime error that surfaces *after* a helper has fired is reported as an
in-sandbox error (not escalated), to avoid re-invoking side-effecting helpers on the next tier.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from .base import (
    Sandbox,
    SandboxCapabilities,
    SandboxEnv,
    SandboxRejected,
    SandboxResult,
)

logger = logging.getLogger(__name__)

# Default per-snippet wall-clock budget. The host-tool timeout (ORCH-1.29) governs the helper
# callbacks; this bounds the *monty* compute itself so a runaway loop can't wedge the process.
_DEFAULT_MAX_DURATION_SECS = 30.0

# Values monty can accept as inputs (its object model is JSON-ish). Anything else in the REPL
# namespace (live module/class refs, callables) is simply not seeded — the snippet must reach
# those host capabilities through the helper callbacks instead.
_MONTYABLE = (str, int, float, bool, type(None), list, dict, tuple)


class MontySandbox(Sandbox):
    """Run a snippet in a fresh monty interpreter with host helpers as external functions."""

    name = "monty"
    capabilities = SandboxCapabilities(
        host_callbacks=True,
        third_party_libs=False,
        classes=False,
        full_stdlib=False,  # ~8 stdlib modules; router relies on escalation for the rest
        network=False,
        isolated=True,
        preference_rank=0,  # fastest acceptable tier — tried first
    )

    def __init__(self, max_duration_secs: float = _DEFAULT_MAX_DURATION_SECS):
        self._max_duration_secs = max_duration_secs
        self._available: bool | None = None

    def is_available(self) -> bool:
        if self._available is None:
            try:
                import pydantic_monty  # noqa: F401

                self._available = True
            except Exception:  # noqa: BLE001 - optional dependency
                self._available = False
        return self._available

    async def execute(self, code: str, env: SandboxEnv) -> SandboxResult:
        from pydantic_monty import (
            CollectString,
            Monty,
            MontyError,
            MontySyntaxError,
            ResourceLimits,
        )

        # Seed the snippet's readable host vars (context, depth, prior FINAL outputs, ...).
        inputs = {k: v for k, v in env.vars.items() if isinstance(v, _MONTYABLE)}
        full_code = self._with_tool_sources(code, env.tool_sources)

        # 1) Parse/compile — rejections here fire BEFORE any helper, so escalation is safe.
        try:
            m = Monty(full_code, inputs=list(inputs))
        except MontyError as e:
            raise SandboxRejected("monty", self._reject_reason(e)) from e

        # 2) Run, tracking whether a (side-effecting) helper fired, so a late runtime error
        #    is reported rather than escalated.
        fired = _FireCounter()
        wrapped_helpers = {n: fired.wrap(fn) for n, fn in env.helpers.items()}
        collected = CollectString()
        try:
            await m.run_async(
                inputs=inputs,
                external_functions=wrapped_helpers,
                limits=ResourceLimits(max_duration_secs=self._max_duration_secs),
                print_callback=collected,
            )
        except MontySyntaxError as e:  # pragma: no cover - construction already parsed
            raise SandboxRejected("monty", self._reject_reason(e)) from e
        except MontyError as e:
            return self._handle_runtime_error(e, fired, collected.output)

        # Host helpers (FINAL_VAR etc.) already mutated env.vars in place; nothing extra to sync.
        return SandboxResult(updated_vars={}, stdout=collected.output, error=None)

    # ── helpers ────────────────────────────────────────────────────────────
    @staticmethod
    def _with_tool_sources(code: str, tool_sources: dict[str, str]) -> str:
        if not tool_sources:
            return code
        return "\n".join([*tool_sources.values(), code])

    @staticmethod
    def _reject_reason(exc: Exception) -> str:
        return f"{type(exc).__name__}: {str(exc)[:120]}"

    def _handle_runtime_error(
        self, exc: Exception, fired: _FireCounter, stdout: str
    ) -> SandboxResult:
        """Classify a monty runtime error: escalate only if it's a capability gap with no side effects."""
        text = str(exc).lower()
        unsupported = (
            "does not yet support" in text
            or "notimplementederror" in text
            or "modulenotfounderror" in text
            or "no module named" in text
        )
        if unsupported and fired.count == 0:
            # A capability gap (e.g. an unsupported stdlib import) hit before any helper ran —
            # safe to escalate to a fuller backend.
            raise SandboxRejected("monty", self._reject_reason(exc))
        # Genuine in-sandbox error (or one after a helper already fired): report it; the model
        # reads the message and retries. Escalating now could double-invoke a side-effecting helper.
        logger.debug("monty runtime error (reported, fired=%d): %s", fired.count, exc)
        return SandboxResult(updated_vars={}, stdout=stdout, error=str(exc))


class _FireCounter:
    """Counts host-helper invocations so the backend can tell side-effecting runs apart."""

    def __init__(self) -> None:
        self.count = 0

    def wrap(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        def tracked(*args: Any, **kwargs: Any) -> Any:
            self.count += 1
            return fn(*args, **kwargs)

        return tracked
