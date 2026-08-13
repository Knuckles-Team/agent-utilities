"""CONCEPT:AU-ORCH.sandbox.tiered-rlm-sandbox — Local (in-process ``exec``) sandbox: the always-available floor.

This is the legacy ``_execute_local`` behaviour, lifted verbatim behind the :class:`Sandbox`
contract. It is NOT a real isolation boundary (CWE-94: it runs model-generated code in this
process), so it advertises ``isolated=False`` and the worst ``preference_rank`` — the router
falls here only when every isolating backend is unavailable or has rejected the snippet. Its
value is that it can do *everything* (full stdlib, third-party libs, classes, native host
helpers), so it never rejects valid Python and guarantees the RLM loop always has a backend.

Stdout capture (B-21). ``forkserver``/``wasm`` own a private OS-level stream: each snippet runs
in a separate forked child / WASI guest (``_bridge.run_child``, ``make_runner_script``), so
swapping ``sys.stdout`` there only ever affects that one throwaway process. ``LocalSandbox``
cannot do the same — ``host_callbacks=True`` here means live, in-process Python closures bound
to orchestrator state (``RLMEnvironment._build_sandbox_env`` hands over bound methods, not
JSON-able RPC targets), so there is no subprocess boundary to hand the snippet without first
building the same bridge protocol docker/forkserver use, which would defeat this backend's
entire reason to exist (the always-available, zero-setup, full-access floor). The in-process
analogue of "own your own stream" is a **single, stable** ``sys.stdout`` replacement — installed
once, never reassigned again — that routes each write to the *calling asyncio Task's* own
buffer via a :class:`contextvars.ContextVar`. asyncio copies the current ``Context`` onto every
new ``Task`` (and into ``run_in_executor`` callables), so two concurrent ``execute()`` calls, or
a concurrent sandbox run racing an unrelated coroutine on the same loop, each see only their own
buffer — never each other's, and never the real terminal's. This replaces the previous
``sys.stdout = X ... finally: sys.stdout = old`` dance, whose ``old`` could already be another
concurrent execution's own swapped buffer, which is exactly the cross-contamination bug.
"""

from __future__ import annotations

import contextvars
import io
import logging
import sys
import threading
import traceback
from typing import IO, Any

from .base import Sandbox, SandboxCapabilities, SandboxEnv, SandboxResult

logger = logging.getLogger(__name__)

#: The active capture buffer for the calling asyncio Task, if any. ``None`` outside of a
#: ``LocalSandbox.execute()`` call, in which case writes pass straight through untouched.
_CAPTURE: contextvars.ContextVar[io.StringIO | None] = contextvars.ContextVar(
    "local_sandbox_stdout_capture", default=None
)

_INSTALL_LOCK = threading.Lock()


class _ContextRoutedStdout:
    """A stable ``sys.stdout`` replacement that dispatches each write by the active
    :data:`_CAPTURE` context instead of by process-global identity.

    Installed on ``sys.stdout`` exactly once (see :func:`_ensure_context_routed_stdout`) and
    never reassigned again, so no execution can race another over ``sys.stdout``'s identity —
    the actual concurrency hazard in the swap-and-restore approach this replaces. A write made
    with no active capture context passes straight through to the real underlying stream.
    """

    def __init__(self, real: IO[str]) -> None:
        self._real = real

    def write(self, s: str) -> int:
        buf = _CAPTURE.get()
        return buf.write(s) if buf is not None else self._real.write(s)

    def flush(self) -> None:
        buf = _CAPTURE.get()
        if buf is None:
            self._real.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def _ensure_context_routed_stdout() -> None:
    """Install :class:`_ContextRoutedStdout` on ``sys.stdout`` if it isn't already there.

    Idempotent and safe to call on every ``execute()``: if something else has since replaced
    ``sys.stdout`` (a test's ``capsys``, a REPL, …), the next call re-wraps whatever is
    currently installed rather than silently writing past it.
    """
    if isinstance(sys.stdout, _ContextRoutedStdout):
        return
    with _INSTALL_LOCK:
        if not isinstance(sys.stdout, _ContextRoutedStdout):
            sys.stdout = _ContextRoutedStdout(sys.stdout)


class LocalSandbox(Sandbox):
    """Run code in a restricted-namespace in-process ``exec`` (no isolation boundary)."""

    name = "local"
    capabilities = SandboxCapabilities(
        host_callbacks=True,
        third_party_libs=True,
        classes=True,
        full_stdlib=True,
        network=True,
        isolated=False,
        preference_rank=30,  # last resort: fast but unsandboxed
    )

    def is_available(self) -> bool:
        return True

    async def execute(self, code: str, env: SandboxEnv) -> SandboxResult:
        """Exec the (async-wrapped) snippet in a namespace of helpers + locals + vars.

        Mirrors the original ``_execute_local``: tool sources and code are wrapped in an
        ``async def`` so ``await`` works, stdout is captured (see module docstring for how this
        stays safe under concurrency), and any name the snippet defined is synced back into
        ``updated_vars`` (everything except builtins, the injected helpers, and the
        local-only globals). In-sandbox exceptions are captured into ``error`` rather than
        raised — the model reads them and retries.
        """
        # Names that are injected scaffolding, not user state to sync back.
        skip = {"__builtins__", "__async_exec__"} | set(env.helpers)

        globals_dict: dict = {
            "__builtins__": __builtins__,
            **env.helpers,
            **env.vars,
        }

        _ensure_context_routed_stdout()
        buf = io.StringIO()
        capture_token = _CAPTURE.set(buf)
        error: str | None = None
        try:
            # Wrap in an async function so the snippet may use top-level ``await``.
            wrapped = "async def __async_exec__():\n"
            for t_src in env.tool_sources.values():
                for line in t_src.splitlines():
                    wrapped += f"    {line}\n"
            for line in code.splitlines():
                wrapped += f"    {line}\n"

            exec(wrapped, globals_dict)  # nosec B102 - RLM REPL, restricted namespace
            await globals_dict["__async_exec__"]()

            updated = {k: v for k, v in globals_dict.items() if k not in skip}
        except Exception as e:  # noqa: BLE001 - surface to the model, keep the loop alive
            traceback.print_exc(file=buf)
            logger.error("LocalSandbox execute error: %s", e)
            error = str(e)
            updated = {k: v for k, v in globals_dict.items() if k not in skip}

        finally:
            _CAPTURE.reset(capture_token)

        return SandboxResult(updated_vars=updated, stdout=buf.getvalue(), error=error)
