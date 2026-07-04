"""CONCEPT:AU-ORCH.sandbox.tiered-rlm-sandbox — Local (in-process ``exec``) sandbox: the always-available floor.

This is the legacy ``_execute_local`` behaviour, lifted verbatim behind the :class:`Sandbox`
contract. It is NOT a real isolation boundary (CWE-94: it runs model-generated code in this
process), so it advertises ``isolated=False`` and the worst ``preference_rank`` — the router
falls here only when every isolating backend is unavailable or has rejected the snippet. Its
value is that it can do *everything* (full stdlib, third-party libs, classes, native host
helpers), so it never rejects valid Python and guarantees the RLM loop always has a backend.
"""

from __future__ import annotations

import io
import logging
import sys
import traceback

from .base import Sandbox, SandboxCapabilities, SandboxEnv, SandboxResult

logger = logging.getLogger(__name__)


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
        ``async def`` so ``await`` works, stdout is captured, and any name the snippet defined
        is synced back into ``updated_vars`` (everything except builtins, the injected helpers,
        and the local-only globals). In-sandbox exceptions are captured into ``error`` rather
        than raised — the model reads them and retries.
        """
        # Names that are injected scaffolding, not user state to sync back.
        skip = (
            {"__builtins__", "__async_exec__"}
            | set(env.helpers)
            | set(env.local_globals)
        )

        globals_dict: dict = {
            "__builtins__": __builtins__,
            **env.local_globals,
            **env.helpers,
            **env.vars,
        }

        old_stdout = sys.stdout
        redirected = io.StringIO()
        sys.stdout = redirected
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
            traceback.print_exc(file=redirected)
            logger.error("LocalSandbox execute error: %s", e)
            error = str(e)
            updated = {k: v for k, v in globals_dict.items() if k not in skip}

        finally:
            sys.stdout = old_stdout

        return SandboxResult(
            updated_vars=updated, stdout=redirected.getvalue(), error=error
        )
