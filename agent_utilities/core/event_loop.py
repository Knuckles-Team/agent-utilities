"""Event-loop compatibility for sync LLM facades.

``pydantic_ai.Agent.run_sync`` (like any ``asyncio.run``-based sync facade)
cannot execute inside an already-running event loop. The historical
workaround sprinkled ``nest_asyncio.apply()`` unconditionally at every such
call site — but ``apply()`` patches asyncio **process-wide** (event-loop
policy, ``Task`` internals), permanently and irreversibly, even when there is
no running loop and the patch buys nothing.

On Python ≥ 3.14 that unconditional patch is actively destructive: the
pure-Python task bookkeeping nest_asyncio installs desynchronizes from the
C-accelerated ``asyncio.current_task()``, which then returns ``None`` inside
perfectly ordinary tasks. Every subsequent ``asyncio.timeout`` /
``asyncio.wait_for`` / anyio blocking portal in the process dies with
``RuntimeError: Timeout should be used inside a task`` or
``AttributeError: 'NoneType' object has no attribute 'set_name'`` — one KG
memento compression in a test run was enough to poison 100+ unrelated tests.

:func:`allow_nested_run_sync` keeps the workaround scoped to the one
situation that actually needs it: it applies nest_asyncio **only when the
calling thread already has a running loop**. Outside a loop it is a no-op and
the process stays unpatched.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

__all__ = ["allow_nested_run_sync"]


def allow_nested_run_sync() -> None:
    """Make a following ``Agent.run_sync``-style call survive a running loop.

    Applies ``nest_asyncio`` iff this thread currently has a running event
    loop (the only case where a sync facade would otherwise raise). With no
    running loop this is a strict no-op — global asyncio state is left
    untouched, so callers in plain sync contexts (CLIs, worker threads,
    pytest) never leak the process-wide patch.

    Best-effort by design: when nest_asyncio is missing or refuses to patch,
    the caller's own ``except Exception`` fallback handles the subsequent
    ``run_sync`` failure exactly as it handles any other LLM-call failure.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop in this thread: run_sync works natively. Do NOT
        # patch — nest_asyncio.apply() here would mutate global asyncio
        # internals for zero benefit (and break asyncio.current_task() on
        # Python >= 3.14).
        return
    try:
        import nest_asyncio

        nest_asyncio.apply()
    except Exception:  # noqa: BLE001 - optional dep; caller degrades gracefully
        logger.debug(
            "nest_asyncio unavailable; nested run_sync may fail", exc_info=True
        )
