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
import concurrent.futures
import logging
from collections.abc import Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

__all__ = ["allow_nested_run_sync", "run_sync_isolated"]

_T = TypeVar("_T")


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


def run_sync_isolated(fn: Callable[[], _T]) -> _T:
    """Run a blocking ``asyncio.run``-based callable (e.g. ``Agent.run_sync``)
    safely regardless of whether the calling thread already has a running loop.

    Not to be confused with :func:`agent_utilities.mcp.concurrency.run_blocking`
    (an ``async def`` helper an *async* caller ``await``s to offload a blocking
    call off the event loop). This one is for plain **sync** call sites that
    themselves invoke a sync-facade-over-asyncio (``Agent.run_sync``) and may be
    reached from a thread that already has a loop running — there is no
    ``await`` available at those call sites to hand off to.

    CONCEPT:AU-KG.query.ask-gateway-rest-twin — the worker-thread sibling of
    :func:`allow_nested_run_sync`, for call sites where patching global asyncio
    state (``nest_asyncio``) is undesirable and a plain per-call escape hatch is
    enough. ``nl_planner.AuNlPlanner._default_run`` established this exact
    pattern for the one-shot NL→query planning call
    (fixing "RuntimeError: This event loop is already running" for `nl_query`);
    this is the same fix extracted so sibling one-shot sync-LLM call sites
    (``nl_query.nl_to_query`` → ``graph_ask``/``ask_data``,
    ``data_analyst.DataAnalystAgent._default_synthesize`` → ``ask_data``) share
    it instead of re-deriving it.

    With no running loop on this thread, ``fn`` runs in-place natively (zero
    overhead). With a running loop, ``fn`` runs on a short-lived single worker
    thread (which has no loop of its own), so the nested ``asyncio.run()``
    inside ``fn`` never collides with the caller's loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return fn()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(fn).result()
