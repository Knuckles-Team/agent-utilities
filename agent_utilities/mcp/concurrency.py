"""Concurrency helpers for MCP tool handlers.

Most fleet MCP tools are ``async def`` (they ``await ctx.*`` helpers) but call a
**blocking** synchronous client (``requests``/SDK). Running that blocking call
inline on the event loop stalls every other concurrent request on the worker —
so one slow upstream call serializes the whole server. :func:`run_blocking`
offloads the blocking call to a worker thread so the event loop stays free and
tool calls actually run concurrently.

Usage in a handler::

    from agent_utilities.mcp_utilities import run_blocking
    response = await run_blocking(client.get_repositories, **kwargs)

CONCEPT:ECO-4.0 — MCP Standardized Interfaces
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TypeVar

import anyio

T = TypeVar("T")


async def run_blocking(func: Callable[..., T], /, *args, **kwargs) -> T:
    """Run a blocking sync callable in a worker thread, off the event loop.

    Returns the callable's result (or re-raises its exception) just like calling
    it directly, but without blocking other concurrent tool calls. ``func`` is
    positional-only so ``func``'s own ``*args``/``**kwargs`` pass through cleanly.
    """
    return await anyio.to_thread.run_sync(functools.partial(func, *args, **kwargs))
