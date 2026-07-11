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

CONCEPT:AU-ECO.mcp.standardized-interfaces — MCP Standardized Interfaces
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import TypeVar

import anyio

T = TypeVar("T")


def _wrap_data_kwargs(func: Callable[..., T], args: tuple, kwargs: dict) -> dict:
    """Fold stray keyword fields into a ``data`` dict when the target expects one.

    Action-routed MCP tools pass the LLM's free-form ``params_json`` fields as
    flat ``**kwargs`` into a client method. Many fleet API-client write methods
    take the REST body as a single ``data: dict`` param (e.g.
    ``create_work_item(project_id, data)``) — so an LLM that naturally passes
    ``{project_id, name, description}`` (mirroring the REST payload) crashes with
    ``unexpected keyword argument 'name'`` and the whole delegated create/update
    fails. This makes the dispatch self-healing: when ``func`` declares an
    explicit ``data`` parameter, no ``data`` was supplied, and there are extra
    fields that don't match a named parameter, those extras are collected into
    ``data``. It is a strict no-op for every method without a ``data`` param (the
    overwhelming majority), so it is safe on the shared dispatch path.
    CONCEPT:AU-ECO.mcp.standardized-interfaces
    """
    if args or "data" in kwargs or not kwargs:
        return kwargs
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return kwargs
    if "data" not in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    ):
        return kwargs
    named = {
        name
        for name, p in params.items()
        if p.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    stray = {k: v for k, v in kwargs.items() if k not in named}
    if not stray:
        return kwargs
    wrapped = {k: v for k, v in kwargs.items() if k in named}
    wrapped["data"] = stray
    return wrapped


async def run_blocking(func: Callable[..., T], /, *args, **kwargs) -> T:
    """Run a blocking sync callable in a worker thread, off the event loop.

    Returns the callable's result (or re-raises its exception) just like calling
    it directly, but without blocking other concurrent tool calls. ``func`` is
    positional-only so ``func``'s own ``*args``/``**kwargs`` pass through cleanly.

    Stray keyword fields are folded into a ``data`` dict when ``func`` expects
    one (see :func:`_wrap_data_kwargs`) so LLM-driven create/update dispatch is
    self-healing; it is a no-op for any callable without a ``data`` parameter.
    """
    kwargs = _wrap_data_kwargs(func, args, kwargs)
    return await anyio.to_thread.run_sync(functools.partial(func, *args, **kwargs))
