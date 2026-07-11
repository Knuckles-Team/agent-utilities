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


# Names fleet API-client methods use for a single REST-body dict parameter. An
# action-routed method takes at most one of these; ``data`` (plane, gitlab-style
# clients) and ``payload`` (the OpenAPI-codegen clients, e.g. atlassian) are the
# two live conventions, ``body`` a common third.
_BODY_PARAM_NAMES = ("data", "payload", "body")


def _wrap_data_kwargs(func: Callable[..., T], args: tuple, kwargs: dict) -> dict:
    """Fold stray keyword fields into the target's REST-body param when it has one.

    Action-routed MCP tools pass the LLM's free-form ``params_json`` fields as
    flat ``**kwargs`` into a client method. Many fleet API-client write methods
    take the REST body as a single dict param — ``data`` (e.g.
    ``create_work_item(project_id, data)``) or ``payload`` (e.g.
    ``jira_cloud_create_issue(update_history, payload)``). An LLM that naturally
    passes ``{project_id, name, description}`` / ``{fields: {...}}`` (mirroring
    the REST payload) then crashes with ``unexpected keyword argument 'name'``
    and the whole delegated create/update fails. This makes the dispatch
    self-healing: when ``func`` declares exactly one body param, none was
    supplied, and there are extra fields that don't match a named parameter,
    those extras are collected into that body param. It is a strict no-op for
    every method without such a param (the overwhelming majority), so it is safe
    on the shared dispatch path. CONCEPT:AU-ECO.mcp.standardized-interfaces
    """
    if args or not kwargs:
        return kwargs
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    body_params = [n for n in _BODY_PARAM_NAMES if n in params]
    # Only act when the body param is unambiguous and wasn't already supplied.
    if len(body_params) != 1 or body_params[0] in kwargs:
        return kwargs
    body_param = body_params[0]
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
    wrapped[body_param] = stray
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
