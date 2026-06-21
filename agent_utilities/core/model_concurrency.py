"""Per-model parallel-call concurrency controller (CONCEPT:KG-2.143).

Each model declares how many parallel calls it can serve — its vLLM per-instance
concurrency (``max_parallel_calls``) times the number of parallel instances
behind the endpoint (``parallel_instances``). That product is the model's
``total_capacity`` (see :mod:`agent_utilities.core.config`).

This module turns that declared capacity into a *shared, cached, per-model gate*
so LLM/embedding fan-out scales with the real backend: hand it 500 items and it
runs ``min(items, capacity)`` concurrently; with capacity ``1`` it stays strictly
sequential (no behaviour change vs the historical for-loop).

Two fan-out helpers are provided, both **order-preserving**:

* :func:`map_concurrent` — async: ``await`` it; fans out ``fn`` (sync or async)
  over ``items`` up to the model's capacity using an :class:`asyncio.Semaphore`.
* :func:`map_concurrent_sync` — sync: a bounded :class:`ThreadPoolExecutor`
  sized to capacity (or a plain loop when capacity is ``1`` / a single item).

Nothing here hardcodes an endpoint or backend; capacity is resolved from config
and falls back to ``1`` whenever it is unknown — always safe.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import Awaitable, Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")

__all__ = [
    "resolve_capacity",
    "get_semaphore",
    "get_thread_pool",
    "reset_controllers",
    "map_concurrent",
    "map_concurrent_sync",
]


def resolve_capacity(model: str | None = None, default: int = 1) -> int:
    """Resolve a model's total parallel-call capacity (CONCEPT:KG-2.143).

    Looks the model up in the live config registry (by id or role; see
    :meth:`Config.model_capacity`). Any failure — no config, unknown model,
    import error — collapses to ``default`` (``1``), i.e. sequential, never zero.
    """
    try:
        from agent_utilities.core.config import config

        cap = config.model_capacity(model)
        return max(1, int(cap))
    except Exception:  # noqa: BLE001 — capacity is best-effort; stay safe-sequential
        return max(1, int(default))


# --- Shared, cached per-model gates -----------------------------------------
# Keyed by (model_key, capacity) so a config reload that changes capacity yields
# a fresh gate instead of silently reusing a stale-sized one. asyncio.Semaphore
# is bound to the running loop, so it is additionally keyed by loop id.

_lock = threading.Lock()
_semaphores: dict[tuple[str, int, int], asyncio.Semaphore] = {}
_pools: dict[tuple[str, int], ThreadPoolExecutor] = {}


def _key(model: str | None) -> str:
    return (model or "__default__").strip().lower() or "__default__"


def _is_async_callable(fn: object) -> bool:
    """True for coroutine functions AND callable instances with async ``__call__``."""
    if inspect.iscoroutinefunction(fn):
        return True
    call = type(fn).__call__ if not inspect.isfunction(fn) else None
    return bool(call is not None and inspect.iscoroutinefunction(call))


def get_semaphore(
    model: str | None = None, capacity: int | None = None
) -> asyncio.Semaphore:
    """Return the shared per-model :class:`asyncio.Semaphore` sized to capacity.

    Cached per (model, capacity, event-loop). CONCEPT:KG-2.143.
    """
    cap = max(1, int(capacity)) if capacity is not None else resolve_capacity(model)
    try:
        loop_id = id(asyncio.get_running_loop())
    except RuntimeError:
        loop_id = 0
    k = (_key(model), cap, loop_id)
    with _lock:
        sem = _semaphores.get(k)
        if sem is None:
            sem = asyncio.Semaphore(cap)
            _semaphores[k] = sem
        return sem


def get_thread_pool(
    model: str | None = None, capacity: int | None = None
) -> ThreadPoolExecutor:
    """Return the shared per-model bounded :class:`ThreadPoolExecutor`.

    Cached per (model, capacity). CONCEPT:KG-2.143.
    """
    cap = max(1, int(capacity)) if capacity is not None else resolve_capacity(model)
    k = (_key(model), cap)
    with _lock:
        pool = _pools.get(k)
        if pool is None:
            pool = ThreadPoolExecutor(
                max_workers=cap, thread_name_prefix=f"modelcap-{k[0]}"
            )
            _pools[k] = pool
        return pool


def reset_controllers() -> None:
    """Drop all cached gates/pools (test isolation; config reload). CONCEPT:KG-2.143."""
    with _lock:
        _semaphores.clear()
        pools = list(_pools.values())
        _pools.clear()
    for p in pools:
        p.shutdown(wait=False)


async def map_concurrent(
    items: Sequence[T],
    fn: Callable[[T], R] | Callable[[T], Awaitable[R]],
    *,
    model: str | None = None,
    capacity: int | None = None,
) -> list[R]:
    """Fan ``fn`` out over ``items`` up to the model's capacity, async.

    CONCEPT:KG-2.143. ``fn`` may be sync or async. At most
    ``min(len(items), capacity)`` calls are in flight at once (gated by the shared
    per-model semaphore). **Results are returned in input order.** With capacity
    ``1`` this is sequential; with capacity ``K`` up to ``K`` run concurrently.

    ``capacity`` overrides the config-resolved value (mainly for tests). A sync
    ``fn`` is offloaded to a thread so it never blocks the event loop.
    """
    if not items:
        return []
    cap = max(1, int(capacity)) if capacity is not None else resolve_capacity(model)
    sem = get_semaphore(model, cap)
    is_coro = _is_async_callable(fn)

    async def _run(item: T) -> R:
        async with sem:
            if is_coro:
                return await fn(item)  # type: ignore[misc]
            return await asyncio.to_thread(fn, item)  # type: ignore[arg-type]

    # gather preserves the order of the awaitables → order of items.
    return list(await asyncio.gather(*(_run(it) for it in items)))


def map_concurrent_sync(
    items: Sequence[T],
    fn: Callable[[T], R],
    *,
    model: str | None = None,
    capacity: int | None = None,
) -> list[R]:
    """Fan ``fn`` out over ``items`` up to the model's capacity, synchronous.

    CONCEPT:KG-2.143. Uses the shared per-model bounded thread pool. At most
    ``min(len(items), capacity)`` calls run at once. **Results are returned in
    input order.** With capacity ``1`` (or a single item) this runs inline with
    no thread hand-off — byte-for-byte the old sequential behaviour.
    """
    n = len(items)
    if n == 0:
        return []
    cap = max(1, int(capacity)) if capacity is not None else resolve_capacity(model)
    if cap <= 1 or n == 1:
        return [fn(it) for it in items]
    pool = get_thread_pool(model, cap)
    # executor.map preserves input order in its result iterator.
    return list(pool.map(fn, items))
