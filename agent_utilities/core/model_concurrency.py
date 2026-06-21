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
import time
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


def _status_of(exc: BaseException) -> int | None:
    """Best-effort HTTP status extraction from a raised exception.

    Works across httpx/openai/requests/urllib without importing any of them: it
    duck-types the common attributes (``status_code``, ``.response.status_code``,
    ``.code``, ``.status``). Returns ``None`` when no status is discernible — the
    controller then treats it as an opaque-under-load congestion-ish failure.
    """
    for attr in ("status_code", "status", "code"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    resp = getattr(exc, "response", None)
    if resp is not None:
        sc = getattr(resp, "status_code", None)
        if isinstance(sc, int):
            return sc
    return None


def _record(
    model: str | None, *, latency_s: float, ok: bool, status: int | None
) -> None:
    """Side-channel: feed an observed call into the adaptive controller.

    Pure observation — never raises, never affects the fan-out contract
    (CONCEPT:KG-2.145).
    """
    try:
        from agent_utilities.core.model_capacity_autoscale import record_sample

        record_sample(model, latency_s=latency_s, ok=ok, status=status)
    except Exception:  # noqa: BLE001 — observation must never break fan-out
        pass


def resolve_capacity(model: str | None = None, default: int = 1) -> int:
    """Resolve a model's parallel-call capacity (CONCEPT:KG-2.143 / KG-2.145).

    Looks the model up in the live config registry (by id or role; see
    :meth:`Config.model_capacity`) for the *static* capacity, then lets the
    adaptive controller (CONCEPT:KG-2.145) raise/lower it toward the model's real
    vLLM serving capacity using that static value as the **floor**. When adaptive
    concurrency is disabled, or the model has no scrapeable endpoint, or any
    metrics scrape fails, this returns the static capacity unchanged — fail-safe,
    never below the configured floor.

    Any failure — no config, unknown model, import error — collapses to
    ``default`` (``1``), i.e. sequential, never zero.
    """
    try:
        from agent_utilities.core.config import config

        static_cap = max(1, int(config.model_capacity(model)))
    except Exception:  # noqa: BLE001 — capacity is best-effort; stay safe-sequential
        return max(1, int(default))
    try:
        from agent_utilities.core.model_capacity_autoscale import adaptive_capacity

        return adaptive_capacity(model, static_cap)
    except Exception:  # noqa: BLE001 — adaptation is best-effort; fall back to static
        return static_cap


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
    """Drop all cached gates/pools (test isolation; config reload). CONCEPT:KG-2.143.

    Also drops the adaptive-capacity controllers (CONCEPT:KG-2.145) so a reload
    re-derives both the gate size and its auto-tune state from fresh config.
    """
    with _lock:
        _semaphores.clear()
        pools = list(_pools.values())
        _pools.clear()
    for p in pools:
        p.shutdown(wait=False)
    try:
        from agent_utilities.core.model_capacity_autoscale import (
            reset_adaptive_controllers,
        )

        reset_adaptive_controllers()
    except Exception:  # noqa: BLE001 — best-effort cleanup
        pass


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
            start = time.monotonic()
            try:
                if is_coro:
                    result = await fn(item)  # type: ignore[misc]
                else:
                    result = await asyncio.to_thread(fn, item)  # type: ignore[arg-type]
            except BaseException as exc:  # noqa: BLE001 — observe then re-raise
                _record(
                    model,
                    latency_s=time.monotonic() - start,
                    ok=False,
                    status=_status_of(exc),
                )
                raise
            _record(model, latency_s=time.monotonic() - start, ok=True, status=None)
            return result

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

    def _timed(item: T) -> R:
        start = time.monotonic()
        try:
            result = fn(item)
        except BaseException as exc:  # noqa: BLE001 — observe then re-raise
            _record(
                model,
                latency_s=time.monotonic() - start,
                ok=False,
                status=_status_of(exc),
            )
            raise
        _record(model, latency_s=time.monotonic() - start, ok=True, status=None)
        return result

    if cap <= 1 or n == 1:
        return [_timed(it) for it in items]
    pool = get_thread_pool(model, cap)
    # executor.map preserves input order in its result iterator.
    return list(pool.map(_timed, items))
