"""Cgroup-aware CPU/memory envelope detection (U-64 / BUG-110).

CONCEPT:AU-OS.host.cgroup-resource-envelope — the live r18/r19 GraphOS pod ran with a
1.5-CPU / 4-GiB cgroup (`cpu.max=150000 100000`), yet every autosizing call site in this
codebase (`knowledge_graph.core.engine_tasks.compute_ingest_worker_count`,
`runtime.warm_registry.compute_warm_parent_count`) read ``os.cpu_count()`` and
``psutil.virtual_memory()`` — the **host's** 192-core / multi-terabyte view, not the
container's. That produced 69 ingest workers inside a pod that could physically run 1.5 of
them concurrently, driving 65.9% throttled scheduling periods and starving interactive
query/engine-call latency (see ``plans/graph-os-completion-program/bug-analysis.md`` U-64/
U-65/U-73).

This module is the ONE place that reads the effective (host-or-container, whichever is
tighter) CPU and memory envelope. Every autosizing helper must anchor to
:func:`effective_cpu_cores`/:func:`effective_memory_limit_bytes` instead of
``os.cpu_count()``/``psutil.virtual_memory()`` directly, and must clamp its own explicit
override (an env var such as ``KG_INGESTION_WORKERS``) to the same ceiling — an explicit
config value is an escape hatch for *raising* concurrency within the box the process
actually has, never a way to force more workers than the cgroup can schedule.

Supports both cgroup v2 (unified hierarchy, ``/sys/fs/cgroup/cpu.max`` +
``/sys/fs/cgroup/memory.max``) and cgroup v1 (``/sys/fs/cgroup/cpu/cpu.cfs_quota_us`` +
``.../memory/memory.limit_in_bytes``). Every read is best-effort: a missing file, an
unreadable path (non-Linux, no cgroup, host process), or an "unlimited" sentinel all
resolve to ``None`` — the caller then falls back to the host view, so a genuine bare-metal
deployment is unaffected.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

__all__ = [
    "cgroup_cpu_limit_cores",
    "cgroup_memory_limit_bytes",
    "effective_cpu_cores",
    "effective_memory_limit_bytes",
]

# cgroup v1 reports "no limit" as a huge sentinel (typically 2**63 - 4096 on 64-bit
# kernels). Anything at or above this is treated as unlimited, matching the kernel's own
# convention rather than guessing a smaller cutoff.
_V1_UNLIMITED_FLOOR = 1 << 62

_V2_CPU_MAX = "/sys/fs/cgroup/cpu.max"
_V1_CFS_QUOTA = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
_V1_CFS_PERIOD = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"
_V2_MEM_MAX = "/sys/fs/cgroup/memory.max"
_V1_MEM_LIMIT = "/sys/fs/cgroup/memory/memory.limit_in_bytes"


def cgroup_cpu_limit_cores() -> float | None:
    """The effective CPU quota in fractional cores, or ``None`` if unset/unlimited/unreadable.

    cgroup v2: ``cpu.max`` is ``"<quota> <period>"`` in microseconds, or ``"max <period>"``
    for no limit. cgroup v1: the same ratio split across ``cpu.cfs_quota_us`` (``-1`` for
    unlimited) and ``cpu.cfs_period_us``.
    """
    try:
        with open(_V2_CPU_MAX, encoding="ascii") as fh:
            parts = fh.read().split()
        if len(parts) == 2:
            quota_str, period_str = parts
            if quota_str != "max":
                quota, period = int(quota_str), int(period_str)
                if period > 0 and quota > 0:
                    return quota / period
                return None
    except (FileNotFoundError, NotADirectoryError, ValueError, OSError):  # noqa: BLE001 — v2 unreadable/malformed, fall through to the v1 read
        pass

    try:
        with open(_V1_CFS_QUOTA, encoding="ascii") as fh:
            quota = int(fh.read().strip())
        with open(_V1_CFS_PERIOD, encoding="ascii") as fh:
            period = int(fh.read().strip())
        if quota > 0 and period > 0:
            return quota / period
    except (FileNotFoundError, NotADirectoryError, ValueError, OSError):  # noqa: BLE001 — neither cgroup version present/readable (bare-metal/non-Linux); caller falls back to the host view
        pass

    return None


def cgroup_memory_limit_bytes() -> int | None:
    """The effective memory ceiling in bytes, or ``None`` if unset/unlimited/unreadable."""
    try:
        with open(_V2_MEM_MAX, encoding="ascii") as fh:
            raw = fh.read().strip()
        if raw and raw != "max":
            value = int(raw)
            if value > 0:
                return value
            return None
    except (FileNotFoundError, NotADirectoryError, ValueError, OSError):  # noqa: BLE001 — v2 unreadable/malformed, fall through to the v1 read
        pass

    try:
        with open(_V1_MEM_LIMIT, encoding="ascii") as fh:
            value = int(fh.read().strip())
        if 0 < value < _V1_UNLIMITED_FLOOR:
            return value
    except (FileNotFoundError, NotADirectoryError, ValueError, OSError):  # noqa: BLE001 — neither cgroup version present/readable (bare-metal/non-Linux); caller falls back to the host view
        pass

    return None


def effective_cpu_cores() -> float:
    """The tighter of the cgroup CPU quota and the host's visible core count.

    Never exceeds ``os.cpu_count()`` (a cgroup quota can legitimately request fractional
    cores, e.g. ``1.5``, that is still smaller than one whole host core count) and never
    exceeds the host even when no cgroup limit is present, so a bare-metal/non-Linux
    process keeps today's host-sized behavior.
    """
    host = float(os.cpu_count() or 4)
    limit = cgroup_cpu_limit_cores()
    if limit is None:
        return host
    return min(limit, host)


def effective_memory_limit_bytes() -> int | None:
    """The tighter of the cgroup memory ceiling and host-visible available memory.

    Returns ``None`` only when neither a cgroup limit nor ``psutil`` is available, in
    which case the caller must fall back to its own conservative default (never an
    unbounded assumption).
    """
    limit = cgroup_memory_limit_bytes()
    try:
        import psutil

        host_available = int(psutil.virtual_memory().available)
    except Exception as exc:  # noqa: BLE001 — psutil optional/best-effort
        logger.debug("psutil memory read failed: %s", exc)
        host_available = None

    if limit is not None and host_available is not None:
        return min(limit, host_available)
    if limit is not None:
        return limit
    return host_available
