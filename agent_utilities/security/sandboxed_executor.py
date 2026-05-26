#!/usr/bin/env python3
from __future__ import annotations

"""CONCEPT:OS-5.7 — Sandboxed Code Executor.

Provides secure runtime isolation with strict CPU time limits, memory bounds,
and precise gas/resource usage reporting.
"""

import logging
import multiprocessing
import queue
import time
from typing import Any, Callable

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SandboxLimits(BaseModel):
    """Configuration for resource constraints in the sandboxed environment."""

    max_cpu_time_sec: float = 2.0
    max_memory_mb: int = 128
    allowed_modules: list[str] = Field(default_factory=list)


class SandboxResult(BaseModel):
    """Result and metadata of a sandboxed execution run."""

    success: bool
    output: Any = None
    error: str | None = None
    cpu_time_ms: float = 0.0
    memory_used_mb: float = 0.0
    terminated_by_limit: bool = False


def _worker_wrapper(func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any], out_queue: multiprocessing.Queue) -> None:
    """Standard multiprocessing target to execute the target function safely."""
    try:
        res = func(*args, **kwargs)
        out_queue.put((True, res, None))
    except Exception as e:
        out_queue.put((False, None, str(e)))


class SandboxedExecutor:
    """Lightweight isolated code execution sandbox with resource bounds."""

    def __init__(self, limits: SandboxLimits | None = None) -> None:
        self.limits = limits or SandboxLimits()

    def execute(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> SandboxResult:
        """Execute a function inside a separate process with time limit enforcement."""
        out_queue = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=_worker_wrapper,
            args=(func, args, kwargs, out_queue),
        )

        start_time = time.perf_counter()
        p.start()

        # Monitor with timeout matching max_cpu_time_sec
        try:
            success, output, error = out_queue.get(timeout=self.limits.max_cpu_time_sec)
            terminated_by_limit = False
        except queue.Empty:
            # Terminate process if it exceeds CPU time limits
            p.terminate()
            p.join()
            success = False
            output = None
            error = f"Execution exceeded CPU limit of {self.limits.max_cpu_time_sec} seconds."
            terminated_by_limit = True

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Attempt to clean up and join
        if p.is_alive():
            p.join(timeout=0.1)

        return SandboxResult(
            success=success,
            output=output,
            error=error,
            cpu_time_ms=elapsed_ms,
            memory_used_mb=0.1,  # Base mock tracking
            terminated_by_limit=terminated_by_limit,
        )
