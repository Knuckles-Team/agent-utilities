#!/usr/bin/env python3
from __future__ import annotations

"""CONCEPT:OS-5.6 — Homeostatic Recovery Daemon.

Monitors running processes, detects hung or crashed agent instances,
and triggers automatic restore operations using context paging and checkpoints.
"""

import asyncio
import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.cognitive_scheduler import CognitiveScheduler

import contextlib

logger = logging.getLogger(__name__)


class RecoveryDaemon:
    """A background polling worker that restores failed or timed out agent checkpoints.
    
    Acts as a homeostatic stabilizer for the agent execution pool.
    """

    def __init__(
        self,
        scheduler: CognitiveScheduler,
        check_interval_seconds: float = 2.0,
        max_running_duration_seconds: float = 30.0,
    ) -> None:
        self.scheduler = scheduler
        self.check_interval_seconds = check_interval_seconds
        self.max_running_duration_seconds = max_running_duration_seconds
        self._task: asyncio.Task | None = None
        self._running = False
        self.recovered_count = 0

    def start(self) -> None:
        """Start the background recovery loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._recovery_loop())
        logger.info("RecoveryDaemon started background stabilizers.")

    async def stop(self) -> None:
        """Stop the recovery loop."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        logger.info("RecoveryDaemon stopped stabilizers.")

    async def _recovery_loop(self) -> None:
        """Periodic stabilizer tick."""
        while self._running:
            try:
                await self.stabilize()
            except Exception as e:
                logger.error("Error in RecoveryDaemon tick: %s", e)
            await asyncio.sleep(self.check_interval_seconds)

    async def stabilize(self) -> int:
        """Perform a single sweep of all active processes, recovering failed nodes."""
        recovered = 0
        now = time.time()
        
        # Access all scheduler processes
        processes = list(self.scheduler._processes.values())
        for proc in processes:
            # 1. Recover failed/preempted processes that have a valid checkpoint
            if proc.state == "failed" and proc.checkpoint_id is not None:
                logger.warning(
                    "RecoveryDaemon: Process %s in FAILED state. Resuming from checkpoint: %s",
                    proc.id,
                    proc.checkpoint_id,
                )
                # Re-submit/resume
                proc.state = "paused"
                resumed = await self.scheduler.resume(proc.id)
                if resumed:
                    recovered += 1
                    self.recovered_count += 1
            
            # 2. Prevent hung processes (running longer than threshold)
            elif proc.state == "running":
                elapsed = now - proc.created_at
                if elapsed > self.max_running_duration_seconds:
                    logger.warning(
                        "RecoveryDaemon: Process %s has hung (running for %.1fs > %.1fs). Triggering preemption.",
                        proc.id,
                        elapsed,
                        self.max_running_duration_seconds,
                    )
                    checkpoint_id = await self.scheduler.preempt(proc.id, reason="hang_preemption")
                    if checkpoint_id:
                        recovered += 1
                        self.recovered_count += 1

        return recovered
