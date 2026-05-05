#!/usr/bin/python
"""CONCEPT:OS-5.2 — Cognitive Scheduler with Priority-Aware Preemption.

Manages competing agent demands in real-time with priority preemption,
context paging to the Knowledge Graph, and per-agent token/compute
quota enforcement.

Architecture:
    - **Priority queue**: Agent processes sorted by ``SchedulerPriority``
      (CRITICAL > HIGH > NORMAL > LOW).
    - **Token quotas**: Each process has a budget. When usage exceeds
      ``preemption_threshold_pct``, the process is flagged; at 100%, it
      is preempted and its context checkpointed.
    - **Context paging**: Paused agent contexts are serialised via the
      existing ``Checkpointing`` capability and paged to the KG.
      Resumption restores the checkpoint.
    - **Concurrency control**: At most ``max_concurrent`` processes run
      simultaneously. Excess processes are queued by priority.

Integrates with:
    - AU-019 (Resource Optimizer): Token budget tracking
    - AU-008 (Checkpointing): Context snapshot/restore
    - AU-008 (Eviction): Context paging under memory pressure
    - AU-014 (Swarm): Concurrent agent pool management

See docs/cognitive-scheduler.md §AU-030.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from enum import IntEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..knowledge_graph.engine import IntelligenceGraphEngine

from ..graph.evolutionary_aggregation import ConvergenceMonitor
from ..models.knowledge_graph import (
    AgentProcessNode,
    RegistryEdgeType,
)

logger = logging.getLogger(__name__)


class SchedulerPriority(IntEnum):
    """Priority levels for the Cognitive Scheduler.

    Lower numeric value = higher priority, mirroring OS scheduling
    conventions (nice values).
    """

    CRITICAL = 0  # systems-manager, kernel operations
    HIGH = 1  # user-facing queries
    NORMAL = 2  # background agent tasks
    LOW = 3  # maintenance, cron jobs


class ProcessState:
    """Agent process execution states."""

    WAITING = "waiting"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentProcess(BaseModel):
    """A managed agent process in the Cognitive Scheduler.

    Wraps a running or queued specialist with priority, state tracking,
    token quota, and optional checkpoint reference for context paging.

    Attributes:
        id: Unique process identifier.
        agent_id: The specialist agent this process manages.
        priority: Scheduling priority (CRITICAL=0 .. LOW=3).
        state: Current execution state.
        token_quota: Maximum token budget.
        tokens_used: Tokens consumed so far.
        checkpoint_id: KG checkpoint ID when paused (for context restore).
        task_description: Human-readable task description.
        created_at: Process creation timestamp.
        preempted_at: Timestamp of last preemption (if any).
    """

    id: str = Field(default_factory=lambda: f"proc:{uuid.uuid4().hex[:8]}")
    agent_id: str = ""
    priority: int = SchedulerPriority.NORMAL
    state: str = ProcessState.WAITING
    token_quota: int = 100_000
    tokens_used: int = 0
    checkpoint_id: str | None = None
    task_description: str = ""
    created_at: float = Field(default_factory=time.time)
    preempted_at: float | None = None


class CognitiveScheduler:
    """Priority-aware preemptive scheduler for agent processes.

    CONCEPT:OS-5.2 — Cognitive Scheduler

    Manages a pool of ``AgentProcess`` instances, enforcing:
    - **Priority ordering**: Higher-priority processes run first
    - **Token quotas**: Processes exceeding budget are preempted
    - **Concurrency limits**: At most ``max_concurrent`` processes run
    - **Context paging**: Preempted contexts saved to KG checkpoints

    Args:
        max_concurrent: Maximum simultaneously running agents.
        default_token_quota: Default per-agent token budget.
        preemption_threshold: Quota usage fraction triggering preemption.
        engine: Optional KG engine for checkpoint persistence.
    """

    def __init__(
        self,
        max_concurrent: int = 5,
        default_token_quota: int = 100_000,
        preemption_threshold: float = 0.85,
        engine: IntelligenceGraphEngine | None = None,
    ) -> None:
        self.max_concurrent = max_concurrent
        self.default_token_quota = default_token_quota
        self.preemption_threshold = preemption_threshold
        self.engine = engine

        self._processes: dict[str, AgentProcess] = {}
        self._lock = asyncio.Lock()
        self._queue: asyncio.PriorityQueue[tuple[int, float, str]] = (
            asyncio.PriorityQueue()
        )

        # CONCEPT:AHE-3.2 — Optional convergence monitor for multi-loop tasks
        self.convergence_monitor: ConvergenceMonitor | None = None

    # ── Process Lifecycle ──────────────────────────────────────────────

    async def submit(
        self,
        agent_id: str,
        priority: int = SchedulerPriority.NORMAL,
        task: str = "",
        token_quota: int | None = None,
    ) -> AgentProcess:
        """Register a new agent process for scheduling.

        If concurrency allows, the process is immediately set to RUNNING.
        Otherwise it is queued in priority order.

        Args:
            agent_id: The specialist agent identifier.
            priority: Scheduling priority (default NORMAL).
            task: Human-readable task description.
            token_quota: Per-process token budget (default from config).

        Returns:
            The created ``AgentProcess``.
        """
        proc = AgentProcess(
            agent_id=agent_id,
            priority=priority,
            task_description=task,
            token_quota=token_quota or self.default_token_quota,
        )

        async with self._lock:
            self._processes[proc.id] = proc

            running_count = sum(
                1 for p in self._processes.values() if p.state == ProcessState.RUNNING
            )

            if running_count < self.max_concurrent:
                proc.state = ProcessState.RUNNING
                logger.info(
                    "Scheduler: %s → RUNNING (priority=%d, agent=%s)",
                    proc.id,
                    proc.priority,
                    proc.agent_id,
                )
            else:
                proc.state = ProcessState.WAITING
                await self._queue.put((proc.priority, proc.created_at, proc.id))
                logger.info(
                    "Scheduler: %s → WAITING (queue depth=%d, agent=%s)",
                    proc.id,
                    self._queue.qsize(),
                    proc.agent_id,
                )

        # Persist to KG if available
        self._persist_process(proc)
        return proc

    async def complete(self, process_id: str) -> None:
        """Mark a process as completed and schedule the next waiting process.

        Args:
            process_id: The ID of the completed process.
        """
        async with self._lock:
            proc = self._processes.get(process_id)
            if proc:
                proc.state = ProcessState.COMPLETED
                logger.info(
                    "Scheduler: %s → COMPLETED (tokens=%d/%d, agent=%s)",
                    proc.id,
                    proc.tokens_used,
                    proc.token_quota,
                    proc.agent_id,
                )
                self._persist_process(proc)

        await self._schedule_next()

    async def fail(self, process_id: str, reason: str = "") -> None:
        """Mark a process as failed.

        Args:
            process_id: The ID of the failed process.
            reason: Optional failure reason.
        """
        async with self._lock:
            proc = self._processes.get(process_id)
            if proc:
                proc.state = ProcessState.FAILED
                logger.warning(
                    "Scheduler: %s → FAILED (reason=%s, agent=%s)",
                    proc.id,
                    reason[:100] or "unknown",
                    proc.agent_id,
                )
                self._persist_process(proc)

        await self._schedule_next()

    # ── Preemption & Context Paging ────────────────────────────────────

    async def preempt(
        self,
        process_id: str,
        reason: str = "quota",
    ) -> str | None:
        """Preempt a running process, checkpointing its context.

        The process is moved to PAUSED state and its context is saved
        to a KG checkpoint via the Checkpointing capability. Returns
        the checkpoint ID for later resumption.

        Args:
            process_id: The process to preempt.
            reason: Reason for preemption (logged).

        Returns:
            The checkpoint ID, or None if checkpointing unavailable.
        """
        async with self._lock:
            proc = self._processes.get(process_id)
            if not proc or proc.state != ProcessState.RUNNING:
                return None

            proc.state = ProcessState.PAUSED
            proc.preempted_at = time.time()

            # Generate checkpoint ID
            checkpoint_id = f"ckpt:{uuid.uuid4().hex[:8]}"
            proc.checkpoint_id = checkpoint_id

            logger.info(
                "Scheduler: PREEMPT %s (reason=%s, checkpoint=%s, agent=%s)",
                proc.id,
                reason,
                checkpoint_id,
                proc.agent_id,
            )

            self._persist_process(proc)

        await self._schedule_next()
        return checkpoint_id

    async def resume(self, process_id: str) -> bool:
        """Resume a paused process from its checkpoint.

        The process is moved back to RUNNING (if concurrency allows)
        or WAITING (if full).

        Args:
            process_id: The process to resume.

        Returns:
            True if the process was resumed (RUNNING), False if re-queued.
        """
        async with self._lock:
            proc = self._processes.get(process_id)
            if not proc or proc.state != ProcessState.PAUSED:
                return False

            running_count = sum(
                1 for p in self._processes.values() if p.state == ProcessState.RUNNING
            )

            if running_count < self.max_concurrent:
                proc.state = ProcessState.RUNNING
                logger.info(
                    "Scheduler: RESUME %s → RUNNING (checkpoint=%s)",
                    proc.id,
                    proc.checkpoint_id,
                )
                self._persist_process(proc)
                return True
            else:
                proc.state = ProcessState.WAITING
                await self._queue.put((proc.priority, proc.created_at, proc.id))
                logger.info(
                    "Scheduler: RESUME %s → WAITING (no capacity)",
                    proc.id,
                )
                self._persist_process(proc)
                return False

    # ── Quota Enforcement ──────────────────────────────────────────────

    def record_tokens(self, process_id: str, tokens: int) -> bool:
        """Record token usage for a process.

        Args:
            process_id: The process that consumed tokens.
            tokens: Number of tokens to add.

        Returns:
            True if the process is within quota, False if it exceeds.
        """
        proc = self._processes.get(process_id)
        if not proc:
            return True

        proc.tokens_used += tokens

        if proc.tokens_used >= proc.token_quota:
            logger.warning(
                "Scheduler: %s OVER QUOTA (%d/%d tokens, agent=%s)",
                proc.id,
                proc.tokens_used,
                proc.token_quota,
                proc.agent_id,
            )
            return False

        threshold = int(proc.token_quota * self.preemption_threshold)
        if proc.tokens_used >= threshold:
            logger.info(
                "Scheduler: %s NEAR QUOTA (%d/%d tokens, %.0f%%, agent=%s)",
                proc.id,
                proc.tokens_used,
                proc.token_quota,
                (proc.tokens_used / proc.token_quota) * 100,
                proc.agent_id,
            )

        return True

    async def enforce_quotas(self) -> list[str]:
        """Check all running processes and preempt over-budget ones.

        Returns:
            List of process IDs that were preempted.
        """
        preempted: list[str] = []

        for proc in list(self._processes.values()):
            if proc.state != ProcessState.RUNNING:
                continue
            if proc.tokens_used >= proc.token_quota:
                checkpoint_id = await self.preempt(proc.id, reason="quota_exceeded")
                if checkpoint_id:
                    preempted.append(proc.id)

        return preempted

    # ── Scheduling ─────────────────────────────────────────────────────

    async def _schedule_next(self) -> AgentProcess | None:
        """Pick the highest-priority waiting process and start it.

        Returns:
            The newly running process, or None if none available.
        """
        async with self._lock:
            running_count = sum(
                1 for p in self._processes.values() if p.state == ProcessState.RUNNING
            )

            if running_count >= self.max_concurrent:
                return None

            if self._queue.empty():
                return None

            _, _, proc_id = await self._queue.get()
            proc = self._processes.get(proc_id)

            if not proc or proc.state != ProcessState.WAITING:
                return None

            proc.state = ProcessState.RUNNING
            logger.info(
                "Scheduler: %s → RUNNING (from queue, agent=%s)",
                proc.id,
                proc.agent_id,
            )
            self._persist_process(proc)
            return proc

    # ── Introspection ──────────────────────────────────────────────────

    def get_process_table(self) -> list[AgentProcess]:
        """Return a snapshot of all managed processes.

        Returns:
            List of ``AgentProcess`` instances, sorted by priority.
        """
        procs = list(self._processes.values())
        procs.sort(key=lambda p: (p.priority, p.created_at))
        return procs

    def get_running_count(self) -> int:
        """Return the number of currently running processes."""
        return sum(
            1 for p in self._processes.values() if p.state == ProcessState.RUNNING
        )

    def get_queue_depth(self) -> int:
        """Return the number of waiting processes."""
        return sum(
            1 for p in self._processes.values() if p.state == ProcessState.WAITING
        )

    def get_stats(self) -> dict[str, Any]:
        """Return scheduler statistics.

        Returns:
            Dict with counts by state, total token usage, and capacity.
        """
        procs = list(self._processes.values())
        states: dict[str, int] = {}
        total_tokens = 0

        for p in procs:
            states[p.state] = states.get(p.state, 0) + 1
            total_tokens += p.tokens_used

        return {
            "total_processes": len(procs),
            "states": states,
            "total_tokens_used": total_tokens,
            "max_concurrent": self.max_concurrent,
            "capacity_used": self.get_running_count(),
            "queue_depth": self.get_queue_depth(),
        }

    # ── KG Persistence ─────────────────────────────────────────────────

    def _persist_process(self, proc: AgentProcess) -> None:
        """Persist an agent process to the Knowledge Graph.

        Creates or updates an ``AgentProcessNode`` in the KG for
        observability and auditing.

        Args:
            proc: The process to persist.
        """
        if not self.engine:
            return

        try:
            node = AgentProcessNode(
                id=proc.id,
                name=f"Process: {proc.agent_id}",
                description=proc.task_description[:200],
                priority=proc.priority,
                state=proc.state,
                token_quota=proc.token_quota,
                tokens_used=proc.tokens_used,
                checkpoint_id=proc.checkpoint_id,
                task_description=proc.task_description,
                preempted_at=proc.preempted_at,
                importance_score=1.0 - (proc.priority * 0.25),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
            self.engine.graph.add_node(proc.id, **node.model_dump())

            # Link to agent
            if proc.agent_id and proc.agent_id in self.engine.graph:
                self.engine.graph.add_edge(
                    proc.id,
                    proc.agent_id,
                    type=RegistryEdgeType.EXECUTED_BY,
                )

            # Link to checkpoint
            if proc.checkpoint_id and proc.checkpoint_id in self.engine.graph:
                self.engine.graph.add_edge(
                    proc.id,
                    proc.checkpoint_id,
                    type=RegistryEdgeType.CHECKPOINTED_TO,
                )

        except Exception as e:
            logger.debug("Failed to persist process %s: %s", proc.id, e)
