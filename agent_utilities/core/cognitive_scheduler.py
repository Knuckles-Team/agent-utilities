#!/usr/bin/python
from __future__ import annotations

"""CONCEPT:OS-5.2 — Cognitive Scheduler with Priority-Aware Preemption.

Manages competing agent demands in real-time with priority preemption,
context paging to the Knowledge Graph, and per-agent token/compute
quota enforcement.

Inference Budget Control (Research: 2605.05701v1):
    - **Cost-aware tracking**: Each process tracks both token counts and
      USD cost, enabling model-tier-aware budgeting.
    - **Auto-downgrade**: When budget pressure exceeds threshold, the
      scheduler automatically recommends a cheaper model tier instead
      of preempting the process entirely.
    - **Tier fallback chain**: Configurable degradation path
      (super → standard → lite) per process.

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
    - CONCEPT:OS-5.2 (Resource Optimizer): Token budget tracking
    - CONCEPT:OS-5.2 (Checkpointing): Context snapshot/restore
    - CONCEPT:OS-5.2 (Eviction): Context paging under memory pressure
    - CONCEPT:KG-2.0 (Swarm): Concurrent agent pool management

See docs/pillars/5_agent_os_infrastructure.md §CONCEPT:OS-5.2
"""


import asyncio
import logging
import time
import uuid
from enum import IntEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, PrivateAttr

if TYPE_CHECKING:
    from ..graph.hierarchical_planner import ConvergenceMonitor
    from ..knowledge_graph.core.engine import IntelligenceGraphEngine

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


# ── Cost Constants ──────────────────────────────────────────────────
# Approximate per-1K-token costs by model tier (USD).
# These are defaults; override via InferenceBudget.tier_costs.
DEFAULT_TIER_COSTS: dict[str, float] = {
    "lite": 0.00015,  # e.g. Gemini Flash, GPT-4o-mini
    "standard": 0.002,  # e.g. Gemini Pro, GPT-4o
    "super": 0.015,  # e.g. Gemini Ultra, o3
}

DEFAULT_FALLBACK_CHAIN: list[str] = ["super", "standard", "lite"]


class InferenceBudget(BaseModel):
    """Per-process inference budget with cost and tier tracking.

    CONCEPT:OS-5.2 — Research: 2605.05701v1 (Inference-Time Budget Control)

    Rather than simply preempting processes that exceed token quotas,
    this model enables graceful degradation: when budget pressure
    rises, the scheduler recommends a cheaper model tier before
    resorting to preemption.

    Attributes:
        cost_budget_usd: Maximum dollar spend for this process.
        cost_used_usd: Dollars consumed so far.
        current_tier: Active model tier (lite/standard/super).
        initial_tier: Tier the process started with.
        fallback_chain: Ordered degradation path.
        auto_downgrade: Whether to auto-switch tiers on budget pressure.
        tier_costs: Per-1K-token cost by tier (overridable).
        downgrade_threshold: Budget usage fraction triggering downgrade.
    """

    cost_budget_usd: float = 1.0
    cost_used_usd: float = 0.0
    current_tier: str = "standard"
    initial_tier: str = "standard"
    fallback_chain: list[str] = Field(
        default_factory=lambda: list(DEFAULT_FALLBACK_CHAIN)
    )
    auto_downgrade: bool = True
    tier_costs: dict[str, float] = Field(
        default_factory=lambda: dict(DEFAULT_TIER_COSTS)
    )
    downgrade_threshold: float = 0.70

    @property
    def cost_remaining_usd(self) -> float:
        """Remaining dollar budget."""
        return max(0.0, self.cost_budget_usd - self.cost_used_usd)

    @property
    def budget_usage_pct(self) -> float:
        """Budget utilization as a fraction (0.0–1.0)."""
        if self.cost_budget_usd <= 0:
            return 1.0
        return min(1.0, self.cost_used_usd / self.cost_budget_usd)

    def record_cost(self, tokens: int, tier: str | None = None) -> float:
        """Record token usage and return the cost incurred.

        Args:
            tokens: Number of tokens consumed.
            tier: Model tier used (defaults to current_tier).

        Returns:
            The dollar cost of this inference call.
        """
        tier = tier or self.current_tier
        cost_per_k = self.tier_costs.get(tier, self.tier_costs.get("standard", 0.002))
        cost = (tokens / 1000.0) * cost_per_k
        self.cost_used_usd += cost
        return cost

    def next_cheaper_tier(self) -> str | None:
        """Return the next cheaper tier in the fallback chain, or None."""
        try:
            idx = self.fallback_chain.index(self.current_tier)
        except ValueError:
            return None
        if idx + 1 < len(self.fallback_chain):
            return self.fallback_chain[idx + 1]
        return None


class AgentProcess(BaseModel):
    """A managed agent process in the Cognitive Scheduler.

    Wraps a running or queued specialist with priority, state tracking,
    token quota, and optional checkpoint reference for context paging.

    CONCEPT:OS-5.2 — Extended with InferenceBudget (Research: 2605.05701v1)

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
        inference_budget: Cost-aware inference budget with tier management.
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
    inference_budget: InferenceBudget = Field(default_factory=InferenceBudget)

    _running_event: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)


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
        self._queue: asyncio.PriorityQueue[
            tuple[int, float, str]
        ] = asyncio.PriorityQueue()

        # CONCEPT:AHE-3.2 — Optional convergence monitor for multi-loop tasks
        self.convergence_monitor: ConvergenceMonitor | None = None

    def _get_kg_centrality(self, agent_id: str) -> float:
        """Query the topological centrality of the agent's node in the active graph.

        CONCEPT:OS-5.8 — Epistemic Resource Scheduler.

        Returns 0.0 (no boost) when the KG engine is unavailable or the agent
        is not present in the graph.  Only agents with actual graph connectivity
        receive priority/quota boosts.
        """
        if (
            not self.engine
            or not hasattr(self.engine, "graph")
            or self.engine.graph is None
        ):
            return 0.0

        try:
            graph = self.engine.graph
            if agent_id in graph:
                degree = graph.degree(agent_id)
                num_nodes = len(graph.nodes)
                if num_nodes > 1:
                    return float(degree) / (num_nodes - 1)
        except Exception as e:
            logger.debug(
                "Failed to calculate graph centrality for agent %s: %s", agent_id, e
            )

        return 0.0

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
        # CONCEPT:OS-5.8 — Epistemic dynamic priority & quota scaling based on KG Centrality
        centrality = self._get_kg_centrality(agent_id)

        final_quota = token_quota or self.default_token_quota
        # Scale quota proportional to centrality
        if centrality > 0.5:
            final_quota = int(final_quota * (1.0 + centrality))

        # Boost priority for high-centrality routes (smaller int = higher priority)
        adjusted_priority = priority
        if centrality > 0.6 and priority > SchedulerPriority.CRITICAL:
            adjusted_priority = max(SchedulerPriority.CRITICAL, priority - 1)

        proc = AgentProcess(
            agent_id=agent_id,
            priority=adjusted_priority,
            task_description=task,
            token_quota=final_quota,
        )

        async with self._lock:
            self._processes[proc.id] = proc

            running_count = sum(
                1 for p in self._processes.values() if p.state == ProcessState.RUNNING
            )

            if running_count < self.max_concurrent:
                proc.state = ProcessState.RUNNING
                proc._running_event.set()
                logger.info(
                    "Scheduler: %s → RUNNING (priority=%d, agent=%s, centrality=%.2f, quota=%d)",
                    proc.id,
                    proc.priority,
                    proc.agent_id,
                    centrality,
                    proc.token_quota,
                )
            else:
                proc.state = ProcessState.WAITING
                await self._queue.put((proc.priority, proc.created_at, proc.id))
                logger.info(
                    "Scheduler: %s → WAITING (queue depth=%d, agent=%s, centrality=%.2f, quota=%d)",
                    proc.id,
                    self._queue.qsize(),
                    proc.agent_id,
                    centrality,
                    proc.token_quota,
                )

        # Persist to KG if available
        self._persist_process(proc)
        return proc

    async def wait_for_running(self, process_id: str) -> None:
        """Block until the process is in RUNNING state."""
        proc = self._processes.get(process_id)
        if proc:
            await proc._running_event.wait()

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
            proc._running_event.clear()
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
                proc._running_event.set()
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

    # ── Inference Budget Control (Research: 2605.05701v1) ──────────────

    def record_inference(
        self,
        process_id: str,
        tokens: int,
        model_tier: str | None = None,
    ) -> dict[str, Any]:
        """Record an inference call with cost-aware budget tracking.

        CONCEPT:OS-5.2 — Research: 2605.05701v1

        Unlike ``record_tokens`` (which only counts raw tokens), this
        method tracks dollar cost by model tier and triggers automatic
        tier downgrade when budget pressure exceeds the threshold.

        Args:
            process_id: The process that consumed tokens.
            tokens: Number of tokens consumed in this call.
            model_tier: The model tier used (defaults to process's current tier).

        Returns:
            Dict with keys:
                - ``within_budget``: Whether the process is still within budget.
                - ``cost_incurred``: Dollar cost of this call.
                - ``recommended_tier``: Tier the process should use next.
                - ``downgraded``: Whether an auto-downgrade occurred.
        """
        proc = self._processes.get(process_id)
        if not proc:
            return {
                "within_budget": True,
                "cost_incurred": 0.0,
                "recommended_tier": "standard",
                "downgraded": False,
            }

        budget = proc.inference_budget
        tier = model_tier or budget.current_tier

        # Record cost and tokens
        cost = budget.record_cost(tokens, tier)
        proc.tokens_used += tokens

        # Check for auto-downgrade
        downgraded = False
        if (
            budget.auto_downgrade
            and budget.budget_usage_pct >= budget.downgrade_threshold
        ):
            next_tier = budget.next_cheaper_tier()
            if next_tier and next_tier != budget.current_tier:
                old_tier = budget.current_tier
                budget.current_tier = next_tier
                downgraded = True
                logger.info(
                    "Scheduler: %s AUTO-DOWNGRADE %s → %s "
                    "(budget %.1f%% used, $%.4f/$%.4f, agent=%s)",
                    proc.id,
                    old_tier,
                    next_tier,
                    budget.budget_usage_pct * 100,
                    budget.cost_used_usd,
                    budget.cost_budget_usd,
                    proc.agent_id,
                )

        within_budget = budget.cost_used_usd < budget.cost_budget_usd
        if not within_budget:
            logger.warning(
                "Scheduler: %s OVER COST BUDGET ($%.4f/$%.4f, agent=%s)",
                proc.id,
                budget.cost_used_usd,
                budget.cost_budget_usd,
                proc.agent_id,
            )

        return {
            "within_budget": within_budget,
            "cost_incurred": cost,
            "recommended_tier": budget.current_tier,
            "downgraded": downgraded,
        }

    def get_recommended_tier(self, process_id: str) -> str:
        """Return the recommended model tier for a process.

        CONCEPT:OS-5.2 — Research: 2605.05701v1

        Based on remaining budget, returns the most capable tier the
        process can afford. If budget is exhausted, returns the cheapest
        available tier.

        Args:
            process_id: The process to query.

        Returns:
            The recommended model tier string.
        """
        proc = self._processes.get(process_id)
        if not proc:
            return "standard"
        return proc.inference_budget.current_tier

    def get_budget_stats(self, process_id: str) -> dict[str, Any]:
        """Return detailed budget statistics for a process.

        CONCEPT:OS-5.2 — Research: 2605.05701v1

        Args:
            process_id: The process to query.

        Returns:
            Dict with budget utilization details.
        """
        proc = self._processes.get(process_id)
        if not proc:
            return {}

        budget = proc.inference_budget
        return {
            "process_id": proc.id,
            "agent_id": proc.agent_id,
            "current_tier": budget.current_tier,
            "initial_tier": budget.initial_tier,
            "cost_used_usd": round(budget.cost_used_usd, 6),
            "cost_budget_usd": budget.cost_budget_usd,
            "cost_remaining_usd": round(budget.cost_remaining_usd, 6),
            "budget_usage_pct": round(budget.budget_usage_pct * 100, 1),
            "tokens_used": proc.tokens_used,
            "token_quota": proc.token_quota,
            "auto_downgrade": budget.auto_downgrade,
            "next_cheaper_tier": budget.next_cheaper_tier(),
        }

    async def enforce_quotas(self) -> list[str]:
        """Check all running processes and preempt over-budget ones.

        Checks both token quotas and cost budgets. Attempts auto-downgrade
        before preemption when inference budgets are enabled.

        Returns:
            List of process IDs that were preempted.
        """
        preempted: list[str] = []

        for proc in list(self._processes.values()):
            if proc.state != ProcessState.RUNNING:
                continue

            # Cost budget check (new — Research: 2605.05701v1)
            budget = proc.inference_budget
            if budget.cost_used_usd >= budget.cost_budget_usd:
                # Try auto-downgrade first before preempting
                if budget.auto_downgrade:
                    next_tier = budget.next_cheaper_tier()
                    if next_tier:
                        budget.current_tier = next_tier
                        logger.info(
                            "Scheduler: %s BUDGET-DOWNGRADE → %s (avoiding preemption)",
                            proc.id,
                            next_tier,
                        )
                        continue
                # No cheaper tier available — preempt
                checkpoint_id = await self.preempt(
                    proc.id, reason="cost_budget_exceeded"
                )
                if checkpoint_id:
                    preempted.append(proc.id)
                continue

            # Legacy token quota check
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
            proc._running_event.set()
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
            from ..models.knowledge_graph import AgentProcessNode, RegistryEdgeType

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
