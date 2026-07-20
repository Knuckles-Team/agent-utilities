#!/usr/bin/python
from __future__ import annotations

"""Resource-Aware Optimization — CONCEPT:AU-OS.state.cognitive-scheduler-preemption

Cost-aware model selection, per-specialist budget allocation,
latency-aware routing, and resource usage tracking with KG persistence.

Design-pattern source: Chapter 16 — Resource-Aware Optimization.

OWL: :ResourceUsage rdfs:subClassOf bfo:Process
See docs/pillars/architecture_c4.md §CONCEPT:AU-OS.state.cognitive-scheduler-preemption
"""


import logging
import time
from typing import Any

from pydantic import BaseModel, Field

from agent_utilities.core.config import setting

logger = logging.getLogger(__name__)

DEFAULT_TOKEN_BUDGET = int(setting("SESSION_TOKEN_BUDGET", "500000"))
DEFAULT_COST_BUDGET = float(setting("SESSION_COST_BUDGET_USD", "5.0"))
DEFAULT_LATENCY_BUDGET = int(setting("SESSION_LATENCY_BUDGET_MS", "30000"))
WEIGHT_COST = float(setting("RESOURCE_WEIGHT_COST", "0.4"))
WEIGHT_LATENCY = float(setting("RESOURCE_WEIGHT_LATENCY", "0.3"))
WEIGHT_QUALITY = float(setting("RESOURCE_WEIGHT_QUALITY", "0.3"))


class AllocationSlice(BaseModel):
    """Budget allocation for a single specialist."""

    specialist_id: str
    token_budget: int = 0
    cost_budget_usd: float = 0.0
    tokens_used: int = 0
    cost_used_usd: float = 0.0
    latency_ms: float = 0.0


class ResourceBudget(BaseModel):
    """Per-session resource budget with allocation tracking."""

    total_token_budget: int = Field(default=DEFAULT_TOKEN_BUDGET)
    total_cost_budget_usd: float = Field(default=DEFAULT_COST_BUDGET)
    latency_budget_ms: int = Field(default=DEFAULT_LATENCY_BUDGET)
    allocated: dict[str, AllocationSlice] = Field(default_factory=dict)
    tokens_used: int = 0
    cost_used_usd: float = 0.0
    elapsed_ms: float = 0.0

    @property
    def tokens_remaining(self) -> int:
        return max(0, self.total_token_budget - self.tokens_used)

    @property
    def cost_remaining(self) -> float:
        return max(0.0, self.total_cost_budget_usd - self.cost_used_usd)

    @property
    def utilization_pct(self) -> float:
        if self.total_token_budget == 0:
            return 0.0
        return (self.tokens_used / self.total_token_budget) * 100


class ResourceUsageRecord(BaseModel):
    """Immutable record of resource consumption for a single operation."""

    specialist_id: str
    model_id: str = ""
    model_tier: str = "medium"
    tokens_input: int = 0
    tokens_output: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    timestamp: float = Field(default_factory=time.time)


class ResourceOptimizer:
    """Cost-aware model selection and resource allocation.

    Parameters
    ----------
    budget : ResourceBudget
        The session-level resource budget.
    model_registry : optional
        If provided, enables tier-based model selection.
    kg_engine : optional
        If provided, usage records are persisted to the KG.
    """

    def __init__(
        self,
        budget: ResourceBudget | None = None,
        model_registry: Any = None,
        kg_engine: Any = None,
    ) -> None:
        self.budget = budget or ResourceBudget()
        self._registry = model_registry
        self._engine = kg_engine
        self._records: list[ResourceUsageRecord] = []

    def allocate_budget(
        self,
        specialist_ids: list[str],
        weights: dict[str, float] | None = None,
    ) -> dict[str, AllocationSlice]:
        """Allocate budget across adaptive_agent_router (equal or weighted)."""
        n = len(specialist_ids)
        if n == 0:
            return {}
        if weights is None:
            weights = {sid: 1.0 / n for sid in specialist_ids}
        total_weight = sum(weights.values()) or 1.0
        allocations: dict[str, AllocationSlice] = {}
        for sid in specialist_ids:
            w = weights.get(sid, 1.0 / n) / total_weight
            allocations[sid] = AllocationSlice(
                specialist_id=sid,
                token_budget=int(self.budget.total_token_budget * w),
                cost_budget_usd=self.budget.total_cost_budget_usd * w,
            )
        self.budget.allocated = allocations
        return allocations

    def select_model_for_step(
        self,
        complexity: str = "medium",
        required_tags: list[str] | None = None,
        confidence_signal: float | None = None,
        routing_percentile: float = 50.0,
    ) -> dict[str, Any] | None:
        """Select optimal model based on remaining budget and complexity.

        When a ``confidence_signal`` is provided and the registry supports
        :meth:`pick_for_task_adaptive`, the confidence signal is forwarded
        for CONCEPT:AU-ORCH.routing.confidence-signal-forwarding confidence-gated routing.  Otherwise falls back to the
        standard tier-based selection.

        Args:
            complexity: Requested model tier.
            required_tags: Tags every candidate must carry.
            confidence_signal: Optional normalised confidence ``[0, 1]``
                from upstream scoring.  CONCEPT:AU-ORCH.routing.confidence-signal-forwarding
            routing_percentile: Threshold for confidence gating.
                CONCEPT:AU-ORCH.routing.confidence-signal-forwarding
        """
        if self._registry is None:
            return None
        remaining_pct = self.budget.cost_remaining / max(
            self.budget.total_cost_budget_usd, 0.01
        )
        if remaining_pct < 0.2:
            effective_complexity = "light"
        elif remaining_pct < 0.5 and complexity in ("heavy", "reasoning"):
            effective_complexity = "medium"
        else:
            effective_complexity = complexity

        # CONCEPT:AU-ORCH.routing.confidence-signal-forwarding — Forward confidence signal when available
        if confidence_signal is not None and hasattr(
            self._registry, "pick_for_task_adaptive"
        ):
            return self._registry.pick_for_task_adaptive(
                complexity=effective_complexity,
                confidence_signal=confidence_signal,
                routing_percentile=routing_percentile,
                required_tags=required_tags or [],
            )

        if hasattr(self._registry, "pick_for_task"):
            return self._registry.pick_for_task(
                complexity=effective_complexity,
                required_tags=required_tags or [],
            )
        return None

    def record_usage(
        self,
        specialist_id: str,
        tokens_input: int = 0,
        tokens_output: int = 0,
        cost_usd: float = 0.0,
        latency_ms: float = 0.0,
        model_id: str = "",
        model_tier: str = "medium",
    ) -> ResourceUsageRecord:
        """Record resource consumption for a specialist operation."""
        total_tokens = tokens_input + tokens_output
        record = ResourceUsageRecord(
            specialist_id=specialist_id,
            model_id=model_id,
            model_tier=model_tier,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
        )
        self._records.append(record)
        self.budget.tokens_used += total_tokens
        self.budget.cost_used_usd += cost_usd
        self.budget.elapsed_ms += latency_ms
        if specialist_id in self.budget.allocated:
            alloc = self.budget.allocated[specialist_id]
            alloc.tokens_used += total_tokens
            alloc.cost_used_usd += cost_usd
            alloc.latency_ms += latency_ms
        logger.debug(
            "Resource usage: specialist=%s tokens=%d cost=$%.4f | %.1f%% used",
            specialist_id,
            total_tokens,
            cost_usd,
            self.budget.utilization_pct,
        )
        return record

    def is_budget_exceeded(self) -> bool:
        """Check if the session budget has been exceeded."""
        return (
            self.budget.tokens_used >= self.budget.total_token_budget
            or self.budget.cost_used_usd >= self.budget.total_cost_budget_usd
        )

    def get_efficiency_score(self, specialist_id: str) -> float:
        """Compute composite efficiency score for a specialist."""
        alloc = self.budget.allocated.get(specialist_id)
        if alloc is None or alloc.cost_budget_usd == 0:
            return 0.5
        cost_eff = max(
            0.0, 1.0 - (alloc.cost_used_usd / max(alloc.cost_budget_usd, 0.01))
        )
        latency_eff = max(
            0.0, 1.0 - (alloc.latency_ms / max(self.budget.latency_budget_ms, 1))
        )
        quality_eff = min(1.0, alloc.tokens_used / max(alloc.token_budget, 1))
        return (
            WEIGHT_COST * cost_eff
            + WEIGHT_LATENCY * latency_eff
            + WEIGHT_QUALITY * quality_eff
        )

    def summary(self) -> dict[str, Any]:
        """Return a summary of resource usage for the session."""
        return {
            "tokens_used": self.budget.tokens_used,
            "tokens_budget": self.budget.total_token_budget,
            "tokens_remaining": self.budget.tokens_remaining,
            "cost_used_usd": round(self.budget.cost_used_usd, 4),
            "cost_budget_usd": self.budget.total_cost_budget_usd,
            "cost_remaining_usd": round(self.budget.cost_remaining, 4),
            "utilization_pct": round(self.budget.utilization_pct, 1),
            "total_operations": len(self._records),
            "budget_exceeded": self.is_budget_exceeded(),
        }
