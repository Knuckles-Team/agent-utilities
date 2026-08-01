from pydantic import BaseModel, Field


class UsageStatistics(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    # CONCEPT:AU-OS.observability.usage-analytics-store (D-54c-1) — provider prompt-cache +
    # reasoning token counts, accumulated the same way as input/output above. Without these,
    # the cost plane (``usage/recorder.py``) and cache-savings telemetry cannot see prompt
    # caching's effect at all, even where the provider reports it.
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    reasoning_tokens: int = 0


class CostModel(BaseModel):
    """Validated per-token price pair from the unified pricing catalog."""

    input_token_price: float = Field(ge=0)
    output_token_price: float = Field(ge=0)

    @classmethod
    def for_model(cls, model: str) -> "CostModel":
        """Build a CostModel from the pricing catalog (per-token from per-Mtok).

        Unknown models are rejected because an unpriced run cannot enforce its
        monetary budget accurately.
        """
        from agent_utilities.pricing import get_pricing_catalog

        pricing = get_pricing_catalog().resolve(model)
        if pricing is None:
            raise LookupError(f"pricing is not configured for model {model!r}")
        return cls(
            input_token_price=pricing.input_per_mtok / 1_000_000,
            output_token_price=pricing.output_per_mtok / 1_000_000,
        )

    def estimate(self, input_tokens: int = 0, output_tokens: int = 0) -> float:
        """Estimate cost in USD for the given token counts."""
        return (
            input_tokens * self.input_token_price
            + output_tokens * self.output_token_price
        )


import time


class ExecutionBudget(BaseModel):
    """CONCEPT:AU-ORCH.execution.execution-budget-caps — Execution Budget.
    Tracks limits for cost, tokens, tool calls, and time to prevent runaway
    recursive executions.

    Every cap defaults to a real, finite ceiling — never unbounded — so a graph
    run is always governed even when a caller never touches this model: a
    5,000-token invoker budget can still be blown by an oversized tool result
    or an unattended research loop without a *graph-level* backstop. Pass an
    explicit ``None`` to deliberately opt a single dimension out for a run that
    genuinely needs it (e.g. an offline batch backfill with no wall-clock
    concern) — enforcement lives in ``graph/_router_impl.py::dispatcher_step``.
    """

    max_cost_usd: float | None = 10.0
    max_total_tokens: int | None = 500_000
    max_node_transitions: int | None = 50
    max_tool_calls: int | None = 200
    """CONCEPT:AU-ORCH.execution.execution-budget-caps — caps ``len(GraphState.tool_calls)``.
    Distinct from ``max_node_transitions``: a single node can invoke several tool
    calls, so a tool-call budget catches a runaway *within* a small number of
    transitions that the transition cap alone would not."""
    start_time: float = Field(default_factory=time.time)
    max_duration_seconds: float | None = 600.0
