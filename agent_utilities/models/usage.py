from pydantic import BaseModel, Field


class UsageStatistics(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


class CostModel(BaseModel):
    """Per-token price pair. Defaults retained for back-compat, but cost is now
    derived from the unified pricing catalog (CONCEPT:AU-ECO.toolkit.model-pricing-catalog) when a model id
    is known. Prefer :meth:`for_model` over the hard-coded defaults.
    """

    input_token_price: float = 0.00000015
    output_token_price: float = 0.0000006

    @classmethod
    def for_model(cls, model: str) -> "CostModel":
        """Build a CostModel from the pricing catalog (per-token from per-Mtok).

        Falls back to the legacy defaults when the model is unknown so callers
        keep working with zero configuration.
        """
        from agent_utilities.pricing import get_pricing_catalog

        pricing = get_pricing_catalog().resolve(model)
        if pricing is None:
            return cls()
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
    Tracks limits for cost, tokens, and time to prevent runaway recursive executions.
    """

    max_cost_usd: float | None = None
    max_total_tokens: int | None = None
    max_node_transitions: int | None = 50
    start_time: float = Field(default_factory=time.time)
    max_duration_seconds: float | None = None
