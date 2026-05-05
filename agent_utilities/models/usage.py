from pydantic import BaseModel, Field


class UsageStatistics(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


class CostModel(BaseModel):
    input_token_price: float = 0.00000015
    output_token_price: float = 0.0000006


import time


class ExecutionBudget(BaseModel):
    """CONCEPT:ORCH-1.3 — Execution Budget.
    Tracks limits for cost, tokens, and time to prevent runaway recursive executions.
    """

    max_cost_usd: float | None = None
    max_total_tokens: int | None = None
    max_node_transitions: int | None = 50
    start_time: float = Field(default_factory=time.time)
    max_duration_seconds: float | None = None
