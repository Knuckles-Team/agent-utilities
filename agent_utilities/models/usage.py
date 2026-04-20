from pydantic import BaseModel


class UsageStatistics(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


class CostModel(BaseModel):
    input_token_price: float = 0.00000015
    output_token_price: float = 0.0000006
