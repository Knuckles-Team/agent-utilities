"""Standardized Quant Agent API (ECO-4.7).

CONCEPT: ECO-4.7 Standardized Quant Agent API (SAAPI)

Base abstract class enforcing `receive_tick()`, `send_order()`, and `evaluate_risk()`
for all quantitative agents.
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class QuantAgentTemplate(BaseModel, ABC):
    """Abstract Base Class enforcing the quantitative trading agent protocol."""

    agent_id: str
    strategy_name: str

    @abstractmethod
    def receive_tick(self, symbol: str, price: float, timestamp: float) -> None:
        """Process incoming market data tick."""
        pass

    @abstractmethod
    def send_order(
        self, symbol: str, side: str, quantity: float, price: float
    ) -> dict[str, Any]:
        """Emit an order execution request."""
        pass

    @abstractmethod
    def evaluate_risk(self, _portfolio: dict[str, Any]) -> float:
        """Calculate the risk exposure of the current portfolio state."""
        pass
