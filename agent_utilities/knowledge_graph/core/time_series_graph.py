"""Time-Series Weighted Graph Structure (KG-2.7).

CONCEPT: KG-2.7 Time-Series Weighted Graph Structure

Adds temporal capabilities to graph execution. Edges use a decay rate
based on market time, prioritizing recent stochastic shifts for HFT logic.
"""

import math
import time
from typing import Any


class TimeSeriesGraph:
    """Graph structure supporting time-decay weighted edges."""

    def __init__(self, decay_constant: float = 0.05):
        self.decay_constant = decay_constant
        self.edges: list[dict[str, Any]] = []

    def add_temporal_edge(self, source: str, target: str, base_weight: float) -> None:
        """Add an edge with a timestamp for future decay calculation."""
        self.edges.append(
            {
                "source": source,
                "target": target,
                "base_weight": base_weight,
                "timestamp": time.time(),
            }
        )

    def calculate_edge_decay(self, edge: dict[str, Any]) -> float:
        """Calculate the current weight of an edge given temporal decay."""
        age = time.time() - edge["timestamp"]
        # Exponential decay formula: W_t = W_0 * e^(-lambda * t)
        decayed_weight = edge["base_weight"] * math.exp(-self.decay_constant * age)
        return max(0.0, decayed_weight)

    def get_active_edges(self, threshold: float = 0.01) -> list[dict[str, Any]]:
        """Return only edges whose decayed weight is above the threshold."""
        active = []
        for edge in self.edges:
            current_weight = self.calculate_edge_decay(edge)
            if current_weight >= threshold:
                active.append({**edge, "current_weight": current_weight})
        return active
