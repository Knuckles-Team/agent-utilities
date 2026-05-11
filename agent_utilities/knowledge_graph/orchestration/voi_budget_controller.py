#!/usr/bin/env python3
"""Value of Information (VOI) Budget Controller.

Implements CONCEPT:KG-2.5 (VOI Budget Controller)
Enforces graph traversal scaling laws by computing the marginal utility of expanding exploration.
"""

import logging

from ..core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)


class VOIBudgetController:
    """Controls graph traversal budgets using Value of Information scaling laws."""

    def __init__(self, engine: IntelligenceGraphEngine, base_budget: int = 100):
        self.engine = engine
        self.base_budget = base_budget

    def compute_marginal_utility(self, current_nodes: int, max_nodes: int) -> float:
        """Compute the marginal utility of continuing traversal.

        Uses an asymptotic scaling law where utility drops as current_nodes approaches max_nodes.
        """
        if current_nodes >= max_nodes:
            return 0.0
        return 1.0 - (current_nodes / max_nodes) ** 2

    def should_continue_traversal(
        self, current_nodes_visited: int, dynamic_budget: int | None = None
    ) -> bool:
        """Determine if graph traversal should continue based on VOI budget.

        Args:
            current_nodes_visited: Number of nodes visited so far.
            dynamic_budget: Optional override for the base budget.

        Returns:
            True if traversal should continue, False if budget is exhausted.
        """
        budget = dynamic_budget if dynamic_budget is not None else self.base_budget

        if current_nodes_visited >= budget:
            logger.info(f"VOI Budget Exhausted: {current_nodes_visited} >= {budget}")
            return False

        utility = self.compute_marginal_utility(current_nodes_visited, budget)

        # Stop if marginal utility drops below 5%
        if utility < 0.05:
            logger.info(f"VOI Utility too low ({utility:.2f}), halting traversal.")
            return False

        return True
