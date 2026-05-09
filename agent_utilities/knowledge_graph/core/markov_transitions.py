#!/usr/bin/env python3
"""Markov Chain Transitions and Vectorized Topologies.

CONCEPT:KG-2.49 — Markov Transition Forecasting

Implements Markov Chain transition matrices over agent interaction traces
(Vectorized Topologies) from *Mathematics for Computer Science* (MCS Ch 21).
Calculates the stationary distribution (Eigenvector) to predict where an
agent is statistically most likely to fail or reach a sink node.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence

import numpy as np

logger = logging.getLogger(__name__)


class MarkovTransitionModel:
    """Predictive model based on Markov Chains and Stationary Distributions.

    Builds a transition matrix from historical agent execution traces
    and uses power iteration to find the stationary distribution.
    """

    def __init__(self):
        self.state_counts: dict[str, int] = defaultdict(int)
        self.transitions: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.states: list[str] = []
        self._state_to_idx: dict[str, int] = {}
        self.transition_matrix: np.ndarray | None = None

    def ingest_trace(self, trace: Sequence[str]) -> None:
        """Ingest a sequential execution trace of states."""
        if len(trace) < 1:
            return

        self.state_counts[trace[0]] += 1
        for i in range(len(trace) - 1):
            src = trace[i]
            dst = trace[i + 1]
            self.transitions[src][dst] += 1
            self.state_counts[dst] += 1

        self._rebuild_matrix()

    def _rebuild_matrix(self) -> None:
        """Rebuild the stochastic transition matrix from observed counts."""
        self.states = sorted(list(self.state_counts.keys()))
        self._state_to_idx = {s: i for i, s in enumerate(self.states)}

        n = len(self.states)
        self.transition_matrix = np.zeros((n, n))

        for src, dsts in self.transitions.items():
            src_idx = self._state_to_idx[src]
            total_transitions = sum(dsts.values())

            if total_transitions > 0:
                for dst, count in dsts.items():
                    dst_idx = self._state_to_idx[dst]
                    self.transition_matrix[src_idx, dst_idx] = count / total_transitions
            else:
                # Absorbing state (sink), stays in itself
                self.transition_matrix[src_idx, src_idx] = 1.0

    def get_transition_probability(self, src: str, dst: str) -> float:
        """Get the empirical probability of transitioning from src to dst."""
        if not self.transition_matrix is not None:
            return 0.0
        if src not in self._state_to_idx or dst not in self._state_to_idx:
            return 0.0

        return float(
            self.transition_matrix[self._state_to_idx[src], self._state_to_idx[dst]]
        )

    def stationary_distribution(
        self, max_iter: int = 1000, tol: float = 1e-6
    ) -> dict[str, float]:
        """Compute the stationary distribution via power iteration.

        The stationary distribution represents the long-term probability
        of the agent being in any particular state. High probabilities on
        error/sink states indicate structural failure points.

        Returns:
            Dictionary mapping state to long-term probability.
        """
        if self.transition_matrix is None or len(self.states) == 0:
            return {}

        n = len(self.states)
        pi = np.ones(n) / n  # Initial uniform distribution

        for _ in range(max_iter):
            # pi * P (left eigenvector for row-stochastic matrix)
            next_pi = pi @ self.transition_matrix
            if np.linalg.norm(next_pi - pi, 1) < tol:
                pi = next_pi
                break
            pi = next_pi

        return {self.states[i]: float(pi[i]) for i in range(n)}

    def predict_sink_nodes(self, threshold: float = 0.1) -> list[tuple[str, float]]:
        """Identify states where the agent gets stuck or terminates.

        Returns a list of (state, probability) sorted by probability descending.
        """
        stat_dist = self.stationary_distribution()
        sinks = [(s, p) for s, p in stat_dist.items() if p >= threshold]
        sinks.sort(key=lambda x: x[1], reverse=True)
        return sinks
