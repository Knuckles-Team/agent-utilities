from __future__ import annotations

"""Deterministic evaluation metrics for native policy optimization.

CONCEPT:AU-AHE.optimization.concept-matching-routing-policy — concept-matching
and routing-policy evaluation.

The engine-owned program optimizer consumes labeled concept pairs and execution
traces. This module retains the dependency-free metrics used to evaluate those
datasets; it does not contain a second optimizer or model transport.
"""

from collections.abc import Sequence
from typing import Any


def classification_accuracy(
    predictions: Sequence[bool], labels: Sequence[bool]
) -> float:
    """Return boolean classification accuracy for aligned observations."""
    pairs = list(zip(predictions, labels, strict=False))
    if not pairs:
        return 0.0
    correct = sum(1 for prediction, label in pairs if bool(prediction) == bool(label))
    return correct / len(pairs)


def routing_success_rate(decisions: Sequence[Any]) -> float:
    """Return the mean realized success of historical routing decisions."""
    items = list(decisions)
    if not items:
        return 0.0
    total = 0.0
    for decision in items:
        succeeded = (
            decision.get("success")
            if isinstance(decision, dict)
            else getattr(decision, "success", False)
        )
        total += 1.0 if succeeded else 0.0
    return total / len(items)


__all__ = ["classification_accuracy", "routing_success_rate"]
