#!/usr/bin/python
from __future__ import annotations

"""Retrieval-time context-token budget (CONCEPT:KG-2.1 / KG-2.3).

The article's hard-won lesson: a bigger memory is useless if retrieval balloons
the context window ("memory ate 40% of context"). The compactor handles *active*
conversation messages, but nothing stopped *retrieval* from over-filling. This
adds an explicit budget at the retrieval boundary: include ranked candidates
greedily until the next would exceed the token budget, and report what was
dropped (no silent truncation).

Pure and dependency-light (reuses the shared token estimator).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..memory.agent_context import estimate_tokens


@dataclass
class BudgetResult:
    kept: list[Any]
    dropped: int
    tokens_used: int
    token_budget: int

    @property
    def truncated(self) -> bool:
        return self.dropped > 0


class RetrievalBudgetManager:
    """Greedily fit ranked candidates within a token budget."""

    def __init__(self, token_budget: int) -> None:
        self.token_budget = max(0, int(token_budget))

    def fit(
        self,
        candidates: list[Any],
        text_of: Callable[[Any], str] | None = None,
    ) -> BudgetResult:
        """Keep the highest-ranked candidates that fit; drop the rest.

        ``candidates`` must already be ranked (best first). ``text_of`` maps a
        candidate to the text whose tokens count against the budget; defaults to
        ``str(candidate)``.
        """
        text_of = text_of or (lambda c: str(c))
        kept: list[Any] = []
        used = 0
        dropped = 0
        for cand in candidates:
            cost = estimate_tokens(text_of(cand))
            if used + cost > self.token_budget and kept:
                dropped = len(candidates) - len(kept)
                break
            # Always allow at least one item even if it alone exceeds budget,
            # so a single large result is returned rather than nothing.
            kept.append(cand)
            used += cost
        return BudgetResult(
            kept=kept, dropped=dropped, tokens_used=used, token_budget=self.token_budget
        )


def fit_within(
    candidates: list[Any],
    token_budget: int | None,
    text_of: Callable[[Any], str] | None = None,
) -> list[Any]:
    """Convenience: return only the kept candidates (no budget → unchanged)."""
    if not token_budget or token_budget <= 0 or not candidates:
        return candidates
    return RetrievalBudgetManager(token_budget).fit(candidates, text_of).kept
