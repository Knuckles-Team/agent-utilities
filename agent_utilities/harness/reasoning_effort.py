#!/usr/bin/python
from __future__ import annotations

"""CONCEPT:AHE-3.1 — Adaptive Reasoning Effort (Test-Time Compute Scaling).

Continuous 0.0–1.0 float scale for reasoning effort budgeting.
Inspired by BrowseComp-Plus (arXiv:2508.06600).

See docs/pillars/3_agentic_harness_engineering.md
"""

import logging
import math
import re

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ReasoningBudget(BaseModel):
    """Discrete parameter set derived from continuous reasoning effort."""

    effort: float = Field(ge=0.0, le=1.0)
    max_search_calls: int = Field(ge=1)
    max_retrieval_depth: int = Field(ge=1)
    context_window: int = Field(ge=2)
    enable_decomposition: bool
    max_decomposition_subtasks: int = Field(ge=1)


def get_budget(effort: float) -> ReasoningBudget:
    """Map continuous effort [0.0, 1.0] to discrete retrieval parameters."""
    effort = max(0.0, min(1.0, effort))
    max_search_calls = max(1, round(1 + 9 * effort**1.5))
    max_retrieval_depth = max(
        1, round(1 + 3 * math.log1p(effort * 2.7) / math.log1p(2.7))
    )
    context_window = max(2, round(5 + 20 * effort))
    enable_decomposition = effort >= 0.3
    max_decomposition_subtasks = max(1, round(2 + 3 * max(0.0, effort - 0.3) / 0.7))
    return ReasoningBudget(
        effort=effort,
        max_search_calls=max_search_calls,
        max_retrieval_depth=max_retrieval_depth,
        context_window=context_window,
        enable_decomposition=enable_decomposition,
        max_decomposition_subtasks=max_decomposition_subtasks,
    )


_COMPLEXITY_SIGNALS = {
    r"\b(and|also|additionally|furthermore|moreover)\b": 0.05,
    r"\b(compare|contrast|difference|versus|vs\.?)\b": 0.10,
    r"\b(why|because|cause|effect|impact|influence|lead to)\b": 0.08,
    r"\b(before|after|during|when|timeline|history|evolution)\b": 0.06,
    r"\b(all|every|each|summary|overview|comprehensive)\b": 0.07,
    r"\b(between|across|among|multiple|several)\b": 0.06,
    r"\b(analyze|evaluate|assess|examine|investigate|diagnose)\b": 0.10,
    r"\b(if|would|could|might|hypothetically|scenario)\b": 0.05,
    r"\b(how many|how much|percentage|ratio|count|statistics)\b": 0.04,
}


def estimate_query_complexity(query: str) -> float:
    """Estimate reasoning effort needed using lightweight heuristics.

    Returns a float in [0.0, 1.0] based on query length, complexity
    patterns, and question nesting.
    """
    if not query or not query.strip():
        return 0.0

    score = 0.0
    query_lower = query.lower()
    words = query.split()
    word_count = len(words)
    length_score = 0.3 / (1 + math.exp(-0.2 * (word_count - 15)))
    score += length_score

    for pattern, weight in _COMPLEXITY_SIGNALS.items():
        if re.search(pattern, query_lower):
            score += weight

    question_marks = query.count("?")
    if question_marks > 1:
        score += 0.05 * min(3, question_marks - 1)

    clause_count = query.count(",") + query.count(";") + query.count("(")
    if clause_count > 2:
        score += 0.03 * min(4, clause_count - 2)

    return max(0.0, min(1.0, score))
