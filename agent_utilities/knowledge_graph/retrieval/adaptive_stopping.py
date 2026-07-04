#!/usr/bin/python
from __future__ import annotations

"""Training-free adaptive stopping for iterative retrieval / RAG loops.

CONCEPT:AU-KG.retrieval.adaptive-stopping-iterative-retrieval — Adaptive Stopping for Iterative Retrieval

Distilled from "TASR: Training-Free Adaptive Stopping for Iterative Retrieval"
(arXiv 2606.x). An iterative retrieve→answer loop normally runs a fixed number of
rounds or relies on an LLM self-judgement to decide it is "done"; both waste rounds
(and tokens) and neither is grounded. TASR's core observation is training-free and
deterministic: **the loop should halt the moment the model repeats its previous
answer** — once an extra round of evidence stops moving the answer, further rounds
are redundant.

This module adds two complementary, equally training-free guards so the stopper is
robust on real loops:

* **answer repeat** — the primary TASR rule: stop when the new answer is the same
  as (or near-identical to, by token-set Jaccard) the prior round's answer.
* **coverage saturation** — stop when a round surfaces fewer than ``min_new_evidence``
  *new* evidence ids for ``patience`` consecutive rounds (the retriever has nothing
  fresh to add, so the answer cannot improve).
* **max rounds** — a hard cap so a pathological loop always terminates.

The stopper is a pure state machine: feed it one round's answer and evidence ids via
:meth:`IterativeStopper.update` and it returns a :class:`StopDecision`. No model, no
I/O, no environment — fully deterministic and unit-testable.
"""

import string
from collections.abc import Iterable
from dataclasses import dataclass

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


@dataclass(frozen=True)
class StopDecision:
    """Outcome of one :meth:`IterativeStopper.update` call.

    Attributes:
        stop: Whether the loop should halt now.
        reason: Empty when ``stop`` is False; otherwise one of
            ``"answer_repeat"``, ``"coverage_saturation"``, ``"max_rounds"``.
    """

    stop: bool
    reason: str  # "" | "answer_repeat" | "coverage_saturation" | "max_rounds"


def normalized_answer(text: str) -> str:
    """Canonicalize an answer for comparison.

    Lowercases, strips punctuation, and collapses all runs of whitespace to a
    single space (trimming the ends). Deterministic and idempotent.

    Args:
        text: Raw answer text (may be empty).

    Returns:
        The normalized form (empty string for empty / whitespace-only input).
    """
    lowered = (text or "").lower().translate(_PUNCT_TABLE)
    return " ".join(lowered.split())


def _token_jaccard(prev: str, cur: str) -> float:
    """Token-set Jaccard similarity of two normalized strings (0.0–1.0)."""
    a = set(prev.split())
    b = set(cur.split())
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def answer_repeats(
    prev: str | None,
    cur: str | None,
    *,
    similarity_threshold: float = 0.95,
) -> bool:
    """Decide whether ``cur`` repeats ``prev`` (the core TASR halt signal).

    Returns True when, after :func:`normalized_answer`, the two answers are equal,
    or their token-set Jaccard similarity is ``>= similarity_threshold``. A missing
    prior answer (``prev is None``) never counts as a repeat — the first round can
    never halt on this rule.

    Args:
        prev: The previous round's answer, or None on the first round.
        cur: The current round's answer.
        similarity_threshold: Jaccard cutoff for a "near" repeat (1.0 = exact only).

    Returns:
        True if the current answer repeats the prior one.
    """
    if prev is None:
        return False
    np = normalized_answer(prev)
    nc = normalized_answer(cur or "")
    if np == nc:
        return True
    return _token_jaccard(np, nc) >= similarity_threshold


class IterativeStopper:
    """Deterministic, training-free stop controller for an iterative loop.

    Drive it one round at a time with :meth:`update`. It tracks the round count,
    the previous answer, the cumulative set of seen evidence ids, and a saturation
    streak, and returns a :class:`StopDecision` each call (CONCEPT:AU-KG.retrieval.adaptive-stopping-iterative-retrieval).
    """

    def __init__(
        self,
        *,
        max_rounds: int = 5,
        similarity_threshold: float = 0.95,
        min_new_evidence: int = 1,
        patience: int = 1,
    ) -> None:
        """Configure the stopper.

        Args:
            max_rounds: Hard cap — stop with ``"max_rounds"`` once the round count
                reaches this value.
            similarity_threshold: Jaccard cutoff passed to :func:`answer_repeats`.
            min_new_evidence: A round contributing fewer than this many *new*
                evidence ids counts as saturated.
            patience: Number of consecutive saturated rounds tolerated before
                stopping with ``"coverage_saturation"``.
        """
        self._max_rounds = max(1, max_rounds)
        self._similarity_threshold = similarity_threshold
        self._min_new_evidence = max(0, min_new_evidence)
        self._patience = max(1, patience)
        self._rounds = 0
        self._prev_answer: str | None = None
        self._seen_evidence: set[str] = set()
        self._saturation_streak = 0

    @property
    def rounds(self) -> int:
        """Number of rounds observed so far (incremented per :meth:`update`)."""
        return self._rounds

    def update(
        self,
        answer: str | None = None,
        evidence_ids: Iterable[str] | None = None,
    ) -> StopDecision:
        """Record one round and decide whether to stop.

        Evaluated in priority order: the primary TASR answer-repeat rule first, then
        coverage saturation, then the hard round cap. The first satisfied rule wins.

        Args:
            answer: This round's answer text (None / empty allowed).
            evidence_ids: Evidence ids retrieved this round (deduped against all
                prior rounds to count *new* coverage).

        Returns:
            A :class:`StopDecision`; ``stop`` is True with a non-empty ``reason``
            when any rule fires, otherwise ``StopDecision(False, "")``.
        """
        self._rounds += 1

        # --- coverage bookkeeping (always, regardless of which rule fires) ---
        new_ids = {str(e) for e in (evidence_ids or []) if str(e)}
        novel = new_ids - self._seen_evidence
        self._seen_evidence |= new_ids
        if len(novel) < self._min_new_evidence:
            self._saturation_streak += 1
        else:
            self._saturation_streak = 0

        # --- rule 1 (primary): the model repeated its previous answer ---
        repeated = answer_repeats(
            self._prev_answer,
            answer,
            similarity_threshold=self._similarity_threshold,
        )
        self._prev_answer = answer

        if repeated:
            return StopDecision(stop=True, reason="answer_repeat")

        # --- rule 2: no fresh evidence for `patience` consecutive rounds ---
        if self._saturation_streak >= self._patience:
            return StopDecision(stop=True, reason="coverage_saturation")

        # --- rule 3: hard cap ---
        if self._rounds >= self._max_rounds:
            return StopDecision(stop=True, reason="max_rounds")

        return StopDecision(stop=False, reason="")


__all__ = [
    "StopDecision",
    "normalized_answer",
    "answer_repeats",
    "IterativeStopper",
]
