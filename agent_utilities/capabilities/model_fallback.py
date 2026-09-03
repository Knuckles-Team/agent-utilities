#!/usr/bin/python
from __future__ import annotations

"""Bounded caller-level fallback for exhausted structured-output repair.

``run_fallback_chain`` executes caller-owned attempts in order and advances only
when an attempt raises ``StructuredOutputRepairExhausted``. All other failures
remain loud. The primitive deliberately owns no model registry or routing policy;
production callers must supply the already-governed attempts they intend to run.
"""

import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

from agent_utilities.capabilities.output_repair import StructuredOutputRepairExhausted

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class FallbackAttemptRecord:
    """One exhausted attempt in a fallback chain, in trace-ready shape."""

    label: str
    error: StructuredOutputRepairExhausted

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "error": str(self.error),
            "repair_attempts": [a.to_dict() for a in self.error.attempts],
        }


class FallbackChainExhausted(RuntimeError):
    """Fail-closed terminal error: every attempt in the fallback chain exhausted
    structured-output repair.

    Carries every attempt's :class:`StructuredOutputRepairExhausted` (which itself
    carries its own classify/repair history), so the full alternate-schema /
    alternate-model story is inspectable from one exception instead of only the
    last attempt tried.
    """

    def __init__(self, records: list[FallbackAttemptRecord]) -> None:
        self.records = list(records)
        labels = ", ".join(r.label for r in records) or "<empty chain>"
        super().__init__(
            f"structured-output repair exhausted for every fallback attempt: {labels}"
        )
        if records:
            self.__cause__ = records[-1].error


async def run_fallback_chain(
    attempts: Sequence[Callable[[], Awaitable[T]]],
    *,
    labels: Sequence[str] | None = None,
) -> T:
    """Run ``attempts`` in order; on :class:`StructuredOutputRepairExhausted`,
    move to the next one. Any other exception propagates immediately — this is a
    fallback for a CLASSIFIED structured-output failure, not a generic retry-
    everything loop.

    Args:
        attempts: Ordered zero-arg callables, each constructing (and running) a
            fresh attempt — e.g. a new ``Agent`` bound to an alternate model or
            an alternate (looser) output schema.
        labels: Optional per-attempt labels for logging/provenance (e.g. model
            ids). Defaults to ``"attempt-<n>"``.

    Returns:
        The first attempt's successful result.

    Raises:
        ValueError: ``attempts`` is empty.
        FallbackChainExhausted: every attempt raised
            ``StructuredOutputRepairExhausted``.
    """
    if not attempts:
        raise ValueError("run_fallback_chain requires at least one attempt")
    resolved_labels = (
        list(labels)
        if labels is not None
        else [f"attempt-{i + 1}" for i in range(len(attempts))]
    )
    if len(resolved_labels) != len(attempts):
        raise ValueError("labels must be the same length as attempts")

    records: list[FallbackAttemptRecord] = []
    for label, attempt in zip(resolved_labels, attempts, strict=True):
        try:
            return await attempt()
        except StructuredOutputRepairExhausted as e:
            logger.warning(
                "Structured-output repair exhausted for %s; trying next "
                "fallback attempt (%d of %d tried so far).",
                label,
                len(records) + 1,
                len(attempts),
            )
            records.append(FallbackAttemptRecord(label=label, error=e))
            continue
    raise FallbackChainExhausted(records)


@dataclass(frozen=True)
class ModelFallbackChain:
    """An ordered, config-driven model-fallback chain (D-47).

    ``attempts``/``model_ids`` are index-aligned, ready to hand straight to
    :func:`run_fallback_chain` — ``run_fallback_chain(chain.attempts,
    labels=chain.model_ids)``.
    """

    attempts: list[Callable[[], Awaitable[Any]]]
    model_ids: list[str] = field(default_factory=list)

    async def run(self) -> Any:
        """Convenience: :func:`run_fallback_chain` over this chain's own attempts."""
        return await run_fallback_chain(self.attempts, labels=self.model_ids)


__all__ = [
    "FallbackAttemptRecord",
    "FallbackChainExhausted",
    "ModelFallbackChain",
    "run_fallback_chain",
]
