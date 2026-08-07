#!/usr/bin/python
from __future__ import annotations

"""Caller-level fallback chain for exhausted structured-output repair (D-47, gate G4).

``StructuredOutputRepair`` (``output_repair.py``) classifies a structured-output
failure and drives a *bounded, in-Agent* classify → targeted ``ModelRetry`` loop,
failing closed with :class:`~agent_utilities.capabilities.output_repair.StructuredOutputRepairExhausted`
when that bound is hit. That capability lives inside ONE ``Agent``/model, so it can
never itself swap models or relax the schema — a genuine alternate-model or
alternate-schema retry needs a caller-level loop that constructs a FRESH ``Agent``
and runs it again. That fourth stage was the gap this module closes
(``reports/deferred/lane-3.3-3.4.md`` D-47): ``StructuredOutputRepairExhausted`` was
raised but had zero ``except`` sites anywhere in the package.

Two pieces, deliberately kept separate (CONCEPT:AU-ORCH.routing.model-fallback-chain):

* :func:`run_fallback_chain` — the generic primitive. Runs an ordered sequence of
  zero-arg attempt callables, catching ONLY ``StructuredOutputRepairExhausted``
  (the seam the deferred item named — a budget-exceeded-mid-output or
  content-filter-refusal failure is deliberately never retried, in-Agent or here,
  matching ``output_repair.py``'s own "never retry a tripped budget" rule) and
  moving to the next attempt. Any other exception propagates immediately
  (fallback is for a classified structured-output failure, not a blanket retry-
  everything policy). No opinion on what an "attempt" IS — a same-model-relaxed-
  schema retry and a different-model retry are both just callables, so this one
  primitive serves the "alternate schema" and "alternate model" halves of D-47's
  title without hardcoding either.
* :func:`model_fallback_chain` — builds that ordered attempt list for the common
  "alternate MODEL" case, config-driven off the SAME :class:`ModelRegistry`
  tier-fallback ordering ``pick_for_task``/``explain_pick_for_task`` already use
  for live routing (CONCEPT:AU-ORCH.routing.rejected-candidate-provenance) — never a
  hardcoded model list, and never able to disagree with the registry's own
  picker since it is built from ``explain_pick_for_task``'s own candidate scores.

Usage::

    from agent_utilities.capabilities.model_fallback import (
        model_fallback_chain, run_fallback_chain,
    )

    async def _attempt(model_id: str) -> MyOutput:
        agent = Agent(model_id, output_type=MyOutput, ...)
        result = await agent.run(prompt)
        return result.output

    chain = model_fallback_chain(_attempt, registry=registry, complexity="medium")
    output = await run_fallback_chain(chain)
"""

import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any, TypeVar

from agent_utilities.capabilities.output_repair import StructuredOutputRepairExhausted
from agent_utilities.models.model_registry import ModelRegistry, ModelTier

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: Default cap on how many alternate models a config-driven chain tries (the
#: primary pick plus this many fallbacks) — bounded for the same reason
#: ``MAX_ROUTING_CANDIDATES`` bounds routing provenance: a runaway registry must
#: never turn one failure into an unbounded retry storm.
DEFAULT_MAX_MODEL_FALLBACKS = 2


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


def model_fallback_chain(
    build_and_run: Callable[[str], Awaitable[T]],
    *,
    registry: ModelRegistry,
    complexity: ModelTier = "medium",
    required_tags: list[str] | None = None,
    max_fallbacks: int = DEFAULT_MAX_MODEL_FALLBACKS,
) -> ModelFallbackChain:
    """Build a config-driven ordered attempt chain for the "alternate MODEL" half
    of D-47: the primary registry pick, then up to ``max_fallbacks`` alternates,
    ranked by the SAME tier-priority :meth:`ModelRegistry.explain_pick_for_task`
    uses for live routing provenance — never a hardcoded model id, and never able
    to disagree with the registry's own picker (the primary entry is always
    ``explain_pick_for_task``'s ``chosen_model_id``).

    Args:
        build_and_run: Given a model id, constructs a fresh ``Agent`` bound to it
            and runs the task, returning the output (or raising
            ``StructuredOutputRepairExhausted`` on repair exhaustion).
        registry: The active :class:`ModelRegistry`.
        complexity: Tier of the task being spawned (see ``pick_for_task``).
        required_tags: Tags every candidate must carry (AND semantics).
        max_fallbacks: How many alternate models to try after the primary pick.

    Returns:
        A :class:`ModelFallbackChain` — pass ``.attempts``/``.model_ids`` to
        :func:`run_fallback_chain`, or call ``.run()`` directly.

    Raises:
        ValueError: The registry is empty (propagated from ``pick_for_task``).
    """
    decision = registry.explain_pick_for_task(
        complexity=complexity, required_tags=required_tags
    )
    ordered_ids = [decision.chosen_model_id]
    for candidate in decision.candidates:
        if candidate.model_id not in ordered_ids:
            ordered_ids.append(candidate.model_id)
    ordered_ids = ordered_ids[: max_fallbacks + 1]

    return ModelFallbackChain(
        attempts=[partial(build_and_run, model_id) for model_id in ordered_ids],
        model_ids=ordered_ids,
    )


__all__ = [
    "DEFAULT_MAX_MODEL_FALLBACKS",
    "FallbackAttemptRecord",
    "FallbackChainExhausted",
    "ModelFallbackChain",
    "model_fallback_chain",
    "run_fallback_chain",
]
