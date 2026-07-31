"""Recomputation-cost estimation — what it would cost to rebuild this context.

CONCEPT:AU-KG.memory.checkpoint-worthiness-scoring (the economics half).

This is the **one** recomputation-cost estimator in the platform, deliberately factored
out of the checkpoint scorer that first needed it so the *other* consumer that wants the
same number does not build a second one. Two live consumers are in view:

* **Checkpoint worthiness** (:mod:`agent_utilities.kvcache.worthiness`) — expensive-to-
  rebuild context is worth keeping. This is the strongest economic signal in the set.
* **Pressure-aware KV eviction** — the engine-side eviction ordering wants exactly this
  quantity as its "importance" input (deferred ``D-5.3-5.6-2`` / ``D-KVR-1``, both
  engine-side/Rust). When that lands, it consumes :class:`RebuildCostEstimate` rather
  than deriving its own, so "expensive to rebuild" means the same thing on both sides
  of the cache.

Measured, not predicted — and absent is not zero
------------------------------------------------
Every input is ``int | None`` / ``float | None`` where **``None`` means "not measured"
and ``0`` means "measured, and it was zero"**. That distinction is the whole point: a
run that made no tool calls is genuinely cheap to rebuild; a run whose tool calls were
never counted tells us nothing. :attr:`RebuildCostEstimate.known` is False when *no*
component was measured, and every consumer must treat that as **abstain**, never as
"cost zero". A scorer that guesses is worse than one that abstains.

Normalization
-------------
Each measured component is normalized against a saturation point in
:class:`RebuildCostScale` — the value at which rebuilding is unambiguously "expensive"
— and the aggregate is the weighted mean over **measured components only**, so an
unmeasured axis dilutes nothing. The scale is a plain dataclass a caller can replace,
not a family of environment variables.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

__all__ = [
    "RebuildCostEstimate",
    "RebuildCostInputs",
    "RebuildCostScale",
    "estimate_rebuild_cost",
]


class RebuildCostScale(BaseModel):
    """Saturation points + weights for normalizing rebuild cost into ``[0, 1]``.

    The defaults describe a long agentic research turn — tens of thousands of prompt
    tokens, a couple of dozen tool calls, a dozen-plus retrievals, a couple of minutes
    of wall time. A deployment whose workloads sit at a different scale constructs its
    own instance and passes it in; there is no env flag, because the correct value is a
    property of a workload, not of a host.
    """

    saturating_tokens: int = Field(
        default=60_000,
        gt=0,
        description="Total tokens at/above which rebuild is unambiguously expensive.",
    )
    saturating_tool_calls: int = Field(default=20, gt=0)
    saturating_retrievals: int = Field(default=15, gt=0)
    saturating_wall_time_s: float = Field(default=120.0, gt=0.0)

    weight_tokens: float = Field(default=0.4, ge=0.0)
    weight_tool_calls: float = Field(default=0.25, ge=0.0)
    weight_retrievals: float = Field(default=0.2, ge=0.0)
    weight_wall_time: float = Field(default=0.15, ge=0.0)

    model_config = ConfigDict(extra="forbid", frozen=True)


#: Process-wide default scale. Module-level constant rather than a flag, per
#: *Configuration discipline* — one correct value, replaceable by passing a ``scale``.
DEFAULT_REBUILD_COST_SCALE = RebuildCostScale()


class RebuildCostInputs(BaseModel):
    """What was actually spent assembling the context under consideration.

    Every field is optional and defaults to ``None`` = **not measured**. Pass ``0``
    only when you genuinely measured zero.
    """

    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    tool_calls: int | None = Field(default=None, ge=0)
    retrievals: int | None = Field(default=None, ge=0)
    wall_time_s: float | None = Field(default=None, ge=0.0)
    model: str = Field(
        default="",
        description="Model identity, used only to price the tokens. Unknown/absent "
        "leaves RebuildCostEstimate.usd as None rather than inventing a price.",
    )

    model_config = ConfigDict(extra="forbid", frozen=True)

    @property
    def total_tokens(self) -> int | None:
        """Prompt + completion tokens, or ``None`` when neither was measured."""
        if self.prompt_tokens is None and self.completion_tokens is None:
            return None
        return (self.prompt_tokens or 0) + (self.completion_tokens or 0)


class RebuildCostEstimate(BaseModel):
    """The normalized cost of rebuilding a context, plus the raw evidence for it."""

    normalized: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Weighted mean of the measured components in [0, 1]. Meaningless "
        "unless `known` is True.",
    )
    known: bool = Field(
        default=False,
        description="False when NO component was measured. Consumers must abstain "
        "rather than read `normalized` as a real zero.",
    )
    total_tokens: int | None = None
    tool_calls: int | None = None
    retrievals: int | None = None
    wall_time_s: float | None = None
    usd: float | None = Field(
        default=None,
        description="Monetary rebuild cost, when the model's pricing is known. None "
        "means unpriced — never a fabricated 0.0.",
    )
    components: dict[str, float] = Field(
        default_factory=dict,
        description="Per-axis normalized contribution, for inspectability. Only "
        "measured axes appear.",
    )
    measured: tuple[str, ...] = Field(
        default=(), description="Names of the axes that carried a real measurement."
    )

    model_config = ConfigDict(extra="forbid", frozen=True)

    def summary(self) -> str:
        """One-line human-readable rendering, safe to put in a prompt or a log."""
        if not self.known:
            return "rebuild cost: unmeasured"
        parts: list[str] = []
        if self.total_tokens is not None:
            parts.append(f"{self.total_tokens} tokens")
        if self.tool_calls is not None:
            parts.append(f"{self.tool_calls} tool calls")
        if self.retrievals is not None:
            parts.append(f"{self.retrievals} retrievals")
        if self.wall_time_s is not None:
            parts.append(f"{self.wall_time_s:.1f}s wall time")
        if self.usd is not None:
            parts.append(f"${self.usd:.4f}")
        return f"rebuild cost {self.normalized:.2f} ({', '.join(parts)})"


def _price_tokens(inputs: RebuildCostInputs) -> float | None:
    """Monetary rebuild cost, or ``None`` when the model isn't in the pricing catalog.

    Reuses the platform's single pricing authority
    (:meth:`~agent_utilities.models.usage.CostModel.for_model`) rather than carrying a
    second price table. An unpriced model is a clean abstain, not a zero.
    """
    if not inputs.model:
        return None
    if inputs.prompt_tokens is None and inputs.completion_tokens is None:
        return None
    from agent_utilities.models.usage import CostModel

    try:
        cost_model = CostModel.for_model(inputs.model)
    except LookupError as exc:  # noqa: BLE001 — deliberate DEBUG: an unpriced model is a documented CLEAN ABSTAIN (return None), not a failure — the docstring above states "An unpriced model is a clean abstain, not a zero." Callers treat None as "no cost signal" and proceed; warning would fire for every model absent from the shared CostModel price table, which is the normal case for local/self-hosted models. The cause is preserved (interpolated).
        logger.debug(
            "[CONCEPT:AU-KG.memory.checkpoint-worthiness-scoring] rebuild cost left "
            "unpriced for model %r: %s",
            inputs.model,
            exc,
        )
        return None
    return cost_model.estimate(
        input_tokens=inputs.prompt_tokens or 0,
        output_tokens=inputs.completion_tokens or 0,
    )


def estimate_rebuild_cost(
    inputs: RebuildCostInputs,
    *,
    scale: RebuildCostScale | None = None,
) -> RebuildCostEstimate:
    """Normalize what was spent building a context into a ``[0, 1]`` rebuild cost.

    Returns an estimate with ``known=False`` when no component was measured — callers
    must abstain in that case rather than treating the context as free to rebuild.
    """
    scale = scale or DEFAULT_REBUILD_COST_SCALE

    total_tokens = inputs.total_tokens
    axes: list[tuple[str, float, float]] = []  # (name, normalized, weight)
    if total_tokens is not None:
        axes.append(
            (
                "tokens",
                min(1.0, total_tokens / scale.saturating_tokens),
                scale.weight_tokens,
            )
        )
    if inputs.tool_calls is not None:
        axes.append(
            (
                "tool_calls",
                min(1.0, inputs.tool_calls / scale.saturating_tool_calls),
                scale.weight_tool_calls,
            )
        )
    if inputs.retrievals is not None:
        axes.append(
            (
                "retrievals",
                min(1.0, inputs.retrievals / scale.saturating_retrievals),
                scale.weight_retrievals,
            )
        )
    if inputs.wall_time_s is not None:
        axes.append(
            (
                "wall_time",
                min(1.0, inputs.wall_time_s / scale.saturating_wall_time_s),
                scale.weight_wall_time,
            )
        )

    if not axes:
        return RebuildCostEstimate(known=False)

    weight_total = sum(weight for _, _, weight in axes)
    if weight_total <= 0.0:
        # Every measured axis was zero-weighted by the caller's scale — that is a
        # deliberate "ignore all of these", which is indistinguishable from having no
        # measurement at all. Abstain rather than divide by zero or invent a value.
        return RebuildCostEstimate(known=False)

    normalized = sum(value * weight for _, value, weight in axes) / weight_total
    return RebuildCostEstimate(
        normalized=round(min(1.0, max(0.0, normalized)), 4),
        known=True,
        total_tokens=total_tokens,
        tool_calls=inputs.tool_calls,
        retrievals=inputs.retrievals,
        wall_time_s=inputs.wall_time_s,
        usd=_price_tokens(inputs),
        components={name: round(value, 4) for name, value, _ in axes},
        measured=tuple(name for name, _, _ in axes),
    )
