"""The flip explainer: the math first, then claims with their sources (EH-419).

A trend flip is a mechanical event: epistemic-graph's trailing-trend signal
changed direction on a closed bar. The explanation therefore has two parts
with different epistemic standing, kept apart:

1. **The math** (:func:`flip_math`) -- read straight off the EG flip record:
   which rule fired, on which bar, at what price against which line. No model
   is involved and nothing is inferred.
2. **The claims** -- an LLM reads the evidence the caller gathered (news,
   filings, posts, each with its URL) and states what may have moved the
   market, as claims. Every claim must cite at least one URL that is in the
   evidence it was given; a claim citing anything else is dropped and counted,
   never shown. With no evidence there are no claims -- the math stands alone.

The explanation is informational only and says so; it carries no order field
and authorises nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "INFORMATIONAL_NOTICE",
    "Evidence",
    "FlipExplanation",
    "FlipMath",
    "SourcedClaim",
    "explain_flip",
    "flip_math",
    "keep_sourced",
]

INFORMATIONAL_NOTICE = (
    "Informational only, not investment advice. The math section is the "
    "mechanical rule that fired; each claim is a model's reading of the cited "
    "sources and may be wrong -- check the sources."
)

_SYSTEM_PROMPT = (
    "You explain a market trend flip that has ALREADY been computed. The math "
    "section is fact; do not restate or dispute it. From the numbered evidence "
    "only, state at most five short claims about what may have contributed to "
    "the move around the flip time. Every claim must cite the exact URL(s) of "
    "the evidence it rests on. Do not cite anything that is not in the evidence. "
    "If the evidence says nothing relevant, return no claims. Never recommend "
    "buying, selling or holding."
)


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Evidence(_Frozen):
    """One source the caller gathered around the flip."""

    url: str = Field(pattern=r"^https://")
    title: str = ""
    snippet: str = ""
    published_at: str | None = None


class SourcedClaim(_Frozen):
    """One model claim and the evidence URLs it rests on."""

    kind: Literal["news", "filing", "social", "macro", "other"] = "news"
    text: str = Field(min_length=1, max_length=600)
    sources: list[str] = Field(min_length=1, max_length=5)


class FlipMath(_Frozen):
    """The mechanical facts of one flip record, from EG."""

    listing_id: str
    timeframe: str
    status: str
    direction_from: str
    direction_to: str
    bar_open_ns: int
    effective_at_ns: int
    close_ticks: int
    line_ticks: int
    rule: str
    revises: str | None = None


class FlipExplanation(_Frozen):
    math: FlipMath
    claims: list[SourcedClaim]
    dropped_claims: int
    evidence_count: int
    notice: str = INFORMATIONAL_NOTICE
    informational_only: Literal[True] = True


def flip_math(alert: dict[str, Any]) -> FlipMath:
    """The math section of one ``finance.flip`` alert; no model involved."""
    record = alert["record"]
    flip = record["flip"]
    crossed = "above" if flip["to"] == "bullish" else "below"
    return FlipMath(
        listing_id=str(alert["listing_id"]),
        timeframe=str(alert["timeframe"]),
        status=str(record["status"]),
        direction_from=str(flip["from"]),
        direction_to=str(flip["to"]),
        bar_open_ns=int(flip["bar_open"]),
        effective_at_ns=int(flip["effective_at"]),
        close_ticks=int(flip["price"]),
        line_ticks=int(flip["line"]),
        rule=(
            f"The {alert['timeframe']} bar closed {crossed} the trailing trend "
            f"line, so the signal flipped {flip['from']} -> {flip['to']}."
        ),
        revises=record.get("revises"),
    )


def keep_sourced(
    claims: Sequence[SourcedClaim], evidence: Sequence[Evidence]
) -> tuple[list[SourcedClaim], int]:
    """Keep only claims whose every source is in ``evidence``."""
    allowed = {item.url for item in evidence}
    kept = [claim for claim in claims if set(claim.sources) <= allowed]
    return kept, len(claims) - len(kept)


def _prompt(math: FlipMath, evidence: Sequence[Evidence]) -> str:
    lines = [f"Math (fact): {math.rule} Listing {math.listing_id}.", "Evidence:"]
    lines.extend(
        f"[{number}] {item.url} | {item.title} | {item.published_at or 'undated'} | {item.snippet}"
        for number, item in enumerate(evidence, start=1)
    )
    return "\n".join(lines)


async def _claims(
    math: FlipMath, evidence: Sequence[Evidence], model: Any
) -> list[SourcedClaim]:
    """The model's claims over ``evidence``.

    The claims' grounding is the evidence itself, enforced afterwards by
    :func:`keep_sourced`; compiled KG context is supplemental here, so the run
    uses the ``best_effort`` grounding policy -- a degraded compile is marked
    in the messages, never silent, and it cannot add an uncited claim.
    """
    from agent_utilities.core.contextual_model import (
        create_context_agent,
        use_grounding_policy,
    )
    from agent_utilities.core.model_factory import create_model

    agent = create_context_agent(
        model=model or create_model(),
        output_type=list[SourcedClaim],
        system_prompt=_SYSTEM_PROMPT,
    )
    with use_grounding_policy("best_effort"):
        result = await agent.run(_prompt(math, evidence))
    return list(result.output)


async def explain_flip(
    alert: dict[str, Any], evidence: Sequence[Evidence], *, model: Any = None
) -> FlipExplanation:
    """Math first; then the model's claims, each backed by given evidence."""
    math = flip_math(alert)
    proposed = await _claims(math, evidence, model) if evidence else []
    claims, dropped = keep_sourced(proposed, evidence)
    return FlipExplanation(
        math=math, claims=claims, dropped_claims=dropped, evidence_count=len(evidence)
    )
