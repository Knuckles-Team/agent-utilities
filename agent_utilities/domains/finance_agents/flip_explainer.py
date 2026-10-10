"""Flip-explainer agent (AU-CONTEXT-R005.2 / AU-CONTEXT-R007).

Computes its answer from math first, then attaches every claim it makes to a
cited source. The underlying price-delta/flip-verdict calculation is sourced
through the typed delegation seam in ``agent_utilities/api/finance_delegation.py``
(``price_delta_via_eg_or_local``), which prefers epistemic-graph's finance
core when it is available and otherwise falls back to the local formula
below -- this module never computes the delta with its own formula directly.

The finance backfill/scan scheduling surfaces presented to users remain
hosted by graph-os; this module only provides the underlying sourced-
explanation primitive consumed by that scheduling work.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent_utilities.api.finance_delegation import (
    PriceDeltaRequest,
    PriceDeltaResponse,
    price_delta_via_eg_or_local,
)


@dataclass(frozen=True)
class SourcedClaim:
    """A single natural-language claim paired with the source it cites."""

    text: str
    source: str

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("SourcedClaim requires a non-empty source citation")
        if not self.text:
            raise ValueError("SourcedClaim requires non-empty claim text")


@dataclass(frozen=True)
class FlipExplanation:
    """The flip-explainer's output: a computed verdict plus cited claims.

    ``computed_value`` and ``verdict`` are derived purely from the numeric
    inputs in :func:`explain_flip` before any claim text is produced, so every
    claim always traces back to the already-computed math rather than being
    asserted independently of it.
    """

    symbol: str
    computed_value: float
    verdict: str
    claims: tuple[SourcedClaim, ...]

    def __post_init__(self) -> None:
        if not self.claims:
            raise ValueError("FlipExplanation requires at least one cited claim")
        for claim in self.claims:
            if not claim.source:
                raise ValueError(f"claim {claim.text!r} has no cited source")


def _local_price_delta(request: PriceDeltaRequest) -> PriceDeltaResponse:
    """Local fallback formula, used only when EG's finance core is absent."""
    delta = request.current_price - request.prior_price
    if delta > 0:
        verdict = "flip_up"
    elif delta < 0:
        verdict = "flip_down"
    else:
        verdict = "no_flip"
    return PriceDeltaResponse(delta=delta, verdict=verdict)


def explain_flip(
    symbol: str,
    prior_price: float,
    current_price: float,
    source: str,
) -> FlipExplanation:
    """Compute a price-flip verdict from math first, then cite every claim.

    Args:
        symbol: Instrument identifier the explanation is about.
        prior_price: Previously observed price.
        current_price: Currently observed price.
        source: Citation (URL, document id, or feed reference) every claim
            produced from this computation attaches to.

    Returns:
        A :class:`FlipExplanation` whose ``verdict``/``computed_value`` come
        from the typed finance-delegation seam (EG's finance core when
        available, else the local fallback formula), and whose ``claims``
        each cite ``source``.
    """
    if not source:
        raise ValueError("explain_flip requires a cited source")

    # Math first: the verdict is derived purely from the numeric delta,
    # sourced through the delegation seam, before any claim sentence is
    # constructed.
    result = price_delta_via_eg_or_local(
        PriceDeltaRequest(prior_price=prior_price, current_price=current_price),
        local_fallback=_local_price_delta,
    )
    delta = result.delta
    verdict = result.verdict

    claims = (
        SourcedClaim(
            text=(
                f"{symbol} moved from {prior_price} to {current_price} (delta={delta})."
            ),
            source=source,
        ),
        SourcedClaim(
            text=f"{symbol} verdict is '{verdict}', computed from delta={delta}.",
            source=source,
        ),
    )
    return FlipExplanation(
        symbol=symbol,
        computed_value=delta,
        verdict=verdict,
        claims=claims,
    )
