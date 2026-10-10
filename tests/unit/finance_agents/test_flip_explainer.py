"""Flip-explainer tests — AU-CONTEXT-R005.2.

Verifies the explainer's output always derives from the computed math
(the verdict is a pure function of the price delta) and that every claim
it produces cites a source.
"""

from __future__ import annotations

import pytest

from agent_utilities.domains.finance_agents.flip_explainer import (
    FlipExplanation,
    SourcedClaim,
    explain_flip,
)


@pytest.mark.spec("AU-CONTEXT-R005.2")
def test_flip_verdict_derives_from_computed_math_and_every_claim_is_sourced() -> None:
    up = explain_flip(
        "ACME", prior_price=10.0, current_price=12.0, source="feed:acme/2026-10-09"
    )
    down = explain_flip(
        "ACME", prior_price=12.0, current_price=10.0, source="feed:acme/2026-10-09"
    )
    flat = explain_flip(
        "ACME", prior_price=10.0, current_price=10.0, source="feed:acme/2026-10-09"
    )

    # The verdict and computed_value are pure functions of the price delta.
    assert up.computed_value == pytest.approx(2.0)
    assert up.verdict == "flip_up"
    assert down.computed_value == pytest.approx(-2.0)
    assert down.verdict == "flip_down"
    assert flat.computed_value == pytest.approx(0.0)
    assert flat.verdict == "no_flip"

    for explanation in (up, down, flat):
        assert isinstance(explanation, FlipExplanation)
        assert explanation.claims, "explanation must carry at least one claim"
        for claim in explanation.claims:
            assert isinstance(claim, SourcedClaim)
            assert claim.source == "feed:acme/2026-10-09"
            # Every claim must reference the computed verdict/value, proving
            # it was derived from the math rather than asserted independently.
            assert (
                str(explanation.computed_value) in claim.text
                or explanation.verdict in claim.text
            )

    # A missing source is rejected outright: no claim may exist unsourced.
    with pytest.raises(ValueError):
        explain_flip("ACME", prior_price=10.0, current_price=12.0, source="")
