"""Unit tests for Rust-compiled Quant FFI Engine.

CONCEPT:KG-2.18
"""

import epistemic_graph.quant as eq
import pytest


def test_quant_moving_averages_and_variance():
    """Verify that moving_average, exponential_moving_average, and rolling_variance calculate correctly."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0]

    # 1. Simple Moving Average (window=3)
    # Expected: [1.0, 1.5, 2.0, 3.0, 4.0]
    ma = eq.moving_average(data, 3)
    assert len(ma) == 5
    assert ma[2] == pytest.approx(2.0)
    assert ma[3] == pytest.approx(3.0)
    assert ma[4] == pytest.approx(4.0)

    # 2. Exponential Moving Average (alpha=0.5)
    # y[0] = x[0] = 1.0
    # y[1] = 0.5 * 2.0 + 0.5 * 1.0 = 1.5
    # y[2] = 0.5 * 3.0 + 0.5 * 1.5 = 2.25
    ema = eq.exponential_moving_average(data, 0.5)
    assert len(ema) == 5
    assert ema[0] == pytest.approx(1.0)
    assert ema[1] == pytest.approx(1.5)
    assert ema[2] == pytest.approx(2.25)

    # 3. Rolling Variance (window=3)
    # Elements at index 2 are [1, 2, 3] -> mean=2. Uses population variance:
    # Var = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = 2/3 ≈ 0.666667
    var = eq.rolling_variance(data, 3)
    assert len(var) == 5
    assert var[2] == pytest.approx(2.0 / 3.0)
    assert var[3] == pytest.approx(2.0 / 3.0)
    assert var[4] == pytest.approx(2.0 / 3.0)

    # 4. Rolling Z-Score (window=3)
    # [1, 2, 3] -> mean=2, std = sqrt(2/3) ≈ 0.81649658.
    # For element 3: (3 - 2) / sqrt(2/3) = 1.22474487
    zscores = eq.rolling_zscore(data, 3)
    assert len(zscores) == 5
    assert zscores[2] == pytest.approx(1.22474487)


def test_quant_orderbook_matching_simulation():
    """Verify that event-driven order book tick simulation matches buy and sell orders properly."""
    # Create L2 order book limits (price, quantity)
    # Bids (buys): buy 10 at 99.0, buy 5 at 98.0
    bids = [(99.0, 10.0), (98.0, 5.0)]
    # Asks (sells): sell 10 at 101.0, sell 5 at 102.0
    asks = [(101.0, 10.0), (102.0, 5.0)]

    # 1. Simulate a buy order (crosses ask price)
    # A market buy order at 101.5 for 5 units should match the ask at 101.0
    # Remaining ask at 101.0 should be 5 units.
    updated_bids, updated_asks, trades = eq.simulate_order_matching(
        bids, asks, 101.5, 5.0, True
    )

    # We expect 1 trade of 5.0 units at 101.0
    assert len(trades) == 1
    trade_price, trade_qty = trades[0]
    assert trade_price == pytest.approx(101.0)
    assert trade_qty == pytest.approx(5.0)

    # The ask at 101.0 should now be reduced to 5.0 units
    assert len(updated_asks) == 2
    assert updated_asks[0][0] == pytest.approx(101.0)
    assert updated_asks[0][1] == pytest.approx(5.0)

    # 2. Simulate a sell order (crosses bid price)
    # A market sell order at 98.5 for 12 units should consume 10 units at 99.0 and leave 2 units unmatched
    # Remaining bids: buy 5 at 98.0 (since the sell limit was 98.5, it won't cross 98.0)
    updated_bids2, updated_asks2, trades2 = eq.simulate_order_matching(
        bids, asks, 98.5, 12.0, False
    )

    # We expect 1 trade of 10.0 units at 99.0
    assert len(trades2) == 1
    assert trades2[0][0] == pytest.approx(99.0)
    assert trades2[0][1] == pytest.approx(10.0)

    # The bid at 99.0 is fully consumed, leaving only 98.0
    assert len(updated_bids2) == 1
    assert updated_bids2[0][0] == pytest.approx(98.0)
    assert updated_bids2[0][1] == pytest.approx(5.0)
