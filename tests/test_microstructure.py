"""Tests for CONCEPT:KG-2.6 — Microstructure methodlogies."""

from agent_utilities.numeric import xp as np
import pytest

from agent_utilities.domains.finance.microstructure import (
    BrierScoreValidator,
    ConvergenceFilter,
    MicroPriceCalculator,
    OrderBookImbalance,
)


class TestOrderBookImbalance:
    def test_positive_imbalance(self):
        obi = OrderBookImbalance.calculate(100.0, 50.0)
        assert np.isclose(obi, 50.0 / 150.0)

    def test_negative_imbalance(self):
        obi = OrderBookImbalance.calculate(50.0, 100.0)
        assert np.isclose(obi, -50.0 / 150.0)

    def test_zero_volume(self):
        obi = OrderBookImbalance.calculate(0.0, 0.0)
        assert obi == 0.0


class TestMicroPriceCalculator:
    def test_micro_price(self):
        # Bid has more volume, price should lean towards ask
        bid_p, ask_p = 100.0, 101.0
        bid_v, ask_v = 100.0, 50.0
        mp = MicroPriceCalculator.calculate(bid_p, ask_p, bid_v, ask_v)
        expected = (100.0 * 101.0 + 50.0 * 100.0) / 150.0
        assert np.isclose(mp, expected)

    def test_micro_price_zero_volume(self):
        mp = MicroPriceCalculator.calculate(100.0, 101.0, 0.0, 0.0)
        assert np.isclose(mp, 100.5)

    def test_from_imbalance(self):
        mp = MicroPriceCalculator.from_imbalance(100.5, 1.0, 1 / 3)
        # 100.5 + 1/3 * 0.5 = 100.5 + 0.1666 = 100.666
        assert np.isclose(mp, 100.66666666666667)


class TestConvergenceFilter:
    def test_agreement_met(self):
        signals = [True, True, True, True, True]
        assert ConvergenceFilter.check_agreement(signals, threshold=5)

    def test_agreement_not_met(self):
        signals = [True, True, True, True, False]
        assert not ConvergenceFilter.check_agreement(signals, threshold=5)

    def test_not_enough_signals(self):
        signals = [True, True]
        assert not ConvergenceFilter.check_agreement(signals, threshold=3)


class TestBrierScoreValidator:
    def test_perfect_predictions(self):
        preds = np.array([1.0, 0.0, 1.0])
        actuals = np.array([1.0, 0.0, 1.0])
        score = BrierScoreValidator.calculate(preds, actuals)
        assert score == 0.0
        assert BrierScoreValidator.is_production_grade(score)

    def test_terrible_predictions(self):
        preds = np.array([0.0, 1.0, 0.0])
        actuals = np.array([1.0, 0.0, 1.0])
        score = BrierScoreValidator.calculate(preds, actuals)
        assert score == 1.0
        assert not BrierScoreValidator.is_production_grade(score)

    def test_invalid_lengths(self):
        with pytest.raises(ValueError):
            BrierScoreValidator.calculate(np.array([0.5]), np.array([]))
