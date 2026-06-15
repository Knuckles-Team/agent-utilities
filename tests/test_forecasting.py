#!/usr/bin/python
from __future__ import annotations

"""Unit tests for the predict-before-run forecasting scoreboard (CONCEPT:AHE-3.34)."""

import math

from agent_utilities.harness.forecasting import Forecast, ForecastBoard


def test_predict_then_resolve() -> None:
    board = ForecastBoard()
    f = board.predict("exp-1", "lr=3e-4 beats baseline", predicted=0.8, confidence=0.7)
    assert isinstance(f, Forecast)
    assert f.experiment_id == "exp-1"
    assert f.predicted == 0.8
    assert f.confidence == 0.7
    assert f.actual is None
    assert f.resolved is False
    assert f.deviation() is None

    resolved = board.resolve("exp-1", 0.75)
    assert resolved is not None
    assert resolved.resolved is True
    assert resolved.actual == 0.75
    assert math.isclose(resolved.deviation() or -1.0, 0.05)


def test_resolve_unknown_returns_none() -> None:
    board = ForecastBoard()
    assert board.resolve("never-predicted", 0.5) is None


def test_brier_score_hand_computed() -> None:
    board = ForecastBoard()
    board.predict("a", "h", predicted=0.9, confidence=0.6)
    board.predict("b", "h", predicted=0.2, confidence=0.6)
    board.resolve("a", 1.0)  # (0.9-1.0)^2 = 0.01
    board.resolve("b", 0.0)  # (0.2-0.0)^2 = 0.04
    # mean = (0.01 + 0.04) / 2 = 0.025
    assert math.isclose(board.brier_score() or -1.0, 0.025)


def test_brier_score_clamps_out_of_range() -> None:
    board = ForecastBoard()
    board.predict("a", "h", predicted=1.5, confidence=0.5)  # clamps to 1.0
    board.resolve("a", -0.5)  # clamps to 0.0 -> (1.0-0.0)^2 = 1.0
    assert math.isclose(board.brier_score() or -1.0, 1.0)


def test_brier_none_when_nothing_resolved() -> None:
    board = ForecastBoard()
    board.predict("a", "h", predicted=0.5, confidence=0.5)
    assert board.brier_score() is None
    assert board.hit_rate() is None
    assert board.calibration_curve() == []


def test_hit_rate_within_tolerance() -> None:
    board = ForecastBoard()
    board.predict("a", "h", predicted=0.50, confidence=0.5)
    board.predict("b", "h", predicted=0.50, confidence=0.5)
    board.predict("c", "h", predicted=0.50, confidence=0.5)
    board.resolve("a", 0.55)  # dev 0.05 <= 0.1 -> hit
    board.resolve("b", 0.59)  # dev 0.09 <= 0.1 -> hit
    board.resolve("c", 0.90)  # dev 0.40 > 0.1  -> miss
    assert math.isclose(board.hit_rate(tolerance=0.1) or -1.0, 2.0 / 3.0)
    # tighter tolerance drops the 0.09 deviation, keeping only the 0.05 one
    assert math.isclose(board.hit_rate(tolerance=0.06) or -1.0, 1.0 / 3.0)


def test_calibration_curve_buckets_and_overconfidence() -> None:
    board = ForecastBoard()
    # High-confidence band (0.9) but consistently wrong predictions -> overconfident.
    for i in range(4):
        board.predict(f"hi-{i}", "h", predicted=0.5, confidence=0.9)
        board.resolve(f"hi-{i}", 0.95)  # dev 0.45 -> all miss
    # Low-confidence band (0.1) but accurate -> under-confident.
    for i in range(2):
        board.predict(f"lo-{i}", "h", predicted=0.5, confidence=0.1)
        board.resolve(f"lo-{i}", 0.52)  # dev 0.02 -> all hit

    curve = board.calibration_curve(bins=5, tolerance=0.1)
    # Two non-empty bins, ordered low->high confidence.
    assert len(curve) == 2
    (low_conf, low_hit, low_n), (high_conf, high_hit, high_n) = curve
    assert low_conf < high_conf
    assert low_n == 2 and high_n == 4
    # Low-confidence bin actually performs well (under-confidence).
    assert math.isclose(low_hit, 1.0)
    assert low_conf < low_hit
    # High-confidence bin performs poorly (overconfidence): conf >> hit-rate.
    assert math.isclose(high_hit, 0.0)
    assert high_conf > high_hit


def test_calibration_curve_top_edge_lands_in_last_bin() -> None:
    board = ForecastBoard()
    board.predict("a", "h", predicted=0.5, confidence=1.0)
    board.resolve("a", 0.5)
    curve = board.calibration_curve(bins=4)
    assert len(curve) == 1
    mean_conf, hit, n = curve[0]
    assert math.isclose(mean_conf, 1.0)
    assert n == 1


def test_surprises_worst_first() -> None:
    board = ForecastBoard()
    board.predict("small", "h", predicted=0.5, confidence=0.5)
    board.predict("big", "h", predicted=0.1, confidence=0.5)
    board.predict("mid", "h", predicted=0.4, confidence=0.5)
    board.predict("tiny", "h", predicted=0.5, confidence=0.5)
    board.resolve("small", 0.55)  # dev 0.05 -> not a surprise (<=0.25)
    board.resolve("big", 0.95)    # dev 0.85 -> biggest surprise
    board.resolve("mid", 0.80)    # dev 0.40 -> surprise
    board.resolve("tiny", 0.50)   # dev 0.0  -> not a surprise

    surprises = board.surprises(tolerance=0.25)
    assert [f.experiment_id for f in surprises] == ["big", "mid"]


def test_confidence_clamping() -> None:
    board = ForecastBoard()
    over = board.predict("over", "h", predicted=0.5, confidence=5.0)
    under = board.predict("under", "h", predicted=0.5, confidence=-3.0)
    assert over.confidence == 1.0
    assert under.confidence == 0.0


def test_predict_overwrites_open_forecast() -> None:
    board = ForecastBoard()
    board.predict("exp", "first guess", predicted=0.2, confidence=0.3)
    board.predict("exp", "revised guess", predicted=0.8, confidence=0.6)
    assert len(board.forecasts) == 1
    f = board.forecasts[0]
    assert f.hypothesis == "revised guess"
    assert f.predicted == 0.8


def test_summary_shape() -> None:
    board = ForecastBoard()
    board.predict("a", "h", predicted=0.5, confidence=0.6)
    board.predict("b", "h", predicted=0.9, confidence=0.8)
    board.resolve("a", 0.5)
    s = board.summary()
    assert set(s) == {
        "total",
        "resolved",
        "brier",
        "hit_rate",
        "mean_confidence",
        "n_surprises",
    }
    assert s["total"] == 2
    assert s["resolved"] == 1
    assert s["brier"] == 0.0
    assert s["hit_rate"] == 1.0
    assert math.isclose(s["mean_confidence"], 0.7)
    assert s["n_surprises"] == 0


def test_summary_empty_board() -> None:
    s = ForecastBoard().summary()
    assert s == {
        "total": 0,
        "resolved": 0,
        "brier": None,
        "hit_rate": None,
        "mean_confidence": None,
        "n_surprises": 0,
    }


def test_determinism() -> None:
    def build() -> dict[str, object]:
        board = ForecastBoard()
        for i in range(6):
            board.predict(f"e{i}", "h", predicted=0.1 * i, confidence=0.1 * i)
            board.resolve(f"e{i}", 0.1 * i + 0.3)
        return {
            "summary": board.summary(),
            "curve": board.calibration_curve(),
            "surprises": [f.experiment_id for f in board.surprises()],
        }

    assert build() == build()
