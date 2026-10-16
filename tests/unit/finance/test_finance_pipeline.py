"""CONCEPT:AU-KG.research.research-pipeline-runner

Torch-free finance tests. The torch-backed TradingLSTM / prepare_sequences /
evaluate_trading_signal / walk_forward_validation assertions were re-homed to
data-science-mcp (``tests/test_trading_lstm.py``) so core stays torch-free —
see AGENTS.md "Dependency discipline".
"""

import math

import pandas as pd
import pytest

# The compiled epistemic_graph.numeric kernel must be built for these tests; skip the whole module cleanly when it isn't, rather than erroring out collection (CONCEPT:AU-KG.compute.numeric-kernel).
pytest.importorskip("epistemic_graph.numeric")

from agent_utilities.numeric import xp as np

try:
    from agent_utilities.domains.finance.execution import (
        calculate_kelly_fraction,
        check_regime_shift,
    )
    from agent_utilities.domains.finance.features import (
        StationaryFeatureEngineer,
    )

    HAS_FINANCE = True
except ImportError:
    HAS_FINANCE = False

pytestmark = pytest.mark.skipif(not HAS_FINANCE, reason="Finance dependencies missing")


def generate_mock_market_data(n_samples: int = 100) -> pd.DataFrame:
    """Generates synthetic random walk OHLCV data."""
    # The native kernel returns builtin lists; build the series with a seeded
    # generator and list comprehensions rather than array arithmetic.
    rng = np.random.default_rng(42)
    steps = rng.normal(0, 0.01, n_samples)
    close = [100.0 * math.exp(c) for c in np.cumsum(steps)]
    high_noise = rng.normal(0, 0.005, n_samples)
    low_noise = rng.normal(0, 0.005, n_samples)
    open_noise = rng.normal(0, 0.002, n_samples)
    high = [c * (1 + abs(e)) for c, e in zip(close, high_noise, strict=True)]
    low = [c * (1 - abs(e)) for c, e in zip(close, low_noise, strict=True)]
    open_price = [c * (1 + e) for c, e in zip(close, open_noise, strict=True)]
    volume = [math.exp(v) for v in rng.normal(10, 1, n_samples)]

    dates = pd.date_range(start="2020-01-01", periods=n_samples, freq="D")
    return pd.DataFrame(
        {
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )


def test_stationary_feature_engineer():
    df = generate_mock_market_data(200)
    engineer = StationaryFeatureEngineer(check_all=False)
    features, target = engineer.transform(df)

    assert not features.empty
    assert not target.empty
    assert len(features) == len(target)
    # Check that expected columns exist
    assert "return_1d" in features.columns
    assert "vol_ratio" in features.columns


def test_kelly_criterion():
    accuracy = 0.55
    win_loss_ratio = 1.2

    f_star = calculate_kelly_fraction(accuracy, win_loss_ratio, half_kelly=True)
    # Full kelly: (0.55 * 1.2 - 0.45) / 1.2 = (0.66 - 0.45) / 1.2 = 0.21 / 1.2 = 0.175
    # Half kelly: 0.0875
    # But max risk is capped at 0.02
    assert f_star == 0.02

    # Check bounds (losing strategy)
    f_star_loss = calculate_kelly_fraction(0.4, 0.8)
    assert f_star_loss == 0.0


def test_regime_shift_detection():
    # Identical distributions should not flag a regime shift
    rng = np.random.default_rng(7)
    hist = rng.normal(0, 1, 100)
    recent = rng.normal(0, 1, 100)

    # 0.1 threshold is quite sensitive, 0.2 or 0.3 for these small samples
    # but let's test a massive shift
    shifted = rng.normal(5, 1, 100)

    assert not check_regime_shift(hist, recent, threshold=0.5)
    assert check_regime_shift(hist, shifted, threshold=0.1)
