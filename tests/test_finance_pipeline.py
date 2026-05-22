"""CONCEPT:KG-2.6"""

import numpy as np
import pandas as pd
import pytest

try:
    import torch

    from agent_utilities.domains.finance.evaluation import evaluate_trading_signal
    from agent_utilities.domains.finance.execution import (
        calculate_kelly_fraction,
        check_regime_shift,
    )
    from agent_utilities.domains.finance.features import (
        StationaryFeatureEngineer,
        check_stationarity,
    )
    from agent_utilities.domains.finance.models import TradingLSTM, prepare_sequences

    HAS_FINANCE = True
except ImportError:
    HAS_FINANCE = False

pytestmark = pytest.mark.skipif(not HAS_FINANCE, reason="Finance dependencies missing")


def generate_mock_market_data(n_samples: int = 100) -> pd.DataFrame:
    """Generates synthetic random walk OHLCV data."""
    np.random.seed(42)
    close = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, n_samples)))
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n_samples)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n_samples)))
    open_price = close * (1 + np.random.normal(0, 0.002, n_samples))
    volume = np.random.lognormal(10, 1, n_samples)

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


def test_trading_lstm_architecture():
    input_size = 5
    seq_length = 10
    batch_size = 32

    model = TradingLSTM(input_size=input_size, hidden_size=16, num_layers=1)

    # Mock input: (batch_size, seq_length, input_size)
    mock_input = torch.randn(batch_size, seq_length, input_size)

    output = model(mock_input)
    # Output should be (batch_size, 1) due to binary classification sigmoid
    assert output.shape == (batch_size, 1)
    # Sigmoid constraint
    assert torch.all((output >= 0) & (output <= 1))


def test_prepare_sequences():
    features = np.random.randn(50, 5)
    target = np.random.randint(0, 2, 50)
    lookback = 10

    X, y = prepare_sequences(features, target, lookback=lookback)

    assert X.shape == (40, 10, 5)
    assert y.shape == (40,)


def test_evaluation_metrics():
    # Mock predictions and actuals
    predictions = np.array([0.9, 0.1, 0.8, 0.2, 0.6])
    actuals = np.array([1, 0, 1, 0, 1])
    returns = np.array([0.01, -0.01, 0.02, -0.02, 0.005])

    accuracy, sharpe, max_dd = evaluate_trading_signal(predictions, actuals, returns)

    assert accuracy == 1.0  # Perfect prediction
    assert sharpe > 0  # Positive returns
    assert max_dd <= 0  # Drawdown is zero or negative


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
    hist = np.random.normal(0, 1, 100)
    recent = np.random.normal(0, 1, 100)

    # 0.1 threshold is quite sensitive, 0.2 or 0.3 for these small samples
    # but let's test a massive shift
    shifted = np.random.normal(5, 1, 100)

    assert not check_regime_shift(hist, recent, threshold=0.5)
    assert check_regime_shift(hist, shifted, threshold=0.1)
