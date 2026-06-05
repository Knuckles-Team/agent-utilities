import numpy as np

try:
    import torch
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    raise ImportError(
        "Finance extra dependencies missing. Please install agent-utilities[finance]"
    ) from e

from .models import TradingLSTM


def evaluate_trading_signal(
    predictions: np.ndarray, actuals: np.ndarray, returns: np.ndarray
) -> tuple[float, float, float]:
    """

    CONCEPT:KG-2.6
        Evaluates signal against directional accuracy, ROC-AUC, Sharpe, and Drawdown.
    """
    pred_direction = (predictions > 0.5).astype(int)

    accuracy = accuracy_score(actuals, pred_direction)
    # Handle single class predictions (e.g. all 1s) for ROC-AUC gracefully
    try:
        roc_auc_score(actuals, predictions)
    except ValueError:
        pass

    signal = np.where(pred_direction == 1, 1, -1)

    # Trim returns to match the length of predictions (due to sequence lookback)
    returns_trimmed = returns[-len(predictions) :]
    strategy_returns = signal * returns_trimmed

    # Calculate Sharpe Ratio (annualized, assuming daily)
    std_dev = strategy_returns.std()
    if std_dev == 0:
        sharpe = 0.0
    else:
        sharpe = (strategy_returns.mean() / std_dev) * np.sqrt(252)

    # Calculate Max Drawdown
    cumulative = (1 + strategy_returns).cumprod()
    rolling_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    return accuracy, sharpe, max_drawdown


def walk_forward_validation(
    features: np.ndarray,
    target: np.ndarray,
    lookback: int,
    train_size: int,
    test_size: int,
    step: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulates live deployment by rolling a training window forward through time.
    Strictly prevents lookahead bias.
    """
    all_predictions = []
    all_actuals: list[float] = []

    for start in range(0, len(features) - train_size - test_size, step):
        train_end = start + train_size
        test_end = train_end + test_size

        X_train = features[start:train_end]
        y_train = target[start:train_end]
        X_test = features[train_end:test_end]
        y_test = target[train_end:test_end]

        # Scale features
        scaler = StandardScaler()
        # reshape for scaling (flattening time series)
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])

        X_train_scaled = scaler.fit_transform(X_train_reshaped)
        X_test_scaled = scaler.transform(X_test_reshaped)

        # rebuild 3D sequence
        from .models import prepare_sequences

        X_train_seq, y_train_seq = prepare_sequences(X_train_scaled, y_train, lookback)
        X_test_seq, y_test_seq = prepare_sequences(X_test_scaled, y_test, lookback)

        if len(X_train_seq) == 0 or len(X_test_seq) == 0:
            continue

        model = TradingLSTM(input_size=X_train.shape[-1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.BCELoss()

        # Simple training loop for the walk-forward step
        model.train()
        for epoch in range(10):  # Minimal epochs for WF demonstration
            optimizer.zero_grad()
            preds = model(torch.FloatTensor(X_train_seq)).squeeze()
            if preds.dim() == 0:
                preds = preds.unsqueeze(0)
            loss = criterion(preds, torch.FloatTensor(y_train_seq))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(torch.FloatTensor(X_test_seq)).squeeze()
            if preds.dim() == 0:
                preds = preds.unsqueeze(0)

        all_predictions.extend(preds.numpy())
        all_actuals.extend(y_test_seq)

    return np.array(all_predictions), np.array(all_actuals)
