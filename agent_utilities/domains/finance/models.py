import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as e:
    raise ImportError(
        "Finance extra dependencies missing. Please install agent-utilities[finance]"
    ) from e


class TradingLSTM(nn.Module):
    """
    LSTM architecture for sequential market data processing.
    Learns conditional expectation functions for topological market regimes.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        # Take the output from the final time step
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return self.sigmoid(out)


def prepare_sequences(
    features: np.ndarray, target: np.ndarray, lookback: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts 2D tabular features into 3D sequences for LSTM consumption.
    Format: (batch_size, sequence_length, features)
    """
    X, y = [], []
    for i in range(len(features) - lookback):
        X.append(features[i : (i + lookback)])
        y.append(target[i + lookback])
    return np.array(X), np.array(y)
