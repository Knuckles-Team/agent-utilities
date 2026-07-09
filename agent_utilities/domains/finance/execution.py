from agent_utilities.numeric import NDArray
from agent_utilities.numeric import xp as np


def calculate_kelly_fraction(
    accuracy: float, win_loss_ratio: float, half_kelly: bool = True
) -> float:
    """
    Calculates the Kelly Criterion for optimal bet sizing.
    f* = (p * b - q) / b
    where p is accuracy, q is 1-p, and b is the win-to-loss ratio.
    """
    if win_loss_ratio <= 0:
        return 0.0

    p = accuracy
    q = 1.0 - p
    f_star = (p * win_loss_ratio - q) / win_loss_ratio

    # Kelly can be aggressive. Half-Kelly is industry standard.
    if half_kelly:
        f_star /= 2.0

    # Cap maximum risk exposure per trade to 2% (0.02)
    max_risk = 0.02
    return max(0.0, min(f_star, max_risk))


def check_regime_shift(
    historical_predictions: NDArray,
    recent_predictions: NDArray,
    threshold: float = 0.1,
) -> bool:
    """
    Uses the Kolmogorov-Smirnov statistic to detect if the distribution of
    recent predictions has significantly drifted from historical validation.
    Returns True if a regime shift is detected.
    """
    if len(historical_predictions) == 0 or len(recent_predictions) == 0:
        return False

    statistic, p_value = np.ks_2samp(historical_predictions, recent_predictions)
    # If the KS statistic is greater than the threshold (0.1), the distributions
    # are meaningfully different, indicating a regime shift.
    return statistic > threshold
