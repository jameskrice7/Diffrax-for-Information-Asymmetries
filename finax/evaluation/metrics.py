"""Evaluation metrics for Finax."""

from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    diff = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(diff**2)))


def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """Compute the Sharpe ratio of a return series."""
    excess = np.asarray(returns) - risk_free
    return float(np.mean(excess) / np.std(excess, ddof=1))
