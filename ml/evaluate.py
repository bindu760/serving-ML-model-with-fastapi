"""Evaluation metrics and forecasting baselines."""

import numpy as np


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Calculate MAE, RMSE, safe MAPE, sMAPE and residual standard deviation."""
    errors = predicted - actual
    nonzero = actual != 0
    denom = np.abs(actual) + np.abs(predicted)
    smape = (
        np.divide(2 * np.abs(errors), denom, out=np.zeros_like(errors, dtype=float), where=denom != 0).mean()
        * 100
    )
    return {
        "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mape": float((np.abs(errors[nonzero] / actual[nonzero]).mean() * 100) if nonzero.any() else 0.0),
        "smape": float(smape),
        "residual_std": float(np.std(errors, ddof=1)) if len(errors) > 1 else 0.0,
    }
