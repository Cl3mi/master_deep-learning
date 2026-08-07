"""Regression metrics reported in hours.

`skill` expresses error relative to the trivial mean predictor: 1 is perfect,
0 means the model is no better than predicting the average, negative means worse.
It exists so a reader can judge a raw MAE without knowing the label spread.
"""

from __future__ import annotations

import numpy as np


def skill_score(model_mae: float, baseline_mae: float) -> float:
    """Fraction of the baseline's error removed by the model."""
    if baseline_mae <= 0:
        return float("nan")
    return 1.0 - model_mae / baseline_mae


def evaluate_predictions(
    truth: np.ndarray, predicted: np.ndarray, baseline_mae: float | None = None
) -> dict[str, float]:
    """Return MAE, RMSE, R2 and - when a baseline is supplied - the skill score."""
    truth = np.asarray(truth, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    residual = predicted - truth

    mae = float(np.abs(residual).mean())
    rmse = float(np.sqrt((residual**2).mean()))
    total_variance = float(((truth - truth.mean()) ** 2).sum())
    r2 = 1.0 - float((residual**2).sum()) / total_variance if total_variance > 0 else float("nan")

    result = {"mae": mae, "rmse": rmse, "r2": r2}
    if baseline_mae is not None:
        result["skill"] = skill_score(mae, baseline_mae)
    return result
