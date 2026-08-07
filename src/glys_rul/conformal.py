"""Distribution-free prediction intervals.

With eleven samples, a point estimate overstates what the model knows. The
jackknife+ residual quantile turns out-of-fold errors into an interval with a
finite-sample coverage guarantee, requiring no assumption about the shape of the
error distribution — which matters here, because six residuals are far too few
to justify a normal approximation.
"""

from __future__ import annotations

import numpy as np

from . import config


def residual_quantile(residuals: np.ndarray, alpha: float = config.CONFORMAL_ALPHA) -> float:
    """The conformal quantile of absolute out-of-fold residuals.

    The (1 - alpha)(n + 1) / n level is the finite-sample correction that makes
    coverage valid rather than merely asymptotic. With no residuals to calibrate
    on, the honest width is infinite.
    """
    absolute = np.abs(np.asarray(residuals, dtype=float))
    count = len(absolute)
    if count == 0:
        return float("inf")
    level = min(1.0, np.ceil((count + 1) * (1.0 - alpha)) / count)
    return float(np.quantile(absolute, level, method="higher"))


def jackknife_plus_interval(
    prediction: float, residuals: np.ndarray, alpha: float = config.CONFORMAL_ALPHA
) -> tuple[float, float]:
    """Symmetric interval around `prediction` at nominal coverage 1 - alpha."""
    width = residual_quantile(residuals, alpha)
    return prediction - width, prediction + width


def interval_table(
    predictions: np.ndarray, residuals: np.ndarray, alpha: float = config.CONFORMAL_ALPHA
) -> np.ndarray:
    """Return an (n, 2) array of lower and upper bounds for a vector of predictions."""
    width = residual_quantile(residuals, alpha)
    predictions = np.asarray(predictions, dtype=float)
    return np.stack([predictions - width, predictions + width], axis=1)


def empirical_coverage(truth: np.ndarray, intervals: np.ndarray) -> float:
    """Fraction of true labels falling inside their interval."""
    truth = np.asarray(truth, dtype=float)
    inside = (truth >= intervals[:, 0]) & (truth <= intervals[:, 1])
    return float(inside.mean())
