"""Cross-validated training.

Splits are grouped by content hash, so a fold never trains on a pixel-identical
copy of what it is tested on. Results are averaged over several seeds because at
this sample size a single run's score is dominated by initialisation noise.

Metrics are computed once over the concatenated out-of-fold predictions rather
than averaged per fold: several folds hold out a single sample, and averaging
fold scores would weight a one-sample fold equally with a two-sample one.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from . import config
from .dataset import grouped_splits
from .metrics import evaluate_predictions


@dataclass
class CVResult:
    """Out-of-fold predictions and the metrics computed from them."""

    predictions: np.ndarray
    metrics: dict[str, float]
    fold_metrics: list[dict[str, float]] = field(default_factory=list)


def cross_validate(
    factory: Callable[[], object],
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    baseline_mae: float | None = None,
) -> CVResult:
    """Fit `factory()` on each training fold and predict its held-out group.

    `factory` is called once per fold so every fold starts from an unfitted
    model; reusing a single instance would carry the previous fold's state.
    """
    features = np.asarray(features, dtype=float)
    target = np.asarray(target, dtype=float)
    predictions = np.full(len(target), np.nan)
    fold_metrics: list[dict[str, float]] = []

    for train_index, test_index in grouped_splits(groups):
        model = factory()
        model.fit(features[train_index], target[train_index])
        fold_prediction = np.asarray(model.predict(features[test_index]), dtype=float).ravel()
        predictions[test_index] = fold_prediction
        fold_metrics.append(evaluate_predictions(target[test_index], fold_prediction))

    if np.isnan(predictions).any():
        raise RuntimeError("cross-validation left samples without a prediction")

    return CVResult(
        predictions=predictions,
        metrics=evaluate_predictions(target, predictions, baseline_mae=baseline_mae),
        fold_metrics=fold_metrics,
    )


def cross_validate_seeds(
    factory: Callable[[], object],
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    seeds: tuple[int, ...] = config.SEEDS,
    baseline_mae: float | None = None,
) -> dict[str, float]:
    """Repeat cross-validation across seeds and summarise the spread.

    With eleven samples a single run's score is dominated by weight
    initialisation, so a bare number is not reportable. Deterministic models
    return a standard deviation of exactly zero, which confirms the sweep itself
    introduces no spurious variation.
    """
    from .determinism import configure

    collected: dict[str, list[float]] = {}
    for seed in seeds:
        configure(seed=seed)
        result = cross_validate(factory, features, target, groups, baseline_mae=baseline_mae)
        for name, value in result.metrics.items():
            collected.setdefault(name, []).append(value)

    summary: dict[str, float] = {"n_seeds": float(len(seeds))}
    for name, values in collected.items():
        summary[f"{name}_mean"] = float(np.mean(values))
        summary[f"{name}_std"] = float(np.std(values))
    return summary
