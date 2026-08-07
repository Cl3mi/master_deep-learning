"""Attribution methods.

Occlusion is preferred over gradient-based saliency because it is model-agnostic
and assumes nothing about differentiability, so the same figure stays comparable
across the convolutional and feature-based models.
"""

from __future__ import annotations

import numpy as np


def permutation_importance(
    model,
    features: np.ndarray,
    target: np.ndarray,
    names: tuple[str, ...],
    repeats: int = 10,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Increase in mean absolute error when each feature is shuffled.

    A feature the model does not consult yields zero: shuffling it changes
    nothing. Averaging over repeats keeps the estimate from hinging on one
    unlucky permutation.
    """
    rng = rng or np.random.default_rng(0)
    features = np.asarray(features, dtype=float)
    target = np.asarray(target, dtype=float)

    baseline = float(np.abs(np.asarray(model.predict(features)).ravel() - target).mean())

    importance: dict[str, float] = {}
    for index, name in enumerate(names):
        losses = []
        for _ in range(repeats):
            shuffled = features.copy()
            shuffled[:, index] = rng.permutation(shuffled[:, index])
            predicted = np.asarray(model.predict(shuffled)).ravel()
            losses.append(float(np.abs(predicted - target).mean()))
        importance[name] = float(np.mean(losses) - baseline)
    return importance


def occlusion_map(
    model, image: np.ndarray, patch: int = 16, stride: int = 8, fill: float = 0.0
) -> np.ndarray:
    """Absolute change in prediction when each patch is masked out.

    High values mark regions the model actually relies on; a model that ignores
    the image entirely produces a flat map of zeros.
    """
    image = np.asarray(image, dtype=float)
    height, width = image.shape[:2]
    base = float(np.asarray(model.predict(image[None, ...])).ravel()[0])

    tops = list(range(0, max(height - patch + 1, 1), stride))
    lefts = list(range(0, max(width - patch + 1, 1), stride))
    heatmap = np.zeros((len(tops), len(lefts)))

    for row_index, top in enumerate(tops):
        for column_index, left in enumerate(lefts):
            occluded = image.copy()
            occluded[top : top + patch, left : left + patch] = fill
            prediction = float(np.asarray(model.predict(occluded[None, ...])).ravel()[0])
            heatmap[row_index, column_index] = abs(prediction - base)
    return heatmap
