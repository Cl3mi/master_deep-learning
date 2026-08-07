"""Actionable maintenance thresholds.

The Glys asked how long an engine will last; what they can act on is how cool a
component must run to reach a target life. Because predicted life decreases
monotonically with temperature, the answer is found by scanning the scale range
from hot to cold and taking the first temperature that meets the target — the
*highest* such temperature, so the advice is not needlessly conservative.
"""

from __future__ import annotations

import numpy as np

from . import config


def required_temperature(
    model,
    current_features: np.ndarray,
    region_index: int,
    target_rul: float,
    resolution: float = 5.0,
) -> float | None:
    """Highest temperature for `region_index` still reaching `target_rul` hours.

    Returns None when the target is unreachable even at 0 degrees — the honest
    answer that cooling this component alone cannot save the engine.
    """
    current_features = np.asarray(current_features, dtype=float).ravel()
    candidates = np.arange(config.SCALE_VMAX, config.SCALE_VMIN - resolution, -resolution)
    candidates = candidates[candidates >= config.SCALE_VMIN]

    trials = np.tile(current_features, (len(candidates), 1))
    trials[:, region_index] = candidates
    predictions = np.asarray(model.predict(trials), dtype=float).ravel()

    reaching = np.flatnonzero(predictions >= target_rul)
    if reaching.size == 0:
        return None
    return float(candidates[reaching[0]])


def threshold_table(
    model, current_features: np.ndarray, region_names: tuple[str, ...], target_rul: float
) -> dict[str, float | None]:
    """Required temperature per region to reach `target_rul` hours."""
    return {
        name: required_temperature(model, current_features, index, target_rul)
        for index, name in enumerate(region_names)
    }
