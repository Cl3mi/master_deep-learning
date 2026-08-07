"""Physical feature extraction.

Region colours are summarised by their *median*, which is robust to the residual
JPEG ringing that survives erosion. Pixel-count features are computed for the
audit but excluded from modelling: geometry never varies, so their small
fluctuation reflects how the white threshold treats anti-aliased edges of dark
versus bright regions - an artefact of the encoder, not of the engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .colorscale import ColorScale
from .segment import regions

#: Features used for modelling. Everything else is diagnostic only.
PHYSICAL_FEATURES = ("cone_c", "body_c", "pylon_c")

#: Diagnostic features retained in the table but never fed to a model.
DIAGNOSTIC_FEATURES = ("cone_px", "body_px", "pylon_px")


def temperature_histogram(temperatures: np.ndarray, bins: int = 16) -> np.ndarray:
    """Normalised histogram of engine-pixel temperatures over the scale range."""
    counts, _ = np.histogram(
        temperatures, bins=bins, range=(config.SCALE_VMIN, config.SCALE_VMAX)
    )
    total = counts.sum()
    return counts / total if total else counts.astype(float)


def extract(
    image: np.ndarray,
    scale: ColorScale,
    min_component_px: int = config.MIN_COMPONENT_PX,
    erode_iterations: int = config.ERODE_ITERATIONS,
    histogram_bins: int = 16,
) -> dict[str, float]:
    """Return the full feature dictionary for one engine image."""
    masks = regions(
        image, min_component_px=min_component_px, erode_iterations=erode_iterations
    )

    features: dict[str, float] = {}
    for name, mask in masks.items():
        features[f"{name}_c"] = scale.to_celsius(np.median(image[mask], axis=0))
        features[f"{name}_px"] = float(mask.sum())

    region_temps = [features[f"{name}_c"] for name in masks]
    features["total_c"] = float(sum(region_temps))
    features["max_c"] = float(max(region_temps))
    features["mean_c"] = float(np.mean(region_temps))
    features["gradient_c"] = float(region_temps[0] - region_temps[-1])

    combined = np.logical_or.reduce(list(masks.values()))
    engine_temperatures = scale.to_map(image)[combined]
    features["hot_fraction"] = float((engine_temperatures > config.HOT_THRESHOLD_C).mean())
    for index, value in enumerate(temperature_histogram(engine_temperatures, histogram_bins)):
        features[f"hist_{index:02d}"] = float(value)

    return features


def degenerate_columns(frame: pd.DataFrame, relative_tolerance: float = 1e-9) -> list[str]:
    """Columns whose spread is negligible relative to their magnitude.

    Such columns cannot carry label information and are excluded from modelling.
    """
    degenerate = []
    for name in frame.columns:
        values = frame[name].to_numpy(dtype=float)
        spread = values.max() - values.min()
        magnitude = max(abs(values).max(), 1e-12)
        if spread / magnitude <= relative_tolerance:
            degenerate.append(name)
    return degenerate
