"""Calibration of the reference temperature scale.

Builds a lookup table from the colour-bar image and inverts pixel colours back
to degrees Celsius. Invertibility is asserted at construction: nearest-neighbour
lookup is only meaningful if distinct temperatures have distinct colours.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from . import config
from .errors import DataContractError
from .io import load_rgb


class ColorScale:
    """Bidirectional map between RGB and degrees Celsius."""

    def __init__(self, lut: np.ndarray, temps: np.ndarray) -> None:
        self.lut = lut
        self.temps = temps
        self._tree = cKDTree(lut)

    @classmethod
    def from_image(
        cls,
        path: Path | str,
        vmin: float = config.SCALE_VMIN,
        vmax: float = config.SCALE_VMAX,
        max_roundtrip_error: float = config.MAX_ROUNDTRIP_ERROR_C,
    ) -> ColorScale:
        """Detect the gradient bar in `path` and build the lookup table."""
        image = load_rgb(path)
        height, width, _ = image.shape

        # The bar is saturated or dark; surrounding margins and axis labels are not.
        coloured = (image.std(axis=2) > 8.0) | (image.mean(axis=2) < 200.0)

        rows = np.flatnonzero(coloured.sum(axis=1) > 0.5 * width)
        if rows.size == 0:
            raise DataContractError(f"no colour bar found in {path}")
        y0, y1 = int(rows.min()), int(rows.max())

        columns = np.flatnonzero(coloured[y0 : y1 + 1].sum(axis=0) > 0.8 * (y1 - y0))
        if columns.size < 2:
            raise DataContractError(f"colour bar in {path} is too narrow to calibrate")
        x0, x1 = int(columns.min()), int(columns.max())

        inset = max(1, (y1 - y0) // 20)  # avoid anti-aliased top and bottom edges
        lut = image[y0 + inset : y1 - inset, x0 : x1 + 1].mean(axis=0)
        temps = np.linspace(vmin, vmax, lut.shape[0])

        scale = cls(lut=lut, temps=temps)
        error = scale.max_roundtrip_error()
        if error > max_roundtrip_error:
            raise DataContractError(
                f"colour scale in {path} is not invertible: a bar colour maps to a "
                f"temperature {error:.0f} °C away from its true value "
                f"(limit {max_roundtrip_error:.0f} °C). Distinct temperatures must "
                f"have distinct colours."
            )
        return scale

    def to_celsius(self, rgb) -> float:
        """Map a single RGB triple to degrees Celsius."""
        _, index = self._tree.query(np.asarray(rgb, dtype=np.float64), workers=1)
        return float(self.temps[index])

    def to_map(self, image: np.ndarray) -> np.ndarray:
        """Map every pixel of an H×W×3 image to degrees Celsius."""
        flat = image.reshape(-1, 3)
        _, indices = self._tree.query(flat, workers=1)
        return self.temps[indices].reshape(image.shape[:2])

    def max_roundtrip_error(self) -> float:
        """Largest temperature error incurred by inverting the bar's own colours.

        For each bar sample, find its nearest *other* colour; if that neighbour
        belongs to a distant temperature, the scale is ambiguous there.
        """
        _, indices = self._tree.query(self.lut, k=2, workers=1)
        nearest_other = indices[:, 1]
        return float(np.abs(self.temps[nearest_other] - self.temps).max())
